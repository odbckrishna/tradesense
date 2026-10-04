"""Model picker: saved per-role choices and the catalog of selectable models."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from tests.fakes import FAST_SETTINGS
from tradesense.config import LLMConfig, ProviderConfig, RoleConfig, parse_llm_config
from tradesense.llm.factory import build_providers
from tradesense.llm.router import LLMRouter
from tradesense.llm.selection import (
    LLMSelection,
    OllamaModel,
    apply_selection,
    build_catalog,
    discover_ollama,
    load_selection,
    save_selection,
)

OLLAMA = "http://localhost:11434"


def base_config() -> LLMConfig:
    return parse_llm_config(
        {
            "roles": {
                "text": {"primary": "cloud", "fallbacks": ["local"]},
                "vision": {"primary": "cloud"},
                "extraction": {"primary": "cloud", "fallbacks": ["local"]},
            },
            "settings": FAST_SETTINGS.model_dump(),
            "providers": {
                "cloud": {
                    "type": "anthropic",
                    "model": "claude-sonnet-5-5",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "capabilities": ["text", "vision"],
                },
                "jev": {
                    "type": "openrouter_decisions",
                    "model": "typesafe/jev-1.13",
                    "api_key_env": "OPENROUTER_API_KEY",
                    "capabilities": ["decision"],
                },
                "local": {
                    "type": "ollama",
                    "base_url": OLLAMA,
                    "model": "nimble-cpu",
                    "timeout_seconds": 600,
                    "options": {"num_gpu": 0, "think": False},
                },
                "local_vl": {
                    "type": "ollama",
                    "base_url": OLLAMA,
                    "model": "qwen2.5vl:7b",
                    "capabilities": ["text", "vision"],
                },
            },
        }
    )


def router_for(config: LLMConfig, env: dict[str, str]) -> LLMRouter:
    return LLMRouter(config, build_providers(config, env))


def test_apply_selection_moves_choice_to_front() -> None:
    discovered = ProviderConfig(type="ollama", base_url=OLLAMA, model="nimble:latest")
    selection = LLMSelection(
        roles={"text": "jev", "extraction": "ollama_nimble_latest", "vision": "ghost"},
        providers={"ollama_nimble_latest": discovered},
    )
    config = apply_selection(base_config(), selection)
    assert config.roles["text"].chain == ["jev", "cloud", "local"]
    assert config.roles["extraction"].chain == ["ollama_nimble_latest", "cloud", "local"]
    assert config.roles["vision"].chain == ["cloud"]  # unknown provider ignored
    assert config.providers["ollama_nimble_latest"].model == "nimble:latest"
    # choosing the llm.yml primary leaves the chain as written
    same = apply_selection(base_config(), LLMSelection(roles={"text": "cloud"}))
    assert same.roles["text"] == RoleConfig(primary="cloud", fallbacks=["local"])


def test_selection_round_trip_and_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "llm_selection.json"
    assert load_selection(path) == LLMSelection()
    selection = LLMSelection(
        roles={"text": "x"},
        providers={"x": ProviderConfig(type="ollama", model="m", options={"num_gpu": 0})},
    )
    save_selection(path, selection)
    assert load_selection(path) == selection
    path.write_text("{not json", encoding="utf-8")
    assert load_selection(path) == LLMSelection()


@respx.mock
async def test_discover_ollama_reads_tags_and_capabilities() -> None:
    respx.get(f"{OLLAMA}/api/tags").mock(
        return_value=httpx.Response(
            200, json={"models": [{"model": "nimble:latest"}, {"model": "bge-m3:latest"}]}
        )
    )
    respx.post(f"{OLLAMA}/api/show").mock(
        side_effect=[
            httpx.Response(200, json={"capabilities": ["decision"]}),
            httpx.Response(500, text="boom"),
        ]
    )
    assert await discover_ollama(OLLAMA) == [
        OllamaModel("nimble:latest", ("decision",)),
        OllamaModel("bge-m3:latest", ()),
    ]
    respx.get("http://localhost:9/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    assert await discover_ollama("http://localhost:9") is None


async def test_catalog_groups_marks_and_discovers(monkeypatch: pytest.MonkeyPatch) -> None:
    async def installed(host: str) -> list[OllamaModel]:
        assert host == OLLAMA
        return [
            OllamaModel("nimble-cpu:latest", ("decision",)),
            OllamaModel("nimble:latest", ("decision",)),
            OllamaModel("llava:7b", ("completion", "vision")),
            OllamaModel("bge-m3:latest", ("embedding",)),
        ]

    monkeypatch.setattr("tradesense.llm.selection.discover_ollama", installed)
    catalog = await build_catalog(router_for(base_config(), {"OPENROUTER_API_KEY": "k"}))
    by_id = {o["id"]: o for o in catalog.options}

    assert by_id["cloud"]["group"] == "cloud"
    assert by_id["cloud"]["available"] is False  # no ANTHROPIC_API_KEY
    assert "ANTHROPIC_API_KEY" in by_id["cloud"]["note"]
    assert by_id["jev"]["roles"] == ["text"]  # decision-only: no vision, no extraction
    assert by_id["jev"]["available"] is True
    assert by_id["local"] == {
        "id": "local",
        "model": "nimble-cpu",
        "type": "ollama",
        "group": "local",
        "capabilities": ["text"],
        "roles": ["text", "extraction"],
        "available": True,  # nimble-cpu == nimble-cpu:latest
        "note": None,
        "healthy": None,
    }
    assert by_id["local_vl"]["note"] == "not installed in Ollama (ollama pull qwen2.5vl:7b)"

    # installed but not in llm.yml: offered, cloned from the llm.yml Ollama entry
    assert set(catalog.discovered) == {"ollama_nimble_latest", "ollama_llava_7b"}
    assert "ollama_bge_m3_latest" not in by_id  # embedding-only
    assert by_id["ollama_llava_7b"]["roles"] == ["text", "vision", "extraction"]
    clone = catalog.discovered["ollama_nimble_latest"]
    assert (clone.model, clone.timeout_seconds) == ("nimble:latest", 600)
    assert clone.options == {"num_gpu": 0, "think": False}


async def test_catalog_when_ollama_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unreachable(host: str) -> None:
        return None

    monkeypatch.setattr("tradesense.llm.selection.discover_ollama", unreachable)
    catalog = await build_catalog(router_for(base_config(), {}))
    by_id = {o["id"]: o for o in catalog.options}
    assert by_id["local_vl"]["available"] is True  # unknown, so not blocked
    assert catalog.discovered == {}
