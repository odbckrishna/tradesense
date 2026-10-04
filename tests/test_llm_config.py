"""llm.yml validation: structure (fail fast) and runtime usability (warn)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import REPO_ROOT
from tradesense.config import ConfigError, load_llm_config, parse_llm_config
from tradesense.llm.base import LLMConfigError
from tradesense.llm.factory import build_providers, validate_providers


def test_repo_llm_yml_builds_all_providers() -> None:
    config = load_llm_config(REPO_ROOT / "config" / "llm.yml")
    providers = build_providers(config, env={})
    assert set(providers) == set(config.providers)
    assert providers["azure_openai"].model == "your-deployment-name"
    assert providers["ollama_vision"].capabilities == {"text", "vision"}


def test_missing_env_var_warns_when_fallback_works() -> None:
    config = load_llm_config(REPO_ROOT / "config" / "llm.yml")
    providers = build_providers(config, env={"OPENAI_API_KEY": "sk-test"})
    issues = validate_providers(config, providers)
    errors = [i for i in issues if i.level == "error"]
    warnings = [i.message for i in issues if i.level == "warning"]
    assert not errors  # openai_main / ollama still serve every role
    assert any("ANTHROPIC_API_KEY is not set" in w for w in warnings)
    assert any("falling back to 'openai_main'" in w for w in warnings)


def test_role_without_usable_provider_is_an_error() -> None:
    config = parse_llm_config(
        {
            "roles": {"text": {"primary": "a"}, "vision": {"primary": "t"}},
            "providers": {
                "a": {"type": "anthropic", "model": "m", "api_key_env": "MISSING"},
                "t": {"type": "ollama", "model": "m", "capabilities": ["text"]},
            },
        }
    )
    issues = validate_providers(config, build_providers(config, env={}))
    errors = {i.role: i.message for i in issues if i.level == "error"}
    assert "no usable provider" in errors["text"]
    assert "no usable provider" in errors["vision"]  # text-only model cannot serve vision
    assert any("no 'vision' capability" in i.message for i in issues)


def test_unknown_provider_type_fails_fast() -> None:
    config = parse_llm_config(
        {
            "roles": {"text": {"primary": "x"}},
            "providers": {"x": {"type": "telepathy", "model": "m"}},
        }
    )
    with pytest.raises(LLMConfigError, match="unknown type 'telepathy'"):
        build_providers(config, env={})


def test_bad_role_reference_fails_fast(tmp_path: Path) -> None:
    path = tmp_path / "llm.yml"
    path.write_text(
        "roles:\n  text: {primary: ghost}\nproviders:\n  real: {type: ollama, model: m}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown provider 'ghost'"):
        load_llm_config(path)


def test_invalid_yaml_fails_fast(tmp_path: Path) -> None:
    path = tmp_path / "llm.yml"
    path.write_text("roles: {text: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_llm_config(path)


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ConfigError, match="temprature"):
        parse_llm_config(
            {
                "roles": {"text": {"primary": "x"}},
                "settings": {"temprature": 0.3},
                "providers": {"x": {"type": "ollama", "model": "m"}},
            }
        )
