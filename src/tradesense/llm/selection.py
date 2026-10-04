"""Model selection from the UI: per-role choices kept outside llm.yml.

``config/llm_selection.json`` (git-ignored) records which provider each role should try
first. Applying it moves that provider to the front of the role's chain and keeps the
rest of the llm.yml chain as fallbacks, so llm.yml (and its comments) is never rewritten.

The picker offers every provider in llm.yml, plus models installed in a local Ollama
that llm.yml does not name. Choosing one of those adds a provider cloned from the
llm.yml Ollama entry on the same host (same options, timeout and concurrency).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tradesense.config import Capability, LLMConfig, ProviderConfig, RoleConfig, RoleName
from tradesense.llm.providers.ollama_provider import DEFAULT_HOST
from tradesense.llm.router import LLMRouter

DISCOVERY_TIMEOUT_SECONDS = 3.0

#: A provider can serve a role if it has any of these capabilities.
ROLE_CAPABILITIES: dict[RoleName, set[Capability]] = {
    "text": {"text", "vision", "decision"},
    "vision": {"vision"},
    "extraction": {"text", "vision"},
}


class LLMSelectionError(ValueError):
    """The requested model cannot be selected for that role (HTTP 400)."""


class LLMSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roles: dict[RoleName, str] = Field(default_factory=dict)
    #: Ollama models chosen in the UI that llm.yml does not define.
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)


def load_selection(path: Path) -> LLMSelection:
    """The saved selection; a missing or corrupt file means "use llm.yml as written"."""
    if not path.is_file():
        return LLMSelection()
    try:
        return LLMSelection.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        logger.warning("Ignoring {}: {}", path.name, exc)
        return LLMSelection()


def save_selection(path: Path, selection: LLMSelection) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(selection.model_dump_json(indent=2, exclude_defaults=True), encoding="utf-8")
    tmp.replace(path)


def apply_selection(base: LLMConfig, selection: LLMSelection) -> LLMConfig:
    """``base`` (llm.yml) with each selected provider moved to the front of its role."""
    providers = {**selection.providers, **base.providers}  # llm.yml wins on a name clash
    roles = dict(base.roles)
    for role, name in selection.roles.items():
        if role not in roles or name not in providers:
            logger.warning("llm_selection.json: ignoring '{}' for role '{}'", name, role)
            continue
        fallbacks = [n for n in roles[role].chain if n != name]
        roles[role] = RoleConfig(primary=name, fallbacks=fallbacks)
    return base.model_copy(update={"roles": roles, "providers": providers})


# --------------------------------------------------------------------------- catalog


@dataclass(frozen=True)
class OllamaModel:
    name: str
    capabilities: tuple[str, ...]  # as Ollama reports them: completion, vision, ...


@dataclass
class Catalog:
    options: list[dict[str, Any]]
    #: Provider configs for the installed-but-unconfigured Ollama models, by option id.
    discovered: dict[str, ProviderConfig]

    def option(self, option_id: str) -> dict[str, Any] | None:
        return next((o for o in self.options if o["id"] == option_id), None)


def _host(config: ProviderConfig) -> str:
    return (config.base_url or DEFAULT_HOST).rstrip("/")


def _tagged(model: str) -> str:
    """Ollama's canonical name: ``llama3`` is ``llama3:latest``."""
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


def _capabilities(ollama_caps: Iterable[str]) -> list[Capability] | None:
    """App capabilities for an installed model; None for embedding-only models."""
    caps = set(ollama_caps)
    if caps and caps <= {"embedding"}:
        return None
    return ["text", "vision"] if "vision" in caps else ["text"]


async def discover_ollama(host: str) -> list[OllamaModel] | None:
    """Models installed on an Ollama server, or None if it cannot be reached."""
    try:
        async with httpx.AsyncClient(base_url=host, timeout=DISCOVERY_TIMEOUT_SECONDS) as client:
            tags = await client.get("/api/tags")
            tags.raise_for_status()
            names = [m["model"] for m in tags.json().get("models", []) if m.get("model")]
            shows = await asyncio.gather(
                *(client.post("/api/show", json={"model": n}) for n in names),
                return_exceptions=True,
            )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.debug("Ollama at {} not reachable for model discovery: {}", host, exc)
        return None
    models = []
    for name, show in zip(names, shows, strict=True):
        caps: list[str] = []
        if isinstance(show, httpx.Response) and show.is_success:
            with contextlib.suppress(ValueError, AttributeError):
                caps = list(show.json().get("capabilities") or [])
        models.append(OllamaModel(name, tuple(caps)))
    return models


def _option(
    option_id: str,
    model: str,
    provider_type: str,
    *,
    local: bool,
    capabilities: Iterable[str],
    roles: Iterable[RoleName],
    note: str | None = None,
    healthy: bool | None = None,
) -> dict[str, Any]:
    caps = set(capabilities)
    return {
        "id": option_id,
        "model": model,
        "type": provider_type,
        "group": "local" if local else "cloud",
        "capabilities": sorted(caps),
        "roles": [r for r in roles if caps & ROLE_CAPABILITIES[r]],
        "available": note is None,
        "note": note,
        "healthy": healthy,
    }


async def build_catalog(router: LLMRouter) -> Catalog:
    """Every selectable model: llm.yml providers, then unconfigured Ollama models."""
    config = router.config
    roles = list(config.roles)
    templates: dict[str, ProviderConfig] = {}  # Ollama host -> first llm.yml entry on it
    for provider_config in config.providers.values():
        if provider_config.type == "ollama":
            templates.setdefault(_host(provider_config), provider_config)
    hosts = list(templates)
    installed = dict(zip(hosts, await asyncio.gather(*map(discover_ollama, hosts)), strict=True))

    options: list[dict[str, Any]] = []
    configured: set[tuple[str, str]] = set()
    for name, provider in router.providers.items():
        note = provider.missing_config_reason()
        if provider.type_name == "ollama":
            host, model = _host(provider.config), _tagged(provider.model)
            configured.add((host, model))
            models = installed.get(host)
            if note is None and models is not None and model not in {m.name for m in models}:
                note = f"not installed in Ollama (ollama pull {provider.model})"
        options.append(
            _option(
                name,
                provider.model,
                provider.type_name,
                local=provider.is_local,
                capabilities=provider.capabilities,
                roles=roles,
                note=note,
                healthy=router.state[name].healthy,
            )
        )

    discovered: dict[str, ProviderConfig] = {}
    taken = set(config.providers)
    for host, models in installed.items():
        for found in models or []:
            caps = _capabilities(found.capabilities)
            if caps is None or (host, found.name) in configured:
                continue
            base = "ollama_" + re.sub(r"[^a-z0-9]+", "_", found.name.lower()).strip("_")
            name, n = base, 2
            while name in taken:
                name, n = f"{base}_{n}", n + 1
            taken.add(name)
            discovered[name] = templates[host].model_copy(
                update={"model": found.name, "capabilities": caps}
            )
            options.append(
                _option(name, found.name, "ollama", local=True, capabilities=caps, roles=roles)
            )
    return Catalog(options, discovered)
