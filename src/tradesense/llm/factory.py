"""Builds providers from ``llm.yml`` via a type registry.

Adding a provider type = one class decorated with ``@register_provider("name")`` in
``tradesense/llm/providers/`` + one YAML entry. Nothing else changes.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from tradesense.config import LLMConfig
from tradesense.llm.base import LLMConfigError, LLMProvider

PROVIDER_TYPES: dict[str, type[LLMProvider]] = {}


def register_provider[P: type[LLMProvider]](type_name: str) -> Callable[[P], P]:
    """Class decorator registering an adapter under ``type_name`` (the YAML ``type``)."""

    def decorator(cls: P) -> P:
        existing = PROVIDER_TYPES.get(type_name)
        if existing is not None and existing.__qualname__ != cls.__qualname__:
            raise ValueError(f"Provider type '{type_name}' is already registered")
        cls.type_name = type_name
        PROVIDER_TYPES[type_name] = cls
        return cls

    return decorator


def provider_types() -> dict[str, type[LLMProvider]]:
    """All registered adapters (imports every module in ``tradesense.llm.providers``)."""
    package = importlib.import_module("tradesense.llm.providers")
    for module in pkgutil.iter_modules(package.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{package.__name__}.{module.name}")
    return dict(PROVIDER_TYPES)


def build_providers(
    config: LLMConfig, env: Mapping[str, str] | None = None
) -> dict[str, LLMProvider]:
    """Instantiate every provider in ``config``. Unknown ``type`` values fail fast."""
    environment = dict(os.environ) if env is None else dict(env)
    types = provider_types()
    providers: dict[str, LLMProvider] = {}
    for name, provider_config in config.providers.items():
        cls = types.get(provider_config.type)
        if cls is None:
            raise LLMConfigError(
                f"llm.yml: provider '{name}' has unknown type '{provider_config.type}'. "
                f"Known types: {', '.join(sorted(types))}"
            )
        try:
            providers[name] = cls(name, provider_config, config.settings, environment)
        except ValueError as exc:
            raise LLMConfigError(f"llm.yml: provider '{name}': {exc}") from exc
    return providers


@dataclass(frozen=True)
class ValidationIssue:
    level: Literal["error", "warning"]
    message: str
    role: str | None = None
    provider: str | None = None


def validate_providers(
    config: LLMConfig, providers: Mapping[str, LLMProvider]
) -> list[ValidationIssue]:
    """Check every provider used by a role is usable (API key present, capabilities).

    A role whose chain still has a usable provider only produces warnings; a role with
    no usable provider at all produces an error-level issue (the app keeps running and
    reports it in the UI instead of crashing).
    """
    issues: list[ValidationIssue] = []
    for role in config.roles:
        chain = config.role_chain(role)
        usable = []
        for name in chain:
            provider = providers[name]
            reason = provider.missing_config_reason()
            if reason:
                issues.append(
                    ValidationIssue("warning", f"provider '{name}': {reason}", role, name)
                )
                continue
            if role == "vision" and "vision" not in provider.capabilities:
                issues.append(
                    ValidationIssue(
                        "warning",
                        f"provider '{name}' has no 'vision' capability and will be skipped "
                        "for image requests",
                        role,
                        name,
                    )
                )
                continue
            if role == "extraction" and not provider.generates_text:
                issues.append(
                    ValidationIssue(
                        "warning",
                        f"provider '{name}' only answers trading-signal decisions and will be "
                        "skipped for strategy extraction",
                        role,
                        name,
                    )
                )
                continue
            usable.append(name)
        if not usable:
            issues.append(
                ValidationIssue(
                    "error", f"role '{role}' has no usable provider ({', '.join(chain)})", role
                )
            )
        elif usable[0] != chain[0]:
            issues.append(
                ValidationIssue(
                    "warning",
                    f"role '{role}' primary '{chain[0]}' is unusable; falling back to "
                    f"'{usable[0]}'",
                    role,
                )
            )
    return issues
