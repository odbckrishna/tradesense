"""Runtime strategy registry: the shared pool every timeframe selects from.

Four kinds of strategies are registered:

* ``builtin``  - Python classes in ``strategies/builtin/`` using :func:`register_strategy`
* ``rule``     - YAML files in ``config/rules/`` (hot-reloaded)
* ``document`` / ``image`` - approved reference-strategy cards from the library

:meth:`StrategyRegistry.reload` rescans rules, ``strategies.yml`` and approved cards
without a server restart. Invalid rules or cards are logged and skipped, never fatal.
"""

from __future__ import annotations

import importlib
import pkgutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from tradesense.config import ConfigError, StrategiesConfig, load_strategies_config
from tradesense.strategies.base import (
    CATEGORIES,
    Strategy,
    StrategyCategory,
    StrategyInfo,
    StrategySource,
    UnknownStrategyError,
    resolve_params,
)
from tradesense.strategies.rule_engine import load_rules_dir

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Built-in strategy classes registered by :func:`register_strategy`.
BUILTIN_STRATEGIES: dict[str, type[Strategy]] = {}

_SOURCE_ORDER: dict[StrategySource, int] = {"builtin": 0, "rule": 1, "document": 2, "image": 3}

#: Loader for approved reference strategies: returns (strategies, error messages).
ReferenceLoader = Callable[[], tuple[list[Strategy], list[str]]]


def register_strategy[S: type[Strategy]](cls: S) -> S:
    """Class decorator registering a built-in strategy under ``cls.name``."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} must define a non-empty 'name'")
    existing = BUILTIN_STRATEGIES.get(name)
    if existing is not None and existing.__qualname__ != cls.__qualname__:
        raise ValueError(f"Duplicate built-in strategy name '{name}'")
    BUILTIN_STRATEGIES[name] = cls
    return cls


def discover_builtins() -> dict[str, type[Strategy]]:
    """Import every module in ``tradesense.strategies.builtin`` (registering its classes)."""
    package = importlib.import_module("tradesense.strategies.builtin")
    for module in pkgutil.iter_modules(package.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{package.__name__}.{module.name}")
    return dict(BUILTIN_STRATEGIES)


@dataclass
class ReloadReport:
    builtin: int = 0
    rules: int = 0
    reference: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.builtin + self.rules + self.reference

    def as_dict(self) -> dict[str, Any]:
        return {
            "builtin": self.builtin,
            "rules": self.rules,
            "reference": self.reference,
            "total": self.total,
            "errors": list(self.errors),
        }


class StrategyRegistry:
    """Holds every selectable strategy and resolves per-timeframe parameters."""

    def __init__(
        self,
        *,
        rules_dir: Path,
        strategies_config_path: Path,
        reference_loader: ReferenceLoader | None = None,
        vision_available: Callable[[], bool] | None = None,
    ) -> None:
        self.rules_dir = rules_dir
        self.strategies_config_path = strategies_config_path
        self.reference_loader = reference_loader
        self.vision_available = vision_available or (lambda: True)
        self._strategies: dict[str, Strategy] = {}
        self._config = StrategiesConfig()
        self._lock = threading.Lock()
        self.last_report = ReloadReport()

    # ------------------------------------------------------------------ loading

    def reload(self) -> ReloadReport:
        report = ReloadReport()
        strategies: dict[str, Strategy] = {}

        for name, cls in sorted(discover_builtins().items()):
            try:
                strategies[name] = cls()
                report.builtin += 1
            except Exception as exc:  # a broken class must not take the pool down
                report.errors.append(f"built-in '{name}': {exc}")

        rules, rule_errors = load_rules_dir(self.rules_dir)
        report.errors.extend(rule_errors)
        for rule in rules:
            if rule.name in strategies:
                report.errors.append(
                    f"rule '{rule.name}' ({rule.path.name if rule.path else '?'}) clashes with "
                    "an existing strategy name; skipped"
                )
                continue
            strategies[rule.name] = rule
            report.rules += 1

        if self.reference_loader is not None:
            try:
                references, ref_errors = self.reference_loader()
            except Exception as exc:
                references, ref_errors = [], [f"reference library: {exc}"]
            report.errors.extend(ref_errors)
            for ref in references:
                if ref.name in strategies:
                    report.errors.append(
                        f"reference strategy '{ref.name}' clashes with an existing name; skipped"
                    )
                    continue
                strategies[ref.name] = ref
                report.reference += 1

        try:
            config = load_strategies_config(self.strategies_config_path)
        except ConfigError as exc:
            report.errors.append(str(exc))
            config = self._config  # keep the previous parameters
        for name in config.strategies:
            if name not in strategies:
                logger.warning("strategies.yml configures unknown strategy '{}'", name)

        with self._lock:
            self._strategies = strategies
            self._config = config
            self.last_report = report

        for error in report.errors:
            logger.warning("Strategy skipped: {}", error)
        logger.info(
            "Strategy pool: {} built-in, {} rule, {} reference ({} errors)",
            report.builtin,
            report.rules,
            report.reference,
            len(report.errors),
        )
        return report

    # ------------------------------------------------------------------ queries

    def __contains__(self, name: object) -> bool:
        return name in self._strategies

    def __iter__(self) -> Iterator[Strategy]:
        return iter(list(self._strategies.values()))

    def __len__(self) -> int:
        return len(self._strategies)

    def names(self) -> list[str]:
        return sorted(self._strategies)

    def get(self, name: str) -> Strategy:
        try:
            return self._strategies[name]
        except KeyError:
            raise UnknownStrategyError(name) from None

    def availability(self, strategy: Strategy) -> tuple[bool, str | None]:
        if strategy.requires_vision and not self.vision_available():
            return False, "No vision-capable LLM provider is reachable"
        return True, None

    def list(
        self,
        category: StrategyCategory | str | None = None,
        source: StrategySource | str | None = None,
    ) -> list[StrategyInfo]:
        """All selectable strategies, sorted by category, kind and name."""
        if category is not None and category not in CATEGORIES:
            raise ValueError(f"Unknown category '{category}'. Use one of: {', '.join(CATEGORIES)}")
        infos = []
        for strategy in self._strategies.values():
            if category and strategy.category != category:
                continue
            if source and strategy.source != source:
                continue
            available, reason = self.availability(strategy)
            infos.append(strategy.info(available=available, reason=reason))
        infos.sort(key=lambda i: (CATEGORIES.index(i.category), _SOURCE_ORDER[i.source], i.name))
        return infos

    def params_for(self, name: str, timeframe: str) -> dict[str, Any]:
        """Parameters for ``name`` on ``timeframe``: override -> params -> class defaults."""
        strategy = self.get(name)
        return resolve_params(strategy, self._config.strategies.get(name), timeframe)
