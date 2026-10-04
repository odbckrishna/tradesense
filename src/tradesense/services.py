"""Composition root: builds and owns every long-lived component.

Shared by the REST API, the WebSocket endpoint and the MCP tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Mapping
from typing import Any

from loguru import logger

from tradesense.backtest.engine import (
    BacktestBusyError,
    Backtester,
    BacktestNotFoundError,
    EventCallback,
    rule_file,
)
from tradesense.backtest.models import BacktestResult
from tradesense.backtest.trace import StrategyTrace
from tradesense.config import LLMConfig, RoleName, Settings, load_llm_config, secret_values
from tradesense.data.oanda_client import OandaClient
from tradesense.data.price_stream import PriceHub, PriceStream
from tradesense.engine.chart_store import ChartStore
from tradesense.engine.orchestrator import AnalysisResult, Orchestrator
from tradesense.llm.decision import DecisionEngine
from tradesense.llm.factory import ValidationIssue, build_providers, validate_providers
from tradesense.llm.router import LLMRouter
from tradesense.llm.selection import (
    LLMSelection,
    LLMSelectionError,
    apply_selection,
    build_catalog,
    load_selection,
    save_selection,
)
from tradesense.llm.usage import UsageTracker
from tradesense.logging_setup import configure_logging
from tradesense.strategies.base import StrategyInfo
from tradesense.strategies.reference import ReferenceLibrary
from tradesense.strategies.registry import ReloadReport, StrategyRegistry


class Services:
    def __init__(
        self,
        settings: Settings,
        *,
        oanda: OandaClient | None = None,
        env: Mapping[str, str] | None = None,
        price_stream: PriceStream | None = None,
        router: LLMRouter | None = None,
    ) -> None:
        self.settings = settings
        self.env = dict(os.environ) if env is None else dict(env)
        self.oanda = oanda or OandaClient.from_settings(settings)
        self.usage = UsageTracker()
        self.charts = ChartStore()
        self.llm_issues: list[ValidationIssue] = []
        # settings.llm is llm.yml with the UI's per-role model choices applied on top
        self.llm_base = settings.llm
        self.llm_selection = load_selection(settings.paths.llm_selection)
        settings.llm = apply_selection(self.llm_base, self.llm_selection)
        self.router = router or self._build_router()
        analysis = settings.app.analysis
        self.decisions = DecisionEngine(
            self.router,
            prompt_candles=analysis.prompt_candles,
            document_prompt_candles=analysis.document_prompt_candles,
        )
        library_cfg = settings.app.library
        self.library = ReferenceLibrary(
            settings.paths.library_dir,
            library_cfg,
            router=self.router,
            import_roots=[settings.resolve_path(p) for p in library_cfg.import_roots],
            reserved_names=lambda: self.registry.names(),
            on_change=self.reload_strategies,
        )
        self.registry = StrategyRegistry(
            rules_dir=settings.paths.rules_dir,
            strategies_config_path=settings.paths.strategies_config,
            reference_loader=self.library.strategies,
            vision_available=lambda: self.router.vision_available(),
        )
        self.orchestrator = Orchestrator(
            source=self.oanda,
            registry=self.registry,
            decisions=self.decisions,
            timeframes=settings.app.timeframes,
            analysis=analysis,
            charts=self.charts,
        )
        self.prices = PriceHub(price_stream or PriceStream(self.oanda))
        self.last_result: AnalysisResult | None = None
        self.backtester = Backtester(
            source=self.oanda,
            registry=self.registry,
            decisions=self.decisions,  # router swaps update it in place
            router=lambda: self.router,
            app=settings.app,
        )
        self.backtest_lock = asyncio.Lock()
        self.last_backtest: BacktestResult | None = None
        self._health_task: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------------ lifecycle

    def _build_router(self) -> LLMRouter:
        """Fail fast on structural problems; warn about unusable providers."""
        providers = build_providers(self.settings.llm, self.env)
        self.llm_issues = validate_providers(self.settings.llm, providers)
        for issue in self.llm_issues:
            log = logger.error if issue.level == "error" else logger.warning
            log("llm.yml: {}", issue.message)
        return LLMRouter(self.settings.llm, providers, self.usage)

    async def start(self, *, health_checks: bool = True) -> None:
        self.reload_strategies()
        if health_checks:
            self._health_task = asyncio.create_task(self._initial_health(), name="llm-health")
        if not self.oanda.configured:
            logger.warning("OANDA_API_KEY is not set: candles and live prices are unavailable")
        env = self.settings.oanda.oanda_environment
        logger.info("TradeSense ready (OANDA {} environment, signals only)", env)

    async def _initial_health(self) -> None:
        await self.router.check_health()
        self.reload_strategies()  # vision availability may have changed
        if not self.router.vision_available():
            logger.warning("No vision-capable LLM is reachable: image strategies are disabled")

    async def stop(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_task
        await self.prices.stop()
        await self.router.aclose()
        await self.oanda.aclose()

    # ------------------------------------------------------------------ reloads

    def reload_strategies(self) -> ReloadReport:
        return self.registry.reload()

    async def reload_llm(self) -> dict[str, Any]:
        """Re-read llm.yml without a restart. Raises ConfigError / LLMConfigError."""
        self.llm_base = load_llm_config(self.settings.paths.llm_config)
        self.llm_selection = load_selection(self.settings.paths.llm_selection)
        await self._apply_llm(apply_selection(self.llm_base, self.llm_selection))
        logger.info("LLM configuration reloaded")
        return self.llm_status()

    async def _apply_llm(self, config: LLMConfig) -> None:
        self.settings.llm = config
        configure_logging(None, secret_values(self.settings))
        new_router = self._build_router()
        old_router = self.router
        self.router = new_router
        self.decisions.router = new_router
        self.library.router = new_router
        await old_router.aclose()
        await new_router.check_health()
        self.reload_strategies()

    # ------------------------------------------------------------------ model picker

    async def llm_models(self) -> dict[str, Any]:
        """Selectable models (llm.yml providers + installed Ollama models) per role."""
        catalog = await build_catalog(self.router)
        return {
            "roles": {role: cfg.primary for role, cfg in self.settings.llm.roles.items()},
            "options": catalog.options,
        }

    async def select_llm(self, role: RoleName, option_id: str) -> dict[str, Any]:
        """Make ``option_id`` the role's primary (the rest of the chain stays as
        fallbacks) and remember the choice in llm_selection.json."""
        if role not in self.settings.llm.roles:
            raise LLMSelectionError(f"llm.yml defines no '{role}' role")
        catalog = await build_catalog(self.router)
        option = catalog.option(option_id)
        if option is None:
            raise LLMSelectionError(f"unknown model '{option_id}'")
        if role not in option["roles"]:
            raise LLMSelectionError(f"'{option_id}' cannot serve the {role} role")
        if not option["available"]:
            raise LLMSelectionError(f"'{option_id}' is not available: {option['note']}")

        roles = dict(self.llm_selection.roles)
        base_role = self.llm_base.roles.get(role)
        if base_role is not None and base_role.primary == option_id:
            roles.pop(role, None)  # back to llm.yml as written
        else:
            roles[role] = option_id
        known = {**self.llm_selection.providers, **catalog.discovered}
        self.llm_selection = LLMSelection(
            roles=roles,
            providers={n: known[n] for n in roles.values() if n in known},
        )
        save_selection(self.settings.paths.llm_selection, self.llm_selection)
        await self._apply_llm(apply_selection(self.llm_base, self.llm_selection))
        logger.info("LLM role '{}' now uses '{}'", role, option_id)
        return self.llm_status()

    # ------------------------------------------------------------------ backtest

    async def run_backtest(
        self,
        instrument: str,
        *,
        llm_check: bool = True,
        llm_rules: bool = True,
        on_event: EventCallback | None = None,
    ) -> BacktestResult:
        """One backtest at a time: they are CPU-heavy and share the LLM budget."""
        if self.backtest_lock.locked():
            raise BacktestBusyError("A backtest is already running; wait for it to finish.")
        async with self.backtest_lock:
            result = await self.backtester.run(
                instrument, llm_check=llm_check, llm_rules=llm_rules, on_event=on_event
            )
        self.last_backtest = result
        return result

    async def replay_strategy(self, instrument: str, timeframe: str, name: str) -> StrategyTrace:
        """One strategy's recent signals on one timeframe, for the chart. Shares the
        backtest lock: both keep every CPU core busy."""
        if self.backtest_lock.locked():
            raise BacktestBusyError("A backtest or replay is running; try again when it finishes.")
        async with self.backtest_lock:
            return await self.backtester.replay_strategy(instrument, timeframe, name)

    def save_backtest_candidate(self, run_id: str, timeframe: str, name: str) -> StrategyInfo:
        """Write a generated rule from the latest backtest to config/rules/ and load it."""
        result = self.last_backtest
        if result is None or result.run_id != run_id:
            raise BacktestNotFoundError("These backtest results expired: run the backtest again.")
        strategy_name, text = rule_file(result, timeframe, name)
        path = self.settings.paths.rules_dir / f"{strategy_name}.yml"
        if strategy_name in self.registry or path.exists():
            raise ValueError(f"A strategy named '{strategy_name}' already exists.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        report = self.reload_strategies()
        if strategy_name not in self.registry:
            path.unlink(missing_ok=True)
            raise ValueError(f"The saved rule did not load: {'; '.join(report.errors)[:300]}")
        logger.info("Saved backtest rule '{}' to {}", strategy_name, path.name)
        return self.registry.get(strategy_name).info()

    # ------------------------------------------------------------------ status

    def llm_status(self) -> dict[str, Any]:
        return {
            "roles": self.router.role_status(),
            "providers": self.router.provider_status(),
            "issues": [
                {"level": i.level, "message": i.message, "role": i.role, "provider": i.provider}
                for i in self.llm_issues
            ],
            "vision_available": self.router.vision_available(),
            "usage": self.usage.snapshot(),
        }
