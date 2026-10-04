from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.helpers import REPO_ROOT
from tradesense.strategies.base import (
    Strategy,
    StrategyContext,
    StrategyFeatures,
    UnknownStrategyError,
)
from tradesense.strategies.registry import StrategyRegistry


class FakeImageStrategy(Strategy):
    name = "double_bottom_ref"
    display_name = "Double Bottom"
    description = "reference chart"
    source = "image"
    category = "pattern"
    min_candles = 50
    requires_vision = True

    async def analyze(
        self, df: pd.DataFrame, params: dict[str, Any], timeframe: str, context: StrategyContext
    ) -> StrategyFeatures:  # pragma: no cover - not executed here
        raise NotImplementedError


@pytest.fixture
def rules_dir(tmp_path: Path) -> Path:
    target = tmp_path / "rules"
    target.mkdir()
    shutil.copy(REPO_ROOT / "config" / "rules" / "rsi_ema_combo.yml", target)
    return target


def make_registry(rules_dir: Path, **kwargs: Any) -> StrategyRegistry:
    registry = StrategyRegistry(
        rules_dir=rules_dir,
        strategies_config_path=REPO_ROOT / "config" / "strategies.yml",
        **kwargs,
    )
    registry.reload()
    return registry


def test_list_contains_all_kinds_sorted(rules_dir: Path) -> None:
    registry = make_registry(rules_dir, reference_loader=lambda: ([FakeImageStrategy()], []))
    infos = registry.list()
    assert len(infos) == 14
    names = [i.name for i in infos]
    assert {"ema_crossover", "rsi_ema_combo", "double_bottom_ref"} <= set(names)
    categories = [i.category for i in infos]
    order = ["trend", "momentum", "mean_reversion", "volatility", "levels", "pattern"]
    assert categories == sorted(categories, key=order.index)
    combo = next(i for i in infos if i.name == "rsi_ema_combo")
    assert combo.source == "rule"
    assert combo.min_candles == 60
    assert combo.default_params["rsi.period"] == 14
    ema = next(i for i in infos if i.name == "ema_crossover")
    assert ema.default_params == {"fast": 9, "slow": 21, "lookback": 20}


def test_filter_by_category_and_source(rules_dir: Path) -> None:
    registry = make_registry(rules_dir)
    trend = registry.list(category="trend")
    assert {i.name for i in trend} == {"ema_crossover", "supertrend", "adx_trend", "ichimoku"}
    assert [i.name for i in registry.list(source="rule")] == ["rsi_ema_combo"]
    with pytest.raises(ValueError, match="Unknown category"):
        registry.list(category="astrology")


def test_reload_picks_up_new_rules_without_restart(rules_dir: Path) -> None:
    registry = make_registry(rules_dir)
    assert "breakout_rule" not in registry
    (rules_dir / "breakout_rule.yml").write_text(
        "name: breakout_rule\ndescription: close above 20-high\ncategory: volatility\n"
        "indicators: [{id: vol, type: volume, period: 20}]\n"
        "rules: {buy: 'close > highest(high, 20)', sell: 'close < lowest(low, 20)'}\n",
        encoding="utf-8",
    )
    (rules_dir / "broken.yml").write_text("::: not yaml :::\n- [", encoding="utf-8")
    report = registry.reload()
    assert "breakout_rule" in registry
    assert report.rules == 2
    assert report.builtin == 12
    assert len(report.errors) == 1
    assert "broken.yml" in report.errors[0]


def test_rule_name_clash_with_builtin_is_skipped(rules_dir: Path) -> None:
    (rules_dir / "clash.yml").write_text(
        "name: ema_crossover\ndescription: x\ncategory: trend\n"
        "indicators: [{id: e, type: ema}]\nrules: {buy: 'e > 1', sell: 'e < 1'}\n",
        encoding="utf-8",
    )
    registry = make_registry(rules_dir)
    assert registry.get("ema_crossover").source == "builtin"
    assert any("clashes" in e for e in registry.last_report.errors)


def test_unknown_strategy(rules_dir: Path) -> None:
    registry = make_registry(rules_dir)
    with pytest.raises(UnknownStrategyError, match="Unknown strategy 'nope'"):
        registry.get("nope")


def test_image_strategies_disabled_without_vision(rules_dir: Path) -> None:
    vision = {"ok": False}
    registry = make_registry(
        rules_dir,
        reference_loader=lambda: ([FakeImageStrategy()], []),
        vision_available=lambda: vision["ok"],
    )
    info = next(i for i in registry.list() if i.name == "double_bottom_ref")
    assert not info.available
    assert "vision" in (info.unavailable_reason or "")
    vision["ok"] = True
    info = next(i for i in registry.list() if i.name == "double_bottom_ref")
    assert info.available


def test_reference_loader_errors_are_not_fatal(rules_dir: Path) -> None:
    def broken_loader() -> tuple[list[Strategy], list[str]]:
        raise RuntimeError("library unreadable")

    registry = make_registry(rules_dir, reference_loader=broken_loader)
    assert len(registry) == 13
    assert any("library unreadable" in e for e in registry.last_report.errors)


def test_params_for_uses_strategies_yml(rules_dir: Path) -> None:
    registry = make_registry(rules_dir)
    assert registry.params_for("ema_crossover", "15m")["slow"] == 200
    assert registry.params_for("ema_crossover", "1m")["slow"] == 21
    assert registry.params_for("rsi_ema_combo", "5m")["rsi.period"] == 14
