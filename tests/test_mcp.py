from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from tests.apptools import build_services, mock_oanda
from tests.helpers import make_image
from tradesense.mcp_server import create_mcp

CARD = (
    '{"name": "Double Bottom", "summary": "Two lows then breakout", "category": "pattern",'
    '"buy_conditions": ["close above neckline"], "sell_conditions": [],'
    '"sideways_conditions": [], "required_indicators": ["swing_points"], "notes": ""}'
)


async def call(client: Client, tool: str, args: dict[str, Any] | None = None) -> Any:
    result = await client.call_tool(tool, args or {})
    return result.data if result.data is not None else result.structured_content


async def test_mcp_tools(tmp_path: Path) -> None:
    services, provider = build_services(tmp_path)
    await services.start(health_checks=False)
    mcp = create_mcp(lambda: services)
    inbox = services.settings.resolve_path("strategy_library/inbox")
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "double_bottom.png").write_bytes(make_image("PNG"))
    try:
        with mock_oanda():
            async with Client(mcp) as client:
                names = {t.name for t in await client.list_tools()}
                assert names == {
                    "list_strategies",
                    "reload_strategies",
                    "get_candles",
                    "get_live_price",
                    "run_analysis",
                    "run_backtest",
                    "add_reference_strategy",
                    "get_reference_strategy",
                    "approve_reference_strategy",
                    "llm_status",
                }
                strategies = await call(client, "list_strategies", {"category": "momentum"})
                assert {s["category"] for s in strategies} == {"momentum"}
                assert (await call(client, "reload_strategies"))["builtin"] == 12

                candles = await call(
                    client, "get_candles", {"instrument": "eurusd", "timeframe": "1m", "count": 30}
                )
                assert candles["instrument"] == "EUR_USD"
                assert len(candles["candles"]) == 30
                price = await call(client, "get_live_price", {"instrument": "EUR_USD"})
                assert price["ask"] > price["bid"]

                analysis = await call(
                    client,
                    "run_analysis",
                    {
                        "instrument": "EUR_USD",
                        "selections": {"5m": ["ema_crossover"]},
                    },
                )
                decision = analysis["timeframes"]["5m"]["strategies"][0]["decision"]
                assert decision["signal"] == "BUY"
                assert decision["decided_by"] == "llm"

                provider.script = [CARD]
                added = await call(
                    client,
                    "add_reference_strategy",
                    {"file_path": str(inbox / "double_bottom.png")},
                )
                card_id = added["card"]["id"]
                assert added["card"]["status"] == "draft"
                fetched = await call(client, "get_reference_strategy", {"id": card_id})
                assert fetched["card"]["buy_conditions"] == ["close above neckline"]
                approved = await call(client, "approve_reference_strategy", {"id": card_id})
                assert approved["status"] == "approved"
                assert card_id in services.registry

                status = await call(client, "llm_status")
                assert status["roles"]["text"]["active"] == "llm"

                with pytest.raises(ToolError, match="outside the allowed import folders"):
                    await client.call_tool(
                        "add_reference_strategy", {"file_path": str(tmp_path / "x.md")}
                    )
                with pytest.raises(ToolError, match="not available"):
                    await client.call_tool("get_live_price", {"instrument": "ZZZ_ZZZ"})
    finally:
        await services.stop()
