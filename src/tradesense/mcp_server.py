"""FastMCP tools exposing TradeSense to MCP clients (signals only, no trading tools).

Mounted inside the FastAPI app at ``/mcp`` (streamable HTTP transport).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from tradesense.data.models import completed
from tradesense.data.oanda_client import InstrumentError, OandaError
from tradesense.services import Services
from tradesense.strategies.base import UnknownStrategyError
from tradesense.strategies.ingestion import UploadError
from tradesense.strategies.reference import LibraryError

INSTRUCTIONS = (
    "TradeSense analyses OANDA market data with pluggable strategies and an LLM, returning "
    "BUY / SELL / SIDEWAYS signals with probabilities. It produces signals only and cannot "
    "place orders. Signals are AI-generated and are not financial advice."
)


def create_mcp(get_services: Callable[[], Services]) -> FastMCP:
    mcp = FastMCP("TradeSense", instructions=INSTRUCTIONS)

    def services() -> Services:
        return get_services()

    def fail(exc: Exception) -> ToolError:
        return ToolError(str(exc))

    @mcp.tool
    async def list_strategies(
        category: Annotated[
            str | None,
            Field(description="trend | momentum | mean_reversion | volatility | levels | pattern"),
        ] = None,
    ) -> list[dict[str, Any]]:
        """List every selectable strategy (built-in, YAML rule and approved references)."""
        try:
            return [i.model_dump() for i in services().registry.list(category=category)]
        except ValueError as exc:
            raise fail(exc) from exc

    @mcp.tool
    async def reload_strategies() -> dict[str, Any]:
        """Rescan config/rules, strategies.yml and approved strategy cards."""
        return services().reload_strategies().as_dict()

    @mcp.tool
    async def get_candles(
        instrument: Annotated[str, Field(description="OANDA instrument, e.g. EUR_USD")],
        timeframe: Annotated[str, Field(description="1m, 5m or 15m")],
        count: Annotated[int, Field(ge=1, le=500)] = 100,
    ) -> dict[str, Any]:
        """Recent OHLCV candles (the last one may still be in progress)."""
        svc = services()
        try:
            resolved = await svc.oanda.resolve_instrument(instrument)
            frame = (await svc.oanda.get_candles(resolved, [timeframe], count=count))[timeframe]
        except (InstrumentError, OandaError, ValueError) as exc:
            raise fail(exc) from exc
        return {
            "instrument": resolved,
            "timeframe": timeframe,
            "complete_candles": len(completed(frame)),
            "candles": [
                {
                    "time": ts.isoformat(),
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                    "complete": bool(row.complete),
                }
                for ts, row in zip(frame.index, frame.itertuples(index=False), strict=True)
            ],
        }

    @mcp.tool
    async def get_live_price(
        instrument: Annotated[str, Field(description="OANDA instrument, e.g. EUR_USD")],
    ) -> dict[str, Any]:
        """Current bid / ask / spread."""
        svc = services()
        try:
            resolved = await svc.oanda.resolve_instrument(instrument)
            return (await svc.oanda.get_price(resolved)).model_dump(mode="json")
        except (InstrumentError, OandaError) as exc:
            raise fail(exc) from exc

    @mcp.tool
    async def run_analysis(
        instrument: Annotated[str, Field(description="OANDA instrument, e.g. EUR_USD")],
        selections: Annotated[
            dict[str, list[str]],
            Field(description='Timeframe -> strategy names, e.g. {"5m": ["ema_crossover"]}'),
        ],
    ) -> dict[str, Any]:
        """Run the selected strategies on each timeframe in parallel and return decisions
        plus a confidence-weighted consensus per timeframe."""
        svc = services()
        try:
            result = await svc.orchestrator.run_analysis(instrument, selections)
        except (InstrumentError, OandaError) as exc:
            raise fail(exc) from exc
        svc.last_result = result
        return result.model_dump(mode="json")

    @mcp.tool
    async def run_backtest(
        instrument: Annotated[str, Field(description="OANDA instrument, e.g. EUR_USD")],
        llm_check: Annotated[
            bool, Field(description="Re-check the top strategies with the LLM (small cost)")
        ] = True,
        llm_rules: Annotated[
            bool, Field(description="Ask a text LLM to propose new rules (small cost)")
        ] = True,
    ) -> dict[str, Any]:
        """Replay the last days on every timeframe, rank strategies by the directional hit
        rate of their BUY / SELL signals and propose new rule strategies. About 2 minutes."""
        try:
            result = await services().run_backtest(
                instrument, llm_check=llm_check, llm_rules=llm_rules
            )
        except (InstrumentError, OandaError, RuntimeError) as exc:
            raise fail(exc) from exc
        return result.model_dump(mode="json")

    @mcp.tool
    async def add_reference_strategy(
        file_path: Annotated[
            str,
            Field(
                description="Path to a .md/.txt/.pdf/.docx/.png/.jpg/.webp file inside "
                "an allowed import folder (library.import_roots)"
            ),
        ],
    ) -> dict[str, Any]:
        """Upload a reference strategy file and extract a DRAFT strategy card for review."""
        try:
            result = await services().library.import_file(file_path)
        except (LibraryError, UploadError) as exc:
            raise fail(exc) from exc
        return {"card": result.card.model_dump(mode="json"), "duplicate": result.duplicate}

    @mcp.tool
    async def get_reference_strategy(
        id: Annotated[str, Field(description="Strategy card id, e.g. double_bottom_ref")],
    ) -> dict[str, Any]:
        """A strategy card with a preview of its original content."""
        library = services().library
        try:
            card = library.get(id)
            preview = library.preview(card)
        except LibraryError as exc:
            raise fail(exc) from exc
        if preview.get("kind") == "markdown":
            preview = {"kind": "text", "text": library.document_text(card)[:20_000]}
        return {"card": card.model_dump(mode="json"), "preview": preview}

    @mcp.tool
    async def approve_reference_strategy(
        id: Annotated[str, Field(description="Strategy card id")],
    ) -> dict[str, Any]:
        """Approve a reviewed card so it appears in strategy selection lists."""
        try:
            return services().library.approve(id).model_dump(mode="json")
        except (LibraryError, UnknownStrategyError) as exc:
            raise fail(exc) from exc

    @mcp.tool
    async def llm_status() -> dict[str, Any]:
        """Active provider per role, provider health and usage totals."""
        return services().llm_status()

    return mcp
