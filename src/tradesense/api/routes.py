"""REST endpoints. Signals only: there is deliberately no order-placement endpoint."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from tradesense import __version__
from tradesense.api.websocket import manager
from tradesense.backtest.models import BacktestResult
from tradesense.backtest.trace import StrategyTrace
from tradesense.config import RoleName
from tradesense.data.models import completed
from tradesense.engine.orchestrator import AnalysisResult
from tradesense.indicators.technical import INDICATORS
from tradesense.llm.schemas import StrategyCard
from tradesense.services import Services
from tradesense.strategies.base import CATEGORIES, StrategyInfo
from tradesense.strategies.reference import CardUpdate, LibraryError

router = APIRouter(prefix="/api")

DISCLAIMER = "Signals are AI-generated and are not financial advice."


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


class LLMSelectRequest(BaseModel):
    role: RoleName
    model: str = Field(min_length=1, max_length=200, description="An option id from /llm/models")


class BacktestRequest(BaseModel):
    instrument: str = Field(min_length=2, max_length=40)
    client_id: str | None = Field(default=None, max_length=64)
    llm_check: bool = True
    llm_rules: bool = True


class ReplayRequest(BaseModel):
    instrument: str = Field(min_length=2, max_length=40)
    timeframe: str = Field(min_length=1, max_length=10)
    strategy: str = Field(min_length=1, max_length=100)


class SaveCandidateRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=64)
    timeframe: str = Field(min_length=1, max_length=10)
    name: str = Field(min_length=1, max_length=40)


class AnalysisRequest(BaseModel):
    instrument: str = Field(min_length=2, max_length=40)
    selections: dict[str, list[str]]
    client_id: str | None = Field(default=None, max_length=64)


# --------------------------------------------------------------------------- meta


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/meta")
async def meta(services: ServicesDep) -> dict[str, Any]:
    app = services.settings.app
    return {
        "version": __version__,
        "disclaimer": DISCLAIMER,
        "oanda": {
            "environment": services.settings.oanda.oanda_environment,
            "configured": services.oanda.configured,
            "account_configured": services.oanda.has_account,
        },
        "timeframes": {
            tf: {"granularity": c.granularity, "refresh_seconds": c.refresh_seconds}
            for tf, c in app.timeframes.items()
        },
        "candle_count": app.oanda.candle_count,
        "categories": list(CATEGORIES),
        "indicators": {name: kind.description for name, kind in INDICATORS.items()},
        "library": {
            "allowed_types": app.library.allowed_types,
            "max_upload_mb": app.library.max_upload_mb,
            "accept": ",".join(f".{t}" for t in app.library.allowed_types),
        },
    }


# --------------------------------------------------------------------------- market data


@router.get("/instruments")
async def instruments(services: ServicesDep) -> list[dict[str, Any]]:
    items = await services.oanda.list_instruments()
    return [i.model_dump() for i in items]


@router.get("/candles/{instrument}")
async def candles(
    instrument: str,
    services: ServicesDep,
    timeframe: str = "1m",
    count: Annotated[int, Query(ge=1, le=5000)] = 300,
) -> dict[str, Any]:
    resolved = await services.oanda.resolve_instrument(instrument)
    frame = (await services.oanda.get_candles(resolved, [timeframe], count=count))[timeframe]
    rows = [
        {
            "time": int(ts.timestamp()),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "complete": bool(row.complete),
        }
        for ts, row in zip(frame.index, frame.itertuples(index=False), strict=True)
    ]
    return {
        "instrument": resolved,
        "timeframe": timeframe,
        "granularity": services.oanda.granularity_for(timeframe),
        "complete_candles": len(completed(frame)),
        "candles": rows,
    }


@router.get("/price/{instrument}")
async def price(instrument: str, services: ServicesDep) -> dict[str, Any]:
    resolved = await services.oanda.resolve_instrument(instrument)
    return (await services.oanda.get_price(resolved)).model_dump(mode="json")


# --------------------------------------------------------------------------- strategies


@router.get("/strategies")
async def strategies(
    services: ServicesDep, category: str | None = None, source: str | None = None
) -> dict[str, Any]:
    items: list[StrategyInfo] = services.registry.list(category=category, source=source)
    return {
        "strategies": [i.model_dump() for i in items],
        "vision_available": services.router.vision_available(),
        "errors": services.registry.last_report.errors,
    }


@router.post("/strategies/reload")
async def reload_strategies(services: ServicesDep) -> dict[str, Any]:
    return services.reload_strategies().as_dict()


@router.post("/analysis")
async def analysis(body: AnalysisRequest, services: ServicesDep) -> AnalysisResult:
    async def forward(event: dict[str, Any]) -> None:
        if body.client_id:
            await manager.send_to(body.client_id, event)

    result = await services.orchestrator.run_analysis(
        body.instrument, body.selections, on_event=forward
    )
    services.last_result = result
    return result


@router.post("/backtest")
async def backtest(body: BacktestRequest, services: ServicesDep) -> BacktestResult:
    """Replay the last days on every timeframe and score each strategy's BUY / SELL calls.
    Progress streams to ``client_id`` over the WebSocket; takes about 2 minutes."""

    async def forward(event: dict[str, Any]) -> None:
        if body.client_id:
            await manager.send_to(body.client_id, event)

    return await services.run_backtest(
        body.instrument, llm_check=body.llm_check, llm_rules=body.llm_rules, on_event=forward
    )


@router.post("/backtest/replay")
async def backtest_replay(body: ReplayRequest, services: ServicesDep) -> StrategyTrace:
    """One strategy's last days on one timeframe: candles, entries marked right / wrong
    and its accuracy. Takes a few seconds."""
    return await services.replay_strategy(body.instrument, body.timeframe, body.strategy)


@router.post("/backtest/strategies")
async def backtest_save(body: SaveCandidateRequest, services: ServicesDep) -> StrategyInfo:
    """Save a generated rule from the latest backtest as a strategy (config/rules/)."""
    return services.save_backtest_candidate(body.run_id, body.timeframe, body.name)


@router.get("/charts/{chart_id}.png")
async def chart(chart_id: str, services: ServicesDep) -> Response:
    png = services.charts.get(chart_id)
    if png is None:
        return JSONResponse({"detail": "Chart expired"}, status_code=404)
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


# --------------------------------------------------------------------------- llm


@router.get("/llm/status")
async def llm_status(services: ServicesDep) -> dict[str, Any]:
    return services.llm_status()


@router.post("/llm/reload")
async def llm_reload(services: ServicesDep) -> dict[str, Any]:
    return await services.reload_llm()


@router.get("/llm/models")
async def llm_models(services: ServicesDep) -> dict[str, Any]:
    return await services.llm_models()


@router.post("/llm/select")
async def llm_select(body: LLMSelectRequest, services: ServicesDep) -> dict[str, Any]:
    return await services.select_llm(body.role, body.model)


# --------------------------------------------------------------------------- library


@router.post("/library/upload")
async def library_upload(
    services: ServicesDep,
    file: Annotated[UploadFile, File()],
    extract: bool = True,
) -> dict[str, Any]:
    limit = services.settings.app.library.max_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise LibraryError(
            f"The file exceeds the {services.settings.app.library.max_upload_mb:g} MB limit.", 413
        )
    result = await services.library.upload(file.filename or "upload", data, extract=extract)
    return {"card": result.card.model_dump(mode="json"), "duplicate": result.duplicate}


@router.get("/library")
async def library_list(services: ServicesDep) -> list[StrategyCard]:
    return services.library.list_cards()


@router.get("/library/{card_id}")
async def library_get(card_id: str, services: ServicesDep) -> dict[str, Any]:
    card = services.library.get(card_id)
    return {"card": card.model_dump(mode="json"), "preview": services.library.preview(card)}


@router.get("/library/{card_id}/file")
async def library_file(card_id: str, services: ServicesDep) -> Response:
    card = services.library.get(card_id)
    data = services.library.read_source(card)
    return Response(
        data,
        media_type=card.media_type or "application/octet-stream",
        headers={
            "Content-Disposition": "inline",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "Cache-Control": "private, max-age=300",
        },
    )


@router.put("/library/{card_id}")
async def library_update(card_id: str, changes: CardUpdate, services: ServicesDep) -> StrategyCard:
    return services.library.update(card_id, changes)


@router.post("/library/{card_id}/extract")
async def library_extract(card_id: str, services: ServicesDep) -> StrategyCard:
    return await services.library.extract(card_id)


@router.post("/library/{card_id}/approve")
async def library_approve(card_id: str, services: ServicesDep) -> StrategyCard:
    return services.library.approve(card_id)


@router.post("/library/{card_id}/disable")
async def library_disable(card_id: str, services: ServicesDep) -> StrategyCard:
    return services.library.disable(card_id)


@router.delete("/library/{card_id}", status_code=204)
async def library_delete(card_id: str, services: ServicesDep) -> Response:
    services.library.delete(card_id)
    return Response(status_code=204)
