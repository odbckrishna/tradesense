"""WebSocket ``/ws``: live prices and streamed analysis results.

Client -> server messages (JSON)::

    {"action": "subscribe_prices", "instrument": "EUR_USD"}
    {"action": "unsubscribe_prices"}
    {"action": "run_analysis", "instrument": "EUR_USD", "selections": {"5m": ["ema_crossover"]}}
    {"action": "auto_refresh", "enabled": true, "instrument": "...", "selections": {...}}
    {"action": "ping"}

Server -> client: ``hello``, ``price``, ``price_status``, ``analysis_started``,
``strategy_result``, ``consensus``, ``analysis_complete``, ``analysis_error``,
``auto_refresh_status``, ``pong``, ``error``.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from tradesense.data.oanda_client import InstrumentError, OandaError
from tradesense.engine.auto_refresh import AutoRefresher

if TYPE_CHECKING:
    from tradesense.services import Services

router = APIRouter()


class ClientMessage(BaseModel):
    action: Literal[
        "subscribe_prices", "unsubscribe_prices", "run_analysis", "auto_refresh", "ping"
    ]
    instrument: str | None = Field(default=None, max_length=40)
    selections: dict[str, list[str]] = Field(default_factory=dict)
    enabled: bool = False


class ClientConnection:
    def __init__(self, websocket: WebSocket, services: Services) -> None:
        self.id = secrets.token_hex(8)
        self.websocket = websocket
        self.services = services
        self._send_lock = asyncio.Lock()
        self.instrument: str | None = None
        self._queue: asyncio.Queue[dict[str, Any]] | None = None
        self._pump: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self.refresher = AutoRefresher(
            services.orchestrator,
            services.settings.app.timeframes,
            self.send,
            grace_seconds=services.settings.app.analysis.refresh_grace_seconds,
        )

    async def send(self, message: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.websocket.send_json(message)

    # ------------------------------------------------------------------ prices

    def subscribe(self, instrument: str) -> None:
        self.unsubscribe()
        self.instrument = instrument
        self._queue = self.services.prices.subscribe(instrument)
        self._pump = asyncio.create_task(self._pump_prices(self._queue), name=f"ws-{self.id}")

    def unsubscribe(self) -> None:
        if self._pump is not None:
            self._pump.cancel()
            self._pump = None
        if self.instrument is not None and self._queue is not None:
            self.services.prices.unsubscribe(self.instrument, self._queue)
        self.instrument = None
        self._queue = None

    async def _pump_prices(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        while True:
            message = await queue.get()
            try:
                await self.send(message)
            except Exception:
                return

    # ------------------------------------------------------------------ analysis

    def spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def run_analysis(self, instrument: str, selections: dict[str, list[str]]) -> None:
        try:
            result = await self.services.orchestrator.run_analysis(
                instrument, selections, on_event=self.send
            )
            self.services.last_result = result
        except (InstrumentError, OandaError) as exc:
            await self.send({"type": "analysis_error", "message": str(exc)})
        except Exception as exc:
            logger.exception("Analysis over WebSocket failed")
            await self.send({"type": "analysis_error", "message": f"Analysis failed: {exc}"})

    def close(self) -> None:
        self.unsubscribe()
        self.refresher.stop()
        for task in list(self._tasks):
            task.cancel()


class ConnectionManager:
    def __init__(self) -> None:
        self.clients: dict[str, ClientConnection] = {}

    def add(self, client: ClientConnection) -> None:
        self.clients[client.id] = client

    def remove(self, client_id: str) -> None:
        self.clients.pop(client_id, None)

    async def send_to(self, client_id: str, message: dict[str, Any]) -> None:
        client = self.clients.get(client_id)
        if client is not None:
            await client.send(message)


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    services: Services = websocket.app.state.services
    await websocket.accept()
    client = ClientConnection(websocket, services)
    manager.add(client)
    await client.send(
        {
            "type": "hello",
            "client_id": client.id,
            "server_time": datetime.now(UTC).isoformat(),
            "stream_available": services.oanda.configured and services.oanda.has_account,
        }
    )
    try:
        while True:
            raw = await websocket.receive_json()
            await _handle(client, raw)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # malformed frames etc.
        logger.debug("WebSocket {} closed: {}", client.id, exc)
    finally:
        client.close()
        manager.remove(client.id)
        with contextlib.suppress(Exception):
            await websocket.close()


async def _handle(client: ClientConnection, raw: Any) -> None:
    try:
        message = ClientMessage.model_validate(raw)
    except ValidationError as exc:
        await client.send(
            {"type": "error", "message": f"Invalid message: {exc.errors()[0]['msg']}"}
        )
        return
    services = client.services
    if message.action == "ping":
        await client.send({"type": "pong", "server_time": datetime.now(UTC).isoformat()})
    elif message.action == "unsubscribe_prices":
        client.unsubscribe()
    elif message.action == "subscribe_prices":
        try:
            instrument = await services.oanda.resolve_instrument(message.instrument or "")
        except (InstrumentError, OandaError) as exc:
            await client.send({"type": "price_status", "state": "unavailable", "message": str(exc)})
            return
        client.subscribe(instrument)
    elif message.action == "run_analysis":
        if not message.instrument:
            await client.send({"type": "error", "message": "instrument is required"})
            return
        client.spawn(client.run_analysis(message.instrument, message.selections))
    elif message.action == "auto_refresh":
        if not message.enabled:
            client.refresher.stop()
            await client.send(client.refresher.status())
            return
        if not message.instrument or not any(message.selections.values()):
            await client.send({"type": "error", "message": "instrument and selections required"})
            return
        try:
            instrument = await services.oanda.resolve_instrument(message.instrument)
        except (InstrumentError, OandaError) as exc:
            await client.send({"type": "analysis_error", "message": str(exc)})
            return
        client.refresher.start(instrument, message.selections)
        await client.send(client.refresher.status())
