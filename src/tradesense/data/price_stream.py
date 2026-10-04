"""Live prices from the OANDA v20 pricing stream.

``GET {stream_url}/v3/accounts/{accountID}/pricing/stream?instruments=EUR_USD,...``
returns chunked JSON lines: ``PRICE`` messages and periodic ``HEARTBEAT`` messages
(ignored). If no data arrives for ``heartbeat_timeout`` seconds, or the connection
drops, the stream reconnects with exponential backoff.

:class:`PriceHub` keeps ONE upstream connection for the union of instruments that
WebSocket clients watch and fans ticks out to per-client queues.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from tradesense.data.models import PriceTick
from tradesense.data.oanda_client import (
    OandaAuthError,
    OandaClient,
    OandaConfigError,
    OandaError,
    parse_price,
)

StatusEvent = dict[str, Any]


class PriceStream:
    def __init__(
        self,
        oanda: OandaClient,
        *,
        heartbeat_timeout: float = 20.0,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.oanda = oanda
        self.heartbeat_timeout = heartbeat_timeout
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.transport = transport

    async def events(self, instruments: list[str]) -> AsyncIterator[PriceTick | StatusEvent]:
        """Yield ticks and status dicts forever (until cancelled)."""
        backoff = self.initial_backoff
        url = f"{self.oanda.stream_url}/v3/accounts/{{account}}/pricing/stream"
        while True:
            try:
                account = self.oanda.account_id
                if not account:
                    raise OandaConfigError("OANDA_ACCOUNT_ID is not set")
                headers = self.oanda.auth_headers()
                timeout = httpx.Timeout(10.0, read=self.heartbeat_timeout)
                async with (
                    httpx.AsyncClient(timeout=timeout, transport=self.transport) as client,
                    client.stream(
                        "GET",
                        url.format(account=account),
                        params={"instruments": ",".join(instruments)},
                        headers=headers,
                    ) as response,
                ):
                    if response.status_code in (401, 403):
                        raise OandaAuthError(
                            f"OANDA rejected the stream request ({response.status_code})",
                            response.status_code,
                        )
                    if response.status_code != 200:
                        body = (await response.aread()).decode(errors="replace")[:200]
                        raise OandaError(
                            f"Stream HTTP {response.status_code}: {body}",
                            response.status_code,
                        )
                    yield {"state": "connected", "instruments": instruments}
                    backoff = self.initial_backoff
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            message = json.loads(line)
                        except json.JSONDecodeError:
                            logger.debug("Ignoring malformed stream line")
                            continue
                        kind = message.get("type")
                        if kind == "HEARTBEAT":
                            continue
                        if kind == "PRICE":
                            tick = parse_price(message)
                            if tick is not None:
                                yield tick
                # Server closed the stream cleanly: reconnect.
                yield {"state": "reconnecting", "message": "stream closed by server"}
            except OandaConfigError as exc:
                yield {"state": "unavailable", "message": str(exc)}
                return
            except (httpx.TransportError, OandaError) as exc:
                if isinstance(exc, OandaAuthError):
                    backoff = self.max_backoff  # do not hammer with a bad key
                message = (
                    "no data within heartbeat timeout"
                    if isinstance(exc, httpx.ReadTimeout)
                    else str(exc)
                )
                logger.warning(
                    "Price stream interrupted ({}); retrying in {:.0f}s", message, backoff
                )
                yield {"state": "reconnecting", "message": message, "retry_in": backoff}
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff)


class PriceHub:
    """Fan-out of one upstream stream to many subscriber queues."""

    def __init__(self, stream: PriceStream, queue_size: int = 100) -> None:
        self.stream = stream
        self.queue_size = queue_size
        self.latest: dict[str, PriceTick] = {}
        self.status: StatusEvent = {"state": "idle"}
        self._subscribers: defaultdict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)
        self._instruments: tuple[str, ...] = ()
        self._task: asyncio.Task[None] | None = None

    @property
    def instruments(self) -> tuple[str, ...]:
        return self._instruments

    def subscribe(self, instrument: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.queue_size)
        self._subscribers[instrument].add(queue)
        if instrument in self.latest:
            self._offer(queue, _tick_message(self.latest[instrument]))
        self._offer(queue, {"type": "price_status", **self.status})
        self._sync()
        return queue

    def unsubscribe(self, instrument: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        subscribers = self._subscribers.get(instrument)
        if subscribers is not None:
            subscribers.discard(queue)
            if not subscribers:
                del self._subscribers[instrument]
        self._sync()

    def _sync(self) -> None:
        wanted = tuple(sorted(self._subscribers))
        running = self._task is not None and not self._task.done()
        if wanted == self._instruments and (running or not wanted):
            return
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._instruments = wanted
        if wanted:
            self._task = asyncio.create_task(self._run(list(wanted)), name="price-stream")
        else:
            self.status = {"state": "idle"}

    async def _run(self, instruments: list[str]) -> None:
        async for item in self.stream.events(instruments):
            if isinstance(item, PriceTick):
                self.latest[item.instrument] = item
                message = _tick_message(item)
                for queue in list(self._subscribers.get(item.instrument, ())):
                    self._offer(queue, message)
            else:
                self.status = item
                event = {"type": "price_status", **item}
                for queues in list(self._subscribers.values()):
                    for queue in list(queues):
                        self._offer(queue, event)

    @staticmethod
    def _offer(queue: asyncio.Queue[dict[str, Any]], message: dict[str, Any]) -> None:
        if queue.full():  # slow consumer: drop the oldest message
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        queue.put_nowait(message)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._subscribers.clear()
        self._instruments = ()


def _tick_message(tick: PriceTick) -> dict[str, Any]:
    return {"type": "price", **tick.model_dump(mode="json")}
