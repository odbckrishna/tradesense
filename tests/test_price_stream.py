from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from tests.apptools import ACCOUNT, HEARTBEAT, STREAM, price_line
from tests.helpers import REPO_ROOT
from tradesense.config import load_settings
from tradesense.data.models import PriceTick
from tradesense.data.oanda_client import OandaClient
from tradesense.data.price_stream import PriceHub, PriceStream


def client(account: str | None = ACCOUNT) -> OandaClient:
    settings = load_settings(REPO_ROOT, load_env=False)
    return OandaClient(
        api_key="test-key",
        account_id=account,
        rest_url="https://x",
        stream_url=STREAM,
        timeframes=settings.app.timeframes,
    )


async def take(stream: PriceStream, instruments: list[str], n: int) -> list[Any]:
    items: list[Any] = []
    async for item in stream.events(instruments):
        items.append(item)
        if len(items) >= n:
            break
    return items


async def test_ticks_heartbeats_and_reconnect() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = (
            "\n".join(
                [
                    HEARTBEAT,
                    price_line(1.1, 1.1002),
                    "",
                    "not json",
                    price_line(1.2, 1.2003, "GBP_USD"),
                ]
            )
            + "\n"
        )
        return httpx.Response(200, stream=httpx.ByteStream(body.encode()))

    stream = PriceStream(client(), initial_backoff=0.01, transport=httpx.MockTransport(handler))
    items = await take(stream, ["EUR_USD", "GBP_USD"], 7)
    assert items[0] == {"state": "connected", "instruments": ["EUR_USD", "GBP_USD"]}
    assert isinstance(items[1], PriceTick)
    assert items[1].instrument == "EUR_USD"
    assert items[1].spread == pytest.approx(0.0002)
    assert isinstance(items[2], PriceTick)
    assert items[2].instrument == "GBP_USD"
    assert items[3]["state"] == "reconnecting"  # server closed the stream
    assert items[4]["state"] == "connected"
    assert len(calls) == 2
    assert calls[0].url.path == f"/v3/accounts/{ACCOUNT}/pricing/stream"
    assert calls[0].url.params["instruments"] == "EUR_USD,GBP_USD"


async def test_http_errors_back_off() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("down")
        if attempts == 2:
            return httpx.Response(503, text="maintenance")
        return httpx.Response(
            200, stream=httpx.ByteStream((price_line(1.1, 1.1001) + "\n").encode())
        )

    stream = PriceStream(client(), initial_backoff=0.01, transport=httpx.MockTransport(handler))
    items = await take(stream, ["EUR_USD"], 4)
    assert items[0]["state"] == "reconnecting"
    assert items[0]["retry_in"] == pytest.approx(0.01)
    assert items[1]["state"] == "reconnecting"
    assert "503" in items[1]["message"]
    assert items[1]["retry_in"] == pytest.approx(0.02)  # exponential backoff
    assert items[2]["state"] == "connected"
    assert isinstance(items[3], PriceTick)


async def test_missing_account_is_unavailable() -> None:
    stream = PriceStream(client(account=None))
    items = [item async for item in stream.events(["EUR_USD"])]
    assert items == [{"state": "unavailable", "message": "OANDA_ACCOUNT_ID is not set"}]


async def test_hub_fans_out_and_shares_one_connection() -> None:
    connections: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        connections.append(request.url.params["instruments"])
        body = "\n".join([price_line(1.1, 1.1001), price_line(150.0, 150.02, "USD_JPY")])
        return httpx.Response(200, stream=httpx.ByteStream((body + "\n").encode()))

    hub = PriceHub(
        PriceStream(client(), initial_backoff=0.05, transport=httpx.MockTransport(handler))
    )
    first = hub.subscribe("EUR_USD")
    second = hub.subscribe("EUR_USD")
    third = hub.subscribe("USD_JPY")

    async def next_price(queue: asyncio.Queue[dict[str, Any]]) -> dict[str, Any]:
        while True:
            message = await asyncio.wait_for(queue.get(), timeout=2)
            if message["type"] == "price":
                return message

    a, b, c = await asyncio.gather(next_price(first), next_price(second), next_price(third))
    assert a["instrument"] == b["instrument"] == "EUR_USD"
    assert c["instrument"] == "USD_JPY"
    assert hub.instruments == ("EUR_USD", "USD_JPY")
    assert set(connections) == {"EUR_USD,USD_JPY"}  # one upstream connection for all
    assert hub.latest["USD_JPY"].bid == pytest.approx(150.0)

    hub.unsubscribe("USD_JPY", third)
    assert hub.instruments == ("EUR_USD",)
    hub.unsubscribe("EUR_USD", first)
    hub.unsubscribe("EUR_USD", second)
    assert hub.instruments == ()
    assert hub.status == {"state": "idle"}
    await hub.stop()
