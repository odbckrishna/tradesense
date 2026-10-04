"""Builders for app-level tests: isolated project home, mocked OANDA, scripted LLM."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import respx

from tests.fakes import VALID_JSON, ScriptedProvider, make_router
from tests.helpers import (
    GRANULARITY,
    REPO_ROOT,
    make_candles,
    oanda_bid_ask_payload,
    oanda_candles_payload,
    oanda_instruments_payload,
)
from tradesense.config import Settings, load_settings
from tradesense.data.oanda_client import OandaClient
from tradesense.data.price_stream import PriceStream
from tradesense.services import Services

REST = "https://api-fxpractice.oanda.com"
STREAM = "https://stream-fxpractice.oanda.com"
ACCOUNT = "001-001-1234567-001"
INSTRUMENTS = ["EUR_USD", "GBP_USD", "US30_USD", "XAU_USD"]


def make_settings(tmp_path: Path, max_upload_mb: float | None = None) -> Settings:
    home = tmp_path / "home"
    # the developer's model-picker choices and backtest-saved rules must not leak into tests
    shutil.copytree(
        REPO_ROOT / "config",
        home / "config",
        ignore=shutil.ignore_patterns("llm_selection*", "bt_*.yml"),
    )
    settings = load_settings(home, load_env=False)
    if max_upload_mb is not None:
        settings.app.library = settings.app.library.model_copy(
            update={"max_upload_mb": max_upload_mb}
        )
    return settings


def price_line(bid: float, ask: float, instrument: str = "EUR_USD") -> str:
    return json.dumps(
        {
            "type": "PRICE",
            "instrument": instrument,
            "time": "2026-03-04T12:00:01.000000000Z",
            "bids": [{"price": f"{bid:.5f}", "liquidity": 1000000}],
            "asks": [{"price": f"{ask:.5f}", "liquidity": 1000000}],
            "status": "tradeable",
            "tradeable": True,
        }
    )


HEARTBEAT = json.dumps({"type": "HEARTBEAT", "time": "2026-03-04T12:00:02.000000000Z"})


def stream_transport(lines: list[str] | None = None) -> httpx.MockTransport:
    body = (
        "\n".join(
            lines
            or [HEARTBEAT, price_line(1.10010, 1.10022), HEARTBEAT, price_line(1.10015, 1.10027)]
        )
        + "\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, stream=httpx.ByteStream(body.encode()))

    return httpx.MockTransport(handler)


def build_services(
    tmp_path: Path,
    *,
    script: list[str | Exception] | None = None,
    max_upload_mb: float | None = None,
) -> tuple[Services, ScriptedProvider]:
    settings = make_settings(tmp_path, max_upload_mb)
    oanda = OandaClient(
        api_key="test-key",
        account_id=ACCOUNT,
        rest_url=REST,
        stream_url=STREAM,
        timeframes=settings.app.timeframes,
        max_attempts=1,
        backoff_initial=0,
    )
    provider = ScriptedProvider("llm", script or [VALID_JSON], capabilities=["text", "vision"])
    router = make_router(
        [provider], roles={"text": ["llm"], "vision": ["llm"], "extraction": ["llm"]}
    )
    stream = PriceStream(oanda, initial_backoff=0.05, max_backoff=0.1, transport=stream_transport())
    services = Services(settings, oanda=oanda, env={}, price_stream=stream, router=router)
    return services, provider


def candles_handler(request: httpx.Request) -> httpx.Response:
    granularity = request.url.params["granularity"]
    timeframe = {v: k for k, v in GRANULARITY.items()}[granularity]
    df = make_candles(int(request.url.params["count"]), timeframe, last_incomplete=True)
    if request.url.params.get("price") == "BA":
        return httpx.Response(200, json=oanda_bid_ask_payload(df, granularity=granularity))
    return httpx.Response(200, json=oanda_candles_payload(df, granularity=granularity))


@contextmanager
def mock_oanda() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=REST, assert_all_called=False) as router:
        router.get(f"/v3/accounts/{ACCOUNT}/instruments", name="instruments").mock(
            return_value=httpx.Response(200, json=oanda_instruments_payload(INSTRUMENTS))
        )
        router.get(path__regex=r"/v3/instruments/[A-Z0-9_]+/candles", name="candles").mock(
            side_effect=candles_handler
        )
        router.get(f"/v3/accounts/{ACCOUNT}/pricing", name="pricing").mock(
            return_value=httpx.Response(
                200, json={"prices": [json.loads(price_line(1.1001, 1.10022))]}
            )
        )
        # Anything not mocked (e.g. the MCP test client) must not hit the network.
        router.route(name="passthrough", host="testserver").pass_through()
        yield router


def receive_until(ws: Any, kind: str, limit: int = 200) -> list[dict[str, Any]]:
    """Read WebSocket messages until one of type ``kind`` arrives (inclusive)."""
    seen = []
    for _ in range(limit):
        message = ws.receive_json()
        seen.append(message)
        if message.get("type") == kind:
            return seen
    raise AssertionError(f"no '{kind}' message in {limit} messages: {seen[-5:]}")
