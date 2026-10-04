from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.apptools import build_services, mock_oanda, receive_until
from tradesense.main import create_app


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    services, _ = build_services(tmp_path)
    app = create_app(services.settings, services, health_checks=False)
    with mock_oanda(), TestClient(app) as test_client:
        yield test_client


def test_hello_and_ping(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["stream_available"] is True
        ws.send_json({"action": "ping"})
        assert ws.receive_json()["type"] == "pong"
        ws.send_json({"action": "launch_rockets"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "Invalid message" in error["message"]


def test_live_prices_stream_and_reconnect(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"action": "subscribe_prices", "instrument": "eur/usd"})
        messages = receive_until(ws, "price")
        prices = [m for m in messages if m["type"] == "price"]
        assert prices[0]["instrument"] == "EUR_USD"
        assert prices[0]["bid"] == pytest.approx(1.1001)
        assert prices[0]["spread"] == pytest.approx(0.00012)
        # the mock stream ends after a few lines: the client sees a reconnect, then more ticks
        later = receive_until(ws, "price_status")
        states = [m.get("state") for m in later if m["type"] == "price_status"]
        assert states[-1] in {"reconnecting", "connected"}
        receive_until(ws, "price")
        ws.send_json({"action": "unsubscribe_prices"})


def test_subscribe_unknown_instrument(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"action": "subscribe_prices", "instrument": "NOPE_NOPE"})
        status = receive_until(ws, "price_status")[-1]
        assert status["state"] == "unavailable"
        assert "not available" in status["message"]


def test_run_analysis_over_websocket(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "action": "run_analysis",
                "instrument": "EUR_USD",
                "selections": {"5m": ["supertrend", "ema_crossover"]},
            }
        )
        events = receive_until(ws, "analysis_complete")
        results = [e for e in events if e["type"] == "strategy_result"]
        assert {r["result"]["strategy"] for r in results} == {"supertrend", "ema_crossover"}
        assert events[-1]["result"]["timeframes"]["5m"]["consensus"]["signal"] == "BUY"


def test_auto_refresh_toggle(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "action": "auto_refresh",
                "enabled": True,
                "instrument": "EUR_USD",
                "selections": {"1m": ["supertrend"], "15m": ["ichimoku"]},
            }
        )
        status = receive_until(ws, "auto_refresh_status")[-1]
        assert status["enabled"] is True
        ws.send_json({"action": "auto_refresh", "enabled": False})
        for _ in range(10):
            status = receive_until(ws, "auto_refresh_status")[-1]
            if status["enabled"] is False:
                break
        assert status["enabled"] is False
