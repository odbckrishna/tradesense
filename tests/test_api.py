from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from tests.apptools import ACCOUNT, build_services, mock_oanda, receive_until
from tests.fakes import ScriptedProvider
from tests.helpers import make_image
from tradesense.backtest.engine import BacktestBusyError
from tradesense.config import BacktestConfig
from tradesense.llm.selection import OllamaModel
from tradesense.main import create_app
from tradesense.services import Services

CARD = (
    '{"name": "Neckline Break", "summary": "Buy the neckline break", "category": "pattern",'
    '"buy_conditions": ["close above neckline"], "sell_conditions": [],'
    '"sideways_conditions": [], "required_indicators": ["swing_points"], "notes": ""}'
)


@pytest.fixture
def app_env(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, Services, ScriptedProvider, respx.MockRouter]]:
    services, provider = build_services(tmp_path, max_upload_mb=0.05)
    app = create_app(services.settings, services, health_checks=False)
    with mock_oanda() as oanda, TestClient(app) as client:
        yield client, services, provider, oanda


# --------------------------------------------------------------------------- basics


def test_index_and_security_headers(app_env: tuple) -> None:
    client = app_env[0]
    response = client.get("/")
    assert response.status_code == 200
    assert "TradeSense" in response.text
    assert "Signals are AI-generated and are not financial advice." in response.text
    assert "script-src 'self' https://unpkg.com" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert client.get("/static/js/app.js").status_code == 200


def test_health_and_meta(app_env: tuple) -> None:
    client = app_env[0]
    assert client.get("/api/health").json() == {"status": "ok"}
    meta = client.get("/api/meta").json()
    assert meta["oanda"] == {
        "environment": "practice",
        "configured": True,
        "account_configured": True,
    }
    assert list(meta["timeframes"]) == ["1m", "5m", "15m"]
    assert "swing_points" in meta["indicators"]
    assert ".webp" in meta["library"]["accept"]


def test_signals_only_no_order_endpoints(app_env: tuple) -> None:
    client = app_env[0]
    paths = list(client.get("/openapi.json").json()["paths"])
    assert "/api/analysis" in paths
    assert not [p for p in paths if any(w in p.lower() for w in ("order", "trade", "position"))]


# --------------------------------------------------------------------------- market data


def test_instruments_and_candles(app_env: tuple) -> None:
    client, _, _, oanda = app_env
    names = [i["name"] for i in client.get("/api/instruments").json()]
    assert names == ["EUR_USD", "GBP_USD", "US30_USD", "XAU_USD"]
    body = client.get("/api/candles/eurusd", params={"timeframe": "5m", "count": 50}).json()
    assert body["instrument"] == "EUR_USD"
    assert body["granularity"] == "M5"
    assert len(body["candles"]) == 50
    assert body["candles"][-1]["complete"] is False
    assert body["complete_candles"] == 49
    assert isinstance(body["candles"][0]["time"], int)
    assert oanda["candles"].calls.last.request.url.params["price"] == "M"


def test_market_data_errors(app_env: tuple) -> None:
    client, _, _, oanda = app_env
    response = client.get("/api/candles/FOOBAR", params={"timeframe": "5m"})
    assert response.status_code == 404
    assert "not available on this OANDA account" in response.json()["detail"]
    assert client.get("/api/candles/eur_usd", params={"timeframe": "3h"}).status_code == 400
    assert client.get("/api/candles/!!!").status_code == 422
    oanda.get(f"/v3/accounts/{ACCOUNT}/pricing").mock(
        return_value=httpx.Response(401, json={"errorMessage": "Insufficient authorization"})
    )
    response = client.get("/api/price/EUR_USD")
    assert response.status_code == 502
    assert "rejected the API key" in response.json()["detail"]


def test_price_snapshot(app_env: tuple) -> None:
    client = app_env[0]
    tick = client.get("/api/price/EUR_USD").json()
    assert tick["bid"] == pytest.approx(1.1001)
    assert tick["spread"] == pytest.approx(0.00012)


# --------------------------------------------------------------------------- strategies / analysis


def test_strategies_list_and_reload(app_env: tuple) -> None:
    client = app_env[0]
    body = client.get("/api/strategies").json()
    names = {s["name"] for s in body["strategies"]}
    assert len(names) == 13
    assert {"ema_crossover", "rsi_ema_combo"} <= names
    assert body["vision_available"] is True
    trend = client.get("/api/strategies", params={"category": "trend"}).json()["strategies"]
    assert {s["category"] for s in trend} == {"trend"}
    assert client.get("/api/strategies", params={"category": "astrology"}).status_code == 400
    assert client.post("/api/strategies/reload").json()["total"] == 13


def test_analysis_streams_to_websocket(app_env: tuple) -> None:
    client, services, provider, _ = app_env
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        response = client.post(
            "/api/analysis",
            json={
                "instrument": "EUR/USD",
                "selections": {
                    "1m": ["ema_crossover"],
                    "5m": ["macd_momentum", "nope"],
                    "15m": ["rsi_ema_combo"],
                },
                "client_id": hello["client_id"],
            },
        )
        assert response.status_code == 200
        result = response.json()
        events = receive_until(ws, "analysis_complete")
    assert result["instrument"] == "EUR_USD"
    assert set(result["timeframes"]) == {"1m", "5m", "15m"}
    statuses = {s["strategy"]: s["status"] for s in result["timeframes"]["5m"]["strategies"]}
    assert statuses == {"macd_momentum": "ok", "nope": "error"}
    assert result["usage"]["decisions"] == 3
    kinds = [e["type"] for e in events]
    assert kinds.count("strategy_result") == 4
    assert kinds.count("consensus") == 3
    assert services.last_result is not None
    assert len(provider.calls) == 3


def test_analysis_validation(app_env: tuple) -> None:
    client = app_env[0]
    assert client.post("/api/analysis", json={"selections": {}}).status_code == 422
    response = client.post(
        "/api/analysis", json={"instrument": "XXX_YYY", "selections": {"5m": ["ema_crossover"]}}
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------- library


def test_library_lifecycle(app_env: tuple) -> None:
    client, _, provider, _ = app_env
    provider.script = [CARD]
    upload = client.post(
        "/api/library/upload",
        files={"file": ("Neckline Break.png", make_image("PNG"), "image/png")},
    )
    assert upload.status_code == 200, upload.text
    card = upload.json()["card"]
    assert card["status"] == "draft"
    card_id = card["id"]
    assert card_id == "neckline_break_ref"

    listed = client.get("/api/library").json()
    assert [c["id"] for c in listed] == [card_id]
    detail = client.get(f"/api/library/{card_id}").json()
    assert detail["preview"] == {"kind": "image", "url": f"/api/library/{card_id}/file"}
    original = client.get(f"/api/library/{card_id}/file")
    assert original.headers["content-type"] == "image/png"
    assert original.headers["x-content-type-options"] == "nosniff"
    assert original.content.startswith(b"\x89PNG")

    edited = client.put(
        f"/api/library/{card_id}",
        json={"notes": "checked", "required_indicators": ["volume", "zigzag"]},
    )
    assert edited.status_code == 200
    assert edited.json()["required_indicators"] == ["volume"]
    assert client.put(f"/api/library/{card_id}", json={"status": "approved"}).status_code == 422

    assert card_id not in {s["name"] for s in client.get("/api/strategies").json()["strategies"]}
    assert client.post(f"/api/library/{card_id}/approve").json()["status"] == "approved"
    assert card_id in {s["name"] for s in client.get("/api/strategies").json()["strategies"]}
    provider.script = [CARD]
    assert client.post(f"/api/library/{card_id}/extract").json()["status"] == "draft"
    assert client.post(f"/api/library/{card_id}/disable").json()["status"] == "disabled"
    assert client.delete(f"/api/library/{card_id}").status_code == 204
    assert client.get(f"/api/library/{card_id}").status_code == 404


def test_library_rejections(app_env: tuple) -> None:
    client = app_env[0]
    wrong = client.post(
        "/api/library/upload", files={"file": ("chart.pdf", make_image("PNG"), "application/pdf")}
    )
    assert wrong.status_code == 415
    assert "does not match" in wrong.json()["detail"]
    big = client.post(
        "/api/library/upload", files={"file": ("big.md", b"#" * 200_000, "text/markdown")}
    )
    assert big.status_code == 413
    assert client.get("/api/library/..%2F..%2Fsecrets").status_code == 404
    assert client.get("/api/library/Bad-Id!").status_code == 404


# --------------------------------------------------------------------------- llm / charts / mcp


def test_llm_status(app_env: tuple) -> None:
    client = app_env[0]
    status = client.get("/api/llm/status").json()
    assert status["roles"]["text"]["active"] == "llm"
    assert status["roles"]["vision"]["fallback_in_use"] is False
    assert status["providers"]["llm"]["model"] == "llm-model"
    assert status["vision_available"] is True


def test_llm_reload(app_env: tuple) -> None:
    client, services, _, _ = app_env
    services.settings.paths.llm_config.write_text(
        "roles:\n  text: {primary: local}\n"
        "providers:\n  local:\n    type: openai_compatible\n"
        "    base_url: http://127.0.0.1:9/v1\n    model: tiny\n",
        encoding="utf-8",
    )
    status = client.post("/api/llm/reload").json()
    assert status["roles"]["text"]["primary"] == "local"
    assert status["providers"]["local"]["healthy"] is False  # nothing listens on port 9
    assert status["vision_available"] is False
    services.settings.paths.llm_config.write_text("roles: [broken", encoding="utf-8")
    response = client.post("/api/llm/reload")
    assert response.status_code == 400
    assert "Invalid YAML" in response.json()["detail"]


def test_llm_model_picker(app_env: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    client, services, _, _ = app_env

    async def installed(host: str) -> list[OllamaModel]:
        return [OllamaModel("nimble-cpu:latest", ("decision",)), OllamaModel("nimble:latest", ())]

    monkeypatch.setattr("tradesense.llm.selection.discover_ollama", installed)
    services.env["OPENROUTER_API_KEY"] = "test-key"
    client.post("/api/llm/reload")  # the repo's llm.yml providers instead of the scripted one

    models = client.get("/api/llm/models").json()
    assert models["roles"] == {
        "text": "openrouter_jev",
        "vision": "anthropic_sonnet",
        "extraction": "anthropic_sonnet",
    }
    by_id = {o["id"]: o for o in models["options"]}
    assert by_id["openrouter_jev"]["group"] == "cloud"
    assert by_id["openrouter_jev"]["roles"] == ["text"]
    assert by_id["ollama_text"]["group"] == "local"
    assert by_id["ollama_vision"]["available"] is False  # qwen2.5vl:7b is not installed
    assert by_id["ollama_nimble_latest"]["roles"] == ["text", "extraction"]

    status = client.post(
        "/api/llm/select", json={"role": "text", "model": "ollama_nimble_latest"}
    ).json()
    assert status["roles"]["text"]["chain"] == [
        "ollama_nimble_latest",
        "openrouter_jev",
        "anthropic_sonnet",
        "openai_main",
        "openrouter",
        "ollama_text",
    ]
    saved = json.loads(services.settings.paths.llm_selection.read_text(encoding="utf-8"))
    assert saved["roles"] == {"text": "ollama_nimble_latest"}
    assert saved["providers"]["ollama_nimble_latest"]["model"] == "nimble:latest"
    assert saved["providers"]["ollama_nimble_latest"]["options"] == {"num_gpu": 0, "think": False}
    reloaded = client.post("/api/llm/reload").json()
    assert reloaded["roles"]["text"]["primary"] == "ollama_nimble_latest"

    for body, error in [
        ({"role": "vision", "model": "openrouter_jev"}, "cannot serve the vision role"),
        ({"role": "vision", "model": "anthropic_sonnet"}, "ANTHROPIC_API_KEY is not set"),
        ({"role": "text", "model": "ghost"}, "unknown model"),
    ]:
        response = client.post("/api/llm/select", json=body)
        assert response.status_code == 400
        assert error in response.json()["detail"]
    assert client.post("/api/llm/select", json={"role": "x", "model": "y"}).status_code == 422

    # picking llm.yml's own primary again drops the override
    client.post("/api/llm/select", json={"role": "text", "model": "openrouter_jev"})
    saved = json.loads(services.settings.paths.llm_selection.read_text(encoding="utf-8"))
    assert saved == {}


def test_backtest_endpoint(app_env: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    client, services, _provider, _ = app_env
    # a short period keeps the candle-by-candle replay of the 12 built-ins quick
    services.backtester.config = BacktestConfig(
        days=0.06, min_signals=4, llm_top_n=1, llm_samples=2
    )
    services.backtester.executor_factory = lambda: ThreadPoolExecutor(max_workers=4)

    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        response = client.post(
            "/api/backtest",
            json={"instrument": "EUR_USD", "client_id": hello["client_id"], "llm_rules": False},
        )
        events = receive_until(ws, "backtest_complete")
    assert response.status_code == 200
    result = response.json()
    assert {e["stage"] for e in events if e["type"] == "backtest_progress"} >= {
        "candles",
        "replay",
        "templates",
        "done",
    }
    one_minute = result["timeframes"]["1m"]
    assert one_minute["candles"] == 86
    assert len(one_minute["strategies"]) == 13  # 12 built-ins + the repo's rule
    assert result["timeframes"]["15m"]["error"].startswith("Not enough candles")
    checked = [s for s in one_minute["strategies"] if s["llm_check"]]
    assert len(checked) == 1
    assert checked[0]["llm_check"]["providers"] == ["llm"]
    assert services.last_backtest is not None
    assert services.last_backtest.run_id == result["run_id"]

    candidate = one_minute["candidates"][0]["name"]
    body = {"run_id": result["run_id"], "timeframe": "1m", "name": candidate}
    saved = client.post("/api/backtest/strategies", json=body)
    assert saved.status_code == 200
    name = saved.json()["name"]
    assert name == f"bt_eur_usd_1m_{candidate}"
    assert (services.settings.paths.rules_dir / f"{name}.yml").is_file()
    names = {s["name"] for s in client.get("/api/strategies").json()["strategies"]}
    assert name in names
    duplicate = client.post("/api/backtest/strategies", json=body)
    assert duplicate.status_code == 400
    assert "already exists" in duplicate.json()["detail"]
    expired = client.post("/api/backtest/strategies", json={**body, "run_id": "old"})
    assert expired.status_code == 404

    async def busy(*args: object, **kwargs: object) -> None:
        raise BacktestBusyError("A backtest is already running")

    monkeypatch.setattr(services, "run_backtest", busy)
    assert client.post("/api/backtest", json={"instrument": "EUR_USD"}).status_code == 409


def test_backtest_replay_endpoint(app_env: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    client, services, _, _ = app_env
    services.backtester.executor_factory = lambda: ThreadPoolExecutor(max_workers=2)
    body = {"instrument": "EUR_USD", "timeframe": "15m", "strategy": "rsi_extremes"}
    response = client.post("/api/backtest/replay", json=body)
    assert response.status_code == 200
    trace = response.json()
    assert trace["strategy"] == "rsi_extremes"
    assert trace["timeframe"] == "15m"
    assert len(trace["candles"]) == len(trace["signals"]) == 192 + 4
    assert trace["stats"]["signals"] == sum(1 for s in trace["signals"][:192] if s)
    assert {"entries", "entry_stats", "horizon_bars", "pip_size"} <= set(trace)

    missing = client.post("/api/backtest/replay", json={**body, "strategy": "nope"})
    assert missing.status_code == 404
    bad_tf = client.post("/api/backtest/replay", json={**body, "timeframe": "4h"})
    assert bad_tf.status_code == 400

    async def busy(*args: object, **kwargs: object) -> None:
        raise BacktestBusyError("A backtest or replay is running")

    monkeypatch.setattr(services, "replay_strategy", busy)
    assert client.post("/api/backtest/replay", json=body).status_code == 409


def test_unknown_chart(app_env: tuple) -> None:
    assert app_env[0].get("/api/charts/deadbeef.png").status_code == 404


def test_mcp_endpoint_is_mounted(app_env: tuple) -> None:
    client = app_env[0]
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
        headers={"accept": "application/json, text/event-stream"},
    )
    assert response.status_code == 200
    assert "TradeSense" in response.text


def test_image_strategy_live_chart_is_served(app_env: tuple) -> None:
    client, _, provider, _ = app_env
    provider.script = [CARD]
    card_id = client.post(
        "/api/library/upload", files={"file": ("pattern.png", make_image("PNG"), "image/png")}
    ).json()["card"]["id"]
    client.post(f"/api/library/{card_id}/approve")
    provider.script = [
        '{"signal": "BUY", "probability": {"buy": 0.6, "sell": 0.2, '
        '"sideways": 0.2}, "confidence": 0.6, "observations": [], '
        '"reasoning": "r", "key_levels": {}, "pattern_match": 0.7}'
    ]
    result = client.post(
        "/api/analysis", json={"instrument": "EUR_USD", "selections": {"5m": [card_id]}}
    ).json()
    item = result["timeframes"]["5m"]["strategies"][0]
    assert item["decision"]["pattern_match"] == 0.7
    assert item["reference_image_url"] == f"/api/library/{card_id}/file"
    chart = client.get(item["live_chart_url"])
    assert chart.status_code == 200
    assert chart.headers["content-type"] == "image/png"
    assert chart.content.startswith(b"\x89PNG")
