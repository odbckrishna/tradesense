from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from tests.helpers import (
    GRANULARITY,
    make_candles,
    oanda_bid_ask_payload,
    oanda_candles_payload,
    oanda_instruments_payload,
)
from tradesense.config import Settings
from tradesense.data.models import completed
from tradesense.data.oanda_client import (
    InstrumentError,
    InstrumentNotFoundError,
    OandaAuthError,
    OandaClient,
    OandaConfigError,
    OandaConnectionError,
    OandaNotFoundError,
    OandaRateLimitError,
    normalize_instrument,
    parse_candles,
)

REST = "https://api-fxpractice.oanda.com"
INSTRUMENTS = ["EUR_USD", "GBP_USD", "SPX500_USD", "US30_USD", "USD_JPY", "XAU_USD"]


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[OandaClient]:
    oanda = OandaClient(
        api_key="test-key",
        account_id="001-001-1234567-001",
        rest_url=REST,
        stream_url="https://stream-fxpractice.oanda.com",
        timeframes=settings.app.timeframes,
        max_attempts=3,
        backoff_initial=0,
    )
    yield oanda
    await oanda.aclose()


@pytest.fixture
def mock_api() -> respx.MockRouter:
    with respx.mock(base_url=REST, assert_all_called=False) as router:
        router.get("/v3/accounts/001-001-1234567-001/instruments", name="instruments").mock(
            return_value=httpx.Response(200, json=oanda_instruments_payload(INSTRUMENTS))
        )
        yield router


# --------------------------------------------------------------------------- normalisation


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("eurusd", "EUR_USD"),
        ("EUR/USD", "EUR_USD"),
        ("eur-usd", "EUR_USD"),
        (" gbp jpy ", "GBP_JPY"),
        ("XAU_USD", "XAU_USD"),
        ("us30_usd", "US30_USD"),
        ("spx500/usd", "SPX500_USD"),
        ("US30USD", "US30USD"),  # resolved later against the account list
    ],
)
def test_normalize_instrument(raw: str, expected: str) -> None:
    assert normalize_instrument(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "EUR_USD; DROP TABLE", "€UR", "E", "A_B_C_D"])
def test_normalize_rejects_garbage(raw: str) -> None:
    with pytest.raises(InstrumentError):
        normalize_instrument(raw)


# --------------------------------------------------------------------------- validation


async def test_resolve_against_account_list(
    client: OandaClient, mock_api: respx.MockRouter
) -> None:
    assert await client.resolve_instrument("eurusd") == "EUR_USD"
    assert await client.resolve_instrument("us30usd") == "US30_USD"
    assert await client.resolve_instrument("SPX500 USD") == "SPX500_USD"
    assert await client.resolve_instrument("xau/usd") == "XAU_USD"
    # The instrument list is fetched once and cached.
    assert mock_api["instruments"].call_count == 1


async def test_resolve_unknown_instrument_suggests(
    client: OandaClient, mock_api: respx.MockRouter
) -> None:
    with pytest.raises(InstrumentNotFoundError) as info:
        await client.resolve_instrument("EUR_USX")
    assert "EUR_USD" in info.value.suggestions
    assert "not available" in str(info.value)


async def test_missing_api_key_fails_without_network(settings: Settings) -> None:
    oanda = OandaClient(
        api_key=None,
        account_id="acc",
        rest_url=REST,
        stream_url=REST,
        timeframes=settings.app.timeframes,
    )
    with respx.mock(assert_all_mocked=True) as router:
        with pytest.raises(OandaConfigError, match="OANDA_API_KEY"):
            await oanda.fetch_candles("EUR_USD", "M5")
        assert not router.calls
    await oanda.aclose()


# --------------------------------------------------------------------------- candles


async def test_get_candles_fetches_all_timeframes_concurrently(
    client: OandaClient, mock_api: respx.MockRouter
) -> None:
    in_flight = 0
    peak = 0
    seen: list[str] = []

    async def candles(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        granularity = request.url.params["granularity"]
        assert request.url.params["price"] == "M"
        assert request.headers["Authorization"] == "Bearer test-key"
        seen.append(granularity)
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        tf = {v: k for k, v in GRANULARITY.items()}[granularity]
        df = make_candles(int(request.url.params["count"]), tf, last_incomplete=True)
        return httpx.Response(200, json=oanda_candles_payload(df, granularity=granularity))

    mock_api.get(path__regex=r"/v3/instruments/EUR_USD/candles").mock(side_effect=candles)
    frames = await client.get_candles("EUR_USD", ["1m", "5m", "15m"], count=120)

    assert sorted(seen) == ["M1", "M15", "M5"]
    assert peak == 3, "timeframes should be requested concurrently"
    for tf in ("1m", "5m", "15m"):
        df = frames[tf]
        assert len(df) == 120
        assert list(df.columns) == ["open", "high", "low", "close", "volume", "complete"]
        assert str(df.index.tz) == "UTC"
        assert not bool(df["complete"].iloc[-1])  # in-progress candle kept for the chart
        assert len(completed(df)) == 119  # analysis uses complete candles only
        assert df.index.is_monotonic_increasing


async def test_get_spreads_from_bid_ask_candles(
    client: OandaClient, mock_api: respx.MockRouter
) -> None:
    df = make_candles(50, "5m")

    def candles(request: httpx.Request) -> httpx.Response:
        assert request.url.params["price"] == "BA"
        assert request.url.params["granularity"] == "M5"
        return httpx.Response(200, json=oanda_bid_ask_payload(df, spread=0.00015))

    mock_api.get(path__regex=r"/v3/instruments/EUR_USD/candles").mock(side_effect=candles)
    spreads = await client.get_spreads("EUR_USD", "5m", count=50)
    assert len(spreads) == 50
    assert spreads.index.equals(df.index)
    assert spreads.to_numpy() == pytest.approx(0.00015, abs=1e-9)


def test_parse_candles_bid_prices() -> None:
    df = make_candles(5)
    payload = oanda_candles_payload(df, price_key="bid")
    parsed = parse_candles(payload, price="B")
    assert len(parsed) == 5
    assert parsed["close"].iloc[-1] == pytest.approx(df["close"].iloc[-1], abs=1e-5)
    assert parse_candles({"candles": []}).empty


# --------------------------------------------------------------------------- errors


async def test_401_raises_auth_error_without_retry(
    client: OandaClient, mock_api: respx.MockRouter
) -> None:
    route = mock_api.get(path__regex=r"/candles").mock(
        return_value=httpx.Response(401, json={"errorMessage": "Insufficient authorization"})
    )
    with pytest.raises(OandaAuthError, match="rejected the API key"):
        await client.fetch_candles("EUR_USD", "M1")
    assert route.call_count == 1


async def test_404_reports_bad_instrument(client: OandaClient, mock_api: respx.MockRouter) -> None:
    mock_api.get(path__regex=r"/candles").mock(
        return_value=httpx.Response(404, json={"errorMessage": "Not found"})
    )
    with pytest.raises(OandaNotFoundError, match="Instrument 'FOO_BAR' was not found"):
        await client.fetch_candles("FOO_BAR", "M1")


async def test_400_invalid_instrument_is_not_found(
    client: OandaClient, mock_api: respx.MockRouter
) -> None:
    mock_api.get(path__regex=r"/candles").mock(
        return_value=httpx.Response(
            400, json={"errorMessage": "Invalid value specified for 'instrument'"}
        )
    )
    with pytest.raises(OandaNotFoundError, match="FOO_BAR"):
        await client.fetch_candles("FOO_BAR", "M1")


async def test_429_is_retried_then_reported(
    client: OandaClient, mock_api: respx.MockRouter
) -> None:
    route = mock_api.get(path__regex=r"/candles").mock(return_value=httpx.Response(429))
    with pytest.raises(OandaRateLimitError, match="rate limit"):
        await client.fetch_candles("EUR_USD", "M1")
    assert route.call_count == 3  # max_attempts


async def test_server_error_recovers_after_retry(
    client: OandaClient, mock_api: respx.MockRouter
) -> None:
    ok = httpx.Response(200, json=oanda_candles_payload(make_candles(10, "1m"), granularity="M1"))
    route = mock_api.get(path__regex=r"/candles").mock(side_effect=[httpx.Response(503), ok])
    df = await client.fetch_candles("EUR_USD", "M1", count=10)
    assert len(df) == 10
    assert route.call_count == 2


async def test_network_errors_are_retried(client: OandaClient, mock_api: respx.MockRouter) -> None:
    route = mock_api.get(path__regex=r"/candles").mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(OandaConnectionError):
        await client.fetch_candles("EUR_USD", "M1")
    assert route.call_count == 3


async def test_unknown_timeframe(client: OandaClient) -> None:
    with pytest.raises(ValueError, match="Unknown timeframe '2h'"):
        await client.get_candles("EUR_USD", ["2h"])


async def test_get_price(client: OandaClient, mock_api: respx.MockRouter) -> None:
    mock_api.get("/v3/accounts/001-001-1234567-001/pricing").mock(
        return_value=httpx.Response(
            200,
            json={
                "prices": [
                    {
                        "type": "PRICE",
                        "instrument": "EUR_USD",
                        "time": "2026-03-04T12:00:01.123456789Z",
                        "bids": [{"price": "1.10010", "liquidity": 1}],
                        "asks": [{"price": "1.10025", "liquidity": 1}],
                        "tradeable": True,
                    }
                ]
            },
        )
    )
    tick = await client.get_price("EUR_USD")
    assert tick.bid == pytest.approx(1.1001)
    assert tick.spread == pytest.approx(0.00015)
    assert tick.mid == pytest.approx(1.100175)
