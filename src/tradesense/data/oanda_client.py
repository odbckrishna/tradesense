"""Async OANDA v20 REST client: instruments, candles and price snapshots.

Docs: https://developer.oanda.com/rest-live-v20/introduction/

This client only *reads* market data. TradeSense produces signals only and has
no order-placement code anywhere.
"""

from __future__ import annotations

import asyncio
import difflib
import re
import time
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any

import httpx
import pandas as pd
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from tradesense.config import Environment, TimeframeConfig
from tradesense.data.models import CANDLE_COLUMNS, InstrumentInfo, PriceTick, empty_candles

if TYPE_CHECKING:
    from tradesense.config import Settings

_PRICE_KEYS = {"M": "mid", "B": "bid", "A": "ask"}
_SEPARATORS = re.compile(r"[\s/\\\-.:|]+")
_VALID_INSTRUMENT = re.compile(r"^[A-Z0-9]{2,12}(_[A-Z0-9]{2,12})?$")


# --------------------------------------------------------------------------- errors


class OandaError(RuntimeError):
    """Base class for OANDA failures. ``status_code`` is the HTTP status, if any."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OandaConfigError(OandaError):
    """Credentials or account id are missing."""


class OandaAuthError(OandaError):
    """401/403: the API key was rejected."""


class OandaNotFoundError(OandaError):
    """404: unknown instrument or account."""


class OandaBadRequestError(OandaError):
    """400: OANDA rejected the request parameters."""


class OandaRateLimitError(OandaError):
    """429: too many requests (retried with backoff before surfacing)."""


class OandaServerError(OandaError):
    """5xx from OANDA (retried with backoff before surfacing)."""


class OandaConnectionError(OandaError):
    """Network failure or timeout (retried with backoff before surfacing)."""


class InstrumentError(ValueError):
    """The instrument string is malformed."""


class InstrumentNotFoundError(InstrumentError):
    """The instrument is not tradeable on this account."""

    def __init__(self, instrument: str, suggestions: Sequence[str] = ()) -> None:
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"Instrument '{instrument}' is not available on this OANDA account.{hint}")
        self.instrument = instrument
        self.suggestions = list(suggestions)


_RETRYABLE = (OandaRateLimitError, OandaServerError, OandaConnectionError)


# --------------------------------------------------------------------------- helpers


def normalize_instrument(raw: str) -> str:
    """Syntactic normalisation to OANDA format.

    ``"eurusd"`` / ``"EUR/USD"`` / ``"eur-usd"`` -> ``"EUR_USD"``; ``"us30_usd"`` -> ``"US30_USD"``.
    Symbols that cannot be split unambiguously (``"US30USD"``) are returned compact and
    resolved against the account instrument list by :meth:`OandaClient.resolve_instrument`.
    """
    if not isinstance(raw, str):
        raise InstrumentError("Instrument must be a string")
    value = _SEPARATORS.sub("_", raw.strip().upper())
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise InstrumentError("Instrument is empty")
    if "_" not in value and len(value) == 6 and value.isalpha():
        value = f"{value[:3]}_{value[3:]}"
    if not _VALID_INSTRUMENT.match(value):
        raise InstrumentError(
            f"'{raw}' is not a valid instrument. Use OANDA format such as EUR_USD, XAU_USD "
            "or US30_USD."
        )
    return value


def parse_candles(payload: Mapping[str, Any], price: str = "M") -> pd.DataFrame:
    """Turn an OANDA candles response into a DataFrame (complete + in-progress rows)."""
    key = _PRICE_KEYS[price]
    rows: list[tuple[str, float, float, float, float, float, bool]] = []
    for candle in payload.get("candles", []) or []:
        ohlc = candle.get(key)
        if not ohlc:
            continue
        rows.append(
            (
                candle["time"],
                float(ohlc["o"]),
                float(ohlc["h"]),
                float(ohlc["l"]),
                float(ohlc["c"]),
                float(candle.get("volume", 0) or 0),
                bool(candle.get("complete", False)),
            )
        )
    if not rows:
        return empty_candles()
    frame = pd.DataFrame(rows, columns=["time", *CANDLE_COLUMNS])
    frame["time"] = pd.to_datetime(frame["time"], utc=True, format="ISO8601")
    frame = frame.set_index("time").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame["complete"] = frame["complete"].astype(bool)
    return frame


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        return str(body.get("errorMessage") or body.get("message") or body)[:300]
    return str(body)[:300]


# --------------------------------------------------------------------------- client


class OandaClient:
    """Read-only async client for the OANDA v20 REST API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        account_id: str | None,
        rest_url: str,
        stream_url: str,
        timeframes: Mapping[str, TimeframeConfig],
        environment: Environment = "practice",
        candle_count: int = 300,
        price: str = "M",
        timeout: float = 15.0,
        instruments_cache_seconds: float = 3600,
        max_attempts: int = 4,
        backoff_initial: float = 0.5,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if price not in _PRICE_KEYS:
            raise ValueError(f"price must be one of {list(_PRICE_KEYS)}")
        self.api_key = api_key
        self.account_id = account_id
        self.environment = environment
        self.rest_url = rest_url.rstrip("/")
        self.stream_url = stream_url.rstrip("/")
        self.timeframes = dict(timeframes)
        self.candle_count = candle_count
        self.price = price
        self.timeout = timeout
        self.instruments_cache_seconds = instruments_cache_seconds
        self.max_attempts = max(1, max_attempts)
        self.backoff_initial = backoff_initial
        self._http = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_http = http_client is None
        self._instruments: list[InstrumentInfo] | None = None
        self._instruments_at = 0.0
        self._instruments_lock = asyncio.Lock()
        self._warned_no_account = False

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> OandaClient:
        env = settings.oanda.oanda_environment
        oanda = settings.app.oanda
        if env not in oanda.rest_url or env not in oanda.stream_url:
            raise OandaConfigError(f"No OANDA URLs configured for environment '{env}'")
        return cls(
            api_key=settings.oanda.api_key,
            account_id=settings.oanda.oanda_account_id,
            environment=env,
            rest_url=oanda.rest_url[env],
            stream_url=oanda.stream_url[env],
            timeframes=settings.app.timeframes,
            candle_count=oanda.candle_count,
            price=oanda.price,
            timeout=oanda.request_timeout_seconds,
            instruments_cache_seconds=oanda.instruments_cache_seconds,
            **kwargs,
        )

    # ------------------------------------------------------------------ lifecycle

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def has_account(self) -> bool:
        return bool(self.account_id)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> OandaClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def auth_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise OandaConfigError("OANDA_API_KEY is not set. Add it to .env (see .env.example).")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept-Datetime-Format": "RFC3339",
        }

    def _require_account(self) -> str:
        if not self.account_id:
            raise OandaConfigError("OANDA_ACCOUNT_ID is not set. Add it to .env.")
        return self.account_id

    # ------------------------------------------------------------------ http

    async def _get_once(self, path: str, params: Mapping[str, Any] | None) -> dict[str, Any]:
        url = f"{self.rest_url}{path}"
        try:
            response = await self._http.get(url, params=params, headers=self.auth_headers())
        except httpx.TimeoutException as exc:
            raise OandaConnectionError(f"Timed out calling OANDA ({path})") from exc
        except httpx.TransportError as exc:
            raise OandaConnectionError(f"Network error calling OANDA ({path}): {exc}") from exc

        status = response.status_code
        if status == 200:
            data = response.json()
            if not isinstance(data, dict):
                raise OandaError("Unexpected OANDA response shape", status)
            return data
        detail = _error_message(response)
        if status in (401, 403):
            raise OandaAuthError(
                f"OANDA rejected the API key ({status}): {detail}. Check OANDA_API_KEY and that "
                f"OANDA_ENVIRONMENT ('{self.environment}') matches the account type.",
                status,
            )
        if status == 404:
            raise OandaNotFoundError(f"OANDA resource not found (404): {detail}", status)
        if status == 429:
            raise OandaRateLimitError(
                "OANDA rate limit hit (429). Requests were retried with backoff; "
                "reduce refresh frequency or try again shortly.",
                status,
            )
        if status == 400:
            raise OandaBadRequestError(f"OANDA rejected the request (400): {detail}", status)
        if status >= 500:
            raise OandaServerError(f"OANDA server error ({status}): {detail}", status)
        raise OandaError(f"Unexpected OANDA response ({status}): {detail}", status)

    async def _get_json(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential_jitter(
                initial=self.backoff_initial, max=8, jitter=self.backoff_initial
            ),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    logger.warning(
                        "Retrying OANDA request {} (attempt {}/{})",
                        path.split("?")[0],
                        attempt.retry_state.attempt_number,
                        self.max_attempts,
                    )
                return await self._get_once(path, params)
        raise AssertionError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------ instruments

    async def list_instruments(self, *, force_refresh: bool = False) -> list[InstrumentInfo]:
        """Instruments tradeable on the account (cached)."""
        account = self._require_account()
        async with self._instruments_lock:
            fresh = (time.monotonic() - self._instruments_at) < self.instruments_cache_seconds
            if self._instruments is not None and fresh and not force_refresh:
                return self._instruments
            data = await self._get_json(f"/v3/accounts/{account}/instruments")
            instruments = [
                InstrumentInfo(
                    name=item["name"],
                    type=item.get("type"),
                    display_name=item.get("displayName"),
                    pip_location=item.get("pipLocation"),
                    display_precision=item.get("displayPrecision"),
                )
                for item in data.get("instruments", [])
                if item.get("name")
            ]
            instruments.sort(key=lambda i: i.name)
            self._instruments = instruments
            self._instruments_at = time.monotonic()
            logger.info("Loaded {} OANDA instruments", len(instruments))
            return instruments

    async def resolve_instrument(self, raw: str) -> str:
        """Normalise ``raw`` and validate it against the account's instrument list."""
        normalized = normalize_instrument(raw)
        if not self.has_account:
            if not self._warned_no_account:
                logger.warning(
                    "OANDA_ACCOUNT_ID not set: instruments are normalised but not validated"
                )
                self._warned_no_account = True
            if "_" not in normalized:
                raise InstrumentError(
                    f"Cannot resolve '{raw}' without the account instrument list; "
                    "use OANDA format such as US30_USD."
                )
            return normalized
        instruments = await self.list_instruments()
        names = {i.name for i in instruments}
        if normalized in names:
            return normalized
        compact = {i.name.replace("_", ""): i.name for i in instruments}
        match = compact.get(normalized.replace("_", ""))
        if match:
            return match
        suggestions = difflib.get_close_matches(normalized, sorted(names), n=3, cutoff=0.6)
        raise InstrumentNotFoundError(normalized, suggestions)

    async def instrument_info(self, instrument: str) -> InstrumentInfo | None:
        if not self.has_account:
            return None
        for info in await self.list_instruments():
            if info.name == instrument:
                return info
        return None

    # ------------------------------------------------------------------ candles

    def granularity_for(self, timeframe: str) -> str:
        try:
            return self.timeframes[timeframe].granularity
        except KeyError as exc:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. Configured: {', '.join(self.timeframes)}"
            ) from exc

    async def fetch_candles(
        self, instrument: str, granularity: str, count: int | None = None
    ) -> pd.DataFrame:
        """Candles for one granularity, including the in-progress candle (``complete=False``)."""
        count = max(1, min(int(count or self.candle_count), 5000))
        params = {"granularity": granularity, "count": count, "price": self.price}
        try:
            data = await self._get_json(f"/v3/instruments/{instrument}/candles", params)
        except OandaNotFoundError as exc:
            raise OandaNotFoundError(
                f"Instrument '{instrument}' was not found on OANDA (404). "
                "Use OANDA format such as EUR_USD.",
                404,
            ) from exc
        except OandaBadRequestError as exc:
            if "instrument" in str(exc).lower():
                raise OandaNotFoundError(
                    f"Instrument '{instrument}' is not valid on OANDA: {exc}", 400
                ) from exc
            raise
        return parse_candles(data, self.price)

    async def get_spreads(
        self, instrument: str, timeframe: str, count: int | None = None
    ) -> pd.Series:
        """Bid/ask spread at each candle's close (price units), from OANDA's bid and ask
        candles: what trading at that moment cost, unlike a snapshot of the live spread."""
        count = max(1, min(int(count or self.candle_count), 5000))
        params = {"granularity": self.granularity_for(timeframe), "count": count, "price": "BA"}
        data = await self._get_json(f"/v3/instruments/{instrument}/candles", params)
        bid, ask = parse_candles(data, "B"), parse_candles(data, "A")
        return (ask["close"] - bid["close"]).dropna().rename("spread")

    async def get_candles(
        self, instrument: str, timeframes: list[str], count: int | None = None
    ) -> dict[str, pd.DataFrame]:
        """Fetch several timeframes concurrently. Keys are timeframe labels (``"5m"``)."""
        unique = list(dict.fromkeys(timeframes))
        granularities = [self.granularity_for(tf) for tf in unique]
        frames = await asyncio.gather(
            *(self.fetch_candles(instrument, g, count) for g in granularities)
        )
        return dict(zip(unique, frames, strict=True))

    # ------------------------------------------------------------------ pricing

    async def get_price(self, instrument: str) -> PriceTick:
        """Latest bid/ask snapshot (non-streaming)."""
        account = self._require_account()
        data = await self._get_json(f"/v3/accounts/{account}/pricing", {"instruments": instrument})
        prices = data.get("prices") or []
        if not prices:
            raise OandaNotFoundError(f"No price returned for {instrument}", 404)
        tick = parse_price(prices[0])
        if tick is None:
            raise OandaError(f"Malformed price payload for {instrument}")
        return tick


def parse_price(payload: Mapping[str, Any]) -> PriceTick | None:
    """Parse a PRICE message (REST snapshot or stream line). Returns None if unusable."""
    try:
        bids = payload.get("bids") or []
        asks = payload.get("asks") or []
        bid = float(bids[0]["price"]) if bids else float(payload["closeoutBid"])
        ask = float(asks[0]["price"]) if asks else float(payload["closeoutAsk"])
        return PriceTick(
            instrument=str(payload["instrument"]),
            time=pd.Timestamp(payload["time"]).floor("us").to_pydatetime(),
            bid=bid,
            ask=ask,
            tradeable=bool(payload.get("tradeable", payload.get("status") == "tradeable")),
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return None
