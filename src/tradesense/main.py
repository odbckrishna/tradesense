"""TradeSense entry point: FastAPI (web UI + REST + WebSocket) with FastMCP mounted.

``uv run tradesense``          start the server (config/app.yml ``server`` section)
``uv run tradesense --check``  validate configuration and connectivity, then exit
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastmcp.utilities.lifespan import combine_lifespans
from loguru import logger

from tradesense import __version__
from tradesense.api import routes, websocket
from tradesense.backtest.engine import BacktestBusyError, BacktestNotFoundError
from tradesense.config import ConfigError, Settings, load_settings, secret_values
from tradesense.data.models import completed
from tradesense.data.oanda_client import (
    InstrumentError,
    InstrumentNotFoundError,
    OandaAuthError,
    OandaClient,
    OandaConfigError,
    OandaError,
    OandaNotFoundError,
    OandaRateLimitError,
)
from tradesense.llm.base import LLMConfigError
from tradesense.logging_setup import configure_logging
from tradesense.mcp_server import create_mcp
from tradesense.services import Services
from tradesense.strategies.base import UnknownStrategyError
from tradesense.strategies.ingestion import UploadError
from tradesense.strategies.reference import LibraryError

WEB_DIR = Path(__file__).parent / "web"

# The hash allows exactly one inline stylesheet: the attribution-logo CSS that
# Lightweight Charts 5.2.1 injects (its license asks to keep the TradingView attribution).
LWC_ATTRIBUTION_STYLE = "'sha256-3pRED1tOXas1FXFoPb9TGCjmYe9XQsmO9OV23khV2nY='"

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self' https://unpkg.com; "
    f"style-src 'self' {LWC_ATTRIBUTION_STYLE}; "
    "img-src 'self' data: blob:; connect-src 'self' ws: wss:; font-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'"
)


def _error_status(exc: Exception) -> int:
    match exc:
        case InstrumentNotFoundError() | OandaNotFoundError():
            return 404
        case InstrumentError():
            return 422
        case OandaConfigError():
            return 503
        case OandaRateLimitError():
            return 429
        case OandaAuthError() | OandaError():
            return 502
        case UploadError() | LibraryError():
            return exc.status_code
        case UnknownStrategyError() | BacktestNotFoundError():
            return 404
        case BacktestBusyError():
            return 409
        case ConfigError() | LLMConfigError() | ValueError():
            return 400
    return 500


def create_app(
    settings: Settings | None = None,
    services: Services | None = None,
    *,
    health_checks: bool = True,
) -> FastAPI:
    """Build the ASGI app. ``services`` may be pre-built (fail fast in ``main``, or
    injected by tests); otherwise they are created at startup."""
    settings = settings or (services.settings if services else load_settings())
    holder: dict[str, Services] = {}

    def current() -> Services:
        return holder["services"]

    mcp = create_mcp(current)
    mcp_app = mcp.http_app(path="/mcp")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc = services or Services(settings)
        holder["services"] = svc
        app.state.services = svc
        await svc.start(health_checks=health_checks)
        try:
            yield
        finally:
            await svc.stop()

    app = FastAPI(
        title="TradeSense",
        version=__version__,
        description="Market analysis signals (no order placement). Not financial advice.",
        lifespan=combine_lifespans(lifespan, mcp_app.lifespan),
    )

    for error_type in (
        InstrumentError,
        OandaError,
        UploadError,
        LibraryError,
        UnknownStrategyError,
        BacktestBusyError,
        BacktestNotFoundError,
        ConfigError,
        LLMConfigError,
        ValueError,
    ):
        app.add_exception_handler(error_type, _domain_error)
    app.add_exception_handler(RequestValidationError, _validation_error)

    max_body = settings.app.library.max_upload_bytes + 1024 * 1024  # multipart overhead

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_body:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if response.headers.get("content-type", "").startswith("text/html"):
            response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
            response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request) -> Response:
        return templates.TemplateResponse(
            request, "index.html", {"version": __version__, "disclaimer": routes.DISCLAIMER}
        )

    app.include_router(routes.router)
    app.include_router(websocket.router)
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
    app.mount("/", mcp_app)  # serves /mcp; must stay last so API routes match first
    return app


async def _domain_error(request: Request, exc: Exception) -> JSONResponse:
    status = _error_status(exc)
    if status >= 500:
        logger.warning("{} {} -> {}: {}", request.method, request.url.path, status, exc)
    return JSONResponse({"detail": str(exc)}, status_code=status)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    errors: Any = exc.errors() if isinstance(exc, RequestValidationError) else str(exc)
    return JSONResponse({"detail": errors}, status_code=422)


# --------------------------------------------------------------------------- CLI


async def run_check(settings: Settings, instrument: str = "EUR_USD") -> bool:
    """Print a configuration summary, probe LLM providers and OANDA. True when healthy."""
    ok = True
    print(f"Home:        {settings.paths.home}")
    print(f"Environment: {settings.oanda.oanda_environment}")
    print(f"Timeframes:  {', '.join(settings.app.timeframes)}")
    services = Services(settings)
    await services.router.check_health()
    for role, status in services.router.role_status().items():
        state = "OK" if status["healthy"] else "UNAVAILABLE"
        fallback = " (fallback)" if status["fallback_in_use"] else ""
        print(f"LLM {role:<10} {status['active']} / {status['active_model']}{fallback}: {state}")
        ok = ok and bool(status["healthy"])
    for issue in services.llm_issues:
        print(f"  llm.yml {issue.level}: {issue.message}")
    oanda: OandaClient = services.oanda
    if not oanda.configured:
        print("OANDA:       NOT CONFIGURED (set OANDA_API_KEY / OANDA_ACCOUNT_ID in .env)")
        ok = False
    else:
        try:
            resolved = await oanda.resolve_instrument(instrument)
            frames = await oanda.get_candles(resolved, list(settings.app.timeframes), count=5)
            for tf, df in frames.items():
                print(
                    f"OANDA {tf:>4}:  {len(df)} candles, {len(completed(df))} complete, "
                    f"last close {df['close'].iloc[-1] if len(df) else 'n/a'}"
                )
        except (OandaError, ValueError) as exc:
            print(f"OANDA:       ERROR {exc}")
            ok = False
    await services.stop()
    return ok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tradesense", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--home", help="project root (default: current directory)")
    parser.add_argument("--host", help="override server.host from config/app.yml")
    parser.add_argument("--port", type=int, help="override server.port from config/app.yml")
    parser.add_argument("--log-level", help="DEBUG, INFO, WARNING, ...")
    parser.add_argument(
        "--check", action="store_true", help="validate configuration and connectivity, then exit"
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    try:
        settings = load_settings(args.home)
    except ConfigError as exc:
        logger.error("{}", exc)
        sys.exit(2)
    configure_logging(args.log_level, secret_values(settings))

    if args.check:
        sys.exit(0 if asyncio.run(run_check(settings)) else 1)

    import uvicorn

    host = args.host or settings.app.server.host
    port = args.port or settings.app.server.port
    try:
        app = create_app(settings, Services(settings))  # llm.yml problems fail fast here
    except (ConfigError, LLMConfigError) as exc:
        logger.error("{}", exc)
        sys.exit(2)
    logger.info("Starting TradeSense on http://{}:{} (MCP endpoint: /mcp)", host, port)
    uvicorn.run(app, host=host, port=port, log_level=(args.log_level or "info").lower())


if __name__ == "__main__":  # pragma: no cover
    main()
