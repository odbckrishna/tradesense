"""Auto-refresh: re-run each timeframe on its ``refresh_seconds`` cycle.

Runs are aligned to the cycle boundary (i.e. just after a candle closes) plus a short
grace period so OANDA has published the completed candle. The next run is scheduled
only after the previous one finishes, so slow LLM calls never stack up.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from tradesense.config import TimeframeConfig
from tradesense.engine.orchestrator import EventCallback, Orchestrator


def next_run_delay(refresh_seconds: int, grace_seconds: float, now: float | None = None) -> float:
    """Seconds until the next ``refresh_seconds`` boundary (UTC epoch aligned) + grace."""
    now = time.time() if now is None else now
    boundary = math.floor(now / refresh_seconds + 1) * refresh_seconds
    return boundary + grace_seconds - now


class AutoRefresher:
    def __init__(
        self,
        orchestrator: Orchestrator,
        timeframes: Mapping[str, TimeframeConfig],
        on_event: EventCallback,
        *,
        grace_seconds: float = 3.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.orchestrator = orchestrator
        self.timeframes = dict(timeframes)
        self.on_event = on_event
        self.grace_seconds = grace_seconds
        self.clock = clock
        self.instrument: str | None = None
        self.selections: dict[str, list[str]] = {}
        self.next_runs: dict[str, datetime] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @property
    def active(self) -> bool:
        return any(not t.done() for t in self._tasks.values())

    def start(self, instrument: str, selections: Mapping[str, Sequence[str]]) -> None:
        self.stop()
        self.instrument = instrument
        self.selections = {
            tf: list(names) for tf, names in selections.items() if names and tf in self.timeframes
        }
        for timeframe, names in self.selections.items():
            self._tasks[timeframe] = asyncio.create_task(
                self._loop(instrument, timeframe, names), name=f"auto-refresh-{timeframe}"
            )
        logger.info("Auto-refresh started for {} ({})", instrument, ", ".join(self.selections))

    def stop(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        self.next_runs.clear()

    def status(self) -> dict[str, Any]:
        return {
            "type": "auto_refresh_status",
            "enabled": self.active,
            "instrument": self.instrument,
            "next_runs": {tf: ts.isoformat() for tf, ts in self.next_runs.items()},
        }

    async def _loop(self, instrument: str, timeframe: str, names: list[str]) -> None:
        refresh = self.timeframes[timeframe].refresh_seconds
        while True:
            delay = next_run_delay(refresh, self.grace_seconds, self.clock())
            self.next_runs[timeframe] = datetime.fromtimestamp(self.clock() + delay, UTC)
            await self.on_event(self.status())
            await asyncio.sleep(delay)
            try:
                await self.orchestrator.run_analysis(
                    instrument, {timeframe: names}, on_event=self.on_event
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Auto-refresh {} {} failed: {}", instrument, timeframe, exc)
                await self.on_event(
                    {"type": "analysis_error", "timeframe": timeframe, "message": str(exc)}
                )
