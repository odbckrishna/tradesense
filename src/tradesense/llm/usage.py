"""Token, latency and cost tracking."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from tradesense.config import PricingConfig
    from tradesense.llm.messages import LLMResponse


def compute_cost(
    pricing: PricingConfig | None, input_tokens: int, output_tokens: int
) -> float | None:
    """USD cost from per-million-token prices, or None when pricing is not configured."""
    if pricing is None:
        return None
    cost = (input_tokens * pricing.input_per_mtok + output_tokens * pricing.output_per_mtok) / 1e6
    return round(cost, 6)


class RunUsage(BaseModel):
    """Usage summary for one analysis run."""

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: int = 0
    llm_calls: int = 0
    decisions: int = 0

    @classmethod
    def from_decisions(cls, decisions: Iterable[Any]) -> RunUsage:
        """Aggregate objects exposing input/output tokens, cost and latency."""
        items = list(decisions)
        latencies = [int(d.latency_ms) for d in items if d.latency_ms]
        input_tokens = sum(int(d.input_tokens or 0) for d in items)
        output_tokens = sum(int(d.output_tokens or 0) for d in items)
        return cls(
            total_tokens=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost_usd=round(sum(float(d.cost_usd or 0.0) for d in items), 6),
            avg_latency_ms=int(sum(latencies) / len(latencies)) if latencies else 0,
            llm_calls=sum(int(getattr(d, "llm_calls", 1) or 0) for d in items),
            decisions=len(items),
        )


@dataclass
class ProviderUsage:
    calls: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    total_latency_ms: int = 0

    @property
    def avg_latency_ms(self) -> int:
        return int(self.total_latency_ms / self.calls) if self.calls else 0


@dataclass
class UsageTracker:
    """Process-wide counters per provider (thread-safe)."""

    by_provider: dict[str, ProviderUsage] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, response: LLMResponse) -> None:
        with self._lock:
            stats = self.by_provider.setdefault(response.provider, ProviderUsage())
            stats.calls += 1
            stats.input_tokens += response.input_tokens
            stats.output_tokens += response.output_tokens
            stats.cost_usd += response.cost_usd or 0.0
            stats.total_latency_ms += response.latency_ms

    def record_failure(self, provider: str) -> None:
        with self._lock:
            self.by_provider.setdefault(provider, ProviderUsage()).failures += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            providers = {
                name: {
                    "calls": s.calls,
                    "failures": s.failures,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cost_usd": round(s.cost_usd, 6),
                    "avg_latency_ms": s.avg_latency_ms,
                }
                for name, s in self.by_provider.items()
            }
        return {
            "since": self.started_at.isoformat(),
            "total_calls": sum(p["calls"] for p in providers.values()),
            "total_tokens": sum(p["input_tokens"] + p["output_tokens"] for p in providers.values()),
            "total_cost_usd": round(sum(p["cost_usd"] for p in providers.values()), 6),
            "providers": providers,
        }
