"""OpenRouter Decisions API (``POST /api/alpha/decisions``), e.g. ``typesafe/jev-1.13``.

Decision models return probabilities for typed questions instead of generated text, so
this adapter only serves trading-signal decisions (``request.task == "decision"``, no
images); give it the ``decision`` capability. The decision prompt becomes the ``state``,
one BUY / SELL / SIDEWAYS ``choice`` question is asked, and the answer is converted into
the decision JSON the app parses. There is no written reasoning and there are no key
levels. Only input tokens are billed: set ``pricing.input_per_mtok`` to track cost.

The health check asks for the API key's details (``GET /v1/key``): it proves the key
works without spending tokens, but cannot prove the model id exists.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

from tradesense.llm.base import (
    LLMBadRequestError,
    LLMProvider,
    LLMResponseError,
    LLMTransientError,
    classify_status,
)
from tradesense.llm.factory import register_provider
from tradesense.llm.messages import LLMRequest
from tradesense.llm.prompts import DECISION_CRITERIA, DECISION_QUESTION_INSTRUCTIONS

DEFAULT_BASE_URL = "https://openrouter.ai/api"
QUESTION = "signal"


@register_provider("openrouter_decisions")
class OpenRouterDecisionsProvider(LLMProvider):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client: httpx.AsyncClient | None = None

    @property
    def base_url(self) -> str:
        return (self.config.base_url or DEFAULT_BASE_URL).rstrip("/")

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        return self._client

    # ------------------------------------------------------------------ translation

    def build_body(self, request: LLMRequest) -> dict[str, Any]:
        # The first user message is the full decision prompt; later turns are only
        # JSON-correction follow-ups, which mean nothing to a decision model.
        return {
            "model": self.model,
            "state": request.messages[0].text,
            "questions": {
                QUESTION: {
                    "type": "choice",
                    "instructions": DECISION_QUESTION_INSTRUCTIONS,
                    "criteria": DECISION_CRITERIA,
                }
            },
        }

    def decision_json(self, answer: Mapping[str, Any]) -> str:
        """The ``choice`` answer as the decision JSON every other provider returns."""
        choice = answer.get("choice")
        raw = answer.get("probabilities")
        if not isinstance(raw, Mapping):
            if choice not in DECISION_CRITERIA:
                raise LLMResponseError(f"{self.name}: no usable answer", provider=self.name)
            raw = {choice: 1.0}
        probability = {k.lower(): float(raw.get(k) or 0) for k in DECISION_CRITERIA}
        confidence = answer.get("confidence")
        if not isinstance(confidence, int | float) or isinstance(confidence, bool):
            confidence = max(probability.values())
        summary = ", ".join(f"{k} {probability[k.lower()]:.0%}" for k in DECISION_CRITERIA)
        return json.dumps(
            {
                "signal": choice if choice in DECISION_CRITERIA else "SIDEWAYS",
                "probability": probability,
                "confidence": confidence,
                "observations": [f"Decision model probabilities: {summary}"],
                "reasoning": (
                    f"{self.model} is a decision model: it returns probabilities only, "
                    "with no written reasoning or key levels."
                ),
                "key_levels": {"support": None, "resistance": None},
                "pattern_match": None,
            }
        )

    # ------------------------------------------------------------------ calls

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self.client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise LLMTransientError(f"{self.name}: request timed out", provider=self.name) from exc
        except httpx.TransportError as exc:
            raise LLMTransientError(
                f"{self.name}: cannot reach {self.base_url}", provider=self.name
            ) from exc
        if response.status_code >= 400:
            message = f"{self.name}: {_error_message(response)}"
            raise classify_status(response.status_code, message, self.name)
        return response

    async def _generate(self, request: LLMRequest) -> tuple[str, int, int]:
        if request.task != "decision" or request.has_images:
            raise LLMBadRequestError(
                f"{self.name}: decision models only answer text trading-signal decisions",
                provider=self.name,
            )
        response = await self._send("POST", "/alpha/decisions", json=self.build_body(request))
        try:
            data = response.json()
            answer = data["answers"][QUESTION]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMResponseError(
                f"{self.name}: unexpected response: {response.text[:200]}", provider=self.name
            ) from exc
        usage = data.get("usage") or {}
        return (
            self.decision_json(answer),
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
        )

    async def health_check(self) -> bool:
        await self._send("GET", "/v1/key")
        return True

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
        message = error.get("message") if isinstance(error, dict) else str(error)
    except (ValueError, AttributeError):
        message = None
    return str(message or response.text[:300] or response.reason_phrase)
