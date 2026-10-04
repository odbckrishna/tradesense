"""New rule candidates proposed by a text LLM, validated by the rule engine.

The request carries no ``task``, so decision-only providers (e.g. Jev) are skipped and
the next text-generating model in the ``text`` chain answers. Output that does not parse
into at least one valid rule gets the router's one correction retry.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from tradesense.llm.prompts import build_rule_design_request
from tradesense.llm.router import LLMRouter
from tradesense.llm.schemas import DecisionParseError, extract_json_object
from tradesense.strategies.rule_engine import RuleStrategy, RuleValidationError, load_rule

MAX_NAME = 16  # saving prefixes "bt_<instrument>_<timeframe>_"


def _slug(value: Any, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return (slug or fallback)[:MAX_NAME].rstrip("_") or fallback


def parse_rule_designs(text: str, limit: int) -> list[RuleStrategy]:
    """``{"rules": [...]}`` -> validated rule strategies named ``llm_<slug>``. Raises
    :class:`DecisionParseError` when no rule is valid."""
    raw_rules = extract_json_object(text).get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise DecisionParseError("'rules' must be a non-empty list")
    rules: list[RuleStrategy] = []
    errors: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rules[:limit], start=1):
        if not isinstance(raw, dict):
            errors.append(f"rule {index} is not an object")
            continue
        name = "llm_" + _slug(raw.get("name"), f"rule{index}")[: MAX_NAME - 4]
        while name in seen:
            name = f"{name[: MAX_NAME - 2]}_{index}"
        spec = {
            "name": name,
            "display_name": str(raw.get("name") or name)[:60],
            "description": str(raw.get("description") or "Rule proposed by the LLM")[:300],
            "category": raw.get("category"),
            "indicators": raw.get("indicators"),
            "rules": {"buy": raw.get("buy"), "sell": raw.get("sell")},
        }
        try:
            rules.append(load_rule(spec, f"rule {index}"))
        except RuleValidationError as exc:
            errors.append(str(exc))
            continue
        seen.add(name)
    if not rules:
        raise DecisionParseError("no valid rule: " + "; ".join(errors)[:600])
    return rules


async def design_rules(
    router: LLMRouter,
    *,
    instrument: str,
    timeframe: str,
    market: Sequence[str],
    tested: Sequence[str],
    count: int,
) -> tuple[list[RuleStrategy], str, float | None]:
    """Rules proposed by the LLM, the provider that answered and the cost."""
    request = build_rule_design_request(
        instrument=instrument, timeframe=timeframe, market=market, tested=tested, count=count
    )
    result = await router.generate_structured(
        "text", request, lambda text: parse_rule_designs(text, count)
    )
    costs = [r.cost_usd for r in result.responses if r.cost_usd is not None]
    return result.value, result.provider, (round(sum(costs), 6) if costs else None)
