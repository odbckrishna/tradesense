from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger

from tradesense.config import (
    ConfigError,
    OandaSecrets,
    Settings,
    load_app_config,
    load_llm_config,
    load_strategies_config,
    parse_llm_config,
    secret_values,
)
from tradesense.logging_setup import SecretRedactor


def test_repo_config_loads(settings: Settings) -> None:
    assert list(settings.app.timeframes) == ["1m", "5m", "15m"]
    assert settings.app.timeframes["5m"].granularity == "M5"
    assert settings.app.timeframes["15m"].granularity_seconds == 900
    assert settings.app.oanda.price == "M"
    assert settings.app.oanda.rest_url["practice"] == "https://api-fxpractice.oanda.com"
    assert settings.app.library.max_upload_bytes == 10 * 1024 * 1024
    assert settings.llm.roles["text"].chain == [
        "openrouter_jev",
        "anthropic_sonnet",
        "openai_main",
        "openrouter",
        "ollama_text",
    ]
    ema = settings.strategies.for_strategy("ema_crossover")
    assert ema.params == {"fast": 9, "slow": 21}
    assert ema.overrides["15m"] == {"fast": 50, "slow": 200}


def test_oanda_secrets_default_to_practice() -> None:
    secrets = OandaSecrets(oanda_api_key="", oanda_account_id=" ", oanda_environment="")
    assert secrets.api_key is None
    assert secrets.oanda_account_id is None
    assert secrets.oanda_environment == "practice"
    live = OandaSecrets(oanda_api_key="abc", oanda_environment=" LIVE ")
    assert live.api_key == "abc"
    assert live.oanda_environment == "live"


def test_invalid_yaml_fails_fast(tmp_path: Path) -> None:
    bad = tmp_path / "app.yml"
    bad.write_text("oanda: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_app_config(bad)


def test_unknown_granularity_rejected(tmp_path: Path) -> None:
    path = tmp_path / "app.yml"
    path.write_text(
        "oanda:\n  rest_url: {practice: https://x}\n  stream_url: {practice: https://y}\n"
        "timeframes:\n  '2m': {granularity: M2X, refresh_seconds: 120}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="granularity"):
        load_app_config(path)


def test_missing_file_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_llm_config(tmp_path / "nope.yml")


def test_strategies_config_tolerates_empty_entries(tmp_path: Path) -> None:
    path = tmp_path / "strategies.yml"
    path.write_text("ema_crossover:\nrsi_extremes:\n  params:\n  overrides:\n", encoding="utf-8")
    cfg = load_strategies_config(path)
    assert cfg.for_strategy("ema_crossover").params == {}
    assert cfg.for_strategy("unknown").overrides == {}


def _llm(providers: dict[str, dict[str, object]], roles: dict[str, object]) -> dict[str, object]:
    return {"roles": roles, "providers": providers}


def test_llm_role_referencing_unknown_provider_fails() -> None:
    data = _llm(
        {"a": {"type": "ollama", "model": "m"}}, {"text": {"primary": "a", "fallbacks": ["ghost"]}}
    )
    with pytest.raises(ConfigError, match="unknown provider 'ghost'"):
        parse_llm_config(data)


def test_llm_text_role_required() -> None:
    data = _llm({"a": {"type": "ollama", "model": "m"}}, {"vision": {"primary": "a"}})
    with pytest.raises(ConfigError, match="'text' role is required"):
        parse_llm_config(data)


def test_llm_inline_api_key_rejected() -> None:
    data = _llm(
        {"a": {"type": "openai", "model": "m", "api_key": "sk-123"}}, {"text": {"primary": "a"}}
    )
    with pytest.raises(ConfigError, match=r"must not be written in llm\.yml"):
        parse_llm_config(data)


def test_llm_azure_requires_deployment() -> None:
    data = _llm(
        {"az": {"type": "openai", "azure": True, "endpoint": "https://x"}},
        {"text": {"primary": "az"}},
    )
    with pytest.raises(ConfigError, match="deployment"):
        parse_llm_config(data)


def test_extraction_role_falls_back_to_text_chain() -> None:
    cfg = parse_llm_config(
        _llm({"a": {"type": "ollama", "model": "m"}}, {"text": {"primary": "a"}})
    )
    assert cfg.role_chain("extraction") == ["a"]
    assert cfg.role_chain("vision") == []


def test_secret_values_and_redaction(settings: Settings) -> None:
    env = [
        ("ANTHROPIC_API_KEY", "sk-ant-secret-123456"),
        ("OANDA_API_KEY", "oanda-token-abcdef"),
        ("UNRELATED", "visible-value"),
    ]
    secrets = secret_values(settings, env)
    assert set(secrets) == {"sk-ant-secret-123456", "oanda-token-abcdef"}

    redactor = SecretRedactor(secrets)
    captured: list[str] = []
    sink_id = logger.add(lambda m: captured.append(str(m)), filter=redactor, format="{message}")
    try:
        logger.info("calling with key sk-ant-secret-123456 and oanda-token-abcdef")
    finally:
        logger.remove(sink_id)
    assert captured
    assert "secret-123456" not in captured[0]
    assert "abcdef" not in captured[0]
    assert captured[0].count("***") == 2
