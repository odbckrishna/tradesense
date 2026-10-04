"""Configuration loading.

Secrets come only from the process environment / ``.env`` (python-dotenv +
pydantic-settings). Everything else comes from YAML files under ``config/``,
validated into pydantic models. Invalid YAML or schema errors raise
:class:`ConfigError` with the offending file in the message ("fail fast").
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["practice", "live"]
#: ``decision``: decision-only models (probabilities, no generated text) that can serve
#: trading-signal decisions but not extraction or summaries.
Capability = Literal["text", "vision", "decision"]
RoleName = Literal["text", "vision", "extraction"]

#: OANDA candle granularities and their length in seconds.
GRANULARITY_SECONDS: dict[str, int] = {
    "S5": 5,
    "S10": 10,
    "S15": 15,
    "S30": 30,
    "M1": 60,
    "M2": 120,
    "M4": 240,
    "M5": 300,
    "M10": 600,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D": 86400,
}

#: File types the strategy library can ingest (validated by signature, not extension).
SUPPORTED_UPLOAD_TYPES = ("md", "txt", "pdf", "docx", "png", "jpg", "jpeg", "webp")


class ConfigError(RuntimeError):
    """Raised for missing, unreadable or invalid configuration."""


# --------------------------------------------------------------------------- paths


@dataclass(frozen=True)
class Paths:
    """Well-known locations, all derived from the project home directory."""

    home: Path

    @property
    def config_dir(self) -> Path:
        return self.home / "config"

    @property
    def app_config(self) -> Path:
        return self.config_dir / "app.yml"

    @property
    def llm_config(self) -> Path:
        return self.config_dir / "llm.yml"

    @property
    def llm_selection(self) -> Path:
        """Per-role model choices made in the UI (git-ignored; see llm/selection.py)."""
        return self.config_dir / "llm_selection.json"

    @property
    def strategies_config(self) -> Path:
        return self.config_dir / "strategies.yml"

    @property
    def rules_dir(self) -> Path:
        return self.config_dir / "rules"

    @property
    def library_dir(self) -> Path:
        return self.home / "strategy_library"

    @property
    def env_file(self) -> Path:
        return self.home / ".env"


def resolve_home(explicit: str | Path | None = None) -> Path:
    """Find the project root.

    Order: explicit argument, ``TRADESENSE_HOME``, the current directory if it holds
    ``config/app.yml``, then the source checkout this package lives in.
    """
    if explicit:
        return Path(explicit).resolve()
    env_home = os.environ.get("TRADESENSE_HOME")
    if env_home:
        return Path(env_home).resolve()
    cwd = Path.cwd()
    if (cwd / "config" / "app.yml").is_file():
        return cwd
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "config" / "app.yml").is_file():
        return checkout
    return cwd


# --------------------------------------------------------------------------- secrets


class OandaSecrets(BaseSettings):
    """OANDA credentials, read from the environment (``.env`` is loaded beforehand)."""

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    oanda_api_key: SecretStr | None = None
    oanda_account_id: str | None = None
    oanda_environment: Environment = "practice"

    @field_validator("oanda_api_key", "oanda_account_id", mode="before")
    @classmethod
    def _blank_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("oanda_environment", mode="before")
    @classmethod
    def _normalise_env(cls, value: Any) -> Any:
        if value is None or (isinstance(value, str) and not value.strip()):
            return "practice"
        return value.strip().lower() if isinstance(value, str) else value

    @property
    def api_key(self) -> str | None:
        return self.oanda_api_key.get_secret_value() if self.oanda_api_key else None


# --------------------------------------------------------------------------- app.yml


def _check_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value.startswith(("https://", "http://")):
        raise ValueError(f"URL must start with http:// or https://: {value!r}")
    return value


class OandaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rest_url: dict[Environment, str]
    stream_url: dict[Environment, str]
    candle_count: int = Field(default=300, ge=20, le=5000)
    price: Literal["M", "B", "A"] = "M"
    request_timeout_seconds: float = Field(default=15.0, gt=0)
    instruments_cache_seconds: int = Field(default=3600, ge=0)

    @field_validator("rest_url", "stream_url")
    @classmethod
    def _urls(cls, value: dict[Environment, str]) -> dict[Environment, str]:
        if "practice" not in value:
            raise ValueError("a 'practice' URL is required")
        return {env: _check_url(url) for env, url in value.items()}


class TimeframeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    granularity: str
    refresh_seconds: int = Field(gt=0)

    @field_validator("granularity")
    @classmethod
    def _granularity(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in GRANULARITY_SECONDS:
            raise ValueError(
                f"unknown OANDA granularity {value!r}; expected one of "
                f"{', '.join(GRANULARITY_SECONDS)}"
            )
        return value

    @property
    def granularity_seconds(self) -> int:
        return GRANULARITY_SECONDS[self.granularity]


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_candles: int = Field(default=20, ge=5, le=200)
    document_prompt_candles: int = Field(default=50, ge=5, le=300)
    chart_candles: int = Field(default=100, ge=20, le=500)
    history_length: int = Field(default=10, ge=1, le=50)
    refresh_grace_seconds: float = Field(default=3.0, ge=0, le=120)


class LibraryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_upload_mb: float = Field(default=10, gt=0, le=200)
    allowed_types: list[str] = Field(default_factory=lambda: list(SUPPORTED_UPLOAD_TYPES))
    max_document_chars: int = Field(default=40_000, ge=2_000)
    import_roots: list[str] = Field(default_factory=lambda: ["strategy_library/inbox"])

    @field_validator("allowed_types")
    @classmethod
    def _types(cls, value: list[str]) -> list[str]:
        cleaned = [v.strip().lower().lstrip(".") for v in value]
        unknown = sorted(set(cleaned) - set(SUPPORTED_UPLOAD_TYPES))
        if unknown:
            raise ValueError(
                f"unsupported upload types {unknown}; supported: {list(SUPPORTED_UPLOAD_TYPES)}"
            )
        return cleaned

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    days: float = Field(default=2, gt=0, le=10)
    #: Candles after a signal at which BUY / SELL is judged right or wrong.
    horizon_bars: dict[str, int] = Field(default_factory=lambda: {"1m": 15, "5m": 6, "15m": 4})
    default_horizon_bars: int = Field(default=5, ge=1, le=500)
    min_signals: int = Field(default=10, ge=1, le=10_000)
    llm_top_n: int = Field(default=3, ge=0, le=10)
    llm_samples: int = Field(default=12, ge=1, le=100)
    llm_rule_candidates: int = Field(default=3, ge=1, le=5)
    llm_timeout_seconds: float = Field(default=180, gt=0, le=1800)  # per LLM stage
    chunk_bars: int = Field(default=360, ge=50, le=5000)  # replay work unit per process
    max_workers: int = Field(default=0, ge=0, le=64)  # 0 = CPU count - 1

    @field_validator("horizon_bars")
    @classmethod
    def _horizons(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not 1 <= v <= 500 for v in value.values()):
            raise ValueError("horizon_bars values must be between 1 and 500")
        return value

    def horizon_for(self, timeframe: str) -> int:
        return self.horizon_bars.get(timeframe, self.default_horizon_bars)

    def scored_bars(self, granularity_seconds: int) -> int:
        """Candles in the backtest period (``days`` of trading time)."""
        return max(1, round(self.days * 86_400 / granularity_seconds))


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    oanda: OandaConfig
    timeframes: dict[str, TimeframeConfig]
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    library: LibraryConfig = Field(default_factory=LibraryConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @field_validator("timeframes")
    @classmethod
    def _timeframes(cls, value: dict[str, TimeframeConfig]) -> dict[str, TimeframeConfig]:
        if not value:
            raise ValueError("at least one timeframe must be configured")
        return value

    def backtest_candles(self, timeframe: str) -> int:
        """Candles one backtest fetches: the period, a live-sized window before it and
        the horizon after it."""
        tf = self.timeframes[timeframe]
        bt = self.backtest
        return (
            bt.scored_bars(tf.granularity_seconds)
            + self.oanda.candle_count
            + bt.horizon_for(timeframe)
        )

    @model_validator(mode="after")
    def _backtest_fits(self) -> AppConfig:
        for timeframe in self.timeframes:
            needed = self.backtest_candles(timeframe)
            if needed > 5000:
                raise ValueError(
                    f"backtest needs {needed} {timeframe} candles but OANDA returns at most "
                    "5000 per request: lower backtest.days or oanda.candle_count"
                )
        return self


# --------------------------------------------------------------------------- strategies.yml


class StrategyParamsConfig(BaseModel):
    """Parameters for one strategy plus optional per-timeframe overrides."""

    model_config = ConfigDict(extra="forbid")

    params: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("params", mode="before")
    @classmethod
    def _none_params(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("overrides", mode="before")
    @classmethod
    def _none_overrides(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {str(k): ({} if v is None else v) for k, v in value.items()}
        return value


class StrategiesConfig(BaseModel):
    strategies: dict[str, StrategyParamsConfig] = Field(default_factory=dict)

    def for_strategy(self, name: str) -> StrategyParamsConfig:
        return self.strategies.get(name) or StrategyParamsConfig()


# --------------------------------------------------------------------------- llm.yml


class PricingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float = Field(default=0.0, ge=0)
    output_per_mtok: float = Field(default=0.0, ge=0)


def _default_capabilities() -> list[Capability]:
    return ["text"]


class ProviderConfig(BaseModel):
    """One LLM provider entry. ``type`` is resolved against the provider registry."""

    model_config = ConfigDict(extra="forbid")

    type: str
    model: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    capabilities: list[Capability] = Field(default_factory=lambda: _default_capabilities())
    max_concurrency: int | None = Field(default=None, ge=1)
    pricing: PricingConfig | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)
    json_mode: bool | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    # Azure OpenAI
    azure: bool = False
    endpoint: str | None = None
    deployment: str | None = None
    api_version: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_inline_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for forbidden in ("api_key", "key", "token", "secret"):
                if forbidden in data:
                    raise ValueError(
                        f"'{forbidden}' must not be written in llm.yml; put the key in .env "
                        "and reference it with api_key_env"
                    )
        return data

    @field_validator("type")
    @classmethod
    def _type(cls, value: str) -> str:
        return value.strip().lower()

    @model_validator(mode="after")
    def _check(self) -> ProviderConfig:
        if self.azure:
            missing = [f for f in ("endpoint", "deployment", "api_version") if not getattr(self, f)]
            if missing:
                raise ValueError(f"azure providers require: {', '.join(missing)}")
        elif not self.model:
            raise ValueError("'model' is required")
        if not self.capabilities:
            raise ValueError("at least one capability is required")
        return self

    @property
    def model_name(self) -> str:
        """The model id sent to the API (the deployment name for Azure)."""
        return (self.deployment if self.azure else self.model) or ""


class RoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary: str
    fallbacks: list[str] = Field(default_factory=list)

    @property
    def chain(self) -> list[str]:
        seen: list[str] = []
        for name in [self.primary, *self.fallbacks]:
            if name not in seen:
                seen.append(name)
        return seen


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_concurrency: int = Field(default=4, ge=1, le=256)
    timeout_seconds: float = Field(default=90, gt=0)
    retries: int = Field(default=2, ge=0, le=10)
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_output_tokens: int = Field(default=1024, ge=64, le=64_000)
    force_json: bool = True
    track_costs: bool = True
    retry_backoff_seconds: float = Field(default=1.0, ge=0, le=60)
    health_check_timeout_seconds: float = Field(default=10, gt=0)


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roles: dict[RoleName, RoleConfig]
    settings: LLMSettings = Field(default_factory=LLMSettings)
    providers: dict[str, ProviderConfig]

    @model_validator(mode="after")
    def _check_roles(self) -> LLMConfig:
        if "text" not in self.roles:
            raise ValueError("a 'text' role is required")
        for role, cfg in self.roles.items():
            for name in cfg.chain:
                if name not in self.providers:
                    raise ValueError(f"role '{role}' references unknown provider '{name}'")
        return self

    def role_chain(self, role: RoleName) -> list[str]:
        """Provider names for a role. ``extraction`` falls back to the text chain."""
        if role in self.roles:
            return self.roles[role].chain
        if role == "extraction":
            return self.roles["text"].chain
        return []

    def used_providers(self) -> list[str]:
        names: list[str] = []
        for cfg in self.roles.values():
            for name in cfg.chain:
                if name not in names:
                    names.append(name)
        return names


# --------------------------------------------------------------------------- loading


def read_yaml(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc}") from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc


def format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def _validate[ModelT: BaseModel](model: type[ModelT], data: Any, source: Path | str) -> ModelT:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid configuration in {source}:\n{format_validation_error(exc)}"
        ) from exc


def load_app_config(path: Path) -> AppConfig:
    data = read_yaml(path)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return _validate(AppConfig, data, path)


def load_strategies_config(path: Path) -> StrategiesConfig:
    if not path.exists():
        return StrategiesConfig()
    data = read_yaml(path) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    normalised = {str(k): ({} if v is None else v) for k, v in data.items()}
    return _validate(StrategiesConfig, {"strategies": normalised}, path)


def parse_llm_config(data: Any, source: Path | str = "llm.yml") -> LLMConfig:
    if not isinstance(data, dict):
        raise ConfigError(f"{source} must contain a mapping at the top level")
    return _validate(LLMConfig, data, source)


def load_llm_config(path: Path) -> LLMConfig:
    return parse_llm_config(read_yaml(path), path)


@dataclass
class Settings:
    """Everything loaded at startup."""

    paths: Paths
    app: AppConfig
    strategies: StrategiesConfig
    llm: LLMConfig
    oanda: OandaSecrets

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a config-relative path (``~`` expanded) against the project home."""
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.paths.home / path)


def load_settings(home: str | Path | None = None, *, load_env: bool = True) -> Settings:
    """Load ``.env`` (without overriding real env vars) and all YAML config files."""
    paths = Paths(resolve_home(home))
    if load_env and paths.env_file.is_file():
        load_dotenv(paths.env_file, override=False, encoding="utf-8")
    return Settings(
        paths=paths,
        app=load_app_config(paths.app_config),
        strategies=load_strategies_config(paths.strategies_config),
        llm=load_llm_config(paths.llm_config),
        oanda=OandaSecrets(),
    )


def secret_values(
    settings: Settings, environ: Iterable[tuple[str, str]] | None = None
) -> list[str]:
    """Values of every secret the app knows about, used to redact them from logs."""
    env = dict(environ) if environ is not None else dict(os.environ)
    names = {"OANDA_API_KEY"}
    names.update(p.api_key_env for p in settings.llm.providers.values() if p.api_key_env)
    names.update(k for k in env if k.endswith(("_API_KEY", "_TOKEN", "_SECRET")))
    values = [env[n] for n in names if env.get(n)]
    return [v for v in values if len(v) >= 6]
