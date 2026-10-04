# TradeSense

Market analysis platform that pulls OANDA candles and live prices, runs any mix of
strategies on the 1-minute, 5-minute and 15-minute timeframes **in parallel**, and asks a
pluggable LLM for a probability-based decision: **BUY / SELL / SIDEWAYS**.

> **Signals only.** TradeSense never places orders: there is no order, trade or position
> endpoint anywhere. Signals are AI-generated and are not financial advice.

- **Data:** OANDA v20 REST (candles, instruments, prices) and the pricing stream (live ticks)
- **Strategies:** one shared pool, selectable per timeframe at runtime, in four kinds:
  built-in Python classes, YAML rules, and uploaded reference documents and chart images
- **LLMs:** Anthropic, OpenAI / Azure OpenAI, Gemini, Ollama, or any OpenAI-compatible API
  (vLLM, Groq, OpenRouter, Together, LM Studio). Roles, fallbacks and limits live in YAML
- **Interfaces:** web dashboard, REST API, WebSocket streaming, and MCP tools (FastMCP)

---

## Quick start

```bash
uv sync                      # install (Python 3.12, managed by uv)
cp .env.example .env         # then fill in OANDA + at least one LLM key
uv run tradesense --check    # validate config, probe LLM providers and OANDA
uv run tradesense            # http://localhost:8000  (MCP endpoint: /mcp)
uv run pytest                # test-suite (OANDA and every LLM SDK are mocked)
```

`uv run tradesense --host 127.0.0.1 --port 8080` overrides `server` from `config/app.yml`.

On Windows you can also run it as a background service that keeps running after the
terminal closes (binds to `127.0.0.1:8000`; output goes to `logs/tradesense.log`):

```bat
scripts\start-local.cmd            :: options: -Port 8080, -BindHost 0.0.0.0
scripts\stop-local.cmd
```

> **Network exposure:** `config/app.yml` ships with `host: 0.0.0.0`, which makes the UI,
> the REST API (uploads, LLM-cost-incurring analysis) and the MCP endpoint reachable from
> your network **without authentication**. Use `127.0.0.1` unless you put TradeSense
> behind an authenticating reverse proxy.

---

## How it works

```
                 ┌──────────── run_analysis(instrument, {tf: [strategies]}) ────────────┐
 OANDA REST ───► │ candles for all timeframes (concurrent)                               │
                 │   └─ per timeframe: feature pack (lazy, once) shared by references    │
                 │ every (timeframe, strategy) pair concurrently (asyncio.gather):       │
                 │   strategy.compute(df) ─► FEATURES ─► LLM router ─► DECISION          │
                 │ per timeframe: confidence-weighted CONSENSUS                          │
                 └──────── each result streamed over WebSocket as soon as it lands ──────┘
```

* Strategies compute **features** (indicator values, ~10 values of history, detected
  events and a rule-based pre-signal). The LLM makes the **decision**.
* Analysis uses completed candles only; the in-progress candle is kept for the chart.
* One failing strategy (or timeframe) never breaks the others: the slot shows an error.
* Fewer candles than a strategy needs gives an `insufficient_data` card instead of a failure.
* If no LLM provider returns valid JSON, the decision falls back to SIDEWAYS with
  confidence 0 and an "LLM parse error" (or "LLM unavailable") observation.

Consensus per timeframe: `p = Σ(confidence_i · p_i) / Σ confidence_i`; the signal is the
highest probability; consensus confidence = mean confidence × the share of confidence
weight that agrees with the consensus signal.

---

## Configuration

| File | Purpose |
|---|---|
| `.env` | Secrets only: `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, `OANDA_ENVIRONMENT` (`practice` default) and LLM keys |
| `config/app.yml` | OANDA URLs, candle count, price type, timeframes + refresh cycle, prompt sizes, upload limits, server |
| `config/llm.yml` | LLM providers, roles (`text`, `vision`, `extraction`) with ordered fallbacks, global settings |
| `config/llm_selection.json` | Per-role model choices made in the UI's model picker (written by the app, git-ignored) |
| `config/strategies.yml` | Strategy parameters with per-timeframe overrides |
| `config/rules/*.yml` | YAML rule strategies (hot-reloaded) |
| `strategy_library/` | Uploaded reference files and their strategy cards (git-ignored) |

Invalid YAML, unknown keys, unknown provider types or a role that points at a missing
provider stop startup with a message naming the file. A provider whose API key env var is
unset is skipped with a warning (the role falls back). If a role has no usable provider,
the UI shows an error banner and its decisions fall back to SIDEWAYS.

### `config/llm.yml`

```yaml
roles:
  text:       { primary: openrouter_jev, fallbacks: [anthropic_sonnet, openai_main, ollama_text] }
  vision:     { primary: anthropic_sonnet, fallbacks: [openai_main, ollama_vision] }
  extraction: { primary: anthropic_sonnet, fallbacks: [ollama_text] }

settings:
  max_concurrency: 4      # global cap on parallel LLM calls
  timeout_seconds: 90
  retries: 2              # per provider, before falling back
  temperature: 0.1
  max_output_tokens: 1024
  force_json: true
  track_costs: true

providers:
  anthropic_sonnet:
    type: anthropic
    model: claude-sonnet-5-5
    api_key_env: ANTHROPIC_API_KEY   # the NAME of the env var, never the key
    capabilities: [text, vision]
    max_concurrency: 4               # optional per-provider cap
    pricing: { input_per_mtok: 0.0, output_per_mtok: 0.0 }   # optional: your USD price per 1M tokens
```

Switching providers is a one-line change of `primary`. Writing `api_key:` in this file is
rejected on purpose.

Routing rules:

* **Transient errors** (timeouts, 429 rate limits, 5xx): retried with exponential backoff,
  then the next fallback is tried.
* **Auth errors** (401/403) and **exhausted credits**: no retries. The router goes straight
  to the next fallback and skips that provider until the LLM config is reloaded.
* **Image requests** skip providers without the `vision` capability.
* **Concurrency:** the global and per-provider semaphores are separate, and neither is
  held during backoff.
* **Status:** the provider that answered is recorded in `decided_by`. The UI shows the
  active provider per role, its health, and whether a fallback is in use.

Optional per-provider keys:

| Key | Applies to | Meaning |
|---|---|---|
| `timeout_seconds` | all | Overrides the global timeout |
| `json_mode` | all | `false` disables native JSON mode (prompt instructions only) |
| `options.structured_output` | anthropic | `false` disables `output_config` JSON-schema output (default on, auto-downgrades if rejected) |
| `options.effort` | anthropic | `low` … `max` |
| `options.token_param` | openai, openai_compatible | `max_tokens` or `max_completion_tokens` |
| `options.send_temperature` | openai, openai_compatible | `false` for reasoning models that reject temperature |
| `options.extra_body` | openai, openai_compatible | Extra request fields passed through |
| `options.thinking_budget` | gemini | e.g. `0` to disable thinking on Flash models |
| `options.think` | ollama | `false` keeps thinking models (Qwen 3, …) fast |
| `options.num_ctx`, `options.keep_alive` | ollama | Context window, model keep-alive |
| `options.num_gpu`, `options.num_thread` | ollama | GPU layers (`0` = CPU only, e.g. when the CUDA runner crashes), CPU threads |

Notes:

* **Anthropic:** the current Messages API has no `temperature` parameter, so it isn't sent.
* **Local servers:** for `openai_compatible` endpoints on `localhost`, the API key is
  optional. Ollama needs none. Local models are assumed to be pulled and running already.
* **Health checks:** a model listing or show call, which never consumes tokens. They run in
  the background at startup and on `POST /api/llm/reload`.

#### Decision models (`openrouter_decisions`)

Decision models such as TypeSafe's `typesafe/jev-1.13` don't generate text. OpenRouter's
Decisions API (`POST /api/alpha/decisions`) takes a `state` plus typed questions and returns
probabilities. TradeSense sends the decision prompt as the state and asks one BUY / SELL /
SIDEWAYS `choice` question. The per-option probabilities and the confidence become the
decision.

```yaml
  openrouter_jev:
    type: openrouter_decisions
    model: typesafe/jev-1.13
    api_key_env: OPENROUTER_API_KEY
    capabilities: [decision]
    pricing: { input_per_mtok: 0.042, output_per_mtok: 0.0 }   # output tokens are free
```

* The `decision` capability means trading-signal decisions only. These providers are
  skipped for strategy extraction, document summaries and image (vision) requests, so keep
  a text model later in the `text` chain and in `extraction`.
* Decisions come with probabilities and confidence only: no written reasoning and no key
  levels.
* The health check (`GET /api/v1/key`) proves that the key works. It can't prove that the
  model id exists.

#### Model picker

Each role in the top bar has a dropdown grouped into **Local** and **Cloud / frontier**. It
lists every provider in `llm.yml` and every model installed in your local Ollama, even
ones `llm.yml` doesn't name.

* Choosing a model makes it the role's primary. The rest of the `llm.yml` chain stays as
  fallbacks.
* The choice is saved in `config/llm_selection.json` and survives restarts and reloads.
  `llm.yml` itself is never rewritten. Choosing the `llm.yml` primary again removes the
  override.
* An Ollama model that isn't in `llm.yml` gets the settings of the `llm.yml` Ollama entry
  on the same host (options, timeout, concurrency).
* Disabled entries show why: a missing API key, or "not installed in Ollama".

### `config/strategies.yml`

```yaml
ema_crossover:
  params: { fast: 9, slow: 21 }
  overrides:
    "15m": { fast: 50, slow: 200 }   # resolution: timeframe override -> params -> class defaults
rsi_ema_combo:                        # YAML rules use "<indicator_id>.<param>" keys
  params: { rsi.period: 10 }
```

Unknown parameter names are logged and ignored; values are coerced to the default's type.

---

## Strategies

| Icon | Kind | Where | Runs as |
|---|---|---|---|
| ⚙ | built-in | `src/tradesense/strategies/builtin/` | Python class |
| 📏 | rule | `config/rules/*.yml` | YAML indicators + boolean rules |
| 📄 | document | uploaded `.md .txt .pdf .docx` | approved strategy card + document text |
| 🖼 | image | uploaded `.png .jpg .webp` | approved card + reference image vs. live chart |

Built-ins: EMA Crossover, RSI Extremes, Bollinger Squeeze Breakout, VWAP Bounce, MACD
Momentum, Supertrend, Stochastic Crossover, RSI Divergence, ADX Trend Strength, Ichimoku
Cloud, Pivot Points S/R, ATR Volatility Breakout.

Strategies are timeframe-agnostic: the timeframe is only context for the prompt and for
parameter overrides. The same strategy can run on several timeframes at once.

### Adding a built-in strategy

Create one file in `src/tradesense/strategies/builtin/`. It is discovered automatically.

```python
from collections.abc import Mapping
from typing import Any

import pandas as pd

from tradesense.indicators.technical import ago, bars_since, crossed_above, ema, last_valid
from tradesense.strategies.base import FeatureBuilder, IndicatorStrategy, require
from tradesense.strategies.registry import register_strategy


@register_strategy
class PriceAboveEma(IndicatorStrategy):
    name = "price_above_ema"
    display_name = "Price Above EMA"
    description = "Close crossing above the EMA is bullish; below it is bearish."
    category = "trend"
    min_candles = 30
    default_params = {"period": 50}  # noqa: RUF012

    def required_candles(self, params: Mapping[str, Any]) -> int:
        return int(params["period"]) + 10

    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        line = ema(df["close"], int(params["period"]))
        (value,) = require(ema=last_valid(line))  # insufficient_data if still warming up
        fb.series("ema", line)  # latest value + last 10 values
        fb.series("close", df["close"])
        crossed = bars_since(crossed_above(df["close"], line), 20)
        if crossed is not None:
            fb.event(f"Close crossed above the EMA {ago(crossed)}")
        close = float(df["close"].iloc[-1])
        fb.signal("BUY" if close > value else "SELL", "Close relative to the EMA")
```

Run `POST /api/strategies/reload` (or restart), then add a test.

### Adding a YAML rule

Drop a file in `config/rules/`, then click **Reload strategies** (no restart needed):

```yaml
name: rsi_ema_combo
description: RSI pullback in direction of EMA trend
category: momentum          # trend | momentum | mean_reversion | volatility | levels | pattern
min_candles: 60
indicators:
  - { id: ema_fast, type: ema, period: 20 }
  - { id: ema_slow, type: ema, period: 50 }
  - { id: rsi,      type: rsi, period: 14 }
rules:
  buy:  "ema_fast > ema_slow and rsi < 40"
  sell: "ema_fast < ema_slow and rsi > 60"
# anything else = SIDEWAYS
```

| Indicator `type` | Parameters (defaults) | Outputs (first = `id` itself) |
|---|---|---|
| `ema`, `sma` | `period: 20` | `value` |
| `rsi` | `period: 14` | `value` |
| `macd` | `fast: 12, slow: 26, signal: 9` | `macd`, `signal`, `hist` |
| `bbands` | `period: 20, std: 2.0` | `middle`, `upper`, `lower`, `width`, `percent_b` |
| `atr` | `period: 14` | `value` |
| `adx` | `period: 14` | `adx`, `plus_di`, `minus_di` |
| `stochastic` | `k: 14, d: 3, smooth: 3` | `k`, `d` |
| `vwap` | – (session VWAP, resets daily UTC) | `value` |
| `supertrend` | `period: 10, multiplier: 3.0` | `value`, `direction` (+1/-1) |
| `swing_points` | `left: 3, right: 3` | `high`, `low` (latest confirmed swing levels) |
| `volume` | `period: 20` | `value`, `sma`, `ratio` |

Expressions can use:

* **Operands:** numbers, `True`/`False`, indicator ids, `id.output` (e.g. `macd.hist`) and
  the price columns `open high low close volume`.
* **Operators:** `+ - * /`, comparisons (chains like `30 < rsi < 70` work) and
  `and or not`.
* **Functions:** `crossed_above(a, b)`, `crossed_below(a, b)`, `prev(x, n)`,
  `rising(x, n)`, `falling(x, n)`, `highest(x, n)`, `lowest(x, n)`, `abs`, `min`, `max`.

Expressions are parsed with `ast` and checked against a whitelist. They are interpreted,
never passed to `eval`. Attribute access, subscripts, lambdas, strings, `**` and unknown
names are all rejected, and files that fail validation are skipped and reported.

### Uploading a reference strategy

Each uploaded file goes through: upload → extract → review → approve → run.

1. **Upload:** drag a file onto **Strategy Library**, or `POST /api/library/upload`, or the
   MCP tool `add_reference_strategy`.
   - The file type is checked by its signature, not its extension, and size is limited
     (`library.max_upload_mb`).
   - The filename is sanitised, and the file is stored under `strategy_library/`.
2. **Extract:** this happens once, and the result is cached; it runs again only on
   **Re-extract**.
   - Documents are parsed (pypdf / python-docx). Text longer than
     `library.max_document_chars` is chunked and summarised, then the `extraction` role
     builds a draft **strategy card**.
   - Images go to the `vision` role, which describes the pattern, trend, levels and the
     entry/exit conditions.
   - Unsupported `required_indicators` are dropped and listed under warnings.
3. **Review:** the original (rendered Markdown, extracted text, or the zoomable image) is
   shown beside the editable card.
4. **Approve:** only approved cards appear in the strategy lists. **Disable** hides a card
   again.
5. **Run:** works on any timeframe.
   - Documents send the card, the original text (if it fits), the shared feature pack, the
     required indicators and the last 50 candles to the `text` role.
   - Images render the live chart (mplfinance, last 100 candles, same style) and send the
     reference image, the live chart, the card and the feature pack to the `vision` role,
     which reports `pattern_match` (0–1).
   - Image strategies are disabled when no vision-capable provider is reachable.

**Prompt-injection safety:** uploaded text, strategy cards and image contents are treated as
DATA.

* They are wrapped in `<untrusted_*_NONCE>` blocks with a random nonce per prompt, and
  look-alike tags inside the content are neutralised.
* The system prompt forbids following instructions found in those blocks.
* Output that doesn't match the decision schema is rejected.

**MCP imports:** `add_reference_strategy(file_path)` only reads files inside
`library.import_roots` (default `strategy_library/inbox`). The folder restriction is checked
before existence, so paths elsewhere cannot be probed.

---

## Backtesting

**Backtest** (in the top bar, or the Backtest tab) replays the last `backtest.days` (2)
trading days of the current instrument on every timeframe. It shows which strategies
called BUY / SELL right lately.

1. **Replay.** Every built-in and YAML rule strategy is re-run on every candle of the
   period, using the same `oanda.candle_count` window the live app sees.
   * Built-ins run candle by candle in a process pool (`backtest.max_workers`). That's
     most of the run, which takes about 2 minutes on a 4-core laptop.
   * Rules are vectorised.
   * Uploaded document / image strategies are listed as "not backtested", because only
     the LLM decides them.
2. **Directional hit rate.** A BUY (SELL) at candle *t* is a hit when the close
   `horizon_bars` later (1m: 15, 5m: 6, 15m: 4) is higher (lower) by more than the
   spread at candle *t*.
   * The signal only uses candles up to *t*.
   * The spread comes from OANDA's historical bid / ask candles, so the result doesn't
     depend on when you click. A wide quote at the Sunday open, for example, doesn't
     penalise two days of history.
   * Only if bid / ask candles are unavailable does the live spread stand in. If the
     market is closed, any favourable move counts.
   * Each strategy shows its BUY, SELL and overall hit rate, its number of signals and the
     average move in pips.
   * Ranking uses the hit rate's Wilson 95% lower bound, so a lucky 3/3 can't beat 70/100.
     Strategies with fewer than `min_signals` signals aren't ranked.
3. **New strategies.** The period is split in time into day 1 and day 2.
   * About 100 rule templates (EMA cross, EMA trend + RSI pullback, RSI cross-back,
     MACD + ADX, Bollinger reversion, Supertrend + RSI, Stochastic + trend, breakouts,
     VWAP trend) are ranked on day 1 and judged on day 2.
   * With **LLM-designed rules** on, a text model proposes `llm_rule_candidates` rules per
     timeframe, judged the same way.
   * A candidate that beats 50% on both days, with at least `min_signals` signals on
     day 2, is marked *held up*.
   * **Save as strategy** writes it to `config/rules/bt_<instrument>_<tf>_<name>.yml`,
     reloads the pool and adds it to that timeframe's column.
4. **LLM check top picks.** For the `llm_top_n` best strategies per timeframe, the live
   LLM decision is replayed on `llm_samples` of their day-2 signals. This compares the
   LLM's hit rate with the raw rule's. It costs about $0.01 with Jev.

**Use best picks on dashboard** selects the top 3 ranked strategies per timeframe.
A couple of days is a small sample, so treat the results as "what fits right now", not as
evidence of an edge. Settings are under `backtest:` in `config/app.yml`.

### Strategy replay on the chart

To see *where* a strategy fired and how each call played out, pick it in **Replay** above
the Dashboard chart. You can also click **Replay on chart** on a strategy card, or click a
row in a Backtest leaderboard. Only that one strategy is replayed over the last
`backtest.days`, on the chart's timeframe. This takes a few seconds and needs no full
backtest.

* **Entries:** an arrow marks the first candle of each BUY (below) or SELL (above) run.
  Green means it was right `horizon_bars` later, red means wrong, grey means too recent
  to judge. With 150 entries or fewer, each arrow is labelled with the pips moved.
* **Signal strip:** the band under the candles shows the strategy's state on every candle
  (BUY / SELL / SIDEWAYS). It replaces the volume bars while replaying.
* **Details:** hovering an entry shows its price, when and where it was judged, and the
  result. **◀ Prev entry / Next entry ▶** step through the entries.
* **Accuracy:** the bar above the chart shows how many entries were right (overall, BUY,
  SELL) and the every-candle hit rate. That's the same number the Backtest ranks by.

Switching the chart timeframe or the instrument re-runs the replay. **Back to live**
returns to the live chart.

## Adding an LLM provider type

1. Create `src/tradesense/llm/providers/my_provider.py`:

   ```python
   from tradesense.llm.base import LLMProvider, classify_status
   from tradesense.llm.factory import register_provider
   from tradesense.llm.messages import ImagePart, LLMRequest, TextPart


   @register_provider("my_api")
   class MyProvider(LLMProvider):
       async def _generate(self, request: LLMRequest) -> tuple[str, int, int]:
           # translate request.system / request.messages (TextPart, ImagePart) to the SDK,
           # map SDK errors with classify_status(status_code, message, self.name)
           ...
           return text, input_tokens, output_tokens

       async def health_check(self) -> bool: ...  # a call that consumes no tokens (e.g. list models)
   ```

2. Add an entry to `config/llm.yml` with `type: my_api`. Nothing else changes.

---

## Web UI

* **Top bar:**
  - Instrument (with suggestions from your account), timeframe toggles, **Run Analysis**,
    **Auto-refresh** (each timeframe re-runs when its candle closes) and the live-stream
    status.
  - LLM status per role (model picker, health, fallback badge) and **Reload LLM config**.
  - Usage of the last run: tokens, estimated cost, average latency.
* **Chart:** candlesticks plus volume for 1m/5m/15m, a live bid/ask/spread ticker, and the
  current candle updating on every tick. Optional overlays: EMA 20/50 and decision markers.
  **Replay** shows any built-in or rule strategy's last days of entries, marked right or
  wrong, with its accuracy. See [Strategy replay](#strategy-replay-on-the-chart).
* **Dashboard:** three columns (one per timeframe), each with a multi-select of the whole
  strategy pool grouped by category.
  - A consensus card sits at the top of each column.
  - Strategy cards show the signal (colour **and** text), a buy/sell/sideways probability
    bar, confidence, observations, reasoning, key levels, and the provider, model and
    latency.
  - Image strategies add the reference and live-chart thumbnails plus the pattern match.
  - Cards update independently as results stream in.
* **Results table:** sortable, with **Export JSON**.
* **Strategy Library:** drag-and-drop upload, a grid of cards, and a split-screen review.
* **Backtest:** progress bar and, per timeframe, the best fit, a sortable leaderboard (hit
  rates, signals, pips, LLM check) and new-strategy candidates with **Save as strategy**.
  See [Backtesting](#backtesting).

The page sends a strict Content-Security-Policy (no inline scripts). The chart library is
pinned with Subresource Integrity, and every server value is rendered as text, never HTML.

---

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/instruments` | Instruments on the OANDA account (cached) |
| GET | `/api/candles/{instrument}?timeframe=1m&count=300` | Candles (last one may be in progress) |
| GET | `/api/price/{instrument}` | Bid / ask / spread snapshot |
| GET | `/api/strategies[?category=&source=]` | The full shared strategy pool |
| POST | `/api/strategies/reload` | Rescan rules, `strategies.yml`, approved cards |
| POST | `/api/analysis` | `{"instrument", "selections": {tf: [names]}, "client_id"?}` → AnalysisResult |
| POST | `/api/llm/reload` | Reload `llm.yml` without restarting |
| GET | `/api/llm/status` | Roles, providers, health, issues, usage |
| GET | `/api/llm/models` | Selectable models (`llm.yml` + installed Ollama), grouped local / cloud, and each role's primary |
| POST | `/api/llm/select` | `{"role", "model"}`: make that model the role's primary (saved in `llm_selection.json`) |
| POST | `/api/backtest` | `{"instrument", "client_id"?, "llm_check"?, "llm_rules"?}` → BacktestResult (progress over the WebSocket; 409 while one runs) |
| POST | `/api/backtest/strategies` | `{"run_id", "timeframe", "name"}`: save a candidate from the latest backtest as a rule strategy |
| POST | `/api/backtest/replay` | `{"instrument", "timeframe", "strategy"}` → candles, per-candle signals, entries (right / wrong / pending) and accuracy |
| POST | `/api/library/upload` | Multipart `file` → draft card (`?extract=false` to skip LLM) |
| GET | `/api/library` | All cards |
| GET / PUT / DELETE | `/api/library/{id}` | Card + preview / save edits / delete |
| GET | `/api/library/{id}/file` | Original file (served with `nosniff` + sandbox CSP) |
| POST | `/api/library/{id}/extract` \| `/approve` \| `/disable` | Lifecycle actions |
| GET | `/api/charts/{chart_id}.png` | Rendered live chart thumbnails |
| GET | `/api/meta`, `/api/health` | UI metadata, liveness |
| WS | `/ws` | Live prices + streamed results (below) |
| — | `/mcp` | MCP (streamable HTTP) |

Interactive OpenAPI docs: `/docs`.

**WebSocket `/ws`.**

* Client → server:
  - `{"action": "subscribe_prices", "instrument": "EUR_USD"}`
  - `unsubscribe_prices`
  - `{"action": "run_analysis", "instrument", "selections"}`
  - `{"action": "auto_refresh", "enabled": true, "instrument", "selections"}`
  - `ping`
* Server → client:
  - `hello` (with `client_id`; pass it to `POST /api/analysis` to stream that run to the
    socket)
  - `price`, `price_status`
  - `analysis_started`, `strategy_result`, `consensus`, `analysis_complete`,
    `analysis_error`
  - `auto_refresh_status`

**MCP tools** (mounted at `/mcp`, streamable HTTP):

* `list_strategies(category?)`
* `reload_strategies()`
* `get_candles(instrument, timeframe, count=100)`
* `get_live_price(instrument)`
* `run_analysis(instrument, selections)`
* `add_reference_strategy(file_path)`
* `get_reference_strategy(id)`
* `approve_reference_strategy(id)`
* `llm_status()`

Connect any MCP client to `http://localhost:8000/mcp`. A client that only speaks stdio can
go through a bridge such as `npx mcp-remote http://localhost:8000/mcp`.

---

## Development

```bash
uv run pytest          # tests; OANDA (respx), every LLM SDK and the price stream are mocked
uv run ruff check .    # lint
uv run ruff format .   # format
uv run mypy            # type-check (src/tradesense)
```

Layout:

```
config/                      app.yml, llm.yml, strategies.yml, rules/
strategy_library/            documents/, images/, cards/, inbox/ (uploads, git-ignored)
src/tradesense/
  main.py                    FastAPI app + CLI (`tradesense`), FastMCP mounted at /mcp
  config.py                  .env + YAML -> pydantic models
  services.py                composition root shared by REST, WebSocket and MCP
  data/                      oanda_client.py (REST), price_stream.py (stream + hub)
  indicators/technical.py    indicators + the indicator registry used by rules/cards
  strategies/                base, registry, rule_engine, reference, ingestion,
                             feature_pack, chart_renderer, builtin/
  llm/                       base, factory, router, messages, prompts, schemas, usage,
                             decision, providers/
  engine/                    orchestrator (parallel runs + consensus), auto_refresh, chart_store
  api/                       routes.py (REST), websocket.py
  web/                       templates/index.html, static/css, static/js
tests/
```

### Safety summary

* **Secrets:** read only from `.env` / the environment. They are never written to YAML,
  and any known key value is masked as `***` in log output.
* **Defaults:** OANDA defaults to the **practice** environment.
* **Signals only:** no order-placement code or endpoints (a test enforces it).
* **Uploads:** validated by signature, size-limited, and never executed. DOCX archives are
  checked for zip bombs, and images are decoded to prove they are images.
* **Prompts:** uploaded content is treated as untrusted data in every prompt.
