/* TradeSense web UI (vanilla JS).
 *
 * Security: every piece of server or user data is rendered with textContent / DOM APIs.
 * The only innerHTML use is the Markdown preview, which the server renders with raw HTML
 * disabled and unsafe URLs stripped.
 */
"use strict";

(() => {
  // ------------------------------------------------------------------ constants

  const ICONS = { builtin: "⚙", rule: "📏", document: "📄", image: "🖼" };
  const SOURCE_LABELS = { builtin: "Built-in", rule: "Rule", document: "Document", image: "Image" };
  const CATEGORY_LABELS = {
    trend: "Trend", momentum: "Momentum", mean_reversion: "Mean reversion",
    volatility: "Volatility", levels: "Levels", pattern: "Pattern",
  };
  const TF_LABELS = { "1m": "1 MIN", "5m": "5 MIN", "15m": "15 MIN" };
  const GRANULARITY_SECONDS = { M1: 60, M5: 300, M15: 900, M30: 1800, H1: 3600, H4: 14400, D: 86400 };
  const DEFAULT_INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD", "EUR_JPY",
    "XAU_USD", "XAG_USD", "US30_USD", "SPX500_USD", "NAS100_USD", "DE30_EUR", "UK100_GBP",
  ];
  const DEFAULT_SELECTIONS = {
    "1m": ["ema_crossover", "rsi_extremes"],
    "5m": ["ema_crossover", "macd_momentum", "rsi_ema_combo"],
    "15m": ["ichimoku", "rsi_ema_combo"],
  };
  const STORAGE_KEY = "tradesense.ui.v1";
  const MODEL_GROUPS = { local: "Local", cloud: "Cloud / frontier" };

  // ------------------------------------------------------------------ state

  const state = {
    meta: null,
    strategies: [],
    strategyByName: new Map(),
    visionAvailable: true,
    instrument: "EUR_USD",
    instrumentInfo: new Map(),
    enabledTfs: ["1m", "5m", "15m"],
    selections: structuredClone(DEFAULT_SELECTIONS),
    chartTf: "5m",
    overlays: { ema: true, markers: true },
    clientId: null,
    ws: null,
    wsBackoff: 1000,
    running: false,
    results: new Map(), // `${tf}|${name}` -> StrategyResult
    consensus: new Map(), // tf -> Decision | null
    latestByTf: {}, // tf -> TimeframeResult
    lastResult: null,
    sort: { key: "timeframe", dir: 1 },
    autoRefresh: false,
    nextRuns: {},
    llmStatus: null,
    llmModels: null, // /api/llm/models: selectable models per role
    backtest: { running: false, result: null, sort: {} },
    // Strategy replay on the main chart; `seq` discards responses that were superseded.
    replay: { active: false, loading: false, strategy: null, trace: null, entryIndex: -1, seq: 0,
      indexByTime: new Map(), entryByTime: new Map() },
    currentCard: null,
    lastPrice: null,
  };

  // ------------------------------------------------------------------ helpers

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function h(tag, props, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") el.className = value;
      else if (key === "text") el.textContent = value;
      else if (key === "dataset") Object.assign(el.dataset, value);
      else if (key.startsWith("on") && typeof value === "function") {
        el.addEventListener(key.slice(2), value);
      } else if (value === true) el.setAttribute(key, "");
      else el.setAttribute(key, String(value));
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
  }

  const pct = (x, digits = 0) => (x === null || x === undefined ? "–" : `${(x * 100).toFixed(digits)}%`);
  const fmtTime = (iso) => (iso ? new Date(iso).toLocaleTimeString([], { hour12: false }) : "–");
  const fmtDate = (iso) => (iso ? new Date(iso).toLocaleString([], { hour12: false }) : "–");

  function priceDecimals(price) {
    const info = state.instrumentInfo.get(state.instrument);
    if (info && Number.isInteger(info.display_precision)) return info.display_precision;
    const p = Math.abs(price || 0);
    return p < 20 ? 5 : p < 1000 ? 3 : 2;
  }
  const fmtPrice = (x) => (x === null || x === undefined ? "–" : Number(x).toFixed(priceDecimals(x)));

  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.status = status;
    }
  }

  async function api(path, options = {}) {
    const init = { ...options };
    if (init.body && !(init.body instanceof FormData) && typeof init.body !== "string") {
      init.body = JSON.stringify(init.body);
      init.headers = { "Content-Type": "application/json", ...(init.headers || {}) };
    }
    const response = await fetch(path, init);
    const isJson = (response.headers.get("content-type") || "").includes("application/json");
    const body = isJson ? await response.json() : null;
    if (!response.ok) {
      let detail = response.statusText;
      if (body && body.detail !== undefined) {
        detail = typeof body.detail === "string" ? body.detail
          : Array.isArray(body.detail) ? body.detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
          : JSON.stringify(body.detail);
      }
      throw new ApiError(detail, response.status);
    }
    return body;
  }

  function toast(message, kind = "info", timeout = 6000) {
    const el = h("div", { class: `toast toast-${kind}`, role: kind === "error" ? "alert" : "status" }, message);
    $("#toasts").append(el);
    setTimeout(() => el.remove(), timeout);
  }

  function loadPrefs() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      if (!saved) return;
      if (typeof saved.instrument === "string") state.instrument = saved.instrument;
      if (Array.isArray(saved.enabledTfs)) state.enabledTfs = saved.enabledTfs;
      if (saved.selections && typeof saved.selections === "object") state.selections = saved.selections;
      if (typeof saved.chartTf === "string") state.chartTf = saved.chartTf;
      if (saved.overlays) state.overlays = { ...state.overlays, ...saved.overlays };
    } catch {
      /* corrupted or blocked storage: keep defaults */
    }
  }

  function savePrefs() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        instrument: state.instrument,
        enabledTfs: state.enabledTfs,
        selections: state.selections,
        chartTf: state.chartTf,
        overlays: state.overlays,
      }));
    } catch {
      /* storage unavailable: preferences are not persisted */
    }
  }

  function timeframes() {
    return state.meta ? Object.keys(state.meta.timeframes) : ["1m", "5m", "15m"];
  }

  function tfSeconds(tf) {
    const granularity = state.meta && state.meta.timeframes[tf] && state.meta.timeframes[tf].granularity;
    return GRANULARITY_SECONDS[granularity] || 60;
  }

  // ------------------------------------------------------------------ banners & status

  function renderBanners() {
    const banners = $("#banners");
    banners.replaceChildren();
    const add = (text, kind = "warn") => banners.append(h("div", { class: `banner banner-${kind}` }, text));
    if (state.meta && !state.meta.oanda.configured) {
      add("OANDA is not configured: set OANDA_API_KEY and OANDA_ACCOUNT_ID in .env and restart. "
        + "Candles, live prices and analysis are unavailable until then.", "error");
    } else if (state.meta && !state.meta.oanda.account_configured) {
      add("OANDA_ACCOUNT_ID is not set: instruments cannot be validated and live prices are off.");
    }
    if (!state.visionAvailable) {
      add("No vision-capable LLM provider is reachable: image reference strategies are disabled.");
    }
    if (state.llmStatus) {
      for (const issue of state.llmStatus.issues || []) {
        if (issue.level === "error") add(`LLM configuration: ${issue.message}`, "error");
      }
    }
  }

  async function refreshLlmStatus() {
    try {
      const status = await api("/api/llm/status");
      state.llmStatus = status;
      state.visionAvailable = status.vision_available;
      renderLlmPanel(status);
      renderBanners();
    } catch (error) {
      $("#llm-panel").replaceChildren(h("span", { class: "muted" }, `LLM status unavailable: ${error.message}`));
    }
  }

  async function refreshLlmModels() {
    try {
      state.llmModels = await api("/api/llm/models");
    } catch {
      state.llmModels = null; // the panel falls back to read-only labels
    }
    if (state.llmStatus) renderLlmPanel(state.llmStatus, true);
  }

  function modelLabel(option) {
    const notes = [];
    if (option.capabilities.length === 1 && option.capabilities[0] === "decision") notes.push("decisions only");
    if (!option.available) notes.push(option.note);
    return `${option.model} · ${option.id}${notes.length ? ` (${notes.join("; ")})` : ""}`;
  }

  function modelPicker(role, primary) {
    if (!state.llmModels) return null;
    const usable = state.llmModels.options.filter((o) => o.roles.includes(role));
    const select = h("select", { id: `llm-model-${role}`, class: "llm-select" });
    if (!usable.some((o) => o.id === primary)) select.append(h("option", { value: primary }, primary));
    for (const [group, label] of Object.entries(MODEL_GROUPS)) {
      const options = usable.filter((o) => o.group === group);
      if (!options.length) continue;
      select.append(h("optgroup", { label }, options.map((o) => h("option", {
        value: o.id,
        disabled: !o.available && o.id !== primary,
        title: `${o.type} · ${o.capabilities.join(", ")}${o.note ? ` · ${o.note}` : ""}`,
      }, modelLabel(o)))));
    }
    select.value = primary;
    select.addEventListener("change", () => selectModel(role, select));
    return select;
  }

  async function selectModel(role, select) {
    const model = select.value;
    select.disabled = true;
    try {
      const status = await api("/api/llm/select", { method: "POST", body: { role, model } });
      state.llmStatus = status;
      state.visionAvailable = status.vision_available;
      await refreshLlmModels();
      await loadStrategies(); // vision availability may have changed
      renderColumns();
      renderBanners();
      toast(`The ${role} role now uses ${model}.`, "ok", 3000);
    } catch (error) {
      toast(`Could not switch the ${role} model: ${error.message}`, "error", 9000);
      renderLlmPanel(state.llmStatus, true); // put the picker back
    }
  }

  function renderLlmPanel(status, force = false) {
    const panel = $("#llm-panel");
    // A periodic refresh must not close a model picker the user has open.
    if (!force && document.activeElement && document.activeElement.matches("#llm-panel select")) return;
    panel.replaceChildren();
    for (const [role, info] of Object.entries(status.roles)) {
      const provider = info.active ? status.providers[info.active] : null;
      const dotClass = !info.available ? "dot-bad"
        : info.healthy === true ? (info.fallback_in_use ? "dot-warn" : "dot-ok")
        : info.healthy === false ? "dot-bad" : "dot-idle";
      const title = [
        `Chain: ${info.chain.join(" → ")}`,
        provider && provider.error ? `Last error: ${provider.error}` : null,
      ].filter(Boolean).join("\n");
      const picker = modelPicker(role, info.primary);
      panel.append(h("span", { class: "llm-role", title },
        h("span", { class: `dot ${dotClass}` }),
        picker ? h("label", { class: "role-name", for: picker.id }, role) : h("span", { class: "role-name" }, role),
        picker || h("span", { class: "role-model" }, info.active ? `${info.active} · ${info.active_model}` : "no provider"),
        info.fallback_in_use
          ? h("span", { class: "badge badge-fallback", title: `Answering: ${info.active} · ${info.active_model}` },
            `fallback: ${info.active}`)
          : null));
    }
  }

  function setStreamStatus(stateName, message) {
    const map = {
      connected: ["dot-ok", "stream live"],
      reconnecting: ["dot-warn", "stream reconnecting"],
      unavailable: ["dot-bad", "stream unavailable"],
      idle: ["dot-idle", "stream idle"],
      offline: ["dot-bad", "server offline"],
    };
    const [dot, label] = map[stateName] || map.idle;
    const el = $("#stream-status");
    el.replaceChildren(h("span", { class: `dot ${dot}` }), h("span", { class: "stream-label" }, label));
    el.title = message ? `Live price stream: ${message}` : "Live price stream";
  }

  function renderUsage(usage) {
    if (!usage) return;
    $("#sum-tokens").textContent = usage.total_tokens.toLocaleString();
    $("#sum-cost").textContent = `$${usage.total_cost_usd.toFixed(4)}`;
    $("#sum-latency").textContent = usage.avg_latency_ms ? `${usage.avg_latency_ms.toLocaleString()} ms` : "–";
  }

  // ------------------------------------------------------------------ chart

  const chart = {
    api: null,
    series: null,
    volume: null,
    ema20: null,
    ema50: null,
    markers: null,
    data: [],
    tf: null,

    init() {
      if (!window.LightweightCharts) {
        this.message("Chart library failed to load (no internet access to unpkg.com?).");
        return;
      }
      const LC = window.LightweightCharts;
      const styles = getComputedStyle(document.documentElement);
      const css = (name) => styles.getPropertyValue(name).trim();
      this.api = LC.createChart($("#chart"), {
        autoSize: true,
        layout: {
          background: { type: LC.ColorType.Solid, color: css("--panel") },
          textColor: css("--muted"),
          fontFamily: css("--font"),
          attributionLogo: true,
        },
        grid: { vertLines: { color: css("--grid") }, horzLines: { color: css("--grid") } },
        rightPriceScale: { borderColor: css("--border") },
        timeScale: { borderColor: css("--border"), timeVisible: true, secondsVisible: false, rightOffset: 8 },
        crosshair: { mode: LC.CrosshairMode.Normal },
      });
      const buy = css("--buy");
      const sell = css("--sell");
      this.series = this.api.addSeries(LC.CandlestickSeries, {
        upColor: buy, downColor: sell, borderVisible: false, wickUpColor: buy, wickDownColor: sell,
      });
      this.volume = this.api.addSeries(LC.HistogramSeries, {
        priceScaleId: "volume", priceFormat: { type: "volume" },
        lastValueVisible: false, priceLineVisible: false,
      });
      this.api.priceScale("volume").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
      const line = { lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false };
      this.ema20 = this.api.addSeries(LC.LineSeries, { ...line, color: "#3b82f6" });
      this.ema50 = this.api.addSeries(LC.LineSeries, { ...line, color: "#f59e0b" });
      this.markers = LC.createSeriesMarkers(this.series, []);
      // Replay mode: the strategy's state on every candle, in a thin band at the bottom.
      this.signals = this.api.addSeries(LC.HistogramSeries, {
        priceScaleId: "signals", lastValueVisible: false, priceLineVisible: false,
      });
      this.api.priceScale("signals").applyOptions({ scaleMargins: { top: 0.94, bottom: 0 }, visible: false });
      this.api.subscribeCrosshairMove((param) => this.onCrosshair(param));
    },

    message(text) {
      const el = $("#chart-message");
      el.textContent = text || "";
      el.hidden = !text;
    },

    async load() {
      if (!this.api) return;
      const tf = state.chartTf;
      this.tf = tf;
      if (state.meta && !state.meta.oanda.configured) {
        this.message("Connect OANDA (OANDA_API_KEY in .env) to see live candles.");
        return;
      }
      try {
        const body = await api(`/api/candles/${encodeURIComponent(state.instrument)}?timeframe=${tf}&count=${state.meta ? state.meta.candle_count : 300}`);
        if (this.tf !== tf || state.replay.active) return; // switched timeframe, or a replay took over
        this.data = body.candles.map((c) => ({
          time: c.time, open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume,
        }));
        this.render();
        this.message(this.data.length ? "" : "No candles returned.");
      } catch (error) {
        this.data = [];
        this.render();
        this.message(`Could not load candles: ${error.message}`);
      }
    },

    precision() {
      const last = this.data.length ? this.data[this.data.length - 1].close : 1;
      const decimals = priceDecimals(last);
      return { type: "price", precision: decimals, minMove: 10 ** -decimals };
    },

    render() {
      if (!this.api) return;
      this.series.applyOptions({ priceFormat: this.precision() });
      this.series.setData(this.data.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
      const trace = state.replay.active ? state.replay.trace : null;
      this.volume.setData(trace ? [] : this.data.map((c) => ({
        time: c.time, value: c.volume,
        color: c.close >= c.open ? "rgba(22,163,74,0.35)" : "rgba(220,38,38,0.35)",
      })));
      this.signals.setData(trace ? signalStrip(trace) : []);
      this.renderEmas();
      this.renderMarkers();
    },

    showTrace(trace) {
      if (!this.api) return;
      this.tf = trace.timeframe;
      this.data = trace.candles.map((c) => ({ ...c }));
      this.render();
      this.info(null);
      const n = this.data.length;
      if (n > 400) this.api.timeScale().setVisibleLogicalRange({ from: n - 400, to: n + 4 });
      else this.api.timeScale().fitContent();
      this.message(n ? "" : "No candles returned.");
    },

    focusEntry(index) {
      const { trace } = state.replay;
      const entry = trace && trace.entries[index];
      if (!this.api || !entry) return;
      state.replay.entryIndex = index;
      const at = state.replay.indexByTime.get(entry.time);
      const range = this.api.timeScale().getVisibleLogicalRange();
      const half = range ? Math.max(20, (range.to - range.from) / 2) : 80;
      this.api.timeScale().setVisibleLogicalRange({ from: at - half, to: at + half });
      this.renderMarkers();
      this.info(entryText(entry));
    },

    info(text) {
      const el = $("#chart-info");
      el.textContent = text || "";
      el.hidden = !text;
    },

    onCrosshair(param) {
      const replay = state.replay;
      if (!replay.active || !replay.trace) return;
      if (!param.time) {
        const focused = replay.trace.entries[replay.entryIndex];
        this.info(focused ? entryText(focused) : null);
        return;
      }
      const entry = replay.entryByTime.get(param.time);
      if (entry) {
        this.info(entryText(entry));
        return;
      }
      const index = replay.indexByTime.get(param.time);
      const code = index === undefined ? null : replay.trace.signals[index];
      const label = code === 1 ? "BUY" : code === -1 ? "SELL" : code === 0 ? "SIDEWAYS" : "not ready";
      this.info(`${fmtDate(param.time * 1000)} · ${replay.trace.display_name}: ${label}`);
    },

    renderEmas() {
      if (!this.api) return;
      const show = state.overlays.ema;
      this.ema20.setData(show ? emaSeries(this.data, 20) : []);
      this.ema50.setData(show ? emaSeries(this.data, 50) : []);
    },

    renderMarkers() {
      if (!this.api) return;
      if (state.replay.active) {
        this.markers.setMarkers(state.replay.trace ? replayMarkers(state.replay.trace) : []);
        return;
      }
      if (!state.overlays.markers) {
        this.markers.setMarkers([]);
        return;
      }
      const times = new Set(this.data.map((c) => c.time));
      const markers = [];
      for (const result of state.results.values()) {
        if (result.timeframe !== state.chartTf || !result.decision || !result.candle_time) continue;
        const signal = result.decision.signal;
        if (signal === "SIDEWAYS") continue;
        const time = Math.floor(Date.parse(result.candle_time) / 1000);
        if (!times.has(time)) continue;
        markers.push({
          time,
          position: signal === "BUY" ? "belowBar" : "aboveBar",
          color: signal === "BUY" ? "#16a34a" : "#dc2626",
          shape: signal === "BUY" ? "arrowUp" : "arrowDown",
          text: `${shortName(result)} ${pct(result.decision.confidence)}`,
        });
      }
      markers.sort((a, b) => a.time - b.time);
      this.markers.setMarkers(markers);
    },

    tick(tick) {
      if (state.replay.active) return; // the replay shows history, not the live candle
      if (!this.api || !this.data.length || tick.instrument !== state.instrument) return;
      const seconds = tfSeconds(this.tf);
      const t = Math.floor(Date.parse(tick.time) / 1000);
      const bucket = t - (t % seconds);
      const last = this.data[this.data.length - 1];
      const price = tick.mid;
      let candle;
      if (bucket === last.time) {
        last.high = Math.max(last.high, price);
        last.low = Math.min(last.low, price);
        last.close = price;
        last.volume += 1; // OANDA volume is tick volume
        candle = last;
      } else if (bucket > last.time) {
        candle = { time: bucket, open: last.close, high: Math.max(last.close, price),
          low: Math.min(last.close, price), close: price, volume: 1 };
        this.data.push(candle);
      } else {
        return; // stale tick
      }
      this.series.update({ time: candle.time, open: candle.open, high: candle.high, low: candle.low, close: candle.close });
      this.volume.update({ time: candle.time, value: candle.volume,
        color: candle.close >= candle.open ? "rgba(22,163,74,0.35)" : "rgba(220,38,38,0.35)" });
    },
  };

  function emaSeries(data, period) {
    const out = [];
    const k = 2 / (period + 1);
    let ema = null;
    data.forEach((candle, index) => {
      if (index < period - 1) return;
      if (ema === null) {
        ema = data.slice(0, period).reduce((sum, c) => sum + c.close, 0) / period;
      } else {
        ema = candle.close * k + ema * (1 - k);
      }
      out.push({ time: candle.time, value: ema });
    });
    return out;
  }

  function shortName(result) {
    const info = state.strategyByName.get(result.strategy);
    const name = (info && info.display_name) || result.display_name || result.strategy;
    return name.length > 11 ? `${name.slice(0, 10)}…` : name;
  }

  // ------------------------------------------------------------------ strategy replay

  const STRIP_COLORS = { 1: "rgba(22,163,74,0.6)", "-1": "rgba(220,38,38,0.6)", 0: "rgba(234,179,8,0.18)" };
  const OUTCOME_COLORS = { hit: "#22c55e", miss: "#ef4444", pending: "#8796a8" };

  function signalStrip(trace) {
    const bars = [];
    trace.candles.forEach((candle, i) => {
      const code = trace.signals[i];
      if (code !== null && code !== undefined) bars.push({ time: candle.time, value: 1, color: STRIP_COLORS[code] });
    });
    return bars;
  }

  function replayMarkers(trace) {
    const labelled = trace.entries.length <= 150;
    return trace.entries.map((e, i) => ({
      time: e.time,
      position: e.signal === "BUY" ? "belowBar" : "aboveBar",
      shape: e.signal === "BUY" ? "arrowUp" : "arrowDown",
      color: OUTCOME_COLORS[e.hit === null ? "pending" : e.hit ? "hit" : "miss"],
      size: i === state.replay.entryIndex ? 2 : 1,
      text: labelled ? (e.hit === null ? "?" : pips(e.move_pips)) : "",
    }));
  }

  function entryText(e) {
    const opened = `${e.signal} ${fmtDate(e.time * 1000)} at ${fmtPrice(e.price)}`;
    const lasted = ` · signal held ${e.candles} candle${e.candles === 1 ? "" : "s"}`;
    if (e.hit === null) return `${opened} → not judged yet${lasted}`;
    return `${opened} → judged ${fmtTime(e.judged_time * 1000)} at ${fmtPrice(e.judged_price)} `
      + `(${pips(e.move_pips)} pips) ${e.hit ? "✓ right" : "✗ wrong"}${lasted}`;
  }

  function refreshChart() {
    if (state.replay.active && state.replay.strategy) runReplay(state.replay.strategy);
    else chart.load();
  }

  function setChartTf(tf) {
    state.chartTf = tf;
    savePrefs();
    $$("#chart-tf button").forEach((b) => b.classList.toggle("active", b.dataset.tf === tf));
  }

  function populateReplayOptions() {
    const select = $("#replay-strategy");
    const off = h("option", { value: "" }, "Off (live chart)");
    const groups = Object.entries(CATEGORY_LABELS).map(([category, label]) => {
      const items = state.strategies.filter((s) => s.category === category);
      if (!items.length) return null;
      return h("optgroup", { label }, items.map((s) => {
        const replayable = s.source === "builtin" || s.source === "rule";
        return h("option", { value: s.name, disabled: !replayable },
          `${ICONS[s.source] || ""} ${s.display_name}${replayable ? "" : " (LLM only)"}`);
      }));
    });
    select.replaceChildren(off, ...groups.filter(Boolean));
    select.value = state.replay.active ? state.replay.strategy : "";
  }

  function setReplayNav(enabled) {
    $("#replay-prev").disabled = !enabled;
    $("#replay-next").disabled = !enabled;
  }

  async function runReplay(name) {
    const replay = state.replay;
    const seq = ++replay.seq;
    Object.assign(replay, { active: true, loading: true, strategy: name, entryIndex: -1 });
    const info = state.strategyByName.get(name);
    $("#replay-strategy").value = name;
    $("#replay-exit").hidden = false;
    $("#replay-bar").hidden = false;
    $("#replay-spinner").hidden = false;
    $("#replay-summary").textContent =
      `Replaying ${info ? info.display_name : name} on ${TF_LABELS[state.chartTf] || state.chartTf}…`;
    setReplayNav(false);
    try {
      const trace = await api("/api/backtest/replay", {
        method: "POST",
        body: { instrument: state.instrument, timeframe: state.chartTf, strategy: name },
      });
      if (seq !== replay.seq || !replay.active) return; // superseded or cancelled
      replay.trace = trace;
      replay.indexByTime = new Map(trace.candles.map((c, i) => [c.time, i]));
      replay.entryByTime = new Map(trace.entries.map((e) => [e.time, e]));
      chart.showTrace(trace);
      renderReplaySummary(trace);
      setReplayNav(trace.entries.length > 0);
    } catch (error) {
      if (seq !== replay.seq) return;
      toast(`Replay failed: ${error.message}`, "error", 9000);
      exitReplay();
    } finally {
      if (seq === replay.seq) {
        replay.loading = false;
        $("#replay-spinner").hidden = true;
      }
    }
  }

  function exitReplay() {
    const replay = state.replay;
    replay.seq += 1; // ignore any replay still in flight
    Object.assign(replay, { active: false, loading: false, strategy: null, trace: null, entryIndex: -1 });
    $("#replay-strategy").value = "";
    $("#replay-exit").hidden = true;
    $("#replay-bar").hidden = true;
    $("#replay-spinner").hidden = true;
    chart.info(null);
    chart.load();
  }

  function renderReplaySummary(trace) {
    const e = trace.entry_stats;
    const s = trace.stats;
    const basis = trace.spread_pips !== null
      ? `a hit must beat the spread at its candle (median ${trace.spread_pips} pips)`
      : "any move in the signalled direction counts";
    const parts = [
      h("b", {}, `${trace.display_name} · ${TF_LABELS[trace.timeframe] || trace.timeframe}`),
      h("span", { class: "muted" }, `${fmtDate(trace.start)} → ${fmtDate(trace.end)}`),
      h("span", {},
        h("b", { class: hitClass(e.hit_rate) }, pct(e.hit_rate)),
        ` of ${e.entries} entries right (BUY ${pct(e.buy_hit_rate)} of ${e.buy_entries} · `
        + `SELL ${pct(e.sell_hit_rate)} of ${e.sell_entries})`),
      h("span", { class: "muted", title: "The Backtest ranking metric: every candle that signalled BUY or SELL" },
        `every signal candle: ${pct(s.hit_rate)} of ${s.signals}`),
      h("span", { class: "muted" }, `judged ${trace.horizon_bars} candles later; ${basis}`
        + (e.pending ? ` · ${e.pending} too recent to judge` : "")),
    ];
    for (const note of trace.notes) parts.push(h("span", { class: "small muted" }, note));
    $("#replay-summary").replaceChildren(...parts);
  }

  function stepEntry(delta) {
    const { trace, entryIndex } = state.replay;
    if (!trace || !trace.entries.length) return;
    const last = trace.entries.length - 1;
    const next = entryIndex < 0 ? (delta > 0 ? 0 : last) : Math.min(last, Math.max(0, entryIndex + delta));
    chart.focusEntry(next);
  }

  function replayOnChart(timeframe, name) {
    showTab("dashboard");
    setChartTf(timeframe);
    runReplay(name);
    $(".chart-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ------------------------------------------------------------------ prices / ticker

  function onPrice(tick) {
    if (tick.instrument !== state.instrument) return;
    const previous = state.lastPrice;
    state.lastPrice = tick;
    const direction = previous ? (tick.mid > previous.mid ? "up" : tick.mid < previous.mid ? "down" : "") : "";
    const decimals = priceDecimals(tick.mid);
    const set = (id, text) => {
      const el = $(id);
      el.textContent = text;
      el.className = direction;
    };
    set("#tick-bid", tick.bid.toFixed(decimals));
    set("#tick-ask", tick.ask.toFixed(decimals));
    const info = state.instrumentInfo.get(tick.instrument);
    const pipLocation = info && Number.isInteger(info.pip_location) ? info.pip_location : null;
    $("#tick-spread").textContent = pipLocation !== null
      ? `${(tick.spread / 10 ** pipLocation).toFixed(1)} pips`
      : tick.spread.toFixed(decimals);
    chart.tick(tick);
  }

  // ------------------------------------------------------------------ websocket

  function connectWebSocket() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${location.host}/ws`);
    state.ws = ws;
    ws.addEventListener("message", (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      handleMessage(message);
    });
    ws.addEventListener("close", () => {
      state.clientId = null;
      setStreamStatus("offline", "connection to TradeSense lost; reconnecting");
      const delay = state.wsBackoff;
      state.wsBackoff = Math.min(state.wsBackoff * 2, 15000);
      setTimeout(connectWebSocket, delay);
    });
  }

  function send(message) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(message));
      return true;
    }
    return false;
  }

  function subscribePrices() {
    state.lastPrice = null;
    for (const id of ["#tick-bid", "#tick-ask", "#tick-spread"]) $(id).textContent = "–";
    $("#ticker-inst").textContent = state.instrument;
    if (state.meta && state.meta.oanda.configured && state.meta.oanda.account_configured) {
      send({ action: "subscribe_prices", instrument: state.instrument });
    } else {
      setStreamStatus("unavailable", "OANDA account not configured");
    }
  }

  function handleMessage(message) {
    switch (message.type) {
      case "hello":
        state.clientId = message.client_id;
        state.wsBackoff = 1000;
        setStreamStatus("idle");
        subscribePrices();
        if (state.autoRefresh) sendAutoRefresh();
        break;
      case "price":
        onPrice(message);
        break;
      case "price_status":
        setStreamStatus(message.state, message.message);
        break;
      case "analysis_started":
        for (const [tf, names] of Object.entries(message.selections || {})) {
          if (names.length) markPending(tf, names);
        }
        break;
      case "strategy_result":
        applyStrategyResult(message.result);
        break;
      case "consensus":
        applyConsensus(message.timeframe, message.consensus);
        break;
      case "analysis_complete":
        applyAnalysisResult(message.result);
        break;
      case "analysis_error":
        toast(message.message, "error");
        break;
      case "backtest_progress":
        if (state.backtest.running) onBacktestProgress(message);
        break;
      case "auto_refresh_status":
        state.nextRuns = message.next_runs || {};
        updateNextRunLabels();
        break;
      case "error":
        toast(message.message, "error");
        break;
      default:
        break;
    }
  }

  // ------------------------------------------------------------------ columns & pickers

  function renderColumns() {
    const container = $("#columns");
    container.replaceChildren();
    for (const tf of timeframes()) {
      const column = h("div", { class: "column", dataset: { tf } },
        h("div", { class: "panel column-head" },
          h("div", { class: "column-title" },
            h("h2", {}, TF_LABELS[tf] || tf.toUpperCase()),
            h("span", { class: "count" }),
            h("span", { class: "next-run" })),
          renderPicker(tf)),
        h("div", { class: "consensus-slot" }),
        h("div", { class: "cards" }));
      container.append(column);
      updateColumnState(tf);
      restoreColumn(tf);
    }
    updateNextRunLabels();
  }

  function renderPicker(tf) {
    const body = h("div", { class: "picker-body" });
    const selected = new Set(state.selections[tf] || []);
    const groups = new Map();
    for (const info of state.strategies) {
      if (!groups.has(info.category)) groups.set(info.category, []);
      groups.get(info.category).push(info);
    }
    for (const [category, items] of groups) {
      const group = h("div", { class: "picker-group" }, h("h3", {}, CATEGORY_LABELS[category] || category));
      for (const info of items) {
        const input = h("input", {
          type: "checkbox", value: info.name,
          checked: selected.has(info.name) && info.available,
          disabled: !info.available,
          onchange: (event) => toggleSelection(tf, info.name, event.target.checked),
        });
        const tooltip = info.available ? `${info.description}\nNeeds ≥ ${info.min_candles} candles`
          : `${info.unavailable_reason || "Unavailable"}`;
        group.append(h("label", { class: `picker-item${info.available ? "" : " unavailable"}`, title: tooltip },
          input,
          h("span", { class: "icon", "aria-label": SOURCE_LABELS[info.source] }, ICONS[info.source] || "•"),
          h("span", {}, info.display_name)));
      }
      body.append(group);
    }
    if (!state.strategies.length) body.append(h("p", { class: "placeholder" }, "No strategies loaded."));
    return h("details", { class: "picker", open: true }, h("summary", {}, "Strategies"), body);
  }

  function toggleSelection(tf, name, checked) {
    const list = (state.selections[tf] || []).filter((n) => n !== name);
    if (checked) list.push(name);
    state.selections[tf] = list;
    savePrefs();
    updateColumnState(tf);
    if (state.autoRefresh) scheduleAutoRefreshUpdate();
  }

  function updateColumnState(tf) {
    const column = $(`.column[data-tf="${tf}"]`);
    if (!column) return;
    const enabled = state.enabledTfs.includes(tf);
    column.classList.toggle("disabled", !enabled);
    const count = (state.selections[tf] || []).length;
    $(".count", column).textContent = enabled ? `${count} selected` : "timeframe off";
  }

  function sanitizeSelections() {
    for (const tf of timeframes()) {
      const list = state.selections[tf] || [];
      state.selections[tf] = list.filter((name) => {
        const info = state.strategyByName.get(name);
        return info && info.available;
      });
    }
  }

  function currentSelections() {
    const selections = {};
    for (const tf of timeframes()) {
      if (!state.enabledTfs.includes(tf)) continue;
      const names = (state.selections[tf] || []).filter((n) => state.strategyByName.has(n));
      if (names.length) selections[tf] = names;
    }
    return selections;
  }

  // ------------------------------------------------------------------ cards

  function columnParts(tf) {
    const column = $(`.column[data-tf="${tf}"]`);
    return column ? { consensus: $(".consensus-slot", column), cards: $(".cards", column) } : null;
  }

  function markPending(tf, names) {
    const parts = columnParts(tf);
    if (!parts) return;
    parts.consensus.replaceChildren(h("div", { class: "card consensus pending" },
      h("div", { class: "card-head" }, h("span", { class: "card-title" }, "Consensus")),
      h("div", { class: "running" }, h("span", { class: "spinner" }), "waiting for strategies…")));
    parts.cards.replaceChildren(...names.map((name) => {
      const info = state.strategyByName.get(name);
      return h("div", { class: "card pending", dataset: { strategy: name } },
        h("div", { class: "card-head" },
          h("span", { class: "card-title" },
            h("span", { class: "icon" }, info ? ICONS[info.source] : "•"),
            info ? info.display_name : name)),
        h("div", { class: "running" }, h("span", { class: "spinner" }), "running…"));
    }));
  }

  function probabilityBar(probability) {
    const segment = (cls, value, letter) => {
      const width = Math.max(0, value * 100);
      const el = h("span", { class: cls, title: `${letter} ${pct(value)}` }, width >= 14 ? `${letter} ${pct(value)}` : "");
      el.style.width = `${width}%`;
      return el;
    };
    return [
      h("div", { class: "probbar", role: "img",
        "aria-label": `Probability: buy ${pct(probability.buy)}, sell ${pct(probability.sell)}, sideways ${pct(probability.sideways)}` },
      segment("p-buy", probability.buy, "BUY"),
      segment("p-sell", probability.sell, "SELL"),
      segment("p-sideways", probability.sideways, "SIDE")),
      h("div", { class: "problegend" },
        h("span", {}, `Buy ${pct(probability.buy)}`),
        h("span", {}, `Sell ${pct(probability.sell)}`),
        h("span", {}, `Sideways ${pct(probability.sideways)}`)),
    ];
  }

  function signalBadge(signal, large = false) {
    return h("span", { class: `signal signal-${signal}${large ? " signal-large" : ""}` }, signal);
  }

  function strategyCard(result) {
    const info = state.strategyByName.get(result.strategy);
    const icon = ICONS[result.source || (info && info.source)] || "•";
    const name = result.display_name || (info && info.display_name) || result.strategy;
    const title = h("span", { class: "card-title", title: info ? info.description : name },
      h("span", { class: "icon" }, icon), name);

    if (result.status === "error") {
      return h("div", { class: "card error", dataset: { strategy: result.strategy } },
        h("div", { class: "card-head" }, title, h("span", { class: "meta" }, "error")),
        h("div", { class: "muted" }, result.message || "Strategy failed"));
    }
    if (result.status === "insufficient_data") {
      return h("div", { class: "card insufficient", dataset: { strategy: result.strategy } },
        h("div", { class: "card-head" }, title, h("span", { class: "meta" }, "insufficient data")),
        h("div", {}, result.message || "Not enough candles."),
        h("div", { class: "small" },
          `Needs ${result.candles_required ?? "?"} completed candles, ${result.candles_available ?? 0} available.`));
    }

    const d = result.decision;
    const card = h("div", { class: `card sig-${d.signal}`, dataset: { strategy: result.strategy } },
      h("div", { class: "card-head" }, title, signalBadge(d.signal)),
      ...probabilityBar(d.probability),
      h("div", { class: "stats" },
        h("span", {}, h("span", { class: "k" }, "Confidence "), h("b", {}, pct(d.confidence))),
        d.pattern_match !== null && d.pattern_match !== undefined
          ? h("span", {}, h("span", { class: "k" }, "Pattern match "), h("b", {}, pct(d.pattern_match))) : null,
        result.pre_signal
          ? h("span", { title: "Rule-based pre-signal (hint given to the LLM)" },
            h("span", { class: "k" }, "Pre-signal "), h("b", {}, result.pre_signal)) : null));

    if (d.observations && d.observations.length) {
      card.append(h("ul", { class: "observations" }, d.observations.map((o) => h("li", {}, o))));
    }
    if (d.reasoning) {
      card.append(h("details", { class: "reasoning" }, h("summary", {}, "Reasoning"), h("p", {}, d.reasoning)));
    }
    const levels = d.key_levels || {};
    if (levels.support != null || levels.resistance != null) {
      card.append(h("div", { class: "stats" },
        h("span", {}, h("span", { class: "k" }, "Support "), h("b", {}, fmtPrice(levels.support))),
        h("span", {}, h("span", { class: "k" }, "Resistance "), h("b", {}, fmtPrice(levels.resistance)))));
    }
    if (result.reference_image_url || result.live_chart_url) {
      const figure = (src, caption) => h("figure", {},
        h("a", { href: src, target: "_blank", rel: "noopener" }, h("img", { src, alt: caption, loading: "lazy" })),
        h("figcaption", {}, caption));
      card.append(h("div", { class: "thumbs" },
        result.reference_image_url ? figure(result.reference_image_url, "Reference") : null,
        result.live_chart_url ? figure(result.live_chart_url, "Live chart") : null));
    }
    card.append(h("div", { class: "card-foot" },
      h("span", { class: "model", title: "Provider (model) that decided" }, `${d.decided_by} · ${d.model}`),
      h("span", {}, `${(d.latency_ms / 1000).toFixed(1)} s`),
      h("span", {}, fmtTime(d.decided_at)),
      d.fallback_used ? h("span", { class: "badge badge-fallback" }, "fallback LLM") : null,
      d.error ? h("span", { title: d.error }, "⚠ LLM error") : null,
      result.source === "builtin" || result.source === "rule"
        ? h("button", {
          type: "button", class: "link-btn",
          title: `Show this strategy's last 2 days on the ${TF_LABELS[result.timeframe] || result.timeframe} chart`,
          onclick: () => replayOnChart(result.timeframe, result.strategy),
        }, "Replay on chart")
        : null));
    return card;
  }

  function consensusCard(tf, decision) {
    if (!decision) {
      return h("div", { class: "card consensus" },
        h("div", { class: "card-head" }, h("span", { class: "card-title" }, "Consensus")),
        h("p", { class: "placeholder" }, "No LLM decisions to combine."));
    }
    return h("div", { class: `card consensus sig-${decision.signal}` },
      h("div", { class: "card-head" },
        h("span", { class: "card-title" }, `Consensus · ${TF_LABELS[tf] || tf}`),
        h("span", { class: "meta" }, `confidence ${pct(decision.confidence)}`)),
      h("div", {}, signalBadge(decision.signal, true)),
      ...probabilityBar(decision.probability),
      decision.observations && decision.observations.length
        ? h("ul", { class: "observations" }, decision.observations.map((o) => h("li", {}, o))) : null);
  }

  function applyStrategyResult(result) {
    state.results.set(`${result.timeframe}|${result.strategy}`, result);
    const parts = columnParts(result.timeframe);
    if (parts) {
      const card = strategyCard(result);
      const existing = $$(".card", parts.cards).find((el) => el.dataset.strategy === result.strategy);
      if (existing) existing.replaceWith(card);
      else parts.cards.append(card);
    }
    renderTable();
    if (result.timeframe === state.chartTf) chart.renderMarkers();
  }

  function applyConsensus(tf, decision) {
    state.consensus.set(tf, decision);
    const parts = columnParts(tf);
    if (parts) parts.consensus.replaceChildren(consensusCard(tf, decision));
  }

  function applyAnalysisResult(result) {
    state.lastResult = result;
    for (const [tf, tfResult] of Object.entries(result.timeframes)) {
      state.latestByTf[tf] = tfResult;
      if (tfResult.error && !tfResult.strategies.length) {
        const parts = columnParts(tf);
        if (parts) parts.consensus.replaceChildren(h("div", { class: "card error" }, tfResult.error));
      }
      for (const item of tfResult.strategies) applyStrategyResult(item);
      if (tfResult.strategies.length) applyConsensus(tf, tfResult.consensus);
    }
    renderUsage(result.usage);
  }

  function restoreColumn(tf) {
    const parts = columnParts(tf);
    if (!parts) return;
    const items = [...state.results.values()].filter((r) => r.timeframe === tf);
    if (!items.length) {
      parts.cards.replaceChildren(h("p", { class: "placeholder" }, "Select strategies and run the analysis."));
      return;
    }
    parts.cards.replaceChildren(...items.map(strategyCard));
    if (state.consensus.has(tf)) parts.consensus.replaceChildren(consensusCard(tf, state.consensus.get(tf)));
  }

  // ------------------------------------------------------------------ run analysis

  async function runAnalysis() {
    if (state.running) return;
    commitInstrument();
    const selections = currentSelections();
    if (!Object.keys(selections).length) {
      toast("Select at least one strategy in an enabled timeframe.", "warn");
      return;
    }
    setRunning(true);
    for (const [tf, names] of Object.entries(selections)) markPending(tf, names);
    try {
      const result = await api("/api/analysis", {
        method: "POST",
        body: { instrument: state.instrument, selections, client_id: state.clientId },
      });
      applyAnalysisResult(result);
      const errors = Object.values(result.timeframes).flatMap((t) => t.strategies).filter((s) => s.status === "error");
      if (errors.length) toast(`${errors.length} strategy slot(s) failed; see the cards for details.`, "warn");
    } catch (error) {
      toast(`Analysis failed: ${error.message}`, "error", 9000);
      for (const tf of Object.keys(selections)) restoreColumn(tf);
    } finally {
      setRunning(false);
    }
  }

  function setRunning(running) {
    state.running = running;
    const button = $("#run-button");
    button.disabled = running;
    $(".spinner", button).hidden = !running;
    $(".label", button).textContent = running ? "Running…" : "Run Analysis";
  }

  // ------------------------------------------------------------------ auto refresh

  let autoRefreshTimer = null;

  function sendAutoRefresh() {
    if (!state.autoRefresh) {
      send({ action: "auto_refresh", enabled: false });
      return;
    }
    const selections = currentSelections();
    if (!Object.keys(selections).length) {
      toast("Auto-refresh needs at least one selected strategy.", "warn");
      $("#auto-refresh").checked = false;
      state.autoRefresh = false;
      return;
    }
    if (!send({ action: "auto_refresh", enabled: true, instrument: state.instrument, selections })) {
      toast("Not connected to the server yet; auto-refresh will start when connected.", "warn");
    }
  }

  function scheduleAutoRefreshUpdate() {
    clearTimeout(autoRefreshTimer);
    autoRefreshTimer = setTimeout(sendAutoRefresh, 800);
  }

  function updateNextRunLabels() {
    for (const tf of timeframes()) {
      const column = $(`.column[data-tf="${tf}"]`);
      if (!column) continue;
      const label = $(".next-run", column);
      const next = state.autoRefresh && state.nextRuns[tf] ? Date.parse(state.nextRuns[tf]) : null;
      if (!next) {
        label.textContent = "";
        continue;
      }
      const seconds = Math.max(0, Math.round((next - Date.now()) / 1000));
      label.textContent = `next ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    }
  }

  // ------------------------------------------------------------------ results table

  function tableRows() {
    const order = timeframes();
    return [...state.results.values()].map((r) => {
      const d = r.decision;
      return {
        result: r,
        timeframe: r.timeframe,
        tfIndex: order.indexOf(r.timeframe),
        strategy: r.display_name || r.strategy,
        source: r.source || "",
        signal: d ? d.signal : r.status,
        buy: d ? d.probability.buy : -1,
        sell: d ? d.probability.sell : -1,
        sideways: d ? d.probability.sideways : -1,
        confidence: d ? d.confidence : -1,
        model: d ? `${d.decided_by} / ${d.model}` : "",
        time: d ? d.decided_at : r.finished_at,
      };
    });
  }

  function renderTable() {
    const rows = tableRows();
    const { key, dir } = state.sort;
    rows.sort((a, b) => {
      const av = key === "timeframe" ? a.tfIndex : a[key];
      const bv = key === "timeframe" ? b.tfIndex : b[key];
      if (av < bv) return -dir;
      if (av > bv) return dir;
      return a.strategy.localeCompare(b.strategy);
    });
    const tbody = $("#results-table tbody");
    if (!rows.length) {
      tbody.replaceChildren(h("tr", { class: "empty" }, h("td", { colspan: 10 }, "Run an analysis to see results.")));
      return;
    }
    const num = (x) => h("td", { class: "num" }, x < 0 ? "–" : pct(x));
    tbody.replaceChildren(...rows.map((row) => h("tr", {},
      h("td", {}, TF_LABELS[row.timeframe] || row.timeframe),
      h("td", {}, row.strategy),
      h("td", {}, `${ICONS[row.source] || ""} ${SOURCE_LABELS[row.source] || ""}`.trim()),
      h("td", {}, ["BUY", "SELL", "SIDEWAYS"].includes(row.signal)
        ? signalBadge(row.signal) : h("span", { class: "muted" }, row.signal.replace("_", " "))),
      num(row.buy), num(row.sell), num(row.sideways), num(row.confidence),
      h("td", { class: "small" }, row.model),
      h("td", { class: "small" }, fmtTime(row.time)))));
    for (const th of $$("#results-table th")) {
      th.setAttribute("aria-sort", th.dataset.key === key ? (dir > 0 ? "ascending" : "descending") : "none");
    }
  }

  function exportJson() {
    if (!state.lastResult) {
      toast("Nothing to export yet: run an analysis first.", "warn");
      return;
    }
    const payload = {
      exported_at: new Date().toISOString(),
      instrument: state.instrument,
      last_run: state.lastResult,
      latest_by_timeframe: state.latestByTf,
      disclaimer: state.meta ? state.meta.disclaimer : "",
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const link = h("a", { href: url, download: `tradesense-${state.instrument}-${stamp}.json` });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // ------------------------------------------------------------------ strategies

  async function loadStrategies() {
    const body = await api("/api/strategies");
    state.strategies = body.strategies;
    state.strategyByName = new Map(body.strategies.map((s) => [s.name, s]));
    state.visionAvailable = body.vision_available;
    for (const error of body.errors || []) console.warn("Strategy skipped:", error);
    sanitizeSelections();
    populateReplayOptions();
  }

  async function reloadStrategies() {
    try {
      const report = await api("/api/strategies/reload", { method: "POST" });
      await loadStrategies();
      renderColumns();
      renderBanners();
      toast(`Strategies reloaded: ${report.builtin} built-in, ${report.rules} rule, ${report.reference} reference`
        + (report.errors.length ? ` (${report.errors.length} skipped)` : ""), report.errors.length ? "warn" : "ok");
    } catch (error) {
      toast(`Reload failed: ${error.message}`, "error");
    }
  }

  // ------------------------------------------------------------------ instrument

  function commitInstrument() {
    const input = $("#instrument");
    const value = input.value.trim().toUpperCase().replace(/[\s/\-.]+/g, "_");
    if (!value) {
      input.value = state.instrument;
      return;
    }
    input.value = value;
    if (value === state.instrument) return;
    state.instrument = value;
    savePrefs();
    state.results.clear();
    state.consensus.clear();
    renderColumns();
    renderTable();
    refreshChart();
    subscribePrices();
    if (state.autoRefresh) sendAutoRefresh();
  }

  async function loadInstruments() {
    const list = $("#instrument-list");
    let names = DEFAULT_INSTRUMENTS;
    try {
      const instruments = await api("/api/instruments");
      for (const info of instruments) state.instrumentInfo.set(info.name, info);
      names = instruments.map((i) => i.name);
    } catch {
      /* OANDA not configured: offer common symbols */
    }
    list.replaceChildren(...names.map((n) => h("option", { value: n })));
  }

  // ------------------------------------------------------------------ backtest

  // Progress-bar share of each backtest stage (replaying built-ins dominates).
  const BT_STAGES = {
    candles: [0, 5], replay: [5, 70], templates: [70, 78], llm_rules: [78, 88], llm_check: [88, 99], done: [100, 100],
  };
  const ORIGIN_LABELS = { builtin: "Built-in", rule: "Rule", template: "Template", llm: "LLM" };

  function showTab(name) {
    for (const tab of $$(".tab")) {
      const active = tab.dataset.tab === name;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
      $(`#tab-${tab.dataset.tab}`).hidden = !active;
    }
    if (name === "library") loadLibrary();
  }

  function setBacktestRunning(running) {
    state.backtest.running = running;
    for (const button of [$("#backtest-button"), $("#bt-run")]) {
      button.disabled = running;
      $(".spinner", button).hidden = !running;
    }
    $("#bt-progress").hidden = !running;
    if (running) {
      $("#bt-progress-bar").style.width = "0%";
      $("#bt-progress-label").textContent = "Starting…";
    }
  }

  function onBacktestProgress(message) {
    const [start, end] = BT_STAGES[message.stage] || [0, 100];
    const share = message.total ? message.done / message.total : 1;
    $("#bt-progress-bar").style.width = `${Math.round(start + (end - start) * share)}%`;
    const count = message.total > 1 ? ` (${message.done}/${message.total})` : "";
    $("#bt-progress-label").textContent = `${message.message}${count}`;
  }

  async function runBacktest() {
    if (state.backtest.running) return;
    commitInstrument();
    showTab("backtest");
    setBacktestRunning(true);
    try {
      const result = await api("/api/backtest", {
        method: "POST",
        body: {
          instrument: state.instrument,
          client_id: state.clientId,
          llm_check: $("#bt-llm-check").checked,
          llm_rules: $("#bt-llm-rules").checked,
        },
      });
      renderBacktest(result);
      $("#bt-progress-label").textContent = `Finished in ${Math.round(result.duration_ms / 1000)} s.`;
    } catch (error) {
      $("#bt-progress-label").textContent = `Backtest failed: ${error.message}`;
      toast(`Backtest failed: ${error.message}`, "error", 9000);
    } finally {
      setBacktestRunning(false);
    }
  }

  const hitClass = (rate) => (rate === null || rate === undefined ? "" : rate >= 0.55 ? "hit-good" : rate < 0.45 ? "hit-bad" : "");
  const hitCell = (rate) => h("td", { class: `num ${hitClass(rate)}` }, pct(rate));
  const pips = (x) => (x === null || x === undefined ? "–" : `${x > 0 ? "+" : ""}${x.toFixed(1)}`);

  function renderBacktest(result) {
    state.backtest.result = result;
    $("#bt-title").textContent = `Backtest · ${result.instrument}`;
    const facts = [
      `Last ${result.days} days`,
      result.spread_pips !== null
        ? `a hit must beat the spread at its candle (median ${result.spread_pips} pips)`
        : "any favourable move counts as a hit",
      `each signal sees the last ${result.window} candles, as live`,
      result.llm_cost_usd ? `LLM cost $${result.llm_cost_usd.toFixed(4)}` : null,
    ].filter(Boolean);
    $("#bt-summary").replaceChildren(
      h("span", {}, `${facts.join(" · ")}.`),
      ...result.notes.map((note) => h("span", { class: "small muted bt-note" }, note)));
    $("#bt-columns").replaceChildren(...Object.values(result.timeframes).map((tf) => backtestColumn(result, tf)));
    $("#bt-apply").disabled = !Object.values(result.timeframes).some((tf) => tf.best);
  }

  function backtestColumn(result, tf) {
    const head = h("div", { class: "panel column-head" },
      h("div", { class: "column-title" },
        h("h2", {}, TF_LABELS[tf.timeframe] || tf.timeframe),
        h("span", { class: "count" }, tf.candles ? `${tf.candles} candles` : ""),
        h("span", { class: "next-run" }, tf.start ? `${fmtDate(tf.start)} → ${fmtDate(tf.end)}` : "")),
      h("span", { class: "small muted" }, `BUY / SELL judged ${tf.horizon_bars} candles later`));
    if (tf.error) return h("div", { class: "bt-column" }, head, h("p", { class: "placeholder" }, tf.error));

    const best = tf.strategies.find((s) => s.name === tf.best);
    const parts = [head, best ? bestCard(best) : h("p", { class: "placeholder" },
      `No strategy produced at least the minimum number of BUY/SELL signals.`)];
    parts.push(h("p", { class: "bt-section-title" }, "Leaderboard"), leaderboard(tf));
    parts.push(h("p", { class: "bt-section-title" }, "New strategy candidates"));
    if (tf.candidates.length) {
      parts.push(...tf.candidates.map((c) => candidateCard(result, tf, c)));
    } else {
      parts.push(h("p", { class: "placeholder" }, "No rule produced enough signals to judge."));
    }
    for (const note of tf.notes) parts.push(h("p", { class: "small muted" }, note));
    if (tf.skipped.length) {
      parts.push(h("details", { class: "small muted bt-skipped" },
        h("summary", {}, `Not backtested (${tf.skipped.length})`),
        h("ul", {}, tf.skipped.map((s) => h("li", { title: s.reason }, `${s.display_name}: ${s.reason}`)))));
    }
    return h("div", { class: "bt-column" }, ...parts);
  }

  function llmCheckLine(check) {
    if (!check) return null;
    const answered = check.samples - check.errors;
    if (!answered) return h("span", { class: "small muted" }, `LLM check: no usable answers (${check.errors} errors)`);
    return h("span", { class: "small", title: `Providers: ${check.providers.join(", ") || "–"}` },
      `LLM check on ${check.samples} of its signals: LLM ${pct(check.llm_hit_rate)} vs rule ${pct(check.rule_hit_rate)}`
      + ` · agrees ${pct(check.agreement)} · ${check.abstained} sideways`);
  }

  function bestCard(best) {
    const s = best.stats;
    return h("div", { class: "card bt-best" },
      h("div", { class: "card-head" },
        h("span", { class: "card-title" }, `Best fit: ${best.display_name}`),
        h("span", { class: "meta" }, ORIGIN_LABELS[best.origin])),
      h("div", { class: "bt-best-row" },
        h("span", { class: `big ${hitClass(s.hit_rate)}` }, pct(s.hit_rate)),
        h("span", { class: "small muted" }, `of ${s.signals} BUY/SELL calls right · BUY ${pct(s.buy_hit_rate)} · `
          + `SELL ${pct(s.sell_hit_rate)} · avg ${pips(s.avg_move_pips)} pips`)),
      llmCheckLine(best.llm_check));
  }

  function leaderboard(tf) {
    const sort = state.backtest.sort[tf.timeframe] || { key: null, dir: -1 };
    const value = (s, key) => (key === "name" ? s.display_name : key === "signals" ? s.stats.signals
      : key === "llm" ? (s.llm_check ? s.llm_check.llm_hit_rate : null) : s.stats[key]);
    const rows = [...tf.strategies];
    if (sort.key) {
      rows.sort((a, b) => {
        const x = value(a, sort.key), y = value(b, sort.key);
        if (x === y) return 0;
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;
        return (x > y ? 1 : -1) * sort.dir;
      });
    }
    const columns = [["#", null], ["Strategy", "name"], ["Hit", "hit_rate"], ["BUY", "buy_hit_rate"],
      ["SELL", "sell_hit_rate"], ["Signals", "signals"], ["Pips", "avg_move_pips"], ["LLM", "llm"]];
    const header = h("tr", {}, columns.map(([label, key]) => h("th", {
      class: key && key !== "name" ? "num" : null,
      "aria-sort": sort.key === key && key ? (sort.dir > 0 ? "ascending" : "descending") : "none",
      title: key === "llm" ? "LLM hit rate on a sample of this strategy's signals" : null,
      onclick: key ? () => {
        state.backtest.sort[tf.timeframe] = { key, dir: sort.key === key ? -sort.dir : -1 };
        renderBacktest(state.backtest.result);
      } : null,
    }, label)));
    const body = rows.map((s, i) => h("tr", {
      class: [s.ranked ? null : "bt-unranked", state.strategyByName.has(s.name) ? "bt-row-link" : null]
        .filter(Boolean).join(" ") || null,
      title: (s.error || (s.ranked ? s.description
        : `Fewer BUY/SELL signals than the minimum: not ranked. ${s.description}`))
        + (state.strategyByName.has(s.name) ? "\nClick to replay it on the Dashboard chart." : ""),
      onclick: state.strategyByName.has(s.name) ? () => replayOnChart(tf.timeframe, s.name) : null,
    },
      h("td", { class: "num" }, s.ranked ? String(i + 1) : "–"),
      h("td", {}, h("span", { class: "icon" }, ICONS[s.origin] || ""), ` ${s.display_name}`),
      hitCell(s.stats.hit_rate), hitCell(s.stats.buy_hit_rate), hitCell(s.stats.sell_hit_rate),
      h("td", { class: "num" }, s.error ? "error" : String(s.stats.signals)),
      h("td", { class: "num" }, pips(s.stats.avg_move_pips)),
      hitCell(s.llm_check ? s.llm_check.llm_hit_rate : null)));
    return h("div", { class: "table-wrap panel" },
      h("table", { class: "results bt-table" }, h("thead", {}, header), h("tbody", {}, body)));
  }

  function candidateCard(result, tf, candidate) {
    const train = candidate.train || {}, validation = candidate.validation || {};
    const robust = candidate.name === tf.best_candidate;
    const rules = candidate.rule ? candidate.rule.rules : { buy: "?", sell: "?" };
    const save = h("button", { type: "button", class: "btn btn-small" }, "Save as strategy");
    save.addEventListener("click", () => saveCandidate(result, tf.timeframe, candidate, save));
    return h("div", { class: `card bt-candidate${robust ? " bt-robust" : ""}` },
      h("div", { class: "card-head" },
        h("span", { class: "card-title", title: candidate.description }, candidate.display_name),
        h("span", { class: `badge ${candidate.origin === "llm" ? "badge-llm" : ""}` }, ORIGIN_LABELS[candidate.origin]),
        robust ? h("span", { class: "badge badge-robust", title: "Better than 50% on both halves of the period" }, "held up") : null),
      h("span", { class: "small" },
        "Day 1 ", h("b", { class: hitClass(train.hit_rate) }, pct(train.hit_rate)), ` (${train.signals || 0})`,
        " → day 2 ", h("b", { class: hitClass(validation.hit_rate) }, pct(validation.hit_rate)), ` (${validation.signals || 0} signals)`),
      h("code", { class: "bt-rule" }, `BUY:  ${rules.buy}\nSELL: ${rules.sell}`),
      h("div", { class: "card-foot" }, save));
  }

  async function saveCandidate(result, timeframe, candidate, button) {
    button.disabled = true;
    try {
      const info = await api("/api/backtest/strategies", {
        method: "POST",
        body: { run_id: result.run_id, timeframe, name: candidate.name },
      });
      button.textContent = "Saved";
      await loadStrategies();
      const picks = state.selections[timeframe] || [];
      if (!picks.includes(info.name)) state.selections[timeframe] = [...picks, info.name];
      savePrefs();
      renderColumns();
      toast(`Saved as ${info.name} and added to the ${TF_LABELS[timeframe] || timeframe} column.`, "ok");
    } catch (error) {
      button.disabled = false;
      toast(`Could not save: ${error.message}`, "error", 9000);
    }
  }

  function applyBestPicks() {
    const result = state.backtest.result;
    if (!result) return;
    const applied = [];
    for (const [tf, r] of Object.entries(result.timeframes)) {
      const picks = r.strategies.filter((s) => s.ranked && state.strategyByName.has(s.name)).slice(0, 3).map((s) => s.name);
      if (!picks.length) continue;
      state.selections[tf] = picks;
      applied.push(TF_LABELS[tf] || tf);
    }
    savePrefs();
    renderColumns();
    if (state.autoRefresh) scheduleAutoRefreshUpdate();
    showTab("dashboard");
    toast(applied.length ? `Top strategies selected for ${applied.join(", ")}.` : "No ranked strategies to apply.", applied.length ? "ok" : "warn");
  }

  // ------------------------------------------------------------------ library

  async function loadLibrary() {
    const grid = $("#library-grid");
    try {
      const cards = await api("/api/library");
      grid.replaceChildren(...cards.map(libraryTile));
      if (!cards.length) grid.append(h("p", { class: "placeholder" }, "No reference strategies yet. Upload one above."));
    } catch (error) {
      grid.replaceChildren(h("p", { class: "placeholder" }, `Could not load the library: ${error.message}`));
    }
  }

  function libraryTile(card) {
    return h("button", { class: "lib-card", type: "button", onclick: () => openReview(card.id) },
      h("div", { class: "lib-title" }, h("span", {}, ICONS[card.source_type]), h("span", {}, card.name)),
      h("div", { class: "lib-summary" }, card.summary || "No summary yet."),
      h("div", { class: "lib-meta" },
        h("span", { class: `badge badge-${card.status}` }, card.status),
        h("span", {}, fmtDate(card.uploaded_at))),
      card.warnings.length ? h("div", { class: "small", title: card.warnings.join("\n") },
        `⚠ ${card.warnings.length} warning(s)`) : null);
  }

  async function uploadFile(file) {
    const status = $("#dropzone-status");
    const limit = state.meta ? state.meta.library.max_upload_mb : 10;
    if (file.size > limit * 1024 * 1024) {
      toast(`${file.name} is larger than the ${limit} MB limit.`, "error");
      return;
    }
    status.replaceChildren(h("span", { class: "spinner" }), `Uploading ${file.name} and extracting a strategy card…`);
    const form = new FormData();
    form.append("file", file);
    try {
      const result = await api("/api/library/upload", { method: "POST", body: form });
      status.textContent = result.duplicate ? "This file is already in the library." : "Draft card created: review it below.";
      await loadLibrary();
      openReview(result.card.id);
    } catch (error) {
      status.textContent = "";
      toast(`Upload failed: ${error.message}`, "error", 9000);
    }
  }

  async function openReview(id) {
    try {
      const body = await api(`/api/library/${encodeURIComponent(id)}`);
      state.currentCard = body.card;
      $("#library-list-view").hidden = true;
      $("#review-view").hidden = false;
      renderPreview(body.card, body.preview);
      renderReviewForm(body.card);
    } catch (error) {
      toast(error.message, "error");
    }
  }

  function closeReview() {
    state.currentCard = null;
    $("#review-view").hidden = true;
    $("#library-list-view").hidden = false;
    loadLibrary();
  }

  function renderPreview(card, preview) {
    const container = $("#review-preview");
    container.replaceChildren(h("h2", {}, card.original_filename || card.source_file));
    if (preview.kind === "image") {
      const img = h("img", { src: preview.url, alt: `Reference chart ${card.name}` });
      const zoom = h("input", { type: "range", min: "50", max: "400", value: "100", step: "10", "aria-label": "Zoom" });
      const label = h("span", {}, "100%");
      zoom.addEventListener("input", () => {
        img.style.width = `${zoom.value}%`;
        label.textContent = `${zoom.value}%`;
      });
      img.style.width = "100%";
      container.append(h("div", { class: "zoom-controls" }, h("span", { class: "muted" }, "Zoom"), zoom, label),
        h("div", { class: "zoom-wrap" }, img));
    } else if (preview.kind === "markdown") {
      const div = h("div", { class: "markdown" });
      div.innerHTML = preview.html; // sanitised server-side: raw HTML escaped, unsafe URLs removed
      container.append(div);
    } else {
      container.append(h("pre", {}, preview.text || "(no extractable text)"));
    }
    if (preview.truncated) container.append(h("p", { class: "muted small" }, "Preview truncated."));
  }

  function conditionEditor(name, values) {
    const list = h("div", { class: "cond-list", dataset: { field: name } });
    const addRow = (value = "") => {
      const input = h("input", { type: "text", value, "aria-label": `${name.replace("_", " ")} item` });
      const row = h("div", { class: "cond-item" }, input,
        h("button", { type: "button", class: "btn btn-small", title: "Remove", onclick: () => row.remove() }, "✕"));
      list.append(row);
      return input;
    };
    values.forEach((v) => addRow(v));
    const add = h("button", { type: "button", class: "btn btn-small", onclick: () => addRow().focus() }, "+ Add condition");
    return [list, add];
  }

  function renderReviewForm(card) {
    $("#review-title").textContent = card.name;
    const badge = $("#review-status");
    badge.textContent = card.status;
    badge.className = `badge badge-${card.status}`;
    const form = $("#review-form");
    const indicators = state.meta ? Object.entries(state.meta.indicators) : [];
    const required = new Set(card.required_indicators);
    const row = (label, ...content) => h("label", { class: "row" }, h("span", {}, label), ...content);
    const block = (label, ...content) => h("div", { class: "row" }, h("span", {}, label), ...content);

    // replaceChildren() would stringify null placeholders, so drop them first.
    form.replaceChildren(...[
      row("Name", h("input", { type: "text", name: "name", value: card.name, required: true, maxlength: 120 })),
      row("Summary", h("textarea", { name: "summary", rows: 3, maxlength: 2000 }, card.summary)),
      row("Category", h("select", { name: "category" }, Object.entries(CATEGORY_LABELS).map(([value, text]) =>
        h("option", { value, selected: value === card.category }, text)))),
      block("Buy conditions", ...conditionEditor("buy_conditions", card.buy_conditions)),
      block("Sell conditions", ...conditionEditor("sell_conditions", card.sell_conditions)),
      block("Sideways conditions", ...conditionEditor("sideways_conditions", card.sideways_conditions)),
      block("Required indicators", h("div", { class: "indicator-grid" }, indicators.map(([name, description]) =>
        h("label", { class: "check", title: description },
          h("input", { type: "checkbox", name: "indicator", value: name, checked: required.has(name) }), name)))),
      row("Notes", h("textarea", { name: "notes", rows: 3, maxlength: 5000 }, card.notes)),
      card.warnings.length ? block("Warnings", h("ul", { class: "warnings" }, card.warnings.map((w) => h("li", {}, w)))) : null,
      h("div", { class: "review-meta" },
        h("span", {}, "Source"), h("span", {}, `${card.source_type} · ${card.source_file}`),
        h("span", {}, "Uploaded"), h("span", {}, fmtDate(card.uploaded_at)),
        h("span", {}, "Extracted"), h("span", {}, card.extracted_at ? `${fmtDate(card.extracted_at)} by ${card.extracted_by}` : "not yet"),
        h("span", {}, "Reviewed"), h("span", {}, card.reviewed_by_user ? "yes" : "no")),
      h("div", { class: "review-actions" },
        h("button", { type: "submit", class: "btn btn-primary" }, "Save"),
        h("button", { type: "button", class: "btn", onclick: reextractCard }, "Re-extract"),
        h("button", { type: "button", class: "btn btn-success", onclick: approveCard, disabled: card.status === "approved" }, "Approve"),
        h("button", { type: "button", class: "btn", onclick: disableCard, disabled: card.status === "disabled" }, "Disable"),
        h("button", { type: "button", class: "btn btn-danger", onclick: deleteCard }, "Delete")),
    ].filter(Boolean));
  }

  function readReviewForm() {
    const form = $("#review-form");
    const list = (field) => $$(`.cond-list[data-field="${field}"] input`, form)
      .map((input) => input.value.trim()).filter(Boolean);
    const field = (name) => form.elements.namedItem(name);
    return {
      name: field("name").value.trim(),
      summary: field("summary").value.trim(),
      category: field("category").value,
      buy_conditions: list("buy_conditions"),
      sell_conditions: list("sell_conditions"),
      sideways_conditions: list("sideways_conditions"),
      required_indicators: $$('input[name="indicator"]:checked', form).map((i) => i.value),
      notes: field("notes").value.trim(),
    };
  }

  async function saveCard(quiet = false) {
    const card = state.currentCard;
    if (!card) return null;
    try {
      const updated = await api(`/api/library/${encodeURIComponent(card.id)}`, { method: "PUT", body: readReviewForm() });
      state.currentCard = updated;
      renderReviewForm(updated);
      if (!quiet) toast("Card saved.", "ok");
      if (updated.status === "approved") await refreshStrategiesUi();
      return updated;
    } catch (error) {
      toast(`Save failed: ${error.message}`, "error");
      return null;
    }
  }

  async function cardAction(path, method, message) {
    const card = state.currentCard;
    if (!card) return;
    try {
      const updated = await api(`/api/library/${encodeURIComponent(card.id)}${path}`, { method });
      state.currentCard = updated;
      renderReviewForm(updated);
      toast(message, "ok");
      await refreshStrategiesUi();
    } catch (error) {
      toast(error.message, "error", 9000);
    }
  }

  async function approveCard() {
    if (await saveCard(true)) await cardAction("/approve", "POST", "Approved: the strategy is now selectable on every timeframe.");
  }

  async function disableCard() {
    await cardAction("/disable", "POST", "Disabled: removed from the strategy lists.");
  }

  async function reextractCard(event) {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "Extracting…";
    const card = state.currentCard;
    try {
      const updated = await api(`/api/library/${encodeURIComponent(card.id)}/extract`, { method: "POST" });
      state.currentCard = updated;
      renderReviewForm(updated);
      toast(updated.warnings.some((w) => w.startsWith("Extraction failed"))
        ? "Extraction failed; see warnings." : "Card re-extracted: please review it again.",
      updated.warnings.length ? "warn" : "ok");
      await refreshStrategiesUi();
    } catch (error) {
      toast(error.message, "error");
      button.disabled = false;
      button.textContent = "Re-extract";
    }
  }

  async function deleteCard() {
    const card = state.currentCard;
    if (!card || !window.confirm(`Delete "${card.name}" and its original file? This cannot be undone.`)) return;
    try {
      await api(`/api/library/${encodeURIComponent(card.id)}`, { method: "DELETE" });
      toast("Deleted.", "ok");
      await refreshStrategiesUi();
      closeReview();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function refreshStrategiesUi() {
    await loadStrategies();
    renderColumns();
  }

  // ------------------------------------------------------------------ wiring

  function bindEvents() {
    $("#run-form").addEventListener("submit", (event) => {
      event.preventDefault();
      runAnalysis();
    });
    $("#instrument").addEventListener("change", commitInstrument);
    $("#instrument").addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        commitInstrument();
      }
    });
    for (const box of $$('input[name="tf"]')) {
      box.checked = state.enabledTfs.includes(box.value);
      box.addEventListener("change", () => {
        state.enabledTfs = $$('input[name="tf"]:checked').map((b) => b.value);
        savePrefs();
        timeframes().forEach(updateColumnState);
        if (state.autoRefresh) scheduleAutoRefreshUpdate();
      });
    }
    $("#auto-refresh").addEventListener("change", (event) => {
      state.autoRefresh = event.target.checked;
      if (!state.autoRefresh) state.nextRuns = {};
      sendAutoRefresh();
      updateNextRunLabels();
    });
    $("#llm-reload").addEventListener("click", async (event) => {
      const button = event.currentTarget;
      button.disabled = true;
      try {
        const status = await api("/api/llm/reload", { method: "POST" });
        state.llmStatus = status;
        state.visionAvailable = status.vision_available;
        await refreshLlmModels();
        await loadStrategies();
        renderColumns();
        renderBanners();
        toast("LLM configuration reloaded.", "ok");
      } catch (error) {
        toast(`LLM reload failed: ${error.message}`, "error", 9000);
      } finally {
        button.disabled = false;
      }
    });

    for (const button of $$("#chart-tf button")) {
      button.classList.toggle("active", button.dataset.tf === state.chartTf);
      button.addEventListener("click", () => {
        setChartTf(button.dataset.tf);
        refreshChart(); // a replay re-runs on the new timeframe
      });
    }
    $("#replay-strategy").addEventListener("change", (event) => {
      if (event.target.value) runReplay(event.target.value);
      else exitReplay();
    });
    $("#replay-exit").addEventListener("click", exitReplay);
    $("#replay-prev").addEventListener("click", () => stepEntry(-1));
    $("#replay-next").addEventListener("click", () => stepEntry(1));
    $("#overlay-ema").checked = state.overlays.ema;
    $("#overlay-ema").addEventListener("change", (event) => {
      state.overlays.ema = event.target.checked;
      savePrefs();
      chart.renderEmas();
    });
    $("#overlay-markers").checked = state.overlays.markers;
    $("#overlay-markers").addEventListener("change", (event) => {
      state.overlays.markers = event.target.checked;
      savePrefs();
      chart.renderMarkers();
    });

    $("#apply-all").addEventListener("click", () => {
      const source = $("#copy-source").value;
      const picks = [...(state.selections[source] || [])];
      for (const tf of timeframes()) state.selections[tf] = [...picks];
      savePrefs();
      renderColumns();
      if (state.autoRefresh) scheduleAutoRefreshUpdate();
      toast(`Copied ${picks.length} strategies from ${TF_LABELS[source] || source} to all timeframes.`, "ok", 3000);
    });
    $("#clear-all").addEventListener("click", () => {
      for (const tf of timeframes()) state.selections[tf] = [];
      savePrefs();
      renderColumns();
    });
    $("#reload-strategies").addEventListener("click", reloadStrategies);
    $("#export-json").addEventListener("click", exportJson);
    for (const th of $$("#results-table th")) {
      th.addEventListener("click", () => {
        const key = th.dataset.key;
        state.sort = { key, dir: state.sort.key === key ? -state.sort.dir : 1 };
        renderTable();
      });
    }

    for (const tab of $$(".tab")) {
      tab.addEventListener("click", () => showTab(tab.dataset.tab));
    }
    $("#backtest-button").addEventListener("click", runBacktest);
    $("#bt-run").addEventListener("click", runBacktest);
    $("#bt-apply").addEventListener("click", applyBestPicks);

    const dropzone = $("#dropzone");
    const input = $("#file-input");
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        input.click();
      }
    });
    input.addEventListener("change", () => {
      if (input.files.length) uploadFile(input.files[0]);
      input.value = "";
    });
    ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    }));
    ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    }));
    dropzone.addEventListener("drop", (event) => {
      const file = event.dataTransfer && event.dataTransfer.files[0];
      if (file) uploadFile(file);
    });
    $("#review-back").addEventListener("click", closeReview);
    $("#review-form").addEventListener("submit", (event) => {
      event.preventDefault();
      saveCard();
    });
  }

  async function boot() {
    loadPrefs();
    $("#instrument").value = state.instrument;
    $("#ticker-inst").textContent = state.instrument;
    try {
      state.meta = await api("/api/meta");
    } catch (error) {
      toast(`Cannot reach the TradeSense API: ${error.message}`, "error", 15000);
      return;
    }
    if (state.meta.library) {
      $("#file-input").setAttribute("accept", state.meta.library.accept);
      $("#dropzone-hint").textContent = `${state.meta.library.allowed_types.map((t) => `.${t}`).join(" ")} · up to ${state.meta.library.max_upload_mb} MB`;
    }
    bindEvents();
    chart.init();
    try {
      await loadStrategies();
    } catch (error) {
      toast(`Could not load strategies: ${error.message}`, "error");
    }
    renderColumns();
    renderBanners();
    await refreshLlmStatus();
    refreshLlmModels(); // Ollama discovery can take a moment: labels show until then
    loadInstruments().then(() => chart.load());
    connectWebSocket();
    setInterval(refreshLlmStatus, 30000);
    setInterval(updateNextRunLabels, 1000);
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
