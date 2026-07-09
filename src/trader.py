"""Main trading loop. Run via scripts/run_paper.py."""
from __future__ import annotations

import logging
import time
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from src.broker.alpaca_client import AlpacaBroker
from src.config import Config, ROOT, load_config
from src.data.earnings import near_earnings, next_earnings_dates
from src.data.market_data import get_history, get_history_many
from src.data.universe import load_universe
from src.politicians.tracker import politician_signals
from src.risk.manager import RiskManager, TradeIntent
from src.signals.classical import classical_signal
from src.signals.hedge_fund import hedge_fund_decision
from src.signals.ml import load_model, ml_signal
from src.trade_log import TradeLogEntry, TradeLogger, trade_logger_from_config

log = logging.getLogger(__name__)

DEFAULT_MARKET_TZ = "America/New_York"


def _consolidate_intents(intents: list[TradeIntent]) -> list[TradeIntent]:
    """Merge duplicate same-side intents so one symbol gets one order per cycle."""
    merged: dict[tuple[str, str], TradeIntent] = {}
    order: list[tuple[str, str]] = []
    for intent in intents:
        key = (intent.symbol, intent.side)
        if key not in merged:
            merged[key] = intent
            order.append(key)
            continue
        prev = merged[key]
        merged[key] = TradeIntent(
            symbol=prev.symbol,
            side=prev.side,
            target_dollars=prev.target_dollars + intent.target_dollars,
            reason=f"{prev.reason}; {intent.reason}",
        )
    return [merged[key] for key in order]


def _setup_logging(cfg: Config) -> None:
    level = getattr(logging, str(cfg.get("logging", "level", default="INFO")).upper(), logging.INFO)
    log_file = Path(cfg.get("logging", "file", default="logs/trader.log"))
    if not log_file.is_absolute():
        log_file = ROOT / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def _last_prices(symbols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not symbols:
        return out
    try:
        data = yf.download(symbols, period="1d", progress=False, threads=False, auto_adjust=True)
        if "Close" in data.columns:
            close = data["Close"]
            if hasattr(close, "iloc"):
                if hasattr(close, "columns"):
                    last = close.iloc[-1]
                    for s in symbols:
                        if s in last.index:
                            out[s] = float(last[s])
                else:
                    out[symbols[0]] = float(close.iloc[-1])
    except Exception as e:
        log.warning("price fetch failed: %s", e)
    return out


def compute_signals(
    cfg: Config,
    symbols: list[str],
    history: dict[str, pd.DataFrame] | None = None,
) -> dict[str, float]:
    """Composite signal in [-1, 1] per symbol."""
    if history is None:
        history = get_history_many(cfg, symbols)
    bundle = load_model(cfg) if cfg.get("strategies", "ml", "enabled", default=True) else None
    use_hedge_fund = bool(cfg.get("strategies", "hedge_fund", "enabled", default=False))

    w_cls = float(cfg.get("strategies", "classical", "weight", default=0.4)) if cfg.get("strategies", "classical", "enabled", default=True) else 0.0
    w_ml = float(cfg.get("strategies", "ml", "weight", default=0.4)) if cfg.get("strategies", "ml", "enabled", default=True) else 0.0
    w_pol = float(cfg.get("strategies", "politicians", "weight", default=0.2)) if cfg.get("strategies", "politicians", "enabled", default=True) else 0.0
    total = max(1e-9, w_cls + w_ml + w_pol)

    pol_scores = politician_signals(cfg, symbols) if w_pol > 0 else {s: 0.0 for s in symbols}

    out: dict[str, float] = {}
    for sym in symbols:
        df = history.get(sym)
        if df is None or df.empty:
            out[sym] = 0.0
            continue
        if use_hedge_fund:
            decision = hedge_fund_decision(cfg, df, bundle=bundle)
            out[sym] = decision.score
            log.debug("hedge_fund_signal %s %s", sym, decision.reasoning)
            continue
        cls = classical_signal(df) if w_cls > 0 else 0.0
        ml = ml_signal(cfg, df, bundle=bundle) if (w_ml > 0 and bundle) else 0.0
        pol = pol_scores.get(sym, 0.0)
        composite = (w_cls * cls + w_ml * ml + w_pol * pol) / total
        out[sym] = max(-1.0, min(1.0, composite))
        log.debug("signal %s cls=%+.2f ml=%+.2f pol=%+.2f -> %+.2f", sym, cls, ml, pol, out[sym])
    return out


def _history_for_all(cfg: Config, symbols: list[str], held_symbols: list[str]) -> dict[str, pd.DataFrame]:
    """History for the universe, plus any held symbols not in the universe (needed for stop math)."""
    hist = get_history_many(cfg, symbols)
    for extra in held_symbols:
        if extra not in hist:
            df = get_history(cfg, extra)
            if not df.empty:
                hist[extra] = df
    return hist


def _apply_earnings_blackout(
    cfg: Config,
    intents: list[TradeIntent],
) -> list[TradeIntent]:
    buy_symbols = [i.symbol for i in intents if i.side == "buy"]
    if not buy_symbols:
        return intents
    blackout_days = int(cfg.get("risk", "earnings_blackout_days", default=0))
    if blackout_days <= 0:
        return intents
    next_dates = next_earnings_dates(cfg, buy_symbols)
    blocked = {s for s in buy_symbols if near_earnings(next_dates, s, blackout_days)}
    if not blocked:
        return intents
    for sym in blocked:
        log.info("[SKIP] BUY %s: earnings within %d days", sym, blackout_days)
    return [i for i in intents if not (i.side == "buy" and i.symbol in blocked)]


def trade_once(cfg: Config) -> None:
    broker = AlpacaBroker(cfg)
    trade_log = trade_logger_from_config(cfg)
    risk = RiskManager(cfg, broker, trade_log=trade_log)
    dry = bool(cfg.get("dry_run", default=False))
    mode = "dry" if dry else ("live" if cfg.is_live else "paper")

    if not broker.is_market_open():
        log.info("market closed; skipping")
        return

    symbols = load_universe(cfg)
    log.info("evaluating %d symbols", len(symbols))
    held_symbols = [p.symbol for p in broker.positions()]
    history = _history_for_all(cfg, symbols, held_symbols)

    risk.apply_stop_losses(history=history, dry_run=dry, mode=mode)

    scores = compute_signals(cfg, symbols, history=history)
    prices = _last_prices(symbols)
    intents = _consolidate_intents(risk.size_orders(scores, prices, history=history))
    intents = _apply_earnings_blackout(cfg, intents)

    if not intents:
        log.info("no trade intents this cycle")
        return

    allow_fractional = bool(cfg.get("execution", "fractional_shares", default=True))
    open_buy_symbols = set() if dry else broker.open_order_symbols(side="buy")

    for intent in intents:
        score = scores.get(intent.symbol)
        if intent.side == "buy" and intent.symbol in open_buy_symbols:
            log.info("[SKIP] BUY %s: open buy order already pending", intent.symbol)
            trade_log.log(TradeLogEntry(
                action="SKIP", symbol=intent.symbol, mode=mode,
                target_dollars=intent.target_dollars, score=score,
                reason=f"buy already pending; {intent.reason}",
            ))
            continue
        price = prices.get(intent.symbol, 0.0)
        qty = RiskManager.intent_to_qty(intent, price, allow_fractional=allow_fractional)
        msg = f"{intent.side.upper()} {intent.symbol} ~${intent.target_dollars:.0f} qty={qty} ({intent.reason})"
        if dry:
            log.info("[DRY] %s", msg)
            trade_log.log(TradeLogEntry(
                action="DRY", symbol=intent.symbol, mode=mode, qty=qty, price=price,
                target_dollars=intent.target_dollars, score=score,
                reason=f"{intent.side} intent; {intent.reason}",
            ))
            continue
        if qty <= 0:
            log.info("[SKIP] %s", msg)
            trade_log.log(TradeLogEntry(
                action="SKIP", symbol=intent.symbol, mode=mode, qty=qty, price=price,
                target_dollars=intent.target_dollars, score=score,
                reason=f"qty<=0; {intent.reason}",
            ))
            continue
        log.info(msg)
        if intent.side == "sell":
            closed = broker.close_position(intent.symbol)
            if closed:
                risk.state.clear_symbol(intent.symbol)
            trade_log.log(TradeLogEntry(
                action="SELL" if closed else "FAIL",
                symbol=intent.symbol, mode=mode, qty=qty, price=price,
                target_dollars=intent.target_dollars, score=score, reason=intent.reason,
            ))
        else:
            order_id = broker.submit_market_order(intent.symbol, qty, "buy")
            trade_log.log(TradeLogEntry(
                action="BUY" if order_id else "FAIL",
                symbol=intent.symbol, mode=mode, qty=qty, price=price,
                target_dollars=intent.target_dollars, score=score, reason=intent.reason,
                order_id=order_id or "",
            ))


def _parse_hhmm(value: str) -> dt_time:
    hour, minute = str(value).split(":", 1)
    return dt_time(hour=int(hour), minute=int(minute))


def _daily_run_times(start: str, end: str, interval_min: int) -> list[dt_time]:
    start_t = _parse_hhmm(start)
    end_t = _parse_hhmm(end)
    base = datetime(2000, 1, 1, start_t.hour, start_t.minute)
    end_dt = datetime(2000, 1, 1, end_t.hour, end_t.minute)
    out: list[dt_time] = []
    current = base
    while current <= end_dt:
        out.append(current.time())
        current += timedelta(minutes=interval_min)
    return out


def _next_scheduled_run(
    now: datetime,
    *,
    start: str = "09:30",
    end: str = "15:30",
    interval_min: int = 60,
    tz_name: str = DEFAULT_MARKET_TZ,
) -> datetime:
    """Return the next weekday scheduled run time in market timezone."""
    market_tz = ZoneInfo(tz_name)
    local_now = now.replace(tzinfo=market_tz) if now.tzinfo is None else now.astimezone(market_tz)
    run_times = _daily_run_times(start, end, interval_min)

    day = local_now.date()
    for offset in range(8):
        candidate_day = day + timedelta(days=offset)
        if candidate_day.weekday() >= 5:
            continue
        for run_time in run_times:
            candidate = datetime.combine(candidate_day, run_time, tzinfo=market_tz)
            if candidate >= local_now:
                return candidate
    raise RuntimeError("could not compute next scheduled run")


def run_loop(cfg: Config) -> None:
    interval_min = int(cfg.get("schedule", "run_interval_minutes", default=60))
    first_run = str(cfg.get("schedule", "first_run_et", default="09:30"))
    last_run = str(cfg.get("schedule", "last_run_et", default="15:30"))
    tz_name = str(cfg.get("schedule", "market_timezone", default=DEFAULT_MARKET_TZ))
    log.info(
        "starting trader loop on %s weekdays from %s to %s every %d minutes",
        tz_name,
        first_run,
        last_run,
        interval_min,
    )
    while True:
        next_run = _next_scheduled_run(
            datetime.now(ZoneInfo(tz_name)),
            start=first_run,
            end=last_run,
            interval_min=interval_min,
            tz_name=tz_name,
        )
        sleep_seconds = max(0.0, (next_run - datetime.now(ZoneInfo(tz_name))).total_seconds())
        log.info("next trade check scheduled for %s", next_run.strftime("%Y-%m-%d %H:%M:%S %Z"))
        time.sleep(sleep_seconds)
        try:
            trade_once(cfg)
        except Exception:
            log.exception("trade cycle failed; continuing")


if __name__ == "__main__":
    cfg = load_config()
    _setup_logging(cfg)
    run_loop(cfg)
