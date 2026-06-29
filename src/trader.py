"""Main trading loop. Run via scripts/run_paper.py."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import yfinance as yf

from src.broker.alpaca_client import AlpacaBroker
from src.config import Config, ROOT, load_config
from src.data.market_data import get_history_many
from src.data.universe import load_universe
from src.politicians.tracker import politician_signals
from src.risk.manager import RiskManager, TradeIntent
from src.signals.classical import classical_signal
from src.signals.hedge_fund import hedge_fund_decision
from src.signals.ml import load_model, ml_signal

log = logging.getLogger(__name__)


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


def compute_signals(cfg: Config, symbols: list[str]) -> dict[str, float]:
    """Composite signal in [-1, 1] per symbol."""
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


def trade_once(cfg: Config) -> None:
    broker = AlpacaBroker(cfg)
    risk = RiskManager(cfg, broker)

    risk.apply_stop_losses()

    if not broker.is_market_open():
        log.info("market closed; skipping")
        return

    symbols = load_universe(cfg)
    log.info("evaluating %d symbols", len(symbols))
    scores = compute_signals(cfg, symbols)
    prices = _last_prices(symbols)
    intents = _consolidate_intents(risk.size_orders(scores, prices))

    if not intents:
        log.info("no trade intents this cycle")
        return

    dry = bool(cfg.get("dry_run", default=False))
    allow_fractional = bool(cfg.get("execution", "fractional_shares", default=True))
    open_buy_symbols = set() if dry else broker.open_order_symbols(side="buy")
    for intent in intents:
        if intent.side == "buy" and intent.symbol in open_buy_symbols:
            log.info("[SKIP] BUY %s: open buy order already pending", intent.symbol)
            continue
        price = prices.get(intent.symbol, 0.0)
        qty = RiskManager.intent_to_qty(intent, price, allow_fractional=allow_fractional)
        msg = f"{intent.side.upper()} {intent.symbol} ~${intent.target_dollars:.0f} qty={qty} ({intent.reason})"
        if dry:
            log.info("[DRY] %s", msg)
            continue
        if qty <= 0:
            log.info("[SKIP] %s", msg)
            continue
        log.info(msg)
        if intent.side == "sell":
            broker.close_position(intent.symbol)
        else:
            broker.submit_market_order(intent.symbol, qty, "buy")


def run_loop(cfg: Config) -> None:
    interval_min = int(cfg.get("schedule", "run_interval_minutes", default=60))
    log.info("starting trader loop every %d minutes", interval_min)
    while True:
        try:
            trade_once(cfg)
        except Exception:
            log.exception("trade cycle failed; continuing")
        time.sleep(interval_min * 60)


if __name__ == "__main__":
    cfg = load_config()
    _setup_logging(cfg)
    run_loop(cfg)
