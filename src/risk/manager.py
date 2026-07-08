"""Position sizing and pre-trade risk checks.

v2 fixes several profit leaks from v1:
  * ATR-based stops with min/max clamp instead of a flat 5%.
  * Trailing highwater lock so winners give back only a bounded amount.
  * Cooldown after a stop-out to kill the "stop then re-buy same cycle" loop.
  * Sector concentration caps so a semi-wide selloff can't hit 15 positions.
  * Entry/exit hysteresis (enter high, exit at zero).
  * Day-start equity persisted across process restarts so the kill switch
    can't be reset by a mid-day restart.
  * dry_run is honored inside stop-loss handling.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.broker.alpaca_client import AlpacaBroker, Position
from src.config import Config, ROOT
from src.data.sectors import sector_for
from src.risk.indicators import gap_pct, latest_atr_pct
from src.risk.state import RiskState
from src.trade_log import TradeLogEntry, TradeLogger

log = logging.getLogger(__name__)

MIN_TRADE_DOLLARS = 100.0


@dataclass(frozen=True)
class TradeIntent:
    symbol: str
    side: str           # 'buy' | 'sell'
    target_dollars: float
    reason: str


def _default_state_path(cfg: Config) -> Path:
    cache_dir = Path(cfg.get("data", "cache_dir", default="data_cache"))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    return cache_dir / "state" / "risk_state.json"


class RiskManager:
    def __init__(
        self,
        cfg: Config,
        broker: AlpacaBroker,
        state: RiskState | None = None,
        trade_log: TradeLogger | None = None,
    ):
        self.cfg = cfg
        self.broker = broker
        self.state = state or RiskState(_default_state_path(cfg))
        self.trade_log = trade_log

    def _r(self, *keys, default):
        return self.cfg.get("risk", *keys, default=default)

    # --- kill switch ------------------------------------------------------

    def kill_switch_tripped(self) -> bool:
        acct = self.broker.account()
        start_equity = self.state.day_start_equity(acct.equity)
        threshold = float(self._r("daily_loss_kill_switch_pct", default=0.03))
        if start_equity <= 0:
            return False
        drop = (start_equity - acct.equity) / start_equity
        if drop >= threshold:
            log.warning(
                "KILL SWITCH: equity drop %.2f%% >= %.2f%% (start=%.2f now=%.2f)",
                drop * 100, threshold * 100, start_equity, acct.equity,
            )
            return True
        return False

    # --- stops ------------------------------------------------------------

    def apply_stop_losses(
        self,
        history: dict[str, pd.DataFrame] | None = None,
        dry_run: bool = False,
        mode: str = "paper",
    ) -> None:
        history = history or {}
        stop_min = float(self._r("stop_min_pct", default=0.04))
        stop_max = float(self._r("stop_max_pct", default=0.12))
        atr_mult = float(self._r("stop_atr_mult", default=2.5))
        trailing_activate = float(self._r("trailing_activate_pct", default=0.08))
        trailing_giveback = float(self._r("trailing_giveback_pct", default=0.04))
        cooldown_days = int(self._r("cooldown_days", default=3))
        gap_skip = float(self._r("gap_skip_pct", default=0.05))

        for p in self.broker.positions():
            hist = history.get(p.symbol)
            stop_pct = self._compute_stop_pct(hist, atr_mult, stop_min, stop_max)
            highwater = self.state.update_highwater(p.symbol, p.unrealized_plpc)

            # Trailing lock takes priority once armed.
            if highwater >= trailing_activate:
                trailing_floor = highwater - trailing_giveback
                if p.unrealized_plpc <= trailing_floor:
                    self._close(
                        p, dry_run, cooldown_days, mode=mode,
                        reason=f"trailing lock hw={highwater:.2%} now={p.unrealized_plpc:.2%}",
                    )
                    continue

            # Gap protection: if the name gapped hard today, skip the stop
            # this cycle. Reopen fills on gap-downs are the worst price of
            # the day; let one bar of price discovery happen first.
            if hist is not None:
                gap = gap_pct(hist)
                if abs(gap) >= gap_skip and p.unrealized_plpc > -stop_pct * 1.5:
                    log.info("gap-skip stop for %s (gap=%.2f%%)", p.symbol, gap * 100)
                    continue

            if p.unrealized_plpc <= -stop_pct:
                self._close(
                    p, dry_run, cooldown_days, mode=mode,
                    reason=f"stop pl={p.unrealized_plpc:.2%} vs -{stop_pct:.2%}",
                )

    @staticmethod
    def _compute_stop_pct(
        hist: pd.DataFrame | None,
        atr_mult: float,
        stop_min: float,
        stop_max: float,
    ) -> float:
        atr_p = latest_atr_pct(hist, window=14) if hist is not None else float("nan")
        if atr_p != atr_p:  # NaN
            return stop_min
        return max(stop_min, min(stop_max, atr_p * atr_mult))

    def _close(
        self,
        p: Position,
        dry_run: bool,
        cooldown_days: int,
        reason: str,
        mode: str = "paper",
    ) -> None:
        log.warning("closing %s: %s", p.symbol, reason)
        if dry_run:
            log.info("[DRY] would close %s", p.symbol)
            if self.trade_log is not None:
                self.trade_log.log(TradeLogEntry(
                    action="DRY", symbol=p.symbol, mode=mode, qty=p.qty,
                    target_dollars=p.market_value, reason=f"stop-loss close; {reason}",
                ))
            return
        closed = self.broker.close_position(p.symbol)
        if closed:
            self.state.record_stop(p.symbol, cooldown_days)
        if self.trade_log is not None:
            self.trade_log.log(TradeLogEntry(
                action="STOP" if closed else "FAIL",
                symbol=p.symbol, mode=mode, qty=p.qty,
                target_dollars=p.market_value, reason=reason,
            ))

    # --- sizing -----------------------------------------------------------

    def size_orders(
        self,
        scores: dict[str, float],
        prices: dict[str, float],
        history: dict[str, pd.DataFrame] | None = None,
    ) -> list[TradeIntent]:
        history = history or {}
        acct = self.broker.account()
        equity = acct.equity
        max_pos_pct = float(self._r("max_position_pct", default=0.05))
        normal_max_gross_pct = float(self._r("max_gross_exposure", default=0.80))
        max_gross_pct = self._regime_adjusted_max_gross_pct(history, normal_max_gross_pct)
        max_positions = int(self._r("max_positions", default=20))
        entry_thr = float(self._r("entry_score_threshold", default=0.35))
        exit_thr = float(self._r("exit_score_threshold", default=0.0))
        gap_skip = float(self._r("gap_skip_pct", default=0.05))
        sector_caps = dict(self._r("sector_caps", default={}) or {})

        held = {p.symbol: p for p in self.broker.positions()}
        held_value = sum(p.market_value for p in held.values())
        max_per_position_dollars = equity * max_pos_pct
        remaining_gross = max(0.0, equity * max_gross_pct - held_value)

        intents: list[TradeIntent] = []

        # 1) Sells: hysteresis — only exit when score has drifted to zero.
        for sym, p in held.items():
            score = scores.get(sym, 0.0)
            if score <= exit_thr:
                intents.append(
                    TradeIntent(
                        sym, "sell", p.market_value,
                        f"score={score:+.2f} <= exit_thr={exit_thr:+.2f}",
                    )
                )

        self._apply_benchmark_core_sells(
            intents=intents,
            held=held,
            max_gross_pct=max_gross_pct,
            normal_max_gross_pct=normal_max_gross_pct,
        )

        if self.kill_switch_tripped():
            log.warning("kill switch tripped; no new buys this cycle")
            return intents

        symbols_to_sell = {i.symbol for i in intents if i.side == "sell"}
        held_active = {
            s: p for s, p in held.items()
            if p.market_value > 0 and s not in symbols_to_sell
        }
        sector_used = _count_by_sector(held_active.keys())
        open_slots = max(0, max_positions - len(held_active))

        core_intent = self._benchmark_core_buy(
            held_active=held_active,
            prices=prices,
            equity=equity,
            remaining_gross=remaining_gross,
            max_gross_pct=max_gross_pct,
            normal_max_gross_pct=normal_max_gross_pct,
            open_slots=open_slots,
        )
        if core_intent is not None:
            intents.append(core_intent)
            remaining_gross -= core_intent.target_dollars
            if core_intent.symbol not in held_active:
                open_slots -= 1

        candidates = sorted(
            ((s, sc) for s, sc in scores.items() if sc >= entry_thr),
            key=lambda x: -x[1],
        )

        for sym, score in candidates:
            if open_slots <= 0 or remaining_gross <= MIN_TRADE_DOLLARS:
                break
            if self.state.in_cooldown(sym):
                log.info("[SKIP] %s in cooldown", sym)
                continue
            price = prices.get(sym, 0.0)
            if price <= 0:
                continue

            if not self._passes_relative_strength(sym, history):
                continue

            hist = history.get(sym)
            if hist is not None:
                gap = gap_pct(hist)
                if abs(gap) >= gap_skip:
                    log.info("[SKIP] %s gap %.2f%% >= %.2f%%", sym, gap * 100, gap_skip * 100)
                    continue

            sector = sector_for(sym)
            cap = int(sector_caps.get(sector, sector_caps.get("other", 4)))
            if sector_used.get(sector, 0) >= cap:
                log.info("[SKIP] %s sector %s at cap %d", sym, sector, cap)
                continue

            # Size: score-scaled with a 50% floor so eligible names get real capital.
            strength = max(0.5, min(1.0, score))
            target = min(max_per_position_dollars * strength, remaining_gross)
            existing = held[sym].market_value if sym in held else 0.0
            delta = target - existing
            if delta < max(MIN_TRADE_DOLLARS, max_per_position_dollars * 0.1):
                continue
            if delta < price:
                # Whole-share sizing would round to zero.
                log.info("[SKIP] %s delta $%.0f < price $%.2f", sym, delta, price)
                continue

            intents.append(
                TradeIntent(sym, "buy", delta, f"score={score:+.2f} sector={sector}")
            )
            remaining_gross -= delta
            open_slots -= 1
            sector_used[sector] = sector_used.get(sector, 0) + 1

        return intents

    def _benchmark_core_cfg(self) -> dict:
        return self._r("benchmark_core", default={}) or {}

    def _benchmark_core_symbol(self) -> str:
        return str(self._benchmark_core_cfg().get("symbol", "QQQ")).upper()

    def _benchmark_core_target_pct(
        self,
        max_gross_pct: float,
        normal_max_gross_pct: float,
    ) -> float:
        core_cfg = self._benchmark_core_cfg()
        if not bool(core_cfg.get("enabled", False)):
            return 0.0
        risk_on = max_gross_pct >= normal_max_gross_pct - 1e-9
        key = "risk_on_target_pct" if risk_on else "risk_off_target_pct"
        return max(0.0, min(max_gross_pct, float(core_cfg.get(key, 0.0))))

    def _apply_benchmark_core_sells(
        self,
        intents: list[TradeIntent],
        held: dict[str, Position],
        max_gross_pct: float,
        normal_max_gross_pct: float,
    ) -> None:
        core_cfg = self._benchmark_core_cfg()
        if not bool(core_cfg.get("enabled", False)):
            return
        symbol = self._benchmark_core_symbol()
        if symbol not in held:
            return
        target_pct = self._benchmark_core_target_pct(max_gross_pct, normal_max_gross_pct)
        if target_pct > 0:
            return
        if any(i.symbol == symbol and i.side == "sell" for i in intents):
            return
        intents.append(TradeIntent(symbol, "sell", held[symbol].market_value, "benchmark core risk-off target=0"))

    def _benchmark_core_buy(
        self,
        held_active: dict[str, Position],
        prices: dict[str, float],
        equity: float,
        remaining_gross: float,
        max_gross_pct: float,
        normal_max_gross_pct: float,
        open_slots: int,
    ) -> TradeIntent | None:
        core_cfg = self._benchmark_core_cfg()
        if not bool(core_cfg.get("enabled", False)):
            return None

        symbol = self._benchmark_core_symbol()
        target_pct = self._benchmark_core_target_pct(max_gross_pct, normal_max_gross_pct)
        if target_pct <= 0:
            return None
        if symbol not in held_active and open_slots <= 0:
            return None

        price = prices.get(symbol, 0.0)
        if price <= 0:
            return None
        existing = held_active[symbol].market_value if symbol in held_active else 0.0
        target = min(equity * target_pct, equity * max_gross_pct)
        delta = min(max(0.0, target - existing), remaining_gross)
        min_trade = float(core_cfg.get("min_trade_dollars", MIN_TRADE_DOLLARS))
        if delta < max(min_trade, price):
            return None
        return TradeIntent(symbol, "buy", delta, f"benchmark core target={target_pct:.0%}")

    def _passes_relative_strength(
        self,
        symbol: str,
        history: dict[str, pd.DataFrame],
    ) -> bool:
        rel_cfg = self._r("relative_strength", default={}) or {}
        if not bool(rel_cfg.get("enabled", False)):
            return True
        symbol = symbol.upper()
        benchmark = str(rel_cfg.get("benchmark_symbol", "QQQ")).upper()
        exempt = {str(s).upper() for s in rel_cfg.get("exempt_symbols", [])}
        if symbol == benchmark or symbol in exempt:
            return True

        lookback = int(rel_cfg.get("lookback_days", 63))
        min_excess = float(rel_cfg.get("min_excess_return", 0.0))
        hist = history.get(symbol)
        bench_hist = history.get(benchmark)
        if hist is None or bench_hist is None:
            log.info("[SKIP] %s missing relative-strength history vs %s", symbol, benchmark)
            return False
        sym_ret = _lookback_return(hist, lookback)
        bench_ret = _lookback_return(bench_hist, lookback)
        if sym_ret is None or bench_ret is None:
            log.info("[SKIP] %s insufficient relative-strength bars vs %s", symbol, benchmark)
            return False
        if sym_ret < bench_ret + min_excess:
            log.info(
                "[SKIP] %s relative strength %.2f%% < %s %.2f%% + %.2f%%",
                symbol, sym_ret * 100, benchmark, bench_ret * 100, min_excess * 100,
            )
            return False
        return True

    def _regime_adjusted_max_gross_pct(
        self,
        history: dict[str, pd.DataFrame],
        normal_max_gross_pct: float,
    ) -> float:
        regime_cfg = self._r("market_regime", default={}) or {}
        if not bool(regime_cfg.get("enabled", False)):
            return normal_max_gross_pct

        benchmark = str(regime_cfg.get("benchmark_symbol", "QQQ")).upper()
        window = int(regime_cfg.get("sma_window", 200))
        risk_off_max = float(regime_cfg.get("risk_off_max_gross_exposure", 0.0))
        hist = history.get(benchmark)
        if hist is None or hist.empty or "close" not in hist.columns or len(hist) < window:
            log.warning("market regime filter skipped: missing %d bars for %s", window, benchmark)
            return normal_max_gross_pct

        close = hist["close"].dropna()
        sma = close.rolling(window).mean().iloc[-1]
        last = close.iloc[-1]
        if sma != sma or last >= sma:
            return normal_max_gross_pct

        adjusted = min(normal_max_gross_pct, risk_off_max)
        log.info(
            "risk-off regime: %s close %.2f < SMA%d %.2f; max gross %.0f%% -> %.0f%%",
            benchmark, last, window, sma, normal_max_gross_pct * 100, adjusted * 100,
        )
        return adjusted

    # --- utils ------------------------------------------------------------

    @staticmethod
    def intent_to_qty(intent: TradeIntent, price: float, allow_fractional: bool = True) -> float:
        if price <= 0:
            return 0.0
        qty = intent.target_dollars / price
        if not allow_fractional:
            return float(floor(qty))
        return round(qty, 4)


def _count_by_sector(symbols: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in symbols:
        sec = sector_for(s)
        counts[sec] = counts.get(sec, 0) + 1
    return counts


def _lookback_return(hist: pd.DataFrame, lookback: int) -> float | None:
    if hist is None or hist.empty or "close" not in hist.columns:
        return None
    close = hist["close"].dropna()
    if len(close) <= lookback:
        return None
    start = close.iloc[-lookback - 1]
    end = close.iloc[-1]
    if start <= 0 or end <= 0:
        return None
    return float(end / start - 1.0)
