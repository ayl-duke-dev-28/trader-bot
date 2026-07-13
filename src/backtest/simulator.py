"""Daily historical simulator for the live bot's current decision path."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type, datetime, timedelta
import logging
from math import sqrt
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config import Config
from src.data.sectors import sector_for
from src.risk.indicators import gap_pct, latest_atr_pct
from src.signals.classical import classical_signal
from src.signals.hedge_fund import hedge_fund_decision
from src.signals.ml import build_features, load_model, ml_signal

log = logging.getLogger(__name__)


@dataclass
class SimPosition:
    symbol: str
    qty: float
    avg_entry_price: float

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_plpc(self, price: float) -> float:
        if self.avg_entry_price <= 0:
            return 0.0
        return price / self.avg_entry_price - 1.0


@dataclass
class SimulationResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    summary: dict[str, float | int | str]


def _cfg_r(cfg: Config, *keys, default):
    return cfg.get("risk", *keys, default=default)


def _stop_pct(cfg: Config, hist: pd.DataFrame | None) -> float:
    atr_mult = float(_cfg_r(cfg, "stop_atr_mult", default=2.5))
    stop_min = float(_cfg_r(cfg, "stop_min_pct", default=0.04))
    stop_max = float(_cfg_r(cfg, "stop_max_pct", default=0.12))
    atr_p = latest_atr_pct(hist, window=14) if hist is not None else float("nan")
    if atr_p != atr_p:
        return stop_min
    return max(stop_min, min(stop_max, atr_p * atr_mult))


def _count_by_sector(symbols: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in symbols:
        sector = sector_for(symbol)
        counts[sector] = counts.get(sector, 0) + 1
    return counts


def _regime_adjusted_max_gross_pct(
    cfg: Config,
    history: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    normal_max_gross_pct: float,
) -> float:
    regime_cfg = _cfg_r(cfg, "market_regime", default={}) or {}
    if not bool(regime_cfg.get("enabled", False)):
        return normal_max_gross_pct

    benchmark = str(regime_cfg.get("benchmark_symbol", "QQQ")).upper()
    window = int(regime_cfg.get("sma_window", 200))
    risk_off_max = float(regime_cfg.get("risk_off_max_gross_exposure", 0.0))
    hist = history.get(benchmark)
    if hist is None or hist.empty or "close" not in hist.columns:
        return normal_max_gross_pct

    close = hist.loc[:date]["close"].dropna()
    if len(close) < window:
        return normal_max_gross_pct
    sma = close.rolling(window).mean().iloc[-1]
    if sma != sma or close.iloc[-1] >= sma:
        return normal_max_gross_pct
    return min(normal_max_gross_pct, risk_off_max)


def _benchmark_core_cfg(cfg: Config) -> dict:
    return _cfg_r(cfg, "benchmark_core", default={}) or {}


def _benchmark_core_symbol(cfg: Config) -> str:
    return str(_benchmark_core_cfg(cfg).get("symbol", "QQQ")).upper()


def _benchmark_core_target_pct(
    cfg: Config,
    max_gross_pct: float,
    normal_max_gross_pct: float,
) -> float:
    core_cfg = _benchmark_core_cfg(cfg)
    if not bool(core_cfg.get("enabled", False)):
        return 0.0
    risk_on = max_gross_pct >= normal_max_gross_pct - 1e-9
    key = "risk_on_target_pct" if risk_on else "risk_off_target_pct"
    return max(0.0, min(max_gross_pct, float(core_cfg.get(key, 0.0))))


def _lookback_return_at(
    history: dict[str, pd.DataFrame],
    symbol: str,
    date: pd.Timestamp,
    lookback: int,
) -> float | None:
    hist = history.get(symbol)
    if hist is None or hist.empty or "close" not in hist.columns:
        return None
    close = hist.loc[:date]["close"].dropna()
    if len(close) <= lookback:
        return None
    start = close.iloc[-lookback - 1]
    end = close.iloc[-1]
    if start <= 0 or end <= 0:
        return None
    return float(end / start - 1.0)


def _passes_relative_strength_at(
    cfg: Config,
    history: dict[str, pd.DataFrame],
    symbol: str,
    date: pd.Timestamp,
) -> bool:
    rel_cfg = _cfg_r(cfg, "relative_strength", default={}) or {}
    if not bool(rel_cfg.get("enabled", False)):
        return True
    symbol = symbol.upper()
    benchmark = str(rel_cfg.get("benchmark_symbol", "QQQ")).upper()
    exempt = {str(s).upper() for s in rel_cfg.get("exempt_symbols", [])}
    if symbol == benchmark or symbol in exempt:
        return True
    lookback = int(rel_cfg.get("lookback_days", 63))
    min_excess = float(rel_cfg.get("min_excess_return", 0.0))
    sym_ret = _lookback_return_at(history, symbol, date, lookback)
    bench_ret = _lookback_return_at(history, benchmark, date, lookback)
    if sym_ret is None or bench_ret is None:
        return False
    return sym_ret >= bench_ret + min_excess


def _near_earnings_at(
    earnings_calendar: dict[str, list[date_type | datetime | pd.Timestamp | str]] | None,
    symbol: str,
    current_date: pd.Timestamp,
    within_days: int,
) -> bool:
    if not earnings_calendar or within_days <= 0:
        return False
    current = current_date.date()
    for raw_date in earnings_calendar.get(symbol.upper(), []):
        try:
            earnings_date = pd.Timestamp(raw_date).date()
        except Exception:
            continue
        if current <= earnings_date <= current + timedelta(days=within_days):
            return True
    return False


def _score_snapshot(cfg: Config, history: dict[str, pd.DataFrame], bundle) -> dict[str, float]:
    """Match trader.compute_signals, but avoid reloading the ML bundle every date."""
    use_hedge_fund = bool(cfg.get("strategies", "hedge_fund", "enabled", default=False))
    if use_hedge_fund:
        out: dict[str, float] = {}
        for sym, df in history.items():
            out[sym] = hedge_fund_decision(cfg, df, bundle=bundle).score if not df.empty else 0.0
        return out

    w_cls = float(cfg.get("strategies", "classical", "weight", default=0.4)) if cfg.get("strategies", "classical", "enabled", default=True) else 0.0
    w_ml = float(cfg.get("strategies", "ml", "weight", default=0.4)) if cfg.get("strategies", "ml", "enabled", default=True) else 0.0
    total = max(1e-9, w_cls + w_ml)

    out: dict[str, float] = {}
    for sym, df in history.items():
        if df is None or df.empty:
            out[sym] = 0.0
            continue
        cls = classical_signal(df) if w_cls > 0 else 0.0
        ml = ml_signal(cfg, df, bundle=bundle) if (w_ml > 0 and bundle is not None) else 0.0
        out[sym] = float(np.clip((w_cls * cls + w_ml * ml) / total, -1.0, 1.0))
    return out


def _clip_series(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _precompute_hedge_fund_scores(cfg: Config, history: dict[str, pd.DataFrame], bundle) -> dict[str, pd.Series]:
    """Vectorized equivalent of hedge_fund_decision(...).score for each historical row."""
    min_probability = float(cfg.get("strategies", "ml", "min_probability", default=0.55))
    score_cache: dict[str, pd.Series] = {}

    for sym, df in history.items():
        if df.empty or "close" not in df.columns or "volume" not in df.columns:
            continue
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        n = pd.Series(np.arange(1, len(df) + 1), index=df.index)

        ema8 = close.ewm(span=8, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema55 = close.ewm(span=55, adjust=False).mean()
        spread = (ema8 - ema55) / ema55.replace(0, np.nan)
        alignment = pd.Series(0.0, index=df.index)
        alignment[(ema8 > ema21) & (ema21 > ema55)] = 1.0
        alignment[(ema8 < ema21) & (ema21 < ema55)] = -1.0
        trend = _clip_series(0.55 * alignment + 0.45 * np.tanh(spread * 20))
        trend[n < 60] = 0.0

        ma50 = close.rolling(50).mean()
        std50 = close.rolling(50).std()
        z = (close - ma50) / std50.replace(0, np.nan)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        band = (close - lower) / (upper - lower).replace(0, np.nan)
        rsi14 = _rsi(close, 14).fillna(50.0)
        mean_rev = pd.Series(-np.tanh(z / 2), index=df.index)
        mean_rev += np.where(rsi14 < 35, 0.25, np.where(rsi14 > 65, -0.25, 0.0))
        mean_rev += np.where(band < 0.2, 0.2, np.where(band > 0.8, -0.2, 0.0))
        mean_rev = _clip_series(mean_rev)
        mean_rev[n < 55] = 0.0

        returns = close.pct_change()
        mom_1m = returns.rolling(21).sum()
        mom_3m = returns.rolling(63).sum()
        mom_6m = returns.rolling(126).sum()
        volume_ratio = (volume / volume.rolling(21).mean()).fillna(1.0)
        raw_momentum = 0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m
        momentum = pd.Series(np.tanh(raw_momentum * 5), index=df.index)
        momentum *= np.where(volume_ratio < 0.8, 0.75, np.where(volume_ratio > 1.2, 1.1, 1.0))
        momentum = _clip_series(momentum)
        momentum[n < 130] = 0.0

        daily_vol = returns.rolling(60).std().fillna(0.025)
        annualized_vol = daily_vol * sqrt(252)
        hist_vol = returns.rolling(21).std() * sqrt(252)
        vol_ma = hist_vol.rolling(63).mean()
        regime = (hist_vol / vol_ma.replace(0, np.nan)).fillna(1.0)
        risk_multiplier = pd.Series(
            np.select(
                [annualized_vol < 0.15, annualized_vol < 0.30, annualized_vol < 0.50],
                [1.10, 1.00, 0.80],
                default=0.60,
            ),
            index=df.index,
        )
        volatility = pd.Series(np.where(regime < 0.8, 0.25, np.where(regime > 1.2, -0.35, 0.0)), index=df.index)
        volatility = _clip_series(volatility)
        volatility[n < 85] = 0.0

        skew = returns.rolling(63).skew()
        recent_return = close.pct_change(20)
        volatility20 = returns.rolling(20).std().fillna(0.02)
        normalized_move = recent_return / volatility20.replace(0, np.nan)
        statistical = pd.Series(
            np.where(
                (normalized_move < -8) & (skew > 0.5),
                0.35,
                np.where((normalized_move > 8) & (skew < -0.5), -0.35, 0.0),
            ),
            index=df.index,
        )
        statistical = _clip_series(statistical)
        statistical[n < 90] = 0.0

        ml = pd.Series(0.0, index=df.index)
        if bundle is not None:
            feats = build_features(df)
            valid = feats[bundle["features"]].dropna()
            if not valid.empty:
                proba_up = bundle["model"].predict_proba(valid)[:, 1]
                raw_ml = np.clip((proba_up - 0.5) * 2, -1.0, 1.0)
                raw_ml[np.abs(proba_up - 0.5) < (min_probability - 0.5)] = 0.0
                ml.loc[valid.index] = raw_ml

        votes = {
            "trend": (trend, 0.25),
            "mean_reversion": (mean_rev, 0.20),
            "momentum": (momentum, 0.20),
            "volatility": (volatility, 0.10),
            "statistical": (statistical, 0.10),
            "ml": (ml, 0.15),
        }
        weighted_sum = pd.Series(0.0, index=df.index)
        total_weight = pd.Series(0.0, index=df.index)
        for vote_score, weight in votes.values():
            confidence = vote_score.abs()
            active = confidence >= 0.1
            weighted_sum += vote_score.where(active, 0.0) * weight * confidence.where(active, 0.0)
            total_weight += weight * confidence.where(active, 0.0)
        raw = weighted_sum / total_weight.replace(0, np.nan)
        adjusted = raw.where(raw <= 0, raw * risk_multiplier)
        score_cache[sym] = _clip_series(adjusted)

    return score_cache


def simulate_current_bot(
    cfg: Config,
    history: dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp | None = None,
    start_capital: float = 100_000.0,
    cost_bps: float = 5.0,
    earnings_calendar: dict[str, list[date_type | datetime | pd.Timestamp | str]] | None = None,
    model_bundle: dict[str, Any] | None = None,
    model_provider: Callable[[pd.Timestamp], dict[str, Any] | None] | None = None,
) -> SimulationResult:
    """Replay daily close-to-close decisions using the live bot's signal/risk rules."""
    if not history:
        raise ValueError("history is empty")

    closes = pd.DataFrame({s: df["close"] for s, df in history.items()}).sort_index()
    closes = closes.dropna(how="all").ffill()
    if closes.empty:
        raise ValueError("no price history")

    end_date = end_date or closes.index.max()
    sim_dates = closes.loc[(closes.index >= start_date) & (closes.index <= end_date)].index
    sim_dates = sim_dates[sim_dates.isin(closes.dropna(how="all").index)]
    if len(sim_dates) < 2:
        raise ValueError("not enough dates in requested simulation window")

    bundle = model_bundle
    if bundle is None and model_provider is None and cfg.get("strategies", "ml", "enabled", default=True):
        bundle = load_model(cfg)
    use_precomputed_scores = bool(cfg.get("strategies", "hedge_fund", "enabled", default=False))
    score_cache = (
        _precompute_hedge_fund_scores(cfg, history, bundle)
        if use_precomputed_scores and model_provider is None
        else {}
    )
    if score_cache:
        log.info("precomputed hedge-fund scores for %d symbols", len(score_cache))
    allow_fractional = bool(cfg.get("execution", "fractional_shares", default=True))
    max_pos_pct = float(_cfg_r(cfg, "max_position_pct", default=0.05))
    max_gross_pct = float(_cfg_r(cfg, "max_gross_exposure", default=0.80))
    max_positions = int(_cfg_r(cfg, "max_positions", default=20))
    entry_thr = float(_cfg_r(cfg, "entry_score_threshold", default=0.35))
    exit_thr = float(_cfg_r(cfg, "exit_score_threshold", default=0.0))
    gap_skip_limit = float(_cfg_r(cfg, "gap_skip_pct", default=0.05))
    cooldown_days = int(_cfg_r(cfg, "cooldown_days", default=3))
    trailing_activate = float(_cfg_r(cfg, "trailing_activate_pct", default=0.08))
    trailing_giveback = float(_cfg_r(cfg, "trailing_giveback_pct", default=0.04))
    earnings_blackout_days = int(_cfg_r(cfg, "earnings_blackout_days", default=0))
    sector_caps = dict(_cfg_r(cfg, "sector_caps", default={}) or {})
    signal_history_bars = int(cfg.get("data", "history_days", default=400))

    cash = float(start_capital)
    positions: dict[str, SimPosition] = {}
    cooldown_until: dict[str, pd.Timestamp] = {}
    highwater: dict[str, float] = {}
    trade_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    def price_on(symbol: str, date: pd.Timestamp) -> float | None:
        value = closes.at[date, symbol] if symbol in closes.columns and date in closes.index else np.nan
        if pd.isna(value) or value <= 0:
            return None
        return float(value)

    def equity_on(date: pd.Timestamp) -> float:
        total = cash
        for sym, pos in positions.items():
            price = price_on(sym, date)
            if price is not None:
                total += pos.market_value(price)
        return float(total)

    def log_trade(date: pd.Timestamp, action: str, symbol: str, qty: float, price: float, score: float | None, reason: str) -> None:
        trade_rows.append({
            "date": date.date().isoformat(),
            "action": action,
            "symbol": symbol,
            "qty": round(float(qty), 6),
            "price": round(float(price), 4),
            "notional": round(float(qty) * float(price), 2),
            "score": None if score is None else round(float(score), 4),
            "reason": reason,
            "cash_after": round(cash, 2),
            "equity_after": round(equity_on(date), 2),
        })

    total_dates = len(sim_dates)
    for date_idx, date in enumerate(sim_dates, start=1):
        if date_idx == 1 or date_idx % 50 == 0:
            log.info("simulating %d/%d %s equity=$%.0f", date_idx, total_dates, date.date(), equity_on(date))
        current_equity = equity_on(date)

        # Stop-loss and trailing exits are evaluated before new entries.
        for sym, pos in list(positions.items()):
            price = price_on(sym, date)
            if price is None:
                continue
            hist_today = history[sym].loc[:date] if sym in history else None
            pnl = pos.unrealized_plpc(price)
            highwater[sym] = max(highwater.get(sym, 0.0), pnl)
            stop = _stop_pct(cfg, hist_today)
            reason = ""
            if highwater[sym] >= trailing_activate and pnl <= highwater[sym] - trailing_giveback:
                reason = f"trailing lock hw={highwater[sym]:.2%} now={pnl:.2%}"
            elif hist_today is not None and not hist_today.empty:
                gap = gap_pct(hist_today)
                if not (abs(gap) >= gap_skip_limit and pnl > -stop * 1.5) and pnl <= -stop:
                    reason = f"stop pl={pnl:.2%} vs -{stop:.2%}"
            elif pnl <= -stop:
                reason = f"stop pl={pnl:.2%} vs -{stop:.2%}"

            if reason:
                proceeds = pos.qty * price * (1.0 - cost_bps / 10_000.0)
                cash += proceeds
                cooldown_until[sym] = date + timedelta(days=cooldown_days)
                positions.pop(sym, None)
                highwater.pop(sym, None)
                log_trade(date, "STOP", sym, pos.qty, price, None, reason)

        if score_cache:
            scores = {}
            for sym, series in score_cache.items():
                prior = series.loc[series.index < date]
                if not prior.empty:
                    scores[sym] = float(prior.iloc[-1])
        else:
            active_bundle = model_provider(date) if model_provider is not None else bundle
            prior_history = {
                sym: df.loc[:date].iloc[:-1].tail(signal_history_bars)
                for sym, df in history.items()
                if not df.loc[:date].iloc[:-1].empty
            }
            scores = _score_snapshot(cfg, prior_history, active_bundle)

        adjusted_max_gross_pct = _regime_adjusted_max_gross_pct(cfg, history, date, max_gross_pct)
        core_symbol = _benchmark_core_symbol(cfg)
        core_target_pct = _benchmark_core_target_pct(cfg, adjusted_max_gross_pct, max_gross_pct)
        if core_target_pct <= 0 and core_symbol in positions:
            price = price_on(core_symbol, date)
            if price is not None:
                pos = positions.pop(core_symbol)
                highwater.pop(core_symbol, None)
                cash += pos.qty * price * (1.0 - cost_bps / 10_000.0)
                log_trade(date, "SELL", core_symbol, pos.qty, price, scores.get(core_symbol), "benchmark core risk-off target=0")

        # Score exits, matching the live hysteresis rule.
        for sym, pos in list(positions.items()):
            score = float(scores.get(sym, 0.0))
            if score <= exit_thr:
                price = price_on(sym, date)
                if price is None:
                    continue
                cash += pos.qty * price * (1.0 - cost_bps / 10_000.0)
                positions.pop(sym, None)
                highwater.pop(sym, None)
                log_trade(date, "SELL", sym, pos.qty, price, score, f"score={score:+.2f} <= exit_thr={exit_thr:+.2f}")

        current_equity = equity_on(date)
        held_value = 0.0
        for sym, pos in positions.items():
            price = price_on(sym, date)
            if price is not None:
                held_value += pos.market_value(price)
        max_per_position = current_equity * max_pos_pct
        remaining_gross = max(0.0, current_equity * adjusted_max_gross_pct - held_value)
        open_slots = max(0, max_positions - len(positions))
        sector_used = _count_by_sector(list(positions))

        core_cfg = _benchmark_core_cfg(cfg)
        core_target_pct = _benchmark_core_target_pct(cfg, adjusted_max_gross_pct, max_gross_pct)
        core_price = price_on(core_symbol, date)
        core_blackout = _near_earnings_at(earnings_calendar, core_symbol, date, earnings_blackout_days)
        if core_target_pct > 0 and core_price is not None and not core_blackout and (core_symbol in positions or open_slots > 0):
            existing = positions[core_symbol].market_value(core_price) if core_symbol in positions else 0.0
            target = min(current_equity * core_target_pct, current_equity * adjusted_max_gross_pct)
            delta = min(max(0.0, target - existing), remaining_gross)
            min_trade = float(core_cfg.get("min_trade_dollars", 100.0))
            if delta >= max(min_trade, core_price):
                spendable = min(delta, cash / (1.0 + cost_bps / 10_000.0))
                qty = spendable / core_price
                if not allow_fractional:
                    qty = float(np.floor(qty))
                else:
                    qty = round(qty, 4)
                if qty > 0:
                    cost = qty * core_price * (1.0 + cost_bps / 10_000.0)
                    if cost <= cash + 1e-6:
                        old_qty = positions[core_symbol].qty if core_symbol in positions else 0.0
                        old_cost = old_qty * positions[core_symbol].avg_entry_price if core_symbol in positions else 0.0
                        new_qty = old_qty + qty
                        positions[core_symbol] = SimPosition(core_symbol, new_qty, (old_cost + qty * core_price) / new_qty)
                        cash -= cost
                        remaining_gross -= qty * core_price
                        if old_qty == 0:
                            open_slots -= 1
                            sector = sector_for(core_symbol)
                            sector_used[sector] = sector_used.get(sector, 0) + 1
                        log_trade(date, "BUY", core_symbol, qty, core_price, scores.get(core_symbol), f"benchmark core target={core_target_pct:.0%}")

        candidates = sorted(
            ((sym, float(score)) for sym, score in scores.items() if float(score) >= entry_thr),
            key=lambda item: -item[1],
        )
        for sym, score in candidates:
            if open_slots <= 0 or remaining_gross <= 100.0:
                break
            if cooldown_until.get(sym) is not None and date < cooldown_until[sym]:
                continue
            price = price_on(sym, date)
            if price is None:
                continue
            hist_today = history[sym].loc[:date]
            if not _passes_relative_strength_at(cfg, history, sym, date):
                continue
            if _near_earnings_at(earnings_calendar, sym, date, earnings_blackout_days):
                continue
            if not hist_today.empty and abs(gap_pct(hist_today)) >= gap_skip_limit:
                continue
            sector = sector_for(sym)
            cap = int(sector_caps.get(sector, sector_caps.get("other", 4)))
            if sector_used.get(sector, 0) >= cap:
                continue

            strength = max(0.5, min(1.0, score))
            target = min(max_per_position * strength, remaining_gross)
            existing = positions[sym].market_value(price) if sym in positions else 0.0
            delta = target - existing
            if delta < max(100.0, max_per_position * 0.1) or delta < price:
                continue
            spendable = min(delta, cash / (1.0 + cost_bps / 10_000.0))
            qty = spendable / price
            if not allow_fractional:
                qty = float(np.floor(qty))
            else:
                qty = round(qty, 4)
            if qty <= 0:
                continue

            cost = qty * price * (1.0 + cost_bps / 10_000.0)
            if cost > cash + 1e-6:
                continue
            old_qty = positions[sym].qty if sym in positions else 0.0
            old_cost = old_qty * positions[sym].avg_entry_price if sym in positions else 0.0
            new_qty = old_qty + qty
            new_avg = (old_cost + qty * price) / new_qty
            positions[sym] = SimPosition(sym, new_qty, new_avg)
            cash -= cost
            remaining_gross -= qty * price
            if old_qty == 0:
                open_slots -= 1
                sector_used[sector] = sector_used.get(sector, 0) + 1
            log_trade(date, "BUY", sym, qty, price, score, f"score={score:+.2f} sector={sector}")

        curve_rows.append({
            "date": date.date().isoformat(),
            "equity": round(equity_on(date), 2),
            "cash": round(cash, 2),
            "positions": len(positions),
        })

    equity_curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    equity_series = equity_curve.set_index(pd.to_datetime(equity_curve["date"]))["equity"]
    rets = equity_series.pct_change().dropna()
    loss_days = rets[rets < 0]
    profit_days = rets[rets > 0]
    total_return = equity_series.iloc[-1] / start_capital - 1.0
    years = max(1e-9, (equity_series.index[-1] - equity_series.index[0]).days / 365.25)
    rolling_max = equity_series.cummax()
    drawdown = equity_series / rolling_max - 1.0
    wins = 0
    losses = 0
    if not trades.empty:
        exits = trades[trades["action"].isin(["SELL", "STOP"])]
        buys_by_symbol: dict[str, list[float]] = {}
        for row in trades.to_dict("records"):
            if row["action"] == "BUY":
                buys_by_symbol.setdefault(str(row["symbol"]), []).append(float(row["price"]))
            elif row["action"] in {"SELL", "STOP"} and buys_by_symbol.get(str(row["symbol"])):
                entry = buys_by_symbol[str(row["symbol"])].pop(0)
                if float(row["price"]) > entry:
                    wins += 1
                else:
                    losses += 1
    else:
        exits = pd.DataFrame()

    summary = {
        "start_date": equity_curve["date"].iloc[0],
        "end_date": equity_curve["date"].iloc[-1],
        "start_capital": float(start_capital),
        "final_equity": float(equity_series.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float((1.0 + total_return) ** (1.0 / years) - 1.0),
        "sharpe": float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "profit_days": int(len(profit_days)),
        "loss_days": int(len(loss_days)),
        "flat_days": int((rets == 0).sum()),
        "loss_day_rate": float(len(loss_days) / len(rets)) if len(rets) > 0 else 0.0,
        "avg_loss_day_return": float(loss_days.mean()) if len(loss_days) > 0 else 0.0,
        "worst_day_return": float(rets.min()) if len(rets) > 0 else 0.0,
        "trades": int(len(trades)),
        "buys": int((trades["action"] == "BUY").sum()) if not trades.empty else 0,
        "sells": int((trades["action"] == "SELL").sum()) if not trades.empty else 0,
        "stops": int((trades["action"] == "STOP").sum()) if not trades.empty else 0,
        "closed_win_rate": float(wins / max(1, wins + losses)),
        "open_positions": int(len(positions)),
        "symbols": int(len(history)),
        "cost_bps": float(cost_bps),
    }
    return SimulationResult(equity_curve=equity_curve, trades=trades, summary=summary)


def write_simulation_report(result: SimulationResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)
    result.trades.to_csv(out_dir / "trades.csv", index=False)
    lines = []
    for key, value in result.summary.items():
        if isinstance(value, float):
            if key in {
                "total_return",
                "cagr",
                "max_drawdown",
                "closed_win_rate",
                "loss_day_rate",
                "avg_loss_day_return",
                "worst_day_return",
            }:
                lines.append(f"{key}: {value:.2%}")
            else:
                lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
