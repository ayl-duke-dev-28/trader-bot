"""AI-hedge-fund-style signal ensemble.

This adapts the architecture pattern from virattt/ai-hedge-fund without adding
LLM/API dependencies: several deterministic "analyst" votes are combined into a
single buy/sell score, then a volatility overlay tempers risky buy signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.signals.ml import ml_signal


@dataclass(frozen=True)
class AnalystVote:
    name: str
    signal: str
    score: float
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class HedgeFundSignal:
    score: float
    signal: str
    confidence: float
    votes: list[AnalystVote]
    risk_multiplier: float
    reasoning: str


def _safe(value: float, default: float = 0.0) -> float:
    try:
        if value != value or np.isinf(value):
            return default
        return float(value)
    except Exception:
        return default


def _clip(value: float) -> float:
    return float(np.clip(_safe(value), -1.0, 1.0))


def _label(score: float) -> str:
    if score > 0.2:
        return "bullish"
    if score < -0.2:
        return "bearish"
    return "neutral"


def _vote(name: str, score: float, reasoning: str) -> AnalystVote:
    score = _clip(score)
    return AnalystVote(
        name=name,
        signal=_label(score),
        score=score,
        confidence=round(abs(score) * 100, 1),
        reasoning=reasoning,
    )


def _latest(series: pd.Series, default: float = 0.0) -> float:
    if series.empty:
        return default
    return _safe(series.iloc[-1], default=default)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _trend_vote(df: pd.DataFrame) -> AnalystVote:
    close = df["close"]
    if len(close) < 60:
        return _vote("trend", 0.0, "not enough data for EMA trend")

    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    e8, e21, e55 = _latest(ema8), _latest(ema21), _latest(ema55)

    spread = (e8 - e55) / e55 if e55 else 0.0
    alignment = 1.0 if e8 > e21 > e55 else (-1.0 if e8 < e21 < e55 else 0.0)
    score = 0.55 * alignment + 0.45 * np.tanh(spread * 20)
    return _vote("trend", score, f"EMA alignment={alignment:+.0f}, 8/55 spread={spread:.2%}")


def _mean_reversion_vote(df: pd.DataFrame) -> AnalystVote:
    close = df["close"]
    if len(close) < 55:
        return _vote("mean_reversion", 0.0, "not enough data for z-score/Bollinger")

    ma50 = close.rolling(50).mean()
    std50 = close.rolling(50).std()
    z = (close - ma50) / std50.replace(0, np.nan)

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    band_width = (upper - lower).replace(0, np.nan)
    price_vs_band = (close - lower) / band_width

    rsi14 = _latest(_rsi(close, 14), default=50.0)
    z_now = _latest(z)
    band_now = _latest(price_vs_band, default=0.5)

    score = -np.tanh(z_now / 2)
    if rsi14 < 35:
        score += 0.25
    elif rsi14 > 65:
        score -= 0.25
    if band_now < 0.2:
        score += 0.2
    elif band_now > 0.8:
        score -= 0.2

    return _vote(
        "mean_reversion",
        score,
        f"z={z_now:.2f}, band={band_now:.2f}, rsi14={rsi14:.1f}",
    )


def _momentum_vote(df: pd.DataFrame) -> AnalystVote:
    close = df["close"]
    volume = df["volume"]
    if len(close) < 130:
        return _vote("momentum", 0.0, "not enough data for 1/3/6 month momentum")

    returns = close.pct_change()
    mom_1m = _latest(returns.rolling(21).sum())
    mom_3m = _latest(returns.rolling(63).sum())
    mom_6m = _latest(returns.rolling(126).sum())
    volume_ratio = _latest(volume / volume.rolling(21).mean(), default=1.0)

    raw_momentum = 0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m
    score = np.tanh(raw_momentum * 5)
    if volume_ratio < 0.8:
        score *= 0.75
    elif volume_ratio > 1.2:
        score *= 1.1

    return _vote(
        "momentum",
        score,
        f"mom1m={mom_1m:.2%}, mom3m={mom_3m:.2%}, mom6m={mom_6m:.2%}, vol_ratio={volume_ratio:.2f}",
    )


def _volatility_vote(df: pd.DataFrame) -> tuple[AnalystVote, float, float]:
    close = df["close"]
    if len(close) < 85:
        return _vote("volatility", 0.0, "not enough data for volatility regime"), 1.0, 0.25

    returns = close.pct_change().dropna()
    recent = returns.tail(min(60, len(returns)))
    daily_vol = _safe(recent.std(), default=0.025)
    annualized_vol = daily_vol * sqrt(252)

    hist_vol = returns.rolling(21).std() * sqrt(252)
    vol_ma = hist_vol.rolling(63).mean()
    regime = _latest(hist_vol / vol_ma.replace(0, np.nan), default=1.0)

    if annualized_vol < 0.15:
        risk_multiplier = 1.10
    elif annualized_vol < 0.30:
        risk_multiplier = 1.00
    elif annualized_vol < 0.50:
        risk_multiplier = 0.80
    else:
        risk_multiplier = 0.60

    if regime < 0.8:
        score = 0.25
    elif regime > 1.2:
        score = -0.35
    else:
        score = 0.0

    vote = _vote(
        "volatility",
        score,
        f"ann_vol={annualized_vol:.1%}, regime={regime:.2f}, risk_mult={risk_multiplier:.2f}",
    )
    return vote, risk_multiplier, annualized_vol


def _statistical_vote(df: pd.DataFrame) -> AnalystVote:
    close = df["close"]
    if len(close) < 90:
        return _vote("statistical", 0.0, "not enough data for skew/reversion")

    returns = close.pct_change().dropna()
    skew = _latest(returns.rolling(63).skew())
    recent_return = _latest(close.pct_change(20))
    volatility = _latest(returns.rolling(20).std(), default=0.02)
    normalized_move = recent_return / volatility if volatility else 0.0

    # Contrarian statistical behavior: stretched upside with bad skew is fragile;
    # stretched downside with positive skew may be a rebound candidate.
    if normalized_move < -8 and skew > 0.5:
        score = 0.35
    elif normalized_move > 8 and skew < -0.5:
        score = -0.35
    else:
        score = 0.0

    return _vote(
        "statistical",
        score,
        f"20d_move={recent_return:.2%}, norm_move={normalized_move:.2f}, skew63={skew:.2f}",
    )


def _ml_vote(cfg: Config, df: pd.DataFrame, bundle: Any | None) -> AnalystVote:
    score = ml_signal(cfg, df, bundle=bundle)
    if score == 0.0:
        return _vote("ml", 0.0, "ML neutral or below confidence threshold")
    direction = "up" if score > 0 else "down"
    return _vote("ml", score, f"model predicts next-day {direction}, score={score:+.2f}")


def hedge_fund_decision(cfg: Config, df: pd.DataFrame, bundle: Any | None = None) -> HedgeFundSignal:
    """Return a multi-analyst ensemble decision in [-1, 1]."""
    if df is None or df.empty or "close" not in df.columns or "volume" not in df.columns:
        return HedgeFundSignal(0.0, "neutral", 0.0, [], 1.0, "missing OHLCV history")

    votes: list[AnalystVote] = [
        _trend_vote(df),
        _mean_reversion_vote(df),
        _momentum_vote(df),
        _volatility_vote(df)[0],
        _statistical_vote(df),
        _ml_vote(cfg, df, bundle),
    ]

    _, risk_multiplier, annualized_vol = _volatility_vote(df)
    weights = {
        "trend": 0.25,
        "mean_reversion": 0.20,
        "momentum": 0.20,
        "volatility": 0.10,
        "statistical": 0.10,
        "ml": 0.15,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    for vote in votes:
        weight = weights.get(vote.name, 0.0)
        confidence = max(vote.confidence / 100.0, 0.20)
        weighted_sum += vote.score * weight * confidence
        total_weight += weight * confidence

    raw_score = weighted_sum / total_weight if total_weight else 0.0
    adjusted_score = raw_score * risk_multiplier if raw_score > 0 else raw_score
    adjusted_score = _clip(adjusted_score)
    signal = _label(adjusted_score)
    confidence = round(abs(adjusted_score) * 100, 1)
    vote_summary = ", ".join(f"{v.name}:{v.signal}/{v.confidence:.0f}" for v in votes)

    return HedgeFundSignal(
        score=adjusted_score,
        signal=signal,
        confidence=confidence,
        votes=votes,
        risk_multiplier=risk_multiplier,
        reasoning=f"{signal} score={adjusted_score:+.2f}, ann_vol={annualized_vol:.1%}, votes=[{vote_summary}]",
    )
