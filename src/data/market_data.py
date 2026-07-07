"""Historical bar fetching with a simple parquet cache to be friendly to free APIs."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import Config, ROOT

log = logging.getLogger(__name__)


def _cache_path(cfg: Config, symbol: str) -> Path:
    cache_dir = Path(cfg.get("data", "cache_dir", default="data_cache"))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{symbol}.parquet"


def get_history(cfg: Config, symbol: str, days: int | None = None) -> pd.DataFrame:
    """Return daily OHLCV for symbol. Re-uses cache if fresh (today already fetched)."""
    days = days or int(cfg.get("data", "history_days", default=400))
    path = _cache_path(cfg, symbol)
    today = datetime.utcnow().date()
    required_start = today - timedelta(days=days)

    if path.exists():
        try:
            df = pd.read_parquet(path)
            if (
                not df.empty
                and df.index.max().date() >= today - timedelta(days=1)
                and df.index.min().date() <= required_start
            ):
                return df[df.index.date >= required_start]
        except Exception as e:
            log.warning("cache read failed for %s: %s", symbol, e)

    start = (datetime.utcnow() - timedelta(days=days + 30)).date()
    try:
        df = yf.download(
            symbol,
            start=start.isoformat(),
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception as e:
        log.error("yfinance download failed for %s: %s", symbol, e)
        return pd.DataFrame()

    if df.empty:
        return df

    # yfinance multi-symbol returns columns as MultiIndex; single symbol returns flat
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    try:
        df.to_parquet(path)
    except Exception as e:
        log.warning("cache write failed for %s: %s", symbol, e)
    return df[df.index.date >= required_start]


def get_history_many(cfg: Config, symbols: list[str], days: int | None = None) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = get_history(cfg, s, days=days)
        if not df.empty:
            out[s] = df
    return out
