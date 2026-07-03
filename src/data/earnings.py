"""Next-earnings lookup with a 24h disk cache.

Used by trader.py to block new buys inside the earnings blackout window.
Only queries yfinance for symbols we actually plan to buy this cycle.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import yfinance as yf

from src.config import Config, ROOT

log = logging.getLogger(__name__)

CACHE_TTL_HOURS = 24


def _cache_path(cfg: Config) -> Path:
    cache_dir = Path(cfg.get("data", "cache_dir", default="data_cache"))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    p = cache_dir / "earnings"
    p.mkdir(parents=True, exist_ok=True)
    return p / "next_earnings.json"


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_cache(path: Path, data: dict) -> None:
    try:
        path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.warning("earnings cache write failed: %s", e)


def _first_future_date(dates_field) -> str | None:
    today = datetime.utcnow().date()
    candidates = dates_field if isinstance(dates_field, (list, tuple)) else [dates_field]
    for entry in candidates:
        if entry is None:
            continue
        try:
            d = entry.date() if hasattr(entry, "date") else date.fromisoformat(str(entry)[:10])
        except Exception:
            continue
        if d >= today:
            return d.isoformat()
    return None


def _fetch_next_earnings(symbol: str) -> str | None:
    try:
        cal = yf.Ticker(symbol).calendar
    except Exception as e:
        log.debug("earnings fetch failed for %s: %s", symbol, e)
        return None
    if cal is None:
        return None
    if isinstance(cal, dict):
        dates_field = cal.get("Earnings Date")
    else:
        try:
            dates_field = cal.loc["Earnings Date"].values[0]
        except Exception:
            return None
    if not dates_field:
        return None
    return _first_future_date(dates_field)


def next_earnings_dates(cfg: Config, symbols: list[str]) -> dict[str, date]:
    """Return {symbol: next earnings date} for the given symbols, using the cache."""
    if not symbols:
        return {}
    path = _cache_path(cfg)
    cache = _load_cache(path)
    now = datetime.utcnow()
    out: dict[str, date] = {}
    dirty = False
    for sym in symbols:
        entry = cache.get(sym)
        if entry:
            try:
                fetched = datetime.fromisoformat(entry["fetched"])
                if (now - fetched) < timedelta(hours=CACHE_TTL_HOURS):
                    iso = entry.get("date")
                    if iso:
                        try:
                            out[sym] = date.fromisoformat(iso)
                        except Exception:
                            pass
                    continue
            except Exception:
                pass
        earning_iso = _fetch_next_earnings(sym)
        cache[sym] = {"fetched": now.isoformat(), "date": earning_iso}
        dirty = True
        if earning_iso:
            try:
                out[sym] = date.fromisoformat(earning_iso)
            except Exception:
                pass
    if dirty:
        _save_cache(path, cache)
    return out


def near_earnings(next_dates: dict[str, date], symbol: str, within_days: int) -> bool:
    d = next_dates.get(symbol)
    if not d:
        return False
    today = datetime.utcnow().date()
    return today <= d <= today + timedelta(days=int(within_days))
