"""Track US congressional stock disclosures (STOCK Act filings).

Free data sources (community-maintained S3-hosted JSON dumps):
  - House:  https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json
  - Senate: https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json

These feeds may move or rate-limit. Endpoints are configurable below.

Output of politician_signal() is in [-1, 1] per symbol:
  - +1 when there's a recent BUY disclosure of meaningful size
  - -1 for a SELL
  - decays linearly over `lookback_days`
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.config import Config, ROOT

log = logging.getLogger(__name__)

HOUSE_URL = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
SENATE_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"

# These feeds use ranges like "$1,001 - $15,000". Map to a midpoint USD.
AMOUNT_BUCKETS = {
    "$1,001 - $15,000": 8_000,
    "$15,001 - $50,000": 32_500,
    "$15,000 - $50,000": 32_500,
    "$50,001 - $100,000": 75_000,
    "$100,001 - $250,000": 175_000,
    "$250,001 - $500,000": 375_000,
    "$500,001 - $1,000,000": 750_000,
    "$1,000,001 - $5,000,000": 3_000_000,
    "$5,000,001 - $25,000,000": 15_000_000,
    "$25,000,001 - $50,000,000": 37_500_000,
    "$50,000,001 +": 75_000_000,
}


def _parse_amount(raw: str | None) -> float:
    if not raw:
        return 0.0
    if raw in AMOUNT_BUCKETS:
        return float(AMOUNT_BUCKETS[raw])
    nums = re.findall(r"[\d,]+", raw)
    if not nums:
        return 0.0
    try:
        vals = [float(n.replace(",", "")) for n in nums]
        return sum(vals) / len(vals)
    except ValueError:
        return 0.0


def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _fetch(url: str, cache_path: Path, max_age_hours: int = 12) -> list[dict]:
    if cache_path.exists():
        age = datetime.utcnow() - datetime.utcfromtimestamp(cache_path.stat().st_mtime)
        if age < timedelta(hours=max_age_hours):
            try:
                return pd.read_json(cache_path).to_dict("records")
            except Exception as e:
                log.warning("politician cache read failed: %s", e)

    log.info("downloading %s", url)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data).to_json(cache_path)
        return data
    except Exception as e:
        log.error("politician fetch failed for %s: %s", url, e)
        return []


def fetch_disclosures(cfg: Config) -> pd.DataFrame:
    """Returns normalized dataframe with columns:
       symbol, chamber, name, txn_date, side ('buy'|'sell'), amount_usd
    """
    cache_dir = Path(cfg.get("data", "cache_dir", default="data_cache"))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir

    rows = []
    if cfg.get("strategies", "politicians", "follow_representatives", default=True):
        for d in _fetch(HOUSE_URL, cache_dir / "house_disclosures.json"):
            sym = (d.get("ticker") or "").upper().strip()
            if not sym or sym == "--":
                continue
            side_raw = (d.get("type") or "").lower()
            side = "buy" if "purchase" in side_raw else ("sell" if "sale" in side_raw else None)
            if side is None:
                continue
            rows.append({
                "symbol": sym,
                "chamber": "house",
                "name": d.get("representative") or "",
                "txn_date": _parse_date(d.get("transaction_date")),
                "side": side,
                "amount_usd": _parse_amount(d.get("amount")),
            })

    if cfg.get("strategies", "politicians", "follow_senators", default=True):
        for d in _fetch(SENATE_URL, cache_dir / "senate_disclosures.json"):
            sym = (d.get("ticker") or "").upper().strip()
            if not sym or sym == "--":
                continue
            side_raw = (d.get("type") or "").lower()
            side = "buy" if "purchase" in side_raw else ("sell" if "sale" in side_raw else None)
            if side is None:
                continue
            rows.append({
                "symbol": sym,
                "chamber": "senate",
                "name": d.get("senator") or "",
                "txn_date": _parse_date(d.get("transaction_date")),
                "side": side,
                "amount_usd": _parse_amount(d.get("amount")),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["txn_date"])
    return df


def politician_signals(cfg: Config, symbols: Iterable[str]) -> dict[str, float]:
    """Compute per-symbol signal in [-1, 1] from recent political disclosures."""
    lookback = int(cfg.get("strategies", "politicians", "lookback_days", default=30))
    min_amt = float(cfg.get("strategies", "politicians", "min_trade_amount_usd", default=50_000))
    df = fetch_disclosures(cfg)
    out: dict[str, float] = {s: 0.0 for s in symbols}
    if df.empty:
        return out

    cutoff = datetime.utcnow() - timedelta(days=lookback)
    recent = df[(df["txn_date"] >= cutoff) & (df["amount_usd"] >= min_amt)]
    if recent.empty:
        return out

    syms = set(symbols)
    for sym, grp in recent.groupby("symbol"):
        if sym not in syms:
            continue
        score = 0.0
        for _, row in grp.iterrows():
            age_days = max(0, (datetime.utcnow() - row["txn_date"]).days)
            decay = max(0.0, 1.0 - age_days / lookback)
            direction = 1.0 if row["side"] == "buy" else -1.0
            # Larger trade -> larger weight, capped
            mag = min(1.0, row["amount_usd"] / 500_000.0)
            score += direction * decay * mag
        # Squash so a flurry of trades doesn't overwhelm
        out[sym] = max(-1.0, min(1.0, score / 2.0))
    return out
