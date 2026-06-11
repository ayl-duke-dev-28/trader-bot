"""Pull recent politician disclosures and rank by recent buy/sell activity."""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import load_config
from src.politicians.tracker import fetch_disclosures

log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    df = fetch_disclosures(cfg)
    if df.empty:
        print("No disclosures fetched (feeds may be down).")
        return 1

    cutoff = datetime.utcnow() - timedelta(days=int(cfg.get("strategies", "politicians", "lookback_days", default=30)))
    recent = df[df["txn_date"] >= cutoff].copy()
    print(f"\nDisclosures in last {(datetime.utcnow() - cutoff).days} days: {len(recent)}")

    if recent.empty:
        return 0

    recent["signed_amount"] = recent.apply(
        lambda r: r["amount_usd"] if r["side"] == "buy" else -r["amount_usd"], axis=1
    )
    agg = (
        recent.groupby("symbol")
        .agg(
            net_dollars=("signed_amount", "sum"),
            n_trades=("symbol", "count"),
            chambers=("chamber", lambda s: ",".join(sorted(set(s)))),
        )
        .sort_values("net_dollars", ascending=False)
    )
    print("\nTop net BUYS (politicians):")
    print(agg.head(15).to_string())
    print("\nTop net SELLS (politicians):")
    print(agg.tail(15).iloc[::-1].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
