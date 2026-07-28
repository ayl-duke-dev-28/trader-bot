"""Download initial-release US macro data for point-in-time backtests."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.config import ROOT
from src.data.macro import (
    build_initial_release_panel,
    fetch_fred_initial_releases,
    write_macro_panel,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data_cache" / "macro" / "fred_initial_releases.csv",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is missing. Add a free FRED API key to .env before downloading macro data."
        )

    releases = fetch_fred_initial_releases(api_key)
    panel = build_initial_release_panel(releases)
    if panel.empty:
        raise RuntimeError("FRED returned no usable initial-release macro observations")

    output = args.out if args.out.is_absolute() else ROOT / args.out
    write_macro_panel(panel, output)
    print(f"wrote {len(panel)} point-in-time macro observations -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

