"""Entry point for paper trading."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.trader import _setup_logging, run_loop


def main() -> int:
    cfg = load_config()
    _setup_logging(cfg)
    if cfg.is_live:
        print("\n!!! LIVE MODE ENABLED — type 'YES' to proceed: ", end="", flush=True)
        if input().strip() != "YES":
            print("aborted.")
            return 1
    run_loop(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
