"""Entry point for paper trading."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.trader import _setup_logging, run_loop

log = logging.getLogger(__name__)


def _start_macos_sleep_preventer(cfg):
    """Keep an open Mac awake while the trading process is running."""
    if not bool(cfg.get("schedule", "prevent_system_sleep", default=True)):
        log.info("macOS idle-sleep prevention disabled by configuration")
        return None
    if sys.platform != "darwin":
        return None
    caffeinate = shutil.which("caffeinate")
    if not caffeinate:
        log.warning("macOS caffeinate command not found; scheduled cycles may be missed during sleep")
        return None
    process = subprocess.Popen(
        [caffeinate, "-i", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log.info("macOS idle sleep prevention active while PID %d is running", os.getpid())
    return process


def main() -> int:
    cfg = load_config()
    _setup_logging(cfg)
    if cfg.is_live:
        print("\n!!! LIVE MODE ENABLED — type 'YES' to proceed: ", end="", flush=True)
        if input().strip() != "YES":
            print("aborted.")
            return 1
    _start_macos_sleep_preventer(cfg)
    run_loop(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
