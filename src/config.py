"""Configuration loader: merges config.yaml with .env secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict[str, Any]
    api_key: str
    api_secret: str
    is_live: bool

    def get(self, *keys, default=None):
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def load_config(path: str | Path = "config.yaml", require_secrets: bool = True) -> Config:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with cfg_path.open() as f:
        raw = yaml.safe_load(f)

    mode = (raw.get("mode") or "paper").lower()
    is_live = mode == "live"

    if is_live:
        key = os.getenv("ALPACA_LIVE_API_KEY", "")
        secret = os.getenv("ALPACA_LIVE_API_SECRET", "")
        if require_secrets and (not key or not secret):
            raise RuntimeError(
                "mode=live in config.yaml but ALPACA_LIVE_API_KEY/SECRET not set in .env. "
                "Refusing to start."
            )
    else:
        key = os.getenv("ALPACA_API_KEY", "")
        secret = os.getenv("ALPACA_API_SECRET", "")
        if require_secrets and (not key or not secret):
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_API_SECRET missing in .env. "
                "Get free paper keys at https://app.alpaca.markets/paper/dashboard/overview"
            )

    return Config(raw=raw, api_key=key, api_secret=secret, is_live=is_live)
