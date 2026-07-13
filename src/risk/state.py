"""Persistent risk state.

Three things live here across process restarts:
  * day-start equity for the kill switch
  * per-symbol cooldowns after a stop-out
  * trailing-stop high-water mark per open position
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


_EMPTY: dict[str, Any] = {
    "cooldowns": {},
    "highwater": {},
    "day_equity": {},
    "portfolio": {},
}


class RiskState:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {k: dict(v) for k, v in _EMPTY.items()}
        try:
            raw = json.loads(self.path.read_text())
        except Exception as e:
            log.warning("state load failed: %s; resetting", e)
            return {k: dict(v) for k, v in _EMPTY.items()}
        for key, default in _EMPTY.items():
            if key not in raw:
                raw[key] = dict(default)
        return raw

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        except Exception as e:
            log.warning("state save failed: %s", e)

    # --- day-start equity -------------------------------------------------

    def day_start_equity(self, current_equity: float, today: date | None = None) -> float:
        """Return today's start-of-day equity; seed with current on first call."""
        today = today or datetime.utcnow().date()
        key = today.isoformat()
        cutoff = (today - timedelta(days=7)).isoformat()
        self._data["day_equity"] = {
            k: v for k, v in self._data["day_equity"].items() if k >= cutoff
        }
        if key not in self._data["day_equity"]:
            self._data["day_equity"][key] = float(current_equity)
            self._save()
        return float(self._data["day_equity"][key])

    # --- cooldowns --------------------------------------------------------

    def in_cooldown(self, symbol: str, now: datetime | None = None) -> bool:
        now = now or datetime.utcnow()
        unlock_iso = self._data["cooldowns"].get(symbol)
        if not unlock_iso:
            return False
        try:
            unlock = datetime.fromisoformat(unlock_iso)
        except Exception:
            return False
        if now >= unlock:
            self._data["cooldowns"].pop(symbol, None)
            self._save()
            return False
        return True

    def record_stop(self, symbol: str, cooldown_days: int, now: datetime | None = None) -> None:
        now = now or datetime.utcnow()
        unlock = now + timedelta(days=int(cooldown_days))
        self._data["cooldowns"][symbol] = unlock.isoformat()
        self._data["highwater"].pop(symbol, None)
        self._save()

    # --- trailing highwater ----------------------------------------------

    def highwater(self, symbol: str) -> float:
        return float(self._data["highwater"].get(symbol, 0.0))

    def update_highwater(self, symbol: str, current_pnl_pct: float) -> float:
        prior = self.highwater(symbol)
        if current_pnl_pct > prior:
            self._data["highwater"][symbol] = float(current_pnl_pct)
            self._save()
            return float(current_pnl_pct)
        return prior

    def clear_symbol(self, symbol: str) -> None:
        """Called when a position is fully closed for reasons other than a stop."""
        changed = False
        if symbol in self._data["highwater"]:
            self._data["highwater"].pop(symbol, None)
            changed = True
        if changed:
            self._save()

    # --- portfolio drawdown guard ----------------------------------------

    def portfolio_highwater(self, current_equity: float) -> float:
        if "highwater" not in self._data["portfolio"]:
            self._data["portfolio"]["highwater"] = float(current_equity)
            self._save()
        highwater = float(self._data["portfolio"]["highwater"])
        if current_equity > highwater:
            highwater = float(current_equity)
            self._data["portfolio"]["highwater"] = highwater
            self._save()
        return highwater

    def portfolio_guard_tripped(self) -> bool:
        return bool(self._data["portfolio"].get("guard_tripped", False))

    def trip_portfolio_guard(self) -> None:
        self._data["portfolio"]["guard_tripped"] = True
        self._save()
