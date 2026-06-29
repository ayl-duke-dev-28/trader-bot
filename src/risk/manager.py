"""Position sizing and pre-trade risk checks.

Two layers of safety:
1. Sizing: cap each position to max_position_pct of equity, total gross to max_gross_exposure.
2. Gates: daily-loss kill switch blocks new buys; per-trade stop-loss closes losers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import floor

from src.broker.alpaca_client import AlpacaBroker, Position
from src.config import Config

log = logging.getLogger(__name__)


@dataclass
class TradeIntent:
    symbol: str
    side: str           # 'buy' | 'sell'
    target_dollars: float
    reason: str


class RiskManager:
    def __init__(self, cfg: Config, broker: AlpacaBroker):
        self.cfg = cfg
        self.broker = broker
        self._start_equity: float | None = None

    def _risk(self, *keys, default):
        return self.cfg.get("risk", *keys, default=default)

    def kill_switch_tripped(self) -> bool:
        acct = self.broker.account()
        if self._start_equity is None:
            self._start_equity = acct.equity
            return False
        threshold = float(self._risk("daily_loss_kill_switch_pct", default=0.03))
        drop = (self._start_equity - acct.equity) / self._start_equity
        if drop >= threshold:
            log.warning("KILL SWITCH: equity drop %.2f%% >= %.2f%%", drop * 100, threshold * 100)
            return True
        return False

    def apply_stop_losses(self) -> None:
        stop_pct = float(self._risk("per_trade_stop_loss_pct", default=0.05))
        for p in self.broker.positions():
            if p.unrealized_plpc <= -stop_pct:
                log.warning("stop-loss closing %s @ pl=%.2f%%", p.symbol, p.unrealized_plpc * 100)
                self.broker.close_position(p.symbol)

    def size_orders(
        self,
        scores: dict[str, float],
        prices: dict[str, float],
    ) -> list[TradeIntent]:
        """Convert composite scores in [-1, 1] -> dollar-sized buy intents.

        v1 is long-only. Negative scores trigger sells of any existing position.
        """
        acct = self.broker.account()
        equity = acct.equity
        max_pos_pct = float(self._risk("max_position_pct", default=0.05))
        max_gross_pct = float(self._risk("max_gross_exposure", default=0.80))
        max_positions = int(self._risk("max_positions", default=20))

        held = {p.symbol: p for p in self.broker.positions()}
        held_value = sum(p.market_value for p in held.values())
        max_per_position_dollars = equity * max_pos_pct
        remaining_gross = max(0.0, equity * max_gross_pct - held_value)

        intents: list[TradeIntent] = []

        # 1. Sells: any held name with score <= -0.2 gets closed
        for sym, p in held.items():
            score = scores.get(sym, 0.0)
            if score <= -0.2:
                intents.append(TradeIntent(sym, "sell", p.market_value, f"score={score:.2f}"))

        # 2. Buys: pick top positive scores we don't already hold (or hold too little of)
        if not self.kill_switch_tripped():
            buy_candidates = sorted(
                ((s, sc) for s, sc in scores.items() if sc >= 0.2),
                key=lambda x: -x[1],
            )
            open_slots = max(0, max_positions - len([p for p in held.values() if p.market_value > 0]))
            for sym, score in buy_candidates[:open_slots]:
                if remaining_gross <= 100:
                    break
                price = prices.get(sym, 0.0)
                if price <= 0:
                    continue
                # Size scaled by score strength
                target = min(max_per_position_dollars * score, remaining_gross)
                existing = held[sym].market_value if sym in held else 0.0
                delta = target - existing
                if delta < max(50.0, max_per_position_dollars * 0.1):
                    continue
                intents.append(TradeIntent(sym, "buy", delta, f"score={score:.2f}"))
                remaining_gross -= delta

        return intents

    @staticmethod
    def intent_to_qty(intent: TradeIntent, price: float, allow_fractional: bool = True) -> float:
        if price <= 0:
            return 0.0
        qty = intent.target_dollars / price
        if not allow_fractional:
            return float(floor(qty))
        # Alpaca supports fractional shares; round to 4 decimals to be safe.
        # Fractional fills may appear as separate whole + fractional activity rows.
        return round(qty, 4)
