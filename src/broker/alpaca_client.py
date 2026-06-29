"""Thin wrapper around alpaca-py covering everything the bot needs.

The same class talks to paper or live based on Config.is_live. Live trading
trips additional guardrails enforced in src/risk/manager.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from src.config import Config

log = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_plpc: float


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float


class AlpacaBroker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = TradingClient(
            api_key=cfg.api_key,
            secret_key=cfg.api_secret,
            paper=not cfg.is_live,
        )
        log.info("Alpaca broker initialized in %s mode", "LIVE" if cfg.is_live else "paper")

    def account(self) -> Account:
        a = self.client.get_account()
        return Account(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            portfolio_value=float(a.portfolio_value),
        )

    def positions(self) -> list[Position]:
        out = []
        for p in self.client.get_all_positions():
            out.append(
                Position(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    market_value=float(p.market_value),
                    unrealized_plpc=float(p.unrealized_plpc),
                )
            )
        return out

    def is_market_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def open_order_symbols(self, side: str | None = None) -> set[str]:
        """Return symbols with currently open orders, optionally filtered by side."""
        try:
            orders = self.client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        except Exception as e:
            log.warning("open orders fetch failed: %s", e)
            return set()

        out: set[str] = set()
        side_filter = side.lower() if side else None
        for order in orders:
            order_side = str(getattr(order, "side", "")).lower()
            if side_filter and side_filter not in order_side:
                continue
            symbol = str(getattr(order, "symbol", "")).upper()
            if symbol:
                out.add(symbol)
        return out

    def submit_market_order(self, symbol: str, qty: float, side: str) -> str | None:
        if qty <= 0:
            return None
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            order = self.client.submit_order(req)
            log.info("submitted %s %s qty=%s id=%s", side, symbol, qty, order.id)
            return str(order.id)
        except Exception as e:
            log.error("order failed %s %s qty=%s: %s", side, symbol, qty, e)
            return None

    def close_position(self, symbol: str) -> bool:
        try:
            self.client.close_position(symbol)
            log.info("closed position %s", symbol)
            return True
        except Exception as e:
            log.error("close_position failed %s: %s", symbol, e)
            return False
