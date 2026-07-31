"""Thin wrapper around alpaca-py covering everything the bot needs.

The same class talks to paper or live based on Config.is_live. Live trading
trips additional guardrails enforced in src/risk/manager.py.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from requests.exceptions import RequestException

from src.config import Config
from src.risk.validation import is_valid_price

log = logging.getLogger(__name__)
T = TypeVar("T")


def _retry_request(label: str, fn: Callable[[], T], *, attempts: int = 3, delay_seconds: float = 1.0) -> T:
    """Retry read-only Alpaca calls that sometimes fail on transient network resets."""
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (RequestException, OSError) as e:
            last_error = e
            if attempt >= attempts:
                break
            log.warning("%s failed on attempt %d/%d: %s; retrying", label, attempt, attempts, e)
            time.sleep(delay_seconds)
    log.error("%s failed after %d attempts: %s", label, attempts, last_error)
    raise last_error or RuntimeError(f"{label} failed")


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
        self.market_data_client = StockHistoricalDataClient(
            api_key=cfg.api_key,
            secret_key=cfg.api_secret,
        )
        log.info("Alpaca broker initialized in %s mode", "LIVE" if cfg.is_live else "paper")

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """Return finite positive latest-trade prices from Alpaca's IEX feed."""
        if not symbols:
            return {}
        request = StockLatestTradeRequest(
            symbol_or_symbols=symbols,
            feed=DataFeed.IEX,
        )
        try:
            trades = _retry_request(
                "Alpaca latest-trade fetch",
                lambda: self.market_data_client.get_stock_latest_trade(request),
            )
        except Exception as e:
            log.warning("latest-trade fetch failed: %s", e)
            return {}

        prices: dict[str, float] = {}
        for symbol in symbols:
            trade = trades.get(symbol)
            raw_price = getattr(trade, "price", None)
            if is_valid_price(raw_price):
                prices[symbol] = float(raw_price)
        return prices

    def account(self) -> Account:
        a = _retry_request("Alpaca account fetch", self.client.get_account)
        return Account(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            portfolio_value=float(a.portfolio_value),
        )

    def positions(self) -> list[Position]:
        out = []
        positions = _retry_request("Alpaca positions fetch", self.client.get_all_positions)
        for p in positions:
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
        clock = _retry_request("Alpaca clock fetch", self.client.get_clock)
        return bool(clock.is_open)

    def open_order_symbols(self, side: str | None = None) -> set[str]:
        """Return symbols with currently open orders, optionally filtered by side."""
        try:
            orders = _retry_request(
                "Alpaca open orders fetch",
                lambda: self.client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)),
            )
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
