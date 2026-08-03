"""Durable broker-order reconciliation tests."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.broker.alpaca_client import Account, AlpacaBroker, BrokerOrderStatus, Position
from src.risk.manager import RiskManager
from src.risk.state import RiskState
from src.trader import _reconcile_pending_orders


class _RiskConfig:
    @staticmethod
    def get(*keys, default=None):
        values = {
            ("risk", "portfolio_drawdown_guard"): {
                "enabled": True,
                "max_drawdown_pct": 0.10,
            },
        }
        return values.get(keys, default)


def test_broker_close_returns_the_submitted_order_id():
    broker = object.__new__(AlpacaBroker)
    broker.client = MagicMock()
    broker.client.close_position.return_value = SimpleNamespace(id="close-1")

    assert broker.close_position("AAPL") == "close-1"


def test_broker_order_status_normalizes_the_alpaca_response():
    broker = object.__new__(AlpacaBroker)
    broker.client = MagicMock()
    broker.client.get_order_by_id.return_value = SimpleNamespace(
        id="order-1",
        symbol="aapl",
        side=SimpleNamespace(value="sell"),
        status=SimpleNamespace(value="partially_filled"),
        qty="10",
        filled_qty="4.5",
    )

    assert broker.order_status("order-1") == BrokerOrderStatus(
        order_id="order-1",
        symbol="AAPL",
        side="sell",
        status="partially_filled",
        qty=10.0,
        filled_qty=4.5,
    )


def test_stop_submission_defers_cooldown_and_highwater_clear_until_fill(tmp_path: Path):
    state = RiskState(tmp_path / "risk_state.json")
    state.update_highwater("AAPL", 0.25)
    risk = object.__new__(RiskManager)
    risk.state = state
    risk.broker = MagicMock()
    risk.broker.close_position.return_value = "stop-1"
    risk.trade_log = MagicMock()
    position = Position("AAPL", 10.0, 100.0, 1_100.0, 0.10)

    assert risk._close(position, False, 3, "trailing stop") == "stop-1"

    assert state.highwater("AAPL") == 0.25
    assert state.in_cooldown("AAPL") is False
    assert state.pending_orders()[0].cooldown_days == 3


def test_stop_scan_skips_pending_symbol_and_returns_newly_submitted_symbol(tmp_path: Path):
    broker = MagicMock()
    broker.positions.return_value = [
        Position("AAPL", 10.0, 100.0, 900.0, -0.10),
        Position("MSFT", 5.0, 200.0, 900.0, -0.10),
    ]
    broker.close_position.return_value = "stop-2"
    state = RiskState(tmp_path / "risk_state.json")
    risk = RiskManager(_RiskConfig(), broker, state=state)

    submitted = risk.apply_stop_losses(blocked_symbols={"MSFT"})

    assert submitted == {"AAPL"}
    broker.close_position.assert_called_once_with("AAPL")
    assert state.pending_orders()[0].order_id == "stop-2"


def test_drawdown_guard_does_not_duplicate_close_for_pending_symbol():
    broker = MagicMock()
    broker.account.return_value = Account(80.0, 0.0, 0.0, 80.0)
    broker.positions.return_value = [
        Position("AAPL", 10.0, 100.0, 800.0, -0.20),
        Position("MSFT", 5.0, 200.0, 800.0, -0.20),
    ]
    state = MagicMock()
    state.portfolio_guard_tripped.return_value = False
    state.portfolio_highwater.return_value = 100.0
    risk = RiskManager(_RiskConfig(), broker, state=state)
    risk._close = MagicMock()

    assert risk.apply_portfolio_drawdown_guard(blocked_symbols={"MSFT"}) is True

    risk._close.assert_called_once()
    assert risk._close.call_args.args[0].symbol == "AAPL"
    state.trip_portfolio_guard.assert_called_once_with()


def test_malformed_pending_order_does_not_disable_valid_reconciliation(tmp_path: Path):
    path = tmp_path / "risk_state.json"
    state = RiskState(path)
    state.record_pending_order(
        order_id="valid",
        symbol="AAPL",
        side="buy",
        qty=1.0,
        full_position=False,
    )
    state._data["pending_orders"]["broken"] = {"symbol": "MSFT"}

    assert [order.order_id for order in state.pending_orders()] == ["valid"]

def test_pending_order_ledger_survives_restart(tmp_path: Path):
    path = tmp_path / "risk_state.json"
    state = RiskState(path)

    state.record_pending_order(
        order_id="order-1",
        symbol="AAPL",
        side="sell",
        qty=10.0,
        full_position=False,
    )

    restored = RiskState(path)
    assert restored.pending_orders()[0].order_id == "order-1"
    assert restored.pending_orders()[0].status == "submitted"
    assert restored.pending_orders()[0].full_position is False
    assert restored.pending_orders()[0].cooldown_days == 0


def test_filled_stop_exit_starts_cooldown_and_resolves_pending_order(tmp_path: Path):
    state = RiskState(tmp_path / "risk_state.json")
    state.update_highwater("AAPL", 0.25)
    state.record_pending_order(
        order_id="order-2",
        symbol="AAPL",
        side="sell",
        qty=25.0,
        full_position=True,
        cooldown_days=3,
    )
    broker = MagicMock()
    broker.order_status.return_value = BrokerOrderStatus(
        order_id="order-2",
        symbol="AAPL",
        side="sell",
        status="filled",
        qty=25.0,
        filled_qty=25.0,
    )

    blocked = _reconcile_pending_orders(
        broker,
        state,
        trade_log=MagicMock(),
        mode="paper",
    )

    assert blocked == set()
    assert state.pending_orders() == []
    assert state.highwater("AAPL") == 0.0
    assert state.in_cooldown("AAPL") is True


def test_filled_non_stop_exit_clears_highwater_without_cooldown(tmp_path: Path):
    state = RiskState(tmp_path / "risk_state.json")
    state.update_highwater("AAPL", 0.25)
    state.record_pending_order(
        order_id="order-2b",
        symbol="AAPL",
        side="sell",
        qty=25.0,
        full_position=True,
    )
    broker = MagicMock()
    broker.order_status.return_value = BrokerOrderStatus(
        order_id="order-2b",
        symbol="AAPL",
        side="sell",
        status="filled",
        qty=25.0,
        filled_qty=25.0,
    )

    blocked = _reconcile_pending_orders(
        broker,
        state,
        trade_log=MagicMock(),
        mode="paper",
    )

    assert blocked == set()
    assert state.pending_orders() == []
    assert state.highwater("AAPL") == 0.0
    assert state.in_cooldown("AAPL") is False


def test_partial_fill_remains_pending_and_blocks_conflicting_orders(tmp_path: Path):
    state = RiskState(tmp_path / "risk_state.json")
    state.record_pending_order(
        order_id="order-3",
        symbol="MSFT",
        side="buy",
        qty=20.0,
        full_position=False,
    )
    broker = MagicMock()
    broker.order_status.return_value = BrokerOrderStatus(
        order_id="order-3",
        symbol="MSFT",
        side="buy",
        status="partially_filled",
        qty=20.0,
        filled_qty=8.0,
    )

    blocked = _reconcile_pending_orders(
        broker,
        state,
        trade_log=MagicMock(),
        mode="paper",
    )

    assert blocked == {"MSFT"}
    assert state.pending_orders()[0].status == "partially_filled"


def test_rejected_order_is_resolved_without_clearing_position_state(tmp_path: Path):
    state = RiskState(tmp_path / "risk_state.json")
    state.update_highwater("NVDA", 0.20)
    state.record_pending_order(
        order_id="order-4",
        symbol="NVDA",
        side="sell",
        qty=15.0,
        full_position=True,
        cooldown_days=3,
    )
    broker = MagicMock()
    broker.order_status.return_value = BrokerOrderStatus(
        order_id="order-4",
        symbol="NVDA",
        side="sell",
        status="rejected",
        qty=15.0,
        filled_qty=0.0,
    )

    blocked = _reconcile_pending_orders(
        broker,
        state,
        trade_log=MagicMock(),
        mode="paper",
    )

    assert blocked == set()
    assert state.pending_orders() == []
    assert state.highwater("NVDA") == 0.20
    assert state.in_cooldown("NVDA") is False


def test_status_lookup_failure_keeps_order_blocked(tmp_path: Path):
    state = RiskState(tmp_path / "risk_state.json")
    state.record_pending_order(
        order_id="order-5",
        symbol="GOOG",
        side="buy",
        qty=5.0,
        full_position=False,
    )
    broker = MagicMock()
    broker.order_status.return_value = None

    blocked = _reconcile_pending_orders(
        broker,
        state,
        trade_log=MagicMock(),
        mode="paper",
    )

    assert blocked == {"GOOG"}
    assert state.pending_orders()[0].order_id == "order-5"
