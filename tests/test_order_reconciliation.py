"""Durable broker-order reconciliation tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.broker.alpaca_client import BrokerOrderStatus
from src.risk.state import RiskState
from src.trader import _reconcile_pending_orders


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


def test_filled_full_exit_clears_symbol_state_and_resolves_pending_order(tmp_path: Path):
    state = RiskState(tmp_path / "risk_state.json")
    state.update_highwater("AAPL", 0.25)
    state.record_pending_order(
        order_id="order-2",
        symbol="AAPL",
        side="sell",
        qty=25.0,
        full_position=True,
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
