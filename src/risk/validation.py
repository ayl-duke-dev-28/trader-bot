"""Validation helpers shared by planning and execution."""
from __future__ import annotations

from math import isfinite
from typing import Any


def is_valid_price(value: Any) -> bool:
    """Return whether a value is a finite, strictly positive price."""
    try:
        price = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(price) and price > 0
