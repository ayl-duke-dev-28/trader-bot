# Non-finite quote guard TDD evidence

## Source

No plan file was supplied. The journey came from the July 29 paper-trading
failure `ValueError: cannot convert float NaN to integer`.

## User journey

As the paper-trading operator, I want missing or invalid quotes skipped safely
so one bad Yahoo Finance value cannot abort a cycle after earlier orders have
already been submitted.

## RED and GREEN

| Stage | Command | Evidence |
| --- | --- | --- |
| RED | `.venv/bin/python tests/test_smoke.py` | Failed in `test_intent_to_qty_rejects_non_finite_prices` with the production exception `ValueError: cannot convert float NaN to integer` |
| GREEN | `.venv/bin/python tests/test_smoke.py` | `smoke tests OK` |
| Regression and coverage | `PYTHONPATH="$(pwd)/.venv/lib/python3.14/site-packages" python3 -m pytest -q tests/test_smoke.py tests/test_macro_cycles.py --cov=src.risk.validation --cov-report=term-missing --cov-fail-under=80` | `27 passed`; price-validation module coverage `100%` |

## Test specification

| # | Guarantee | Test | Type | Result |
| --- | --- | --- | --- | --- |
| 1 | Whole and fractional sizing return zero for NaN and infinite prices | `test_intent_to_qty_rejects_non_finite_prices` | Unit | PASS |
| 2 | The common validator rejects non-numeric, non-positive, NaN, and infinite values | `test_price_validation_rejects_non_numeric_and_non_positive_values` | Unit | PASS |
| 3 | Quote ingestion retains only finite positive prices | `test_last_prices_drops_non_finite_quotes` | Unit with mocked Yahoo response | PASS |
| 4 | Order planning does not create an intent for a NaN quote | `test_size_orders_skips_non_finite_quote` | Integration | PASS |
| 5 | Buy execution maps a NaN quote to zero quantity while sell execution falls back to position value | `test_execution_qty_rejects_nan_buy_and_falls_back_for_sell` | Integration | PASS |

## Known warnings

The suite still emits the pre-existing XGBoost serialized-model version warning
and a websockets deprecation warning. Neither warning affects this price guard.

No Git checkpoint commits were created because the user requested workspace
changes, not repository-history changes. RED and GREEN evidence is preserved
above.
