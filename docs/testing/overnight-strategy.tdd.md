# Overnight strategy TDD evidence

## Source and user journey

No source plan was provided. The journey was derived from the request:

> As a strategy researcher, I want a reproducible close-to-next-open backtest with explicit two-sided costs, so that I can determine whether the reported overnight edge survives realistic turnover.

## Task report

### Return alignment and transaction costs

- RED command: `python3 -m pytest tests/test_overnight_backtest.py -q`
- RED evidence: collection failed with `ModuleNotFoundError: No module named 'src.backtest.overnight'` before the implementation existed.
- GREEN command: `python3 -m pytest tests/test_overnight_backtest.py -q`
- GREEN evidence: `5 passed in 0.19s`.
- Guarantee: the close at day *t-1* is paired with the open at day *t*, and a cost is charged on both executions of every round trip.

### Historical report

- Validation command: `PYTHONPATH=.venv/lib/python3.14/site-packages python3 scripts/backtest_overnight.py --start 2006-08-20 --end 2026-08-21`
- Result: PASS; the command downloaded 5,031 adjusted daily bars per ETF and wrote the Markdown, CSV, and equity-curve outputs under `reports/backtests/overnight_20y/`.
- Guarantee: SPY, QQQ, IWM, DIA, and VTI are compared over the same requested period at 0, 1, 2, and 5 basis points per side.

## Test specification

| # | What is guaranteed | Test target | Type | Result |
|---|---|---|---|---|
| 1 | Overnight returns use the prior close and next open without look-ahead | `test_session_returns_use_previous_close_for_overnight_leg` | Unit | PASS |
| 2 | Each daily round trip pays the configured cost twice | `test_backtest_charges_cost_on_both_daily_executions` | Unit | PASS |
| 3 | Invalid negative or greater-than-100% side costs are rejected | `test_backtest_rejects_invalid_costs` | Unit | PASS |
| 4 | Missing and nonpositive prices fail rather than silently corrupt returns | `test_backtest_rejects_missing_or_nonpositive_prices` | Unit | PASS |
| 5 | The historical CLI completes and emits research artifacts | historical report command above | Integration | PASS |

## Coverage and regression suite

- Coverage command: `python3 -m coverage run -m pytest tests/test_overnight_backtest.py -q && python3 -m coverage report -m src/backtest/overnight.py`
- Result: 96% statement coverage for `src/backtest/overnight.py`.
- Full-suite command: `PYTHONPATH=.venv/lib/python3.14/site-packages python3 -m pytest -q`
- Result: `108 passed, 7 warnings in 17.64s`.
- Known gap: Yahoo Finance is a free research feed, not an execution-quality auction dataset. The integration run exercises the live download but cannot prove auction fills, historical bid/ask spreads, taxes, or broker-specific fees.

No Git checkpoint commits were created because the research request did not authorize modifying branch history; RED/GREEN evidence is preserved here instead.
