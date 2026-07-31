# Commodity walk-forward TDD evidence

## Source and user journey

No plan file was supplied. The user journey was derived from the request:

> As a trader, I want a commodity strategy evaluated over at least ten years in
> consecutive four-month walk-forward windows, so I can inspect genuinely
> out-of-sample performance rather than a full-history fit.

## RED and GREEN evidence

| Behavior | RED evidence | GREEN evidence | Guarantee |
| --- | --- | --- | --- |
| Calendar walk-forward construction | `python -m unittest tests.test_commodity_backtest -v` failed with `ModuleNotFoundError: src.backtest.commodities` | Five tests passed | Training ends before testing and test blocks span four calendar months |
| Commodity allocation | Same RED run | `test_target_weights_*` passed | Allocations are long-only, capped, favor stronger positive trends, and may hold cash |
| Leakage guard | Same RED run | `test_walk_forward_selection_never_receives_test_or_future_prices` passed | The selector receives a frame ending before its test window |
| Trading costs | Same RED run | `test_transaction_costs_reduce_walk_forward_equity` passed | Positive turnover costs reduce final equity and are reported |
| Per-window reporting | Targeted test initially failed with `KeyError: test_observations` | Targeted test and full file passed | Every completed test block reports observations, return, Sharpe, and drawdown |

## Validation

Commands run:

```text
python3 -m pytest tests/test_commodity_backtest.py -q
5 passed in 11.57s

python3 -m pytest tests/test_commodity_backtest.py \
  --cov=src.backtest.commodities --cov-report=term-missing \
  --cov-fail-under=80 -q
5 passed in 11.57s
src/backtest/commodities.py: 86%
```

`python -m compileall -q src scripts tests` and `git diff --check` also passed.
The full repository suite passed with the virtual environment's dependencies on
the Python path: `49 passed, 2 warnings in 19.08s`. The warnings are the existing
websockets deprecation and cross-version XGBoost serialization warning.

## Data-run status and known gap

The real-data command was attempted on 2026-07-31. The sandbox could not
resolve Yahoo's host, and none of the eleven required adjusted-close histories
were already cached. No synthetic performance figures were substituted. The
runner supports `--prices-csv` so the exact evaluation can be reproduced from a
local wide adjusted-close export when network access is available.

This implementation tests commodity ETF returns. It does not model direct
futures margin, contract selection, or a separate futures roll engine; roll and
fund expenses are reflected only through the ETF price series.
