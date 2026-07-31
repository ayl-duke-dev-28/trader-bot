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
the Python path: `50 passed, 2 warnings in 19.19s`. The warnings are the existing
websockets deprecation and cross-version XGBoost serialization warning.

## Cache fallback follow-up

The network-enabled run downloaded prices but exposed a missing optional
Parquet engine under the system Python. A new test first failed because
`GLD.csv` was not created after `DataFrame.to_parquet` raised `ImportError`.
After the fix, `tests/test_market_data_cache.py` passed and proves that the CSV
fallback is both written and reused without another download.

## Data-run status and known gap

The first sandbox attempt could not resolve Yahoo's host. The user then ran the
same command in a network-enabled terminal, producing 31 completed four-month
windows from 2015-12-16 through 2026-04-15. The retained raw report and analysis
contain the measured baseline results. The runner also supports `--prices-csv`
and now retains `adjusted_closes.csv` for exact variant reproduction.

This implementation tests commodity ETF returns. It does not model direct
futures margin, contract selection, or a separate futures roll engine; roll and
fund expenses are reflected only through the ETF price series.

## Diversification follow-up

The retained baseline averaged only 2.21 effective positions. Precious metals
averaged 28.30% of capital while agriculture averaged 4.49%; one losing window
held 67.57% in precious metals and another held 57.94% in industrial metals.
This motivated a diversified variant rather than adding leverage.

A new test first failed because `default_diversified_candidates` did not exist.
It now proves the variant holds at least four qualifying assets in the supplied
scenario, caps individual funds at 25%, caps the energy group at 35%, remains
long-only, and never exceeds 100% gross exposure. The original five commodity
tests and the new diversification test pass together.
