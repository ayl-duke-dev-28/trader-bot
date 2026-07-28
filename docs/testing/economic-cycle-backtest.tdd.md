# Economic-cycle backtest TDD evidence

## Source

No plan file was supplied. The user journey and guarantees were derived from
the request to backtest long- and short-term economic cycles using growth,
inflation, unemployment, and interest rates.

## User journey

As a strategy researcher, I want a backtest-only economic-cycle exposure
overlay based on historically available macroeconomic observations so that I
can compare cycle-aware risk limits without giving the simulation future or
revised information.

## Task report

| Behavior | Validation | Evidence |
| --- | --- | --- |
| Missing cycle implementation produces RED | `python3 -m pytest -q tests/test_macro_cycles.py` | Collection failed with `ModuleNotFoundError: No module named 'src.data.macro'` |
| Long and short scores distinguish a synthetic expansion from contraction | `python3 -m pytest -q tests/test_macro_cycles.py` | `6 passed` after implementation |
| Exposure uses the latest observation available on or before each simulation date and never raises another cap | `python3 -m pytest -q tests/test_macro_cycles.py` | Point-in-time and cap tests passed |
| FRED data requests initial-release observations and aligns series on release dates | `python3 -m pytest -q tests/test_macro_cycles.py` | Downloader and panel-alignment tests passed |
| The live-path simulator applies the contraction cap | `.venv/bin/python tests/test_smoke.py` | `smoke tests OK`; integration assertion verifies a `30%` minimum cap and no QQQ core buy in contraction |
| CLI entry points expose macro download and backtest inputs | `.venv/bin/python scripts/backtest.py --help` and `.venv/bin/python scripts/fetch_macro_data.py --help` | Both commands loaded successfully and displayed the new options |

## Test specification

| # | Guarantee | Test | Type | Result |
| --- | --- | --- | --- | --- |
| 1 | Expansion produces positive long/short scores and contraction produces negative scores | `test_macro_cycle_scores_long_and_short_expansion_then_contraction` | Unit | PASS |
| 2 | Dates before the first macro release retain the existing cap | `test_macro_exposure_uses_only_information_available_as_of_date` | Unit | PASS |
| 3 | A macro overlay cannot increase an existing market-regime cap | `test_macro_overlay_never_increases_existing_exposure_cap` | Unit | PASS |
| 4 | Several releases in one calendar month count as one monthly cycle observation | `test_multiple_release_events_in_one_month_do_not_count_as_multiple_months` | Unit | PASS |
| 5 | Growth and inflation year-over-year values appear only after their release dates | `test_initial_release_panel_aligns_series_by_release_date_without_lookahead` | Unit | PASS |
| 6 | FRED requests use `output_type=4` (initial release only) | `test_fred_fetch_requests_initial_release_observations` | Integration boundary with fake HTTP session | PASS |
| 7 | The live-path backtester applies a contraction exposure cap | Macro assertions in `test_backtest_uses_live_path_benchmark_core` | Integration | PASS |

## Coverage and known gaps

`python3 -m pytest -q tests/test_macro_cycles.py --cov=src.data.macro
--cov-report=term-missing --cov-fail-under=80` passed with **87.59%** coverage
of `src/data/macro.py`.

A real historical comparison was not run in this implementation session because
the workspace has no `FRED_API_KEY`. The downloader deliberately does not fall
back to latest-revision CSV data because that would weaken the point-in-time
guarantee. Once the key is configured, the README commands generate the macro
panel and baseline/macro reports.

No Git checkpoint commits were created because the user requested workspace
changes, not repository-history changes. RED and GREEN evidence is preserved
above.
