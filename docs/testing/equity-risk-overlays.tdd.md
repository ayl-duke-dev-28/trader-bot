# Equity macro and sector-risk overlays — TDD evidence

## Source and user journeys

No external plan file was supplied. The journeys were derived from the request:

- As a paper/live trader, I want current point-in-time macro conditions to cap
  equity exposure so that contractions do not receive normal risk-on sizing.
- As a trader, I want weak or unusually volatile industry groups to receive
  smaller new positions so that correlated sector risk is reduced.
- As a backtest user, I want the historical simulator to apply the same sector
  sizing formula as the paper/live path.

## RED and GREEN report

| Behavior | RED evidence | GREEN evidence | Guarantee |
|---|---|---|---|
| Sector breadth/volatility model and live macro cap | `PYTHONPATH=.venv/lib/python3.14/site-packages python3 -m pytest tests/test_equity_risk_overlays.py -q` failed during collection with `ModuleNotFoundError: src.risk.sector` | The same target passed: `6 passed` | Weak breadth and elevated volatility only reduce sizing; insufficient membership is neutral; contraction caps gross exposure without overriding a stricter market cap. |
| Live macro handoff | `... pytest tests/test_smoke.py::test_trade_once_passes_alpaca_price_fallback_to_quote_loader -q` failed because `_load_live_macro_cycles` did not exist | The focused equity and live-handoff target passed: `5 passed` | The trading loop supplies loaded macro cycles to `RiskManager.size_orders`. |
| Regression suite | N/A | `PYTHONPATH=.venv/lib/python3.14/site-packages python3 -m pytest -q` passed: `57 passed` | Existing signal, stop, VaR, cache, stress, commodity, and backtest behavior remains green. |

## Test specification

| # | What is guaranteed | Test | Type | Result |
|---|---|---|---|---|
| 1 | A sector below minimum breadth receives the configured 0.50 multiplier | `test_sector_risk_combines_breadth_and_realized_volatility` | Unit | PASS |
| 2 | Sector volatility above its ceiling reduces sizing proportionally | `test_sector_risk_combines_breadth_and_realized_volatility` | Unit | PASS |
| 3 | Too few valid sector members results in a neutral overlay | `test_sector_risk_is_neutral_when_too_few_members_have_history` | Unit | PASS |
| 4 | Macro contraction caps 80% gross at 30%, but cannot raise a 20% QQQ cap | `test_live_risk_manager_caps_gross_exposure_for_macro_contraction` | Unit | PASS |
| 5 | The live buy target and audit reason include the sector multiplier | `test_live_buy_size_is_reduced_in_a_weak_sector` | Integration | PASS |
| 6 | A fresh point-in-time macro cache is converted to cycle regimes | `test_live_macro_loader_builds_cycles_from_fresh_point_in_time_cache` | Integration | PASS |
| 7 | A stale macro cache is rejected | `test_live_macro_loader_rejects_stale_cache` | Error path | PASS |
| 8 | The paper/live cycle passes macro regimes into risk sizing | `test_trade_once_passes_alpaca_price_fallback_to_quote_loader` | Integration | PASS |

## Coverage and known gaps

`src/risk/sector.py` measured 91% statement coverage. The existing macro module
measured 88% in the focused macro/smoke run. Full regression passed with two
pre-existing warnings: a websockets deprecation and the serialized XGBoost
version warning.

The FRED downloader remains a separately invoked network operation and was not
called by tests. The paper/live process deliberately falls back to the existing
QQQ regime rule when its local macro file is absent, malformed, or stale. No
return improvement is claimed; the new factors are risk-reduction overlays and
should be compared in a new walk-forward equity run before thresholds are made
more aggressive.

## Checkpoints

- RED checkpoint commit: `3bb53c3 test: specify equity macro and sector risk overlays`.
- A later GREEN checkpoint could not be written because this session's managed
  filesystem made `.git/index.lock` read-only. The production changes and test
  evidence remain present in the worktree.
