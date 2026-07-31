# trader-bot

Long-only Alpaca trading bot with paper/live execution, scheduled intraday
checks, persistent risk controls, and live-path historical backtesting.

The repository is paper-first: `config.yaml` currently uses `mode: paper` and
`dry_run: false`. Live mode requires separate live Alpaca credentials and a
typed `YES` confirmation at startup.

## Effective strategy

The configured signal path is:

- **Hedge-fund ensemble** — enabled and therefore authoritative. It combines
  trend, mean-reversion, momentum, volatility-regime, statistical, and ML votes.
- **ML direction prediction** — enabled inside the ensemble. The configured
  XGBoost model at `models/xgb_direction.joblib` predicts the direction five
  trading sessions ahead and returns a neutral vote when its up-probability is
  between `0.45` and `0.55`.
- **Benchmark-aware risk layer** — enabled; keeps a `QQQ` core sleeve in
  risk-on regimes and requires individual names to beat `QQQ`.
- **Momentum breakout** — implemented but disabled. Enabling it replaces the
  ensemble output with the breakout ranking.
- **Classical weighted blend** — configured but bypassed while the hedge-fund
  ensemble is enabled. It becomes the fallback signal path if the ensemble is
  disabled.
- **Politician disclosures** — implemented but disabled. Their configured
  weight is only used by the fallback weighted blend.
- **Economic-cycle overlay** — enabled for paper/live trading and available in
  backtests. It combines growth, inflation, unemployment, and interest rates
  into separate long- and short-cycle scores and can only reduce gross exposure.
- **Dynamic sector-risk overlay** — enabled. Sector breadth versus the 50-day
  SMA and median 20-day realized volatility can reduce new single-name position
  sizes; static sector position-count caps remain a separate safeguard.
- **Historical portfolio VaR gate** — enabled for paper trading and backtests.
  It blocks projected buys when one-day 99% historical VaR exceeds 3% of equity
  or expected shortfall exceeds 4%; sells are never blocked.

## Current State

- Mode: `paper`
- Dry run: disabled
- Universe: `src/data/tech_universe.txt`, capped at `250` symbols
- Execution: whole-share buys; fractional shares disabled
- Position sizing: max `5%` per position, max `80%` gross exposure, max `20`
  positions
- Signal thresholds: enter at `0.55` or higher; exit when the score reaches
  `0.00` or lower
- Market regime filter: `QQQ` above/below its `200`-day SMA
- Benchmark core: `QQQ`, `50%` target in risk-on regimes
- Relative strength: enabled versus `QQQ` over `63` trading days
- Macro cycle: enabled; neutral and contraction regimes cap gross exposure at
  `60%` and `30%`, after the existing QQQ trend cap
- Sector risk: enabled; below-40% breadth halves new-position size, annualized
  sector volatility above 50% scales it down further, with a 25% multiplier floor
- Daily loss kill switch: `3%`
- Stops: ATR-scaled, floored at `4%`, capped at `12%`
- Trailing lock: arms at an `8%` gain and exits after a `4%` giveback
- Stop cooldown: `3` days
- Entry filters: skip buys within `3` days of earnings; gap protection at `5%`
- Quote recovery and safety: missing or invalid Yahoo prices are retried in one
  Alpaca IEX latest-trade request; unresolved, non-numeric, non-positive, `NaN`,
  and infinite prices are rejected before sizing without aborting the cycle
- Portfolio drawdown guard: implemented but disabled
- Portfolio VaR: enabled; one-day 99% historical simulation over 252 aligned
  returns, minimum 60 observations, 3% VaR and 4% expected-shortfall limits,
  fail-closed when an enabled pre-trade check lacks enough aligned history
- Schedule: weekdays from `09:30` through `15:30` ET, hourly

## Approved Next Experiment — volatility-targeted exposure

An approved design proposes replacing the fixed `80%` risk-on exposure ceiling
with a volatility-scaled target between `60%` and `95%`. The existing QQQ
`200`-day-SMA regime rule remains authoritative, including the `20%` risk-off
target. The goal is higher bull-market participation without materially worsening
the strategy's historical drawdown.

This feature is **designed but not implemented or enabled**. It must be built
behind a disabled configuration flag and pass pre-registered discovery,
withheld-period, stressed-cost, live/backtest-parity, and paper-trading gates.
Failure leaves the current `80%` risk-on / `20%` risk-off policy unchanged.

See [the approved volatility-targeting design](docs/design-volatility-targeted-exposure.md)
for the formula, order-accounting rules, safeguards, and acceptance criteria.

## Setup — native Python

Requires Python 3.11 or newer.

```bash
cd /path/to/trader-bot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# edit .env with your paper keys from https://app.alpaca.markets/paper/dashboard/overview
```

<!-- AUTO-GENERATED: environment variables from .env.example -->
| Variable | Required | Used for |
| --- | --- | --- |
| `ALPACA_API_KEY` | Yes | Paper trading and commands that load broker-backed configuration |
| `ALPACA_API_SECRET` | Yes | Paper trading and commands that load broker-backed configuration |
| `ALPACA_LIVE_API_KEY` | Live mode only | Live trading |
| `ALPACA_LIVE_API_SECRET` | Live mode only | Live trading |
| `FRED_API_KEY` | Macro download only | Initial-release economic data from FRED |
<!-- END AUTO-GENERATED -->

## Setup — Docker (recommended for deploy)

```bash
cd /path/to/trader-bot
cp .env.example .env       # then edit with your Alpaca paper keys
docker compose build
docker compose run --rm trader python scripts/train_models.py
docker compose up -d
docker compose logs -f
```

To deploy on a free Oracle Cloud VM see [docs/ORACLE_DEPLOY.md](docs/ORACLE_DEPLOY.md).

## Usage

<!-- AUTO-GENERATED: commands from scripts/ and tests/ -->
| Command | Description | Alpaca keys required |
| --- | --- | --- |
| `python scripts/backtest.py` | Run the primary live-path backtest; walk-forward ML is on by default | No |
| `python scripts/backtest_commodities.py` | Run the commodity ETF strategy in rolling four-month out-of-sample windows | No |
| `python scripts/fetch_macro_data.py` | Download point-in-time initial-release macro history from FRED | No Alpaca keys; requires `FRED_API_KEY` |
| `python scripts/simulate_backtest.py` | Run the simpler current-decision simulator without walk-forward retraining | No |
| `python scripts/stress_test.py` | Run deterministic offline market, data, cost, and macro stress scenarios | No |
| `python scripts/train_models.py` | Download the configured training history and write the XGBoost model | Yes |
| `python scripts/politicians_analyze.py` | Fetch and rank recent politician disclosures | Yes, because it loads the broker-backed configuration |
| `python scripts/run_paper.py` | Start the scheduled paper/live trading loop | Yes |
| `python tests/test_smoke.py` | Run the offline smoke tests | No |
<!-- END AUTO-GENERATED -->

(In Docker: prefix any of these with `docker compose run --rm trader`.)

The model file is ignored by Git. Train it before relying on the ML vote; if the
file is absent, ML contributes a neutral vote.

## Trading schedule

The live/paper loop uses fixed regular-session Eastern Time slots instead of
running relative to process start time. By default it runs on weekdays at:

```text
09:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30 ET
```

`15:30 ET` is the final cycle, 30 minutes before the normal `16:00 ET` close.
The schedule is configurable in `config.yaml`:

```yaml
schedule:
  market_timezone: America/New_York
  first_run_et: "09:30"
  last_run_et: "15:30"
  run_interval_minutes: 60
  prevent_system_sleep: true
  max_start_delay_seconds: 300
```

Each scheduled cycle still checks Alpaca's market clock before trading, so
holidays, weekends, and unexpected closures are skipped safely.

On macOS, the runner starts an idle-sleep assertion while it is active. Closing
the laptop lid or manually sleeping the Mac can still suspend it. If a wake-up
occurs more than five minutes after a slot, the stale cycle is skipped and the
next scheduled slot is used.

## Backtesting

`python scripts/backtest.py` is the primary trust check. By default it:

- fetches the configured universe plus warmup history;
- runs the currently configured strategy stack from `config.yaml`;
- replays the live-path risk rules: sizing, sector caps, stop/trailing exits,
  cooldowns, whole/fractional-share sizing, and trading costs.

When ML is enabled, the backtester does **not** use a single model trained on the
full dataset. It trains ML only on rolling prior windows and tests only the
immediately following window.

`scripts/backtest.py` options:

<!-- AUTO-GENERATED: options from scripts/backtest.py -->
| Option | Default | Purpose |
| --- | ---: | --- |
| `--years` | `5` | Simulation length in calendar years |
| `--start-capital` | `100000` | Starting cash |
| `--cost-bps` | `5` | Per-trade cost assumption in basis points |
| `--max-symbols` | all configured | Limit the universe for a faster run |
| `--out-dir` | timestamped directory | Select the report directory |
| `--train-window-days` | `756` | Override the walk-forward ML training window |
| `--test-window-days` | `63` | Override the walk-forward ML test window |
| `--no-walk-forward` | off | Reuse the saved model instead of retraining rolling prior windows |
| `--macro-data` | off | Enable the economic-cycle overlay with a point-in-time macro CSV |
<!-- END AUTO-GENERATED -->

For example:

```bash
python scripts/backtest.py --years 5 --out-dir reports/backtests/walk_forward_5y
python scripts/backtest.py --years 20 --out-dir reports/backtests/walk_forward_20y
python scripts/backtest.py --years 1 --max-symbols 50
```

### Commodity walk-forward strategy

`python scripts/backtest_commodities.py` is a separate research backtest. It
does not pass commodity funds through the tech bot's QQQ benchmark sleeve,
relative-strength rule, earnings blackout, or tech-sector caps.

The tradable universe spans precious metals (`GLD`, `SLV`, `PPLT`), energy
(`USO`, `BNO`, `UNG`), agriculture (`DBA`), and industrial metals (`DBB`,
`CPER`). The strategy:

- ranks positive absolute momentum over candidate 3-, 6-, and 12-month
  lookbacks;
- requires price to remain above its 200-session average;
- holds the strongest two to four funds, selected using only the preceding
  five-year training window;
- sizes positions by inverse trailing volatility, with a 40% position cap and
  15% annualized portfolio-volatility target;
- holds unused capital as cash and moves fully to cash when no fund qualifies;
- rebalances monthly and charges 10 basis points on every dollar of turnover;
  and
- freezes the selected parameters for the next four calendar months before
  retraining on the next rolling prior window.

The default command now compares that baseline with a diversified variant. The
variant selects five to nine qualifying funds, caps each position at 25%, and
caps each correlated commodity group at 35%. The group limits keep precious
metals, energy, industrial metals, or agriculture from dominating the risk
budget merely because several closely related funds have the same trend.

The default request uses 15.5 years of prices: five years for the first training
window followed by at least ten years of non-overlapping, four-month
out-of-sample windows. `DBC` is the commodity benchmark; `SPY`, `VOO`, `QQQ`,
and `QQQM` are reported as market context. Because `QQQM` began in October
2020, its output explicitly shows a shorter start date; `QQQ` supplies the
full-window Nasdaq-100 comparison.

```bash
python scripts/backtest_commodities.py

# Run only one portfolio construction method.
python scripts/backtest_commodities.py --variant diversified

# Reproduce from a local wide adjusted-close file. The first column is Date;
# remaining columns must include the nine strategy tickers above.
python scripts/backtest_commodities.py \
  --prices-csv /path/to/adjusted_closes.csv \
  --out-dir reports/backtests/commodities_walk_forward_10y
```

The comparison report writes `comparison.md`, `comparison.json`, the downloaded
`adjusted_closes.csv`, and separate `baseline/` and `diversified/` directories.
Each variant directory contains `summary.md`, `results.json`,
`walk_forward_windows.csv`, `equity_curve.csv`, `daily_returns.csv`, and
`weights.csv`. Each window records its isolated out-of-sample return, Sharpe,
and drawdown as well as the training-only parameters selected for it.

The historical data cache prefers Parquet. If the active Python environment
does not have `pyarrow` or `fastparquet`, it automatically writes and reuses a
CSV cache instead.

This tests exchange-traded commodity products, not spot commodities or direct
futures execution. Futures-based products can diverge materially from spot
returns because expiring contracts must be rolled; contango, backwardation,
fees, and fund construction remain embedded in the adjusted ETF price history.

### Economic-cycle overlay

Add a free FRED API key to `.env`, download initial-release observations, then
run the baseline and macro-aware variants against the same dates, universe, and
cost assumptions:

```bash
python scripts/fetch_macro_data.py

python scripts/backtest.py \
  --years 20 \
  --out-dir reports/backtests/baseline_20y

python scripts/backtest.py \
  --years 20 \
  --macro-data data_cache/macro/fred_initial_releases.csv \
  --out-dir reports/backtests/macro_cycles_20y
```

The same downloaded file is used by `scripts/run_paper.py`. Refresh it at least
monthly. If the file is missing, malformed, or more than 75 days old, the bot
logs a warning and retains the existing QQQ market-regime controls instead of
making a decision from stale macro data.

The four monthly inputs are:

- growth: year-over-year industrial production (`INDPRO`);
- inflation: year-over-year CPI (`CPIAUCSL`);
- unemployment: U-3 unemployment rate (`UNRATE`);
- interest rates: effective federal-funds rate (`FEDFUNDS`).

The long-cycle score emphasizes levels over a rolling 120-month context. The
short-cycle score measures six-month changes. In expansion the existing risk
cap remains authoritative; neutral and contraction regimes cap gross exposure
at `60%` and `30%` respectively. The QQQ market-regime cap can reduce exposure
further.

The downloader requests FRED's initial-release output and indexes every value by
its historical availability date. Do not substitute a latest-revision FRED CSV:
that would introduce revision and look-ahead bias. The equity report records
the daily macro regime, both cycle scores, the composite score, and the applied
gross-exposure cap.

### Dynamic sector risk

Before sizing a new equity position, the bot groups symbols using
`src/data/sectors.py` and measures each group across members with enough price
history:

- breadth: fraction of members trading at or above their 50-day SMA;
- risk: median annualized volatility from the latest 20 daily returns.

If fewer than three group members have usable history, the overlay is neutral.
Otherwise breadth below 40% applies a 0.50 size multiplier, and sector
volatility above 50% applies `50% / observed volatility`; the stricter result
wins, bounded below at 0.25. This affects new buys only. It does not enlarge
positions, liquidate holdings, replace VaR, or replace static sector caps. Buy
reasons record the multiplier, breadth, and sector volatility for auditability.

New backtest reports include daily P/L diagnostics in addition to total return,
Sharpe, and max drawdown:

- `profit_days`, `loss_days`, `flat_days`
- `loss_day_rate`
- `avg_loss_day_return`
- `worst_day_return`
- `historical_var_pct`, `expected_shortfall_pct`, and `var_observations` when
  the VaR gate is enabled
- `var_blocked_buys`, `max_historical_var_pct`, and
  `max_expected_shortfall_pct` in the summary

These are risk diagnostics, not an optimization guarantee. A strategy can have
zero losing days by staying in cash, but any active long-equity strategy should
expect some negative mark-to-market days.

Current saved strategy reports are limited to the retained benchmark and
walk-forward runs. Temporary parameter-comparison reports are not retained.

| Report | Period | Final equity | CAGR | Sharpe | Max drawdown | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `reports/backtests/benchmark_aware_5y/` | `2021-07-08` to `2026-07-07` | `$165,185.92` | `10.57%` | `0.8520` | `-20.32%` | Benchmark-core and relative-strength risk layers |
| `reports/backtests/walk_forward_5y/` | `2021-07-09` to `2026-07-09` | `$168,761.46` | `11.04%` | `0.8857` | `-21.26%` | Selected `0.55` entry threshold; 29 walk-forward ML windows |
| `reports/backtests/walk_forward_20y/` | `2006-07-10` to `2026-07-09` | `$511,336.91` | `8.50%` | `0.7187` | `-21.08%` | 116 walk-forward ML windows |
| `reports/backtests/daily_metrics_1y_qqq/` | `2025-07-14` to `2026-07-13` | `$130,358.84` | `30.48%` | `1.7966` | `-6.84%` | 1-year daily P/L diagnostic |

The remaining report files are `summary.txt`, `equity_curve.csv`, and
`trades.csv` for each report. `walk_forward_20y` also has `benchmarks.csv`.

## Portfolio VaR and expected shortfall

The bot applies a one-day historical-simulation risk gate to the whole projected
portfolio before submitting buys. For every aligned daily observation it
multiplies each symbol's return by that position's projected dollar exposure as
a fraction of account equity, then sums those contributions into a portfolio
return. VaR is the configured loss quantile; expected shortfall is the average
loss at or beyond that cutoff.

<!-- AUTO-GENERATED: value_at_risk settings from config.yaml -->
| Setting | Current value | Meaning |
| --- | ---: | --- |
| `enabled` | `true` | Apply the gate to paper/live sizing and backtests |
| `confidence` | `0.99` | Report the 99th-percentile one-day loss |
| `lookback_days` | `252` | Use up to one trading year of returns |
| `min_observations` | `60` | Require this many fully aligned portfolio returns |
| `max_var_pct` | `0.03` | Block a buy when projected VaR exceeds 3% of equity |
| `max_expected_shortfall_pct` | `0.04` | Block a buy when projected tail loss exceeds 4% of equity |
| `fail_closed` | `true` | Block buys when the enabled calculation lacks sufficient data |
<!-- END AUTO-GENERATED -->

The gate evaluates proposed buys sequentially, so an accepted buy becomes part
of the projected exposure used to assess the next candidate. Existing sell
intents reduce projected exposure first and are always preserved; VaR never
prevents the bot from reducing risk. A blocked order is written to
`logs/trader.log` with its projected VaR and expected shortfall, or with an
insufficient-history explanation.

Backtests use only returns available before each simulated trade decision. Their
daily equity diagnostics include `historical_var_pct`,
`expected_shortfall_pct`, and `var_observations`; summaries include the maximum
values and `var_blocked_buys`. This keeps live sizing and historical evaluation
on the same gate implementation in `src/risk/var.py`.

VaR is not a maximum-loss estimate. Because it is backward-looking, it cannot
anticipate a new flash crash before comparable losses enter the observation
window. Expected shortfall describes the severity of the observed tail, while
gap controls, stops, exposure caps, drawdown controls, and stress tests address
different failure modes.

## Stress testing

`python scripts/stress_test.py` runs entirely offline against deterministic
synthetic OHLCV data. It does not contact Yahoo, Alpaca, or submit orders. The
default suite replays the live-path simulator through:

- an ordinary correlated market baseline;
- a synchronized 32% flash crash;
- a prolonged 48% bear-market decline;
- repeated market-wide volatility spikes;
- missing OHLC observations for several symbols;
- extreme transaction costs of 100 basis points per trade; and
- a severe macro contraction with a 30% gross-exposure cap.

The saved ML model is included by default when present. Use `--no-model` to
isolate the deterministic non-ML strategy path. Each run writes `summary.md`,
`results.csv`, and `results.json` under a timestamped directory in
`reports/stress_tests/`.

```bash
python scripts/stress_test.py
python scripts/stress_test.py --no-model
python scripts/stress_test.py --scenario flash_crash --scenario volatility_spike
```

| Option | Default | Purpose |
| --- | ---: | --- |
| `--periods` | `520` | Number of synthetic business days, including warmup history |
| `--simulation-bars` | `252` | Number of measured trading days per scenario |
| `--seed` | `7` | Reproduce the same synthetic price paths |
| `--start-capital` | `100000` | Starting portfolio cash |
| `--runtime-budget` | `10` | Per-scenario warning threshold in seconds |
| `--no-model` | off | Disable the saved ML vote |
| `--scenario` | all | Run one named scenario; repeat the option to select several |
| `--out-dir` | timestamped directory | Select the report directory |

`PASS` means accounting values remained finite, equity and cash stayed valid,
and position/exposure caps held. `WARN` means the bot stayed operational but
crossed the default 35% drawdown, 20% worst-day, or 10-second runtime budget.
`FAIL` means an exception, insolvency, non-finite accounting value, negative
cash, or a configured cap violation. These are controlled behavior tests, not
return forecasts or substitutes for historical out-of-sample backtests.

The retained VaR-enabled 2026-07-31 run passed every software safety invariant.
In the volatility-spike scenario the gate blocked 64 projected buys and reduced
maximum drawdown from `58.39%` in the original run to `31.93%`. It did not reduce
the instantaneous flash-crash loss because a backward-looking estimate cannot
anticipate a shock absent from its return window. The 100 bps cost scenario still
lost `27.05%`, indicating substantial turnover sensitivity. See
`reports/stress_tests/2026-07-31-var/summary.md` for the complete scenario table.

## Current Files

Important tracked files:

- `config.yaml` — current strategy, risk, universe, schedule, and logging config
- `requirements.txt` — Python dependencies
- `Dockerfile` and `docker-compose.yml` — containerized runner
- `docs/ORACLE_DEPLOY.md` — Oracle VM deployment notes
- `docs/RESEARCH_COCKPIT_DESIGN.md` — research UI/design notes
- `docs/design-volatility-targeted-exposure.md` — approved, not-yet-implemented
  volatility-targeting experiment
- `scripts/backtest.py` — live-path historical backtester
- `scripts/backtest_commodities.py` — four-month commodity walk-forward runner
- `scripts/fetch_macro_data.py` — downloads initial-release FRED macro history
- `scripts/simulate_backtest.py` — simulation report runner
- `scripts/stress_test.py` — deterministic offline stress-suite runner
- `scripts/train_models.py` — trains `models/xgb_direction.joblib`
- `scripts/run_paper.py` — starts the scheduled paper/live loop
- `scripts/politicians_analyze.py` — inspects disclosure feeds

Important local/generated files:

- `.env` — local Alpaca credentials; ignored by git
- `logs/trader.log` — runtime log; ignored by git
- `logs/trades.xlsx` — trade activity workbook; ignored by git
- `models/xgb_direction.joblib` — trained ML artifact; ignored by git
- `data_cache/` — yfinance cache plus persisted risk state; ignored by git
- `.venv/` — local Python environment; ignored by git

## Trade activity log

Broker-facing buys, sells, stop-loss closes, dry-run intents, failures, and
selected execution skips are appended to an Excel file so you can review why an
order was or was not submitted.

- Default path: `logs/trades.xlsx` (configurable via `logging.trades_file` in `config.yaml`).
- Columns: `timestamp, mode, action, symbol, qty, price, target_dollars, score, reason, order_id`.
- Actions: `BUY`, `SELL`, `STOP` (stop-loss / trailing lock), `SKIP`, `DRY`, `FAIL`.
- The `reason` column carries the exact signal/sizing/stop trigger (e.g. `score=+0.42 sector=tech`, `stop pl=-6.20% vs -4.00%`).
- The file is created on the first logged action — until then it won't exist on disk.

Market-closed cycles, earnings-blackout exclusions, and cycles with no intents
are written to `logs/trader.log`, not to the Excel activity log. Yahoo quote
misses recovered from Alpaca IEX are summarized there at `INFO`; unresolved
invalid quotes are logged at `WARNING`. An unresolved quote does not produce an
order or Excel activity row because it is rejected before order planning.

## Going live

1. Add `ALPACA_LIVE_API_KEY` / `ALPACA_LIVE_API_SECRET` to `.env`.
2. Set `mode: live` in `config.yaml`.
3. Run `python scripts/run_paper.py` — it prompts for a typed `YES` before submitting any order.

**Read `config.yaml` end-to-end before going live.** The defaults are conservative but you own the financial risk.

## Layout

```
.env                     local secrets and Alpaca keys (ignored)
.env.example             environment template
config.yaml              current mode, universe, strategies, risk, schedule
requirements.txt         Python dependencies
Dockerfile               container image
docker-compose.yml       container runner
docs/
  ORACLE_DEPLOY.md       Oracle Cloud VM deployment notes
  RESEARCH_COCKPIT_DESIGN.md
                          research cockpit design notes
  design-volatility-targeted-exposure.md
                          approved volatility-targeting experiment design
models/
  xgb_direction.joblib   trained ML model artifact (ignored)
data_cache/
  state/risk_state.json  day equity, stop cooldowns, and high-water marks (ignored)
reports/backtests/
  benchmark_aware_5y/    current-strategy 5-year benchmark-aware run
  walk_forward_5y/       current-strategy 5-year walk-forward run
  walk_forward_20y/      current-strategy 20-year walk-forward run
  daily_metrics_1y_qqq/  current-strategy 1-year daily metrics run
logs/
  trader.log             runtime log (ignored)
  trades.xlsx            activity log workbook (ignored)
src/
  config.py              .env + yaml loader
  broker/alpaca_client.py
                          alpaca-py wrapper
  data/                  universe + cached yfinance fetcher
  data/macro.py          point-in-time macro panel + cycle scoring
  signals/classical.py   technical-analysis composite signal
  signals/hedge_fund.py  current ensemble scoring path
  signals/ml.py          XGBoost direction model
  signals/momentum_breakout.py
                          disabled breakout strategy implementation
  politicians/tracker.py STOCK Act feeds -> per-symbol signal
  risk/manager.py        sizing, kill switch, stop-losses
  risk/sector.py         dynamic sector breadth/volatility sizing overlay
  risk/state.py          persisted risk state helpers
  risk/validation.py     shared finite-positive price validation
  risk/var.py            historical portfolio VaR and expected-shortfall gate
  backtest/engine.py     walk-forward backtester
  backtest/commodities.py
                          commodity ETF strategy and walk-forward evaluator
  backtest/simulator.py  live-path historical simulator
  trade_log.py           Excel activity log writer
  trader.py              main loop
scripts/                 entry points
tests/                   smoke tests (no network)
```

## Caveats

- yfinance is unofficial and may rate-limit; the data layer caches to `data_cache/`.
- Per-symbol yfinance prices that are missing or invalid are recovered through a
  single Alpaca IEX latest-trade request for only the affected symbols. Any
  unresolved, non-numeric, non-positive, `NaN`, or infinite price is logged and
  skipped. The same finite-positive validation is repeated during risk planning
  and immediately before execution.
- Read-only Alpaca calls (account, positions, clock, open orders, latest trades)
  retry on transient network errors. Order submission does **not** retry: a reset
  mid-submit leaves the order's fate unknown, and a blind retry risks duplicating
  a filled order. Those failures are logged as `FAIL` in the trade log and left
  for the next cycle.
- A serialized XGBoost model may warn when loaded by a different XGBoost version.
  Retrain with `python scripts/train_models.py` after dependency upgrades rather
  than relying on cross-version pickle compatibility.
- Historical VaR is a backward-looking loss quantile, not a maximum-loss bound.
  It can react only after stressed returns enter its observation window and may
  understate regime shifts, liquidity gaps, and unprecedented shocks. Expected
  shortfall and the separate stress suite are retained because VaR alone does
  not describe the severity of losses beyond its cutoff.
- Politician-disclosure feeds are community-maintained and may move; URLs are in `src/politicians/tracker.py`.
- Universe defaults to a curated tech-heavy list from `src/data/tech_universe.txt`. Broad universes work in principle but invite rate-limiting on free APIs.
- Backtests use today's configured universe and available historical data, so old periods exclude symbols that did not yet have enough history.
- This is a tool, not investment advice.
