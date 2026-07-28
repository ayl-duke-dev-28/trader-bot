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
- **Economic-cycle overlay** — implemented for backtests but disabled by
  default. It combines growth, inflation, unemployment, and interest rates into
  separate long- and short-cycle scores and can only reduce gross exposure.

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
- Daily loss kill switch: `3%`
- Stops: ATR-scaled, floored at `4%`, capped at `12%`
- Trailing lock: arms at an `8%` gain and exits after a `4%` giveback
- Stop cooldown: `3` days
- Entry filters: skip buys within `3` days of earnings; gap protection at `5%`
- Portfolio drawdown guard: implemented but disabled
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
| `python scripts/fetch_macro_data.py` | Download point-in-time initial-release macro history from FRED | No Alpaca keys; requires `FRED_API_KEY` |
| `python scripts/simulate_backtest.py` | Run the simpler current-decision simulator without walk-forward retraining | No |
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

New backtest reports include daily P/L diagnostics in addition to total return,
Sharpe, and max drawdown:

- `profit_days`, `loss_days`, `flat_days`
- `loss_day_rate`
- `avg_loss_day_return`
- `worst_day_return`

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
- `scripts/fetch_macro_data.py` — downloads initial-release FRED macro history
- `scripts/simulate_backtest.py` — simulation report runner
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
are written to `logs/trader.log`, not to the Excel activity log.

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
  risk/state.py          persisted risk state helpers
  backtest/engine.py     walk-forward backtester
  backtest/simulator.py  live-path historical simulator
  trade_log.py           Excel activity log writer
  trader.py              main loop
scripts/                 entry points
tests/                   smoke tests (no network)
```

## Caveats

- yfinance is unofficial and may rate-limit; the data layer caches to `data_cache/`.
- Read-only Alpaca calls (account, positions, clock, open orders) retry on transient
  network errors. Order submission does **not** retry: a reset mid-submit leaves the
  order's fate unknown, and a blind retry risks duplicating a filled order. Those
  failures are logged as `FAIL` in the trade log and left for the next cycle.
- Politician-disclosure feeds are community-maintained and may move; URLs are in `src/politicians/tracker.py`.
- Universe defaults to a curated tech-heavy list from `src/data/tech_universe.txt`. Broad universes work in principle but invite rate-limiting on free APIs.
- Backtests use today's configured universe and available historical data, so old periods exclude symbols that did not yet have enough history.
- This is a tool, not investment advice.
