# trader-bot

Free-tier Alpaca trading bot for paper/live Alpaca accounts.

Current configured strategy:

- **Hedge-fund ensemble** — enabled; multi-analyst scoring path.
- **Classical quant signals** — enabled; momentum, mean-reversion, MA-cross,
  RSI, and MACD.
- **ML direction prediction** — enabled; XGBoost model at
  `models/xgb_direction.joblib`.
- **Benchmark-aware risk layer** — enabled; keeps a `QQQ` core sleeve in
  risk-on regimes and requires individual names to beat `QQQ`.
- **Momentum breakout** — present in code but disabled in `config.yaml`.
- **Politician trade tracker** — present in code but disabled in `config.yaml`.

Built **paper-first, live-ready**: `mode: paper` is the current default, while
live trading requires live Alpaca keys, `mode: live`, and a typed `YES`
confirmation before orders are submitted.

## Current State

- Mode: `paper`
- Universe: `src/data/tech_universe.txt`, capped at `250` symbols
- Position sizing: max `5%` per position, max `80%` gross exposure, max `20`
  positions
- Market regime filter: `QQQ` above/below its `200`-day SMA
- Benchmark core: `QQQ`, `50%` target in risk-on regimes
- Relative strength: enabled versus `QQQ` over `63` trading days
- Daily loss kill switch: `3%`
- Stops: ATR-scaled, floored at `4%`, capped at `12%`
- Schedule: weekdays from `09:30` through `15:30` ET, hourly

## Setup — native Python

```bash
cd ~/Documents/Projects/trader-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your paper keys from https://app.alpaca.markets/paper/dashboard/overview
```

## Setup — Docker (recommended for deploy)

```bash
cd ~/Documents/Projects/trader-bot
cp .env.example .env       # then edit with your Alpaca paper keys
docker compose build
docker compose run --rm trader python scripts/train_models.py
docker compose up -d
docker compose logs -f
```

To deploy on a free Oracle Cloud VM see [docs/ORACLE_DEPLOY.md](docs/ORACLE_DEPLOY.md).

## Usage

```bash
# 1. Walk-forward backtest (no broker connection needed)
python scripts/backtest.py

# 2. Train the ML model on the configured universe
python scripts/train_models.py

# 3. Inspect recent politician disclosures
python scripts/politicians_analyze.py

# 4. Start the paper trading loop
python scripts/run_paper.py
```

(In Docker: prefix any of these with `docker compose run --rm trader`.)

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
```

Each scheduled cycle still checks Alpaca's market clock before trading, so
holidays, weekends, and unexpected closures are skipped safely.

## Backtesting

`python scripts/backtest.py` is the primary trust check. By default it:

- fetches the configured universe plus warmup history;
- runs the currently configured strategy stack from `config.yaml`;
- replays the live-path risk rules: sizing, sector caps, stop/trailing exits,
  cooldowns, whole/fractional-share sizing, and trading costs.

When ML is enabled, the backtester does **not** use a single model trained on the
full dataset. It trains ML only on rolling prior windows and tests only the
immediately following window.

Useful options:

```bash
python scripts/backtest.py --years 5 --out-dir reports/backtests/walk_forward_5y
python scripts/backtest.py --years 20 --out-dir reports/backtests/walk_forward_20y
python scripts/backtest.py --years 1 --out-dir reports/backtests/daily_metrics_1y_qqq
```

Backtest reports include daily P/L diagnostics in addition to total return,
Sharpe, and max drawdown:

- `profit_days`, `loss_days`, `flat_days`
- `loss_day_rate`
- `avg_loss_day_return`
- `worst_day_return`

These are risk diagnostics, not an optimization guarantee. A strategy can have
zero losing days by staying in cash, but any active long-equity strategy should
expect some negative mark-to-market days.

Current saved strategy reports:

| Report | Period | Final equity | CAGR | Sharpe | Max drawdown | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `reports/backtests/benchmark_aware_5y/` | `2021-07-08` to `2026-07-07` | `$165,185.92` | `10.57%` | `0.8520` | `-20.32%` | Benchmark-core and relative-strength risk layers |
| `reports/backtests/walk_forward_5y/` | `2021-07-09` to `2026-07-09` | `$168,761.46` | `11.04%` | `0.8857` | `-21.26%` | 29 walk-forward ML windows |
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
- `scripts/backtest.py` — live-path historical backtester
- `scripts/simulate_backtest.py` — simulation report runner
- `scripts/train_models.py` — trains `models/xgb_direction.joblib`
- `scripts/run_paper.py` — starts the scheduled paper/live loop
- `scripts/politicians_analyze.py` — inspects disclosure feeds

Important local/generated files:

- `.env` — local Alpaca credentials; ignored by git
- `logs/trader.log` — runtime log; ignored by git
- `logs/trades.xlsx` — trade activity workbook; ignored by git
- `models/xgb_direction.joblib` — trained ML artifact; ignored by git
- `data_cache/` — recreated on demand by yfinance fetches; ignored by git
- `.venv/` — local Python environment; ignored by git and currently absent after cleanup

## Trade activity log

Every buy, sell, stop-loss close, skip, and dry-run intent is appended to an Excel
file so you can review *why* each trade was made after the fact.

- Default path: `logs/trades.xlsx` (configurable via `logging.trades_file` in `config.yaml`).
- Columns: `timestamp, mode, action, symbol, qty, price, target_dollars, score, reason, order_id`.
- Actions: `BUY`, `SELL`, `STOP` (stop-loss / trailing lock), `SKIP`, `DRY`, `FAIL`.
- The `reason` column carries the exact signal/sizing/stop trigger (e.g. `score=+0.42 sector=tech`, `stop pl=-6.20% vs -4.00%`).
- The file is created on the first logged action — until then it won't exist on disk.

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
models/
  xgb_direction.joblib   trained ML model artifact (ignored)
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
