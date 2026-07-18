# trader-bot

Free-tier Alpaca trading bot that combines:

- **Classical quant signals** — momentum, mean-reversion, MA-cross, RSI, MACD
- **ML direction prediction** — XGBoost on engineered price/volume features
- **Politician trade tracker** — Senate/House STOCK Act disclosures (free S3 datasets)
- **Walk-forward backtester** — replays the live trading rules and retrains ML on rolling prior windows before each test window

Built **paper-first, live-ready**: a single config flag flips to live, gated by extra env vars, a typed `YES` confirmation, a daily-loss kill switch, per-trade stop-losses, and exposure caps.

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

- `reports/backtests/benchmark_aware_5y/` — 5-year run with benchmark-core and
  relative-strength risk layers; final equity `$165,185.92`, CAGR `10.57%`,
  max drawdown `-20.32%`.
- `reports/backtests/walk_forward_5y/` — 5-year walk-forward ML run; final
  equity `$168,761.46`, CAGR `11.04%`, max drawdown `-21.26%`.
- `reports/backtests/walk_forward_20y/` — 20-year walk-forward ML run; final
  equity `$511,336.91`, CAGR `8.50%`, max drawdown `-21.08%`.
- `reports/backtests/daily_metrics_1y_qqq/` — 1-year current-strategy diagnostic;
  final equity `$130,358.84`, CAGR `30.48%`, max drawdown `-6.84%`.

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
config.yaml              tunables (universe, weights, risk caps)
src/
  config.py              .env + yaml loader
  broker/alpaca_client.py  alpaca-py wrapper
  data/                  universe + cached yfinance fetcher
  signals/classical.py   technical-analysis composite signal
  signals/ml.py          XGBoost direction model
  politicians/tracker.py STOCK Act feeds -> per-symbol signal
  risk/manager.py        sizing, kill switch, stop-losses
  backtest/engine.py     walk-forward backtester
  backtest/simulator.py  live-path historical simulator
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
