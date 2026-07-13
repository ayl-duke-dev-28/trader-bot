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
- runs the configured sparse `momentum_breakout` strategy;
- ranks symbols by prior 252-day return;
- buys only the top qualifying symbol when it is above its 100-day SMA, up at
  least `300%` over the lookback, below the volatility cap, and `QQQ` is above
  its 200-day SMA;
- applies a portfolio drawdown circuit breaker that liquidates and blocks new
  buys after a configured account-level drawdown;
- replays the live-path risk rules: sizing, sector caps, stop/trailing exits,
  cooldowns, whole/fractional-share sizing, and trading costs.

When ML is re-enabled, the backtester does **not** use a single model trained on
the full dataset. It trains ML only on rolling prior windows and tests only the
immediately following window.

Useful options:

```bash
python scripts/backtest.py --years 20 --out-dir reports/backtests/momentum_breakout_portfolio_guard_55_20y
python scripts/backtest.py --years 5 --train-window-days 756 --test-window-days 63
python scripts/backtest.py --years 1 --max-symbols 25
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

Latest saved 20-year momentum-breakout run with portfolio guard:

- Report: `reports/backtests/momentum_breakout_portfolio_guard_55_20y/`
- Period: `2006-07-13` to `2026-07-13`
- Start capital: `$100,000`
- Trading cost: `5 bps`
- Portfolio guard: liquidate and block new buys after `55%` account drawdown
- Benchmark comparison: `reports/backtests/momentum_breakout_portfolio_guard_55_20y/benchmarks.csv`

| Strategy / Benchmark | Final equity | Total return | CAGR | Loss days | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trader bot | `$2,450,743.00` | `2350.74%` | `17.35%` | `9.54%` | `-57.46%` |
| Dow proxy (`DIA`) | `$753,715.90` | `653.72%` | `10.63%` | `44.82%` | `-51.87%` |
| S&P 500 proxy (`SPY`) | `$872,409.40` | `772.41%` | `11.44%` | `44.54%` | `-55.19%` |
| Nasdaq-100 proxy (`QQQ`) | `$2,292,458.90` | `2192.46%` | `16.95%` | `43.83%` | `-53.40%` |

The guarded bot still beat the strongest benchmark (`QQQ`) by about `6.90%` on
final equity and kept loss days below `20%`. It no longer clears the prior
`QQQ + 10%` target, but it reduced max drawdown from the raw breakout run's
`-68.64%` to `-57.46%` and cut loss days from `15.17%` to `9.54%`.

Bot trade stats for that run:

- Trades: `250`
- Buys / sells / stops: `125 / 123 / 2`
- Closed win rate: `48.00%`
- Loss days: `480` of `5,029` return days (`9.54%`)
- Average loss day: `-2.50%`
- Worst day: `-14.51%`
- Symbols tested: `233`

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
- Politician-disclosure feeds are community-maintained and may move; URLs are in `src/politicians/tracker.py`.
- Universe defaults to a curated tech-heavy list from `src/data/tech_universe.txt`. Broad universes work in principle but invite rate-limiting on free APIs.
- Backtests use today's configured universe and available historical data, so old periods exclude symbols that did not yet have enough history.
- This is a tool, not investment advice.
