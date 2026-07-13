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

`python scripts/backtest.py` is the primary trust check. It does **not** use a
single model trained on the full dataset. By default it:

- fetches the configured universe plus warmup history;
- trains ML only on a rolling prior window (`756` calendar days by default);
- tests only the immediately following window (`63` calendar days by default);
- slides forward through time and repeats;
- replays the live-path risk rules: benchmark core sleeve, regime filter,
  relative strength, sector caps, gap skips, stop/trailing exits, cooldowns,
  whole/fractional-share sizing, and trading costs.

Useful options:

```bash
python scripts/backtest.py --years 20 --out-dir reports/backtests/walk_forward_20y
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

Latest saved 20-year walk-forward run:

- Report: `reports/backtests/walk_forward_20y/`
- Period: `2006-07-10` to `2026-07-09`
- Start capital: `$100,000`
- Windows: `116` walk-forward windows, `756d` train / `63d` test
- Trading cost: `5 bps`

| Strategy / Benchmark | Final equity | Total return | CAGR | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trader bot | `$511,336.91` | `411.34%` | `8.50%` | `0.7187` | `-21.08%` |
| Dow proxy (`DIA`) | `$734,096.11` | `634.10%` | `10.48%` | `0.6324` | `-51.87%` |
| S&P 500 proxy (`SPY`) | `$855,802.69` | `755.80%` | `11.33%` | `0.6508` | `-55.19%` |
| Nasdaq-100 proxy (`QQQ`) | `$2,264,156.07` | `2164.16%` | `16.88%` | `0.8185` | `-53.40%` |

The bot's 20-year run had materially lower drawdown than the benchmarks, but
lower final equity and CAGR. It beat `DIA` and `SPY` on Sharpe, but not `QQQ`.

Bot trade stats for that run:

- Trades: `6,699`
- Buys / sells / stops: `3,543 / 1,785 / 1,371`
- Closed win rate: `75.63%`
- Loss days: `2,172` of `5,030` return days (`43.18%`)
- Average loss day: `-0.586%`
- Worst day: `-5.88%`
- Symbols tested: `227`

Representative 1-year daily-loss check including the `QQQ` benchmark core:

- Report: `reports/backtests/daily_metrics_1y_qqq/`
- Period: `2025-07-14` to `2026-07-13`
- Total return: `30.36%`
- Max drawdown: `-6.84%`
- Loss days: `103` of `250` return days (`41.20%`)
- Average loss day: `-0.78%`
- Worst day: `-3.67%`

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
