# trader-bot

Free-tier Alpaca trading bot that combines:

- **Classical quant signals** — momentum, mean-reversion, MA-cross, RSI, MACD
- **ML direction prediction** — XGBoost on engineered price/volume features
- **Politician trade tracker** — Senate/House STOCK Act disclosures (free S3 datasets)
- **Walk-forward backtester** — sanity-check strategies before risking capital

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
# 1. Backtest (no broker connection needed)
python scripts/backtest.py

# 2. Train the ML model on the configured universe
python scripts/train_models.py

# 3. Inspect recent politician disclosures
python scripts/politicians_analyze.py

# 4. Start the paper trading loop
python scripts/run_paper.py
```

(In Docker: prefix any of these with `docker compose run --rm trader`.)

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
  trader.py              main loop
scripts/                 entry points
tests/                   smoke tests (no network)
```

## Caveats

- yfinance is unofficial and may rate-limit; the data layer caches to `data_cache/`.
- Politician-disclosure feeds are community-maintained and may move; URLs are in `src/politicians/tracker.py`.
- Universe defaults to ~50 liquid NYSE names — "all NYSE" works in principle but invites rate-limiting on free APIs. Expand `src/data/nyse_universe.txt` at your own risk.
- This is a tool, not investment advice.
