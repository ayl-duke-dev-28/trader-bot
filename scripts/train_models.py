"""Train the ML direction model on the configured universe."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_config
from src.data.market_data import get_history_many
from src.data.universe import load_universe
from src.signals.ml import train_model

log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    symbols = load_universe(cfg)
    log.info("downloading history for %d symbols...", len(symbols))
    hist = get_history_many(cfg, symbols)
    log.info("got history for %d symbols", len(hist))

    model_path = Path(cfg.get("strategies", "ml", "model_path", default="models/xgb_direction.joblib"))
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    train_model(hist, model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
