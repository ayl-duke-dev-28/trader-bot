"""ML direction-prediction signal.

Trains an XGBoost classifier per-symbol-pooled to predict P(next-day return > 0).
Output signal range: [-1, 1] where +1 = very confident up.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config import Config, ROOT

log = logging.getLogger(__name__)

DEFAULT_HORIZON_DAYS = 5

FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "vol_5", "vol_20",
    "rsi_14",
    "ma_ratio_20_50",
    "volume_ratio_20",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    vol = df["volume"]
    out = pd.DataFrame(index=df.index)
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["ret_20"] = close.pct_change(20)
    out["vol_5"] = close.pct_change().rolling(5).std()
    out["vol_20"] = close.pct_change().rolling(20).std()
    out["rsi_14"] = _rsi(close, 14) / 100.0
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    out["ma_ratio_20_50"] = ma20 / ma50 - 1.0
    out["volume_ratio_20"] = vol / vol.rolling(20).mean()
    return out


def build_training_set(
    hist: dict[str, pd.DataFrame],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Target is P(close_{t+horizon} > close_t). Longer horizon has better signal-to-noise."""
    rows = []
    for sym, df in hist.items():
        if len(df) < 80 + horizon_days:
            continue
        feats = build_features(df)
        future_close = df["close"].shift(-horizon_days)
        target = (future_close > df["close"]).astype("float")
        target[future_close.isna()] = np.nan
        joined = feats.join(target.rename("y")).dropna()
        if joined.empty:
            continue
        joined = joined.assign(_date=pd.to_datetime(joined.index), _symbol=sym)
        rows.append(joined)
    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int)
    data = pd.concat(rows).sort_values(["_date", "_symbol"])
    return data[FEATURES], data["y"].astype(int)


def train_model(
    hist: dict[str, pd.DataFrame],
    model_path: Path,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> XGBClassifier | None:
    bundle = train_model_bundle(hist, horizon_days=horizon_days)
    if bundle is None:
        return None

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    log.info("saved model -> %s (horizon=%d days)", model_path, horizon_days)
    return bundle["model"]


def train_model_bundle(
    hist: dict[str, pd.DataFrame],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> dict[str, Any] | None:
    """Train an in-memory model bundle for backtests or later persistence."""
    X, y = build_training_set(hist, horizon_days=horizon_days)
    if X.empty:
        log.error("no training data assembled")
        return None
    if y.nunique() < 2:
        log.error("training data has only one target class")
        return None
    # Hold out last 20% chronologically for a sanity check
    cut = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y.iloc[:cut], y.iloc[cut:]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_tr, y_tr)

    if not X_te.empty:
        acc = (model.predict(X_te) == y_te).mean()
        log.info("ML holdout accuracy: %.3f on n=%d", acc, len(X_te))

    return {"model": model, "features": FEATURES, "horizon_days": horizon_days}


def load_model(cfg: Config):
    path = Path(cfg.get("strategies", "ml", "model_path", default="models/xgb_direction.joblib"))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    return joblib.load(path)


def ml_signal(cfg: Config, df: pd.DataFrame, bundle=None) -> float:
    """Return signal in [-1, 1] based on P(up) - 0.5, gated by min_probability."""
    bundle = bundle or load_model(cfg)
    if bundle is None or df is None or df.empty:
        return 0.0
    feats = build_features(df).iloc[[-1]].dropna()
    if feats.empty:
        return 0.0
    proba_up = float(bundle["model"].predict_proba(feats[bundle["features"]])[0, 1])
    threshold = float(cfg.get("strategies", "ml", "min_probability", default=0.55))
    if abs(proba_up - 0.5) < (threshold - 0.5):
        return 0.0
    # Map [0,1] -> [-1, 1]
    return float(np.clip((proba_up - 0.5) * 2, -1.0, 1.0))
