"""Tests for historical-price cache portability."""
from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.config import Config
from src.data.market_data import get_history


def _download_frame(days: int = 40) -> pd.DataFrame:
    end = pd.Timestamp(datetime.now(UTC).date())
    index = pd.date_range(end - pd.Timedelta(days=days), end, freq="B")
    return pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 1_000_000,
        },
        index=index,
    )


class MarketDataCacheTests(unittest.TestCase):
    def test_csv_cache_is_written_and_reused_when_parquet_engine_is_missing(self):
        with TemporaryDirectory() as tmp:
            cfg = Config(raw={"data": {"cache_dir": tmp}}, api_key="", api_secret="", is_live=False)
            download = _download_frame()

            with (
                patch("src.data.market_data.yf.download", return_value=download) as downloader,
                patch.object(pd.DataFrame, "to_parquet", side_effect=ImportError("no parquet engine")),
            ):
                first = get_history(cfg, "GLD", days=30)

            csv_path = Path(tmp) / "GLD.csv"
            self.assertTrue(csv_path.exists())
            self.assertFalse(first.empty)
            downloader.assert_called_once()

            # A stale Parquet file may remain from a different Python
            # environment. Once CSV fallback exists, do not try that file and
            # emit the same missing-engine warning on every run.
            (Path(tmp) / "GLD.parquet").write_text("stale")
            with (
                patch("src.data.market_data.yf.download") as downloader,
                patch("src.data.market_data.pd.read_parquet") as parquet_reader,
            ):
                second = get_history(cfg, "GLD", days=30)

            downloader.assert_not_called()
            parquet_reader.assert_not_called()
            pd.testing.assert_frame_equal(first, second, check_freq=False)

            with patch("src.data.market_data.yf.download") as downloader:
                partial = get_history(
                    cfg,
                    "GLD",
                    days=365,
                    allow_partial_cache=True,
                )

            downloader.assert_not_called()
            self.assertFalse(partial.empty)


if __name__ == "__main__":
    unittest.main()
