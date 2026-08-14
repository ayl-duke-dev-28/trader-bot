from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config
from src.data.sectors import sector_for
from src.data.universe import load_universe


def _config(universe: dict[str, object]) -> Config:
    return Config(raw={"universe": universe}, api_key="", api_secret="", is_live=False)


def test_multiple_universe_files_are_merged_in_order_and_deduplicated(tmp_path: Path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("# first\naapl\nMSFT\n")
    second.write_text("MSFT\nJPM\n")

    cfg = _config({"source": "file", "file_paths": [str(first), str(second)], "max_symbols": 10})

    assert load_universe(cfg) == ["AAPL", "MSFT", "JPM"]


def test_combined_universe_applies_cap_after_deduplication(tmp_path: Path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("AAPL\nMSFT\n")
    second.write_text("AAPL\nJPM\nXOM\n")

    cfg = _config({"source": "file", "file_paths": [str(first), str(second)], "max_symbols": 3})

    assert load_universe(cfg) == ["AAPL", "MSFT", "JPM"]


def test_legacy_single_file_path_remains_supported(tmp_path: Path):
    universe_file = tmp_path / "symbols.txt"
    universe_file.write_text("QQQ\nNVDA\n")

    cfg = _config({"source": "file", "file_path": str(universe_file), "max_symbols": 10})

    assert load_universe(cfg) == ["QQQ", "NVDA"]


def test_csv_universe_reads_symbol_column(tmp_path: Path):
    universe_file = tmp_path / "symbols.csv"
    universe_file.write_text("Symbol,GICS Sector\naapl,Information Technology\nJPM,Financials\n")

    cfg = _config({"source": "file", "file_paths": [str(universe_file)], "max_symbols": 10})

    assert load_universe(cfg) == ["AAPL", "JPM"]


def test_default_config_loads_expanded_diversified_universe():
    symbols = load_universe(load_config(require_secrets=False))

    assert len(symbols) == 687
    assert {"AAPL", "JPM", "XOM", "LLY", "CAT", "QQQ", "APD", "AFL"}.issubset(symbols)
    assert sector_for("JPM") == "financials"
    assert sector_for("XOM") == "energy"
    assert sector_for("LLY") == "healthcare"
    assert sector_for("APD") == "materials"
    assert sector_for("AOS") == "industrials"
