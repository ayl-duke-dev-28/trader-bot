"""Load a deterministic trading universe from one or more static symbol files."""
from __future__ import annotations

from pathlib import Path

from src.config import Config, ROOT


def load_universe(cfg: Config) -> list[str]:
    source = cfg.get("universe", "source", default="file")
    max_n = int(cfg.get("universe", "max_symbols", default=50))
    if source != "file":
        raise NotImplementedError(f"universe.source={source} not supported yet")

    configured_paths = cfg.get("universe", "file_paths", default=None)
    if configured_paths is None:
        configured_paths = [
            cfg.get("universe", "file_path", default="src/data/nyse_universe.txt")
        ]
    if isinstance(configured_paths, (str, Path)):
        configured_paths = [configured_paths]
    if not configured_paths:
        raise ValueError("universe.file_paths must contain at least one file")

    seen: set[str] = set()
    symbols: list[str] = []
    for configured_path in configured_paths:
        path = Path(configured_path)
        if not path.is_absolute():
            path = ROOT / path

        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sym = line.upper()
            if sym in seen:
                continue
            seen.add(sym)
            symbols.append(sym)
            if len(symbols) >= max_n:
                return symbols
    return symbols
