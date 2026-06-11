"""Universe loading. v1 supports a static file; sp500_nyse option can be added later."""
from __future__ import annotations

from pathlib import Path

from src.config import Config, ROOT


def load_universe(cfg: Config) -> list[str]:
    source = cfg.get("universe", "source", default="file")
    max_n = int(cfg.get("universe", "max_symbols", default=50))
    if source != "file":
        raise NotImplementedError(f"universe.source={source} not supported yet")

    path = Path(cfg.get("universe", "file_path", default="src/data/nyse_universe.txt"))
    if not path.is_absolute():
        path = ROOT / path

    seen: set[str] = set()
    symbols: list[str] = []
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
            break
    return symbols
