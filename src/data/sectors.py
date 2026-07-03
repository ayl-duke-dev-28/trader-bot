"""Sector classification for the tech-focused universe.

Bucket names must match keys under `risk.sector_caps` in config.yaml.
Names not listed fall through to `other`.
"""
from __future__ import annotations

SECTOR_MAP: dict[str, str] = {
    # Mega-cap tech: move with QQQ / broad market.
    "AAPL": "mega_cap_tech",
    "MSFT": "mega_cap_tech",
    "GOOGL": "mega_cap_tech",
    "GOOG": "mega_cap_tech",
    "AMZN": "mega_cap_tech",
    "META": "mega_cap_tech",
    "TSLA": "mega_cap_tech",
    "NVDA": "mega_cap_tech",
    # AI / accelerated compute infra.
    "AMD": "ai_infra",
    "AVGO": "ai_infra",
    "TSM": "ai_infra",
    "ASML": "ai_infra",
    "SMCI": "ai_infra",
    "DELL": "ai_infra",
    "ANET": "ai_infra",
    "VRT": "ai_infra",
    # Broader semis — more cyclical, higher beta.
    "INTC": "semiconductors",
    "QCOM": "semiconductors",
    "MU": "semiconductors",
    "AMAT": "semiconductors",
    "LRCX": "semiconductors",
    "KLAC": "semiconductors",
    "MRVL": "semiconductors",
    "ADI": "semiconductors",
    "TXN": "semiconductors",
    "NXPI": "semiconductors",
    "MCHP": "semiconductors",
    "ON": "semiconductors",
    "SWKS": "semiconductors",
    "STM": "semiconductors",
    "MTSI": "semiconductors",
    "DIOD": "semiconductors",
    "COHR": "semiconductors",
    "SITM": "semiconductors",
    "ARM": "semiconductors",
    "GFS": "semiconductors",
    "MPWR": "semiconductors",
    "ENTG": "semiconductors",
    "TER": "semiconductors",
    # Software / SaaS.
    "CRM": "software",
    "ADBE": "software",
    "INTU": "software",
    "NOW": "software",
    "ORCL": "software",
    "SAP": "software",
    "WDAY": "software",
    "SNPS": "software",
    "CDNS": "software",
    "TEAM": "software",
    "SHOP": "software",
    "XYZ": "software",
    "PYPL": "software",
    "PLTR": "software",
    "U": "software",
    "APPS": "software",
    # Cloud infra / data.
    "NET": "cloud_infra",
    "SNOW": "cloud_infra",
    "DDOG": "cloud_infra",
    "MDB": "cloud_infra",
    "ESTC": "cloud_infra",
    "EQIX": "cloud_infra",
    "DLR": "cloud_infra",
    "AKAM": "cloud_infra",
    # Cybersecurity.
    "CRWD": "cybersecurity",
    "PANW": "cybersecurity",
    "ZS": "cybersecurity",
    "S": "cybersecurity",
    "OKTA": "cybersecurity",
    "FTNT": "cybersecurity",
    # Crypto-adjacent — heavily BTC-correlated, high vol.
    "MSTR": "crypto_miners",
    "COIN": "crypto_miners",
    "MARA": "crypto_miners",
    "CIFR": "crypto_miners",
    "RIOT": "crypto_miners",
    "CLSK": "crypto_miners",
    "HUT": "crypto_miners",
    "WULF": "crypto_miners",
    "IREN": "crypto_miners",
    # Networking / hardware.
    "CSCO": "other",
    "HPE": "other",
    "JBL": "other",
    "IBM": "other",
    # Streaming / consumer internet.
    "NFLX": "other",
    "SPOT": "other",
    "UBER": "other",
    "ABNB": "other",
    "DASH": "other",
    "ROKU": "other",
    # Fintech rails.
    "V": "other",
    "MA": "other",
}


def sector_for(symbol: str) -> str:
    """Return the bucket for a symbol, defaulting to `other`."""
    return SECTOR_MAP.get(symbol.upper(), "other")
