from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


OFFLINE_DEMO_DATA_STATUS = "offline_demo"
OFFLINE_DEMO_SUPPORTED_TICKERS = ("AAPL", "GME", "MSFT", "NVDA")
OFFLINE_DEMO_DISCLAIMER = (
    "Offline demo mode uses static local fixture data for educational/demo use only. "
    "It is not current market data."
)


def _fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "offline_demo"


def get_offline_demo_fixture_path(ticker: str) -> Path:
    return _fixture_dir() / f"{ticker.upper()}.json"


@lru_cache(maxsize=None)
def load_offline_demo_fixture(ticker: str) -> dict[str, Any]:
    path = get_offline_demo_fixture_path(ticker)
    if not path.exists():
        raise FileNotFoundError(f"No offline demo fixture exists for ticker '{ticker.upper()}'.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Offline demo fixture for '{ticker.upper()}' must be a JSON object.")
    return payload


def has_offline_demo_fixture(ticker: str) -> bool:
    return get_offline_demo_fixture_path(ticker).exists()
