"""
core/data_loader.py

yfinance wrapper with local pickle caching. Pulls max-available history
per ticker plus the TSX Composite benchmark used for bull/bear regime
classification.

FRED macro series and Fama-French factor pulls are deferred until the
hypothesis-testing phase (analysis/hypothesis_tests.py) — not needed
for the core OOP skeleton / risk framework.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_TICKER = "^GSPTSE"  # TSX Composite — used for bull/bear regime classification


def _cache_path(key: str) -> Path:
    safe_key = key.replace("^", "").replace("/", "_").replace(" ", "_")
    return CACHE_DIR / f"{safe_key}.pkl"


def _load_cache(key: str) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(key: str, df: pd.DataFrame) -> None:
    with open(_cache_path(key), "wb") as f:
        pickle.dump(df, f)


def get_price_history(
    ticker: str,
    start: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Pull max-available daily OHLCV history for a ticker via yfinance,
    caching locally as a pickle. Set force_refresh=True to bypass cache.
    """
    cache_key = f"prices_{ticker}"
    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    tk = yf.Ticker(ticker)
    hist = tk.history(period="max", start=start, auto_adjust=False)
    if hist.empty:
        raise ValueError(
            f"No price data returned for {ticker}. Check the ticker is "
            f"correct for its listing exchange (e.g. '.TO' suffix for TSX)."
        )
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    _save_cache(cache_key, hist)
    return hist


def get_benchmark_history(force_refresh: bool = False) -> pd.DataFrame:
    """Convenience wrapper for the TSX Composite benchmark series."""
    return get_price_history(BENCHMARK_TICKER, force_refresh=force_refresh)


def get_price_histories(
    tickers: list[str],
    start: Optional[str] = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Bulk convenience wrapper. Pulls each ticker individually (not via
    yf.download batch mode) so a single bad/delisted ticker doesn't
    blow up the whole batch — failures are collected and reported
    rather than silently dropped.
    """
    results: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    for ticker in tickers:
        try:
            results[ticker] = get_price_history(ticker, start=start, force_refresh=force_refresh)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            failures[ticker] = str(exc)

    if failures:
        print(f"[data_loader] Failed to fetch {len(failures)} ticker(s):")
        for ticker, msg in failures.items():
            print(f"  {ticker}: {msg}")

    return results
