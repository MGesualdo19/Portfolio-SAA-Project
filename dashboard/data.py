"""
dashboard/data.py

Cached access to the analysis layer.

The dashboard runs no analysis of its own. Every number it draws comes
from `core.saa.run_saa()` or from a function in `core/` or `analysis/`,
so the dashboard and the notebook can never disagree about a weight --
there is only one implementation of each calculation and both call it.

Caching matters here for a practical reason: a full run with resampling
solves five constrained optimisations across 120 bootstrap resamples,
which takes a couple of minutes. Streamlit re-executes the entire script
on every widget interaction, so without `@st.cache_resource` keyed on the
model settings, moving a slider would re-solve everything.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.saa import CACHE_PATH, load_result, run_saa, save_result


@st.cache_resource(show_spinner=False)
def get_saa(bear_weight: float, drawdown_threshold: float, n_resamples: int,
            risk_aversion: float, view_confidence_scale: float, rf: float):
    """
    Run (or reuse) the SAA for a given set of model settings.

    Keyed on every judgement parameter, so changing a slider produces a
    genuinely new run rather than a stale cached one -- and holding the
    sliders still costs nothing.
    """
    result = run_saa(
        bear_weight=bear_weight,
        drawdown_threshold=drawdown_threshold,
        n_resamples=n_resamples,
        resample=n_resamples > 0,
        rf=rf,
        risk_aversion=risk_aversion,
        view_confidence_scale=view_confidence_scale,
        verbose=False,
    )
    try:
        save_result(result)
    except Exception:
        pass  # the cache file is a convenience; failing to write it is not fatal
    return result


@st.cache_resource(show_spinner=False)
def get_cached_saa():
    """Last saved run, for instant first paint before any slider is touched."""
    return load_result(CACHE_PATH)


@st.cache_data(show_spinner=False)
def security_frame(_securities, ticker: str) -> pd.DataFrame:
    """Full OHLCV plus CAD and native total-return indices for one holding."""
    from core.fx import to_cad
    from core.returns import total_return_index, total_return_series

    sec = next((s for s in _securities if s.ticker == ticker), None)
    if sec is None:
        raise KeyError(ticker)
    px = sec.prices.dropna(subset=["Close"])
    out = pd.DataFrame({
        "close_native": px["Close"],
        "volume": px["Volume"],
        "distribution": px["Dividends"].fillna(0.0),
    })
    out["close_cad"] = to_cad(px["Close"], sec.currency)
    out["tr_native"] = total_return_index(px, currency=sec.currency, in_cad=False)
    out["tr_cad"] = total_return_index(px, currency=sec.currency, in_cad=True)
    out["ret_native"] = total_return_series(px, currency=sec.currency, in_cad=False)
    out["ret_cad"] = total_return_series(px, currency=sec.currency, in_cad=True)
    return out


def fmt_pct(x, digits: int = 1) -> str:
    return "-" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def fmt_money(x, currency: str = "CAD", digits: int = 0) -> str:
    if pd.isna(x):
        return "-"
    return f"${x:,.{digits}f} {currency}"


def fmt_num(x, digits: int = 2) -> str:
    return "-" if pd.isna(x) else f"{x:,.{digits}f}"
