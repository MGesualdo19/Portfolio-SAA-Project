"""
core/returns.py

One canonical definition of "a return" for the whole project.

Two things were being conflated before this module existed:

1. PRICE return vs TOTAL return. `Close.pct_change()` silently discards
   every distribution. For this book that is not a rounding error --
   CASH.TO and XFR.TO deliver essentially ALL of their return as
   distributions and almost none as price appreciation, so a
   price-return optimiser sees them as assets with ~zero expected
   return and a small positive volatility, and refuses to hold them.
   Every return in this project is a total return: (P_t + D_t)/P_{t-1} - 1,
   where D_t includes ordinary distributions and capital-gains
   distributions.

   Yahoo's `Close` is already split-adjusted, so no split factor is
   applied here (the `Stock Splits` column is informational and, for
   XIC.TO, contains at least one spurious non-split entry). `Adj Close`
   is used only as an independent cross-check via `reconciliation_report`.

2. NATIVE currency vs CAD. See core/fx.py. `total_returns(..., in_cad=True)`
   is the portfolio-level view; `in_cad=False` is the per-security view.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from core.fx import to_cad

TRADING_DAYS = 252


def _distributions(prices: pd.DataFrame) -> pd.Series:
    dist = prices["Dividends"].fillna(0.0)
    if "Capital Gains" in prices.columns:
        dist = dist + prices["Capital Gains"].fillna(0.0)
    return dist


def total_return_series(prices: pd.DataFrame, currency: str = "CAD", in_cad: bool = False) -> pd.Series:
    """
    Daily total return from a yfinance OHLCV frame. `currency` is the
    security's own quote currency; `in_cad=True` additionally converts
    the price and the distribution into CAD before differencing, so the
    result is the return an investor whose base currency is CAD
    actually earned.
    """
    px = prices["Close"].dropna()
    dist = _distributions(prices).reindex(px.index).fillna(0.0)

    if in_cad and currency != "CAD":
        px_c = to_cad(px, currency)
        # The distribution is converted at the same day's rate it was paid.
        dist_c = dist * (px_c / px)
        px, dist = px_c, dist_c

    return ((px + dist) / px.shift(1) - 1.0).dropna()


def total_return_index(prices: pd.DataFrame, currency: str = "CAD", in_cad: bool = False,
                       base: float = 100.0) -> pd.Series:
    """Growth-of-`base` total-return index, i.e. dividends reinvested."""
    r = total_return_series(prices, currency=currency, in_cad=in_cad)
    return base * (1.0 + r).cumprod()


def returns_frame(securities: Iterable, in_cad: bool = True, align: bool = False) -> pd.DataFrame:
    """
    Total-return matrix, one column per security.

    align=False (default) keeps each column's own full history with NaNs
    elsewhere -- correct for any single-asset statistic. align=True inner-
    joins to the dates on which EVERY security traded, which is required
    for a covariance matrix but throws away history down to the shortest-
    lived holding, so it should be an explicit choice at the call site
    rather than a side effect.
    """
    cols = {}
    for s in securities:
        cols[s.ticker] = total_return_series(s.prices, currency=s.currency, in_cad=in_cad)
    frame = pd.DataFrame(cols).sort_index()
    return frame.dropna(how="any") if align else frame


def price_frame(securities: Iterable, in_cad: bool = False) -> pd.DataFrame:
    """Close prices, native currency by default (this is the quoted price)."""
    cols = {}
    for s in securities:
        px = s.prices["Close"].dropna()
        cols[s.ticker] = to_cad(px, s.currency) if in_cad else px
    return pd.DataFrame(cols).sort_index()


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------

def annualise_return(daily: pd.Series, geometric: bool = True) -> float:
    """
    Geometric (CAGR-equivalent) annualisation by default. The arithmetic
    mean * 252 convention overstates compounded outcomes by roughly
    sigma^2/2, which for a 20%-vol asset is ~2%/yr of pure artefact --
    enough to reorder an optimiser's asset ranking on its own.
    """
    r = daily.dropna()
    if r.empty:
        return np.nan
    if not geometric:
        return float(r.mean() * TRADING_DAYS)
    growth = float((1.0 + r).prod())
    if growth <= 0:  # a total wipeout (see VOLX.TO) has no real CAGR
        return -1.0
    return growth ** (TRADING_DAYS / len(r)) - 1.0


def annualise_vol(daily: pd.Series) -> float:
    return float(daily.dropna().std() * np.sqrt(TRADING_DAYS))


def summary_stats(daily: pd.Series, rf_annual: float = 0.0) -> dict:
    r = daily.dropna()
    ann_ret = annualise_return(r)
    ann_vol = annualise_vol(r)
    downside = r[r < 0]
    down_vol = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else np.nan
    idx = (1 + r).cumprod()
    dd = (idx / idx.cummax() - 1.0)
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "downside_vol": down_vol,
        "sharpe": (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else np.nan,
        "sortino": (ann_ret - rf_annual) / down_vol if down_vol and down_vol > 0 else np.nan,
        "max_drawdown": float(dd.min()),
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurtosis()),
        "n_obs": int(len(r)),
        "start": r.index.min(),
        "end": r.index.max(),
    }


def reconciliation_report(securities: Iterable) -> pd.DataFrame:
    """
    Cross-checks this module's reconstructed total return against Yahoo's
    own `Adj Close`. They are computed from different inputs, so a large
    gap means one of the two data columns is corrupt for that ticker and
    the number should not be trusted until it is explained.
    """
    rows = []
    for s in securities:
        px = s.prices.dropna(subset=["Close"])
        mine = total_return_series(px, currency=s.currency, in_cad=False)
        theirs = px["Adj Close"].pct_change().dropna()
        joined = pd.concat([mine, theirs], axis=1).dropna()
        rows.append({
            "ticker": s.ticker,
            "cum_total_return_reconstructed": float((1 + joined.iloc[:, 0]).prod() - 1),
            "cum_total_return_adj_close": float((1 + joined.iloc[:, 1]).prod() - 1),
            "max_daily_abs_diff": float((joined.iloc[:, 0] - joined.iloc[:, 1]).abs().max()),
        })
    df = pd.DataFrame(rows).set_index("ticker")
    df["flag"] = np.where(df["max_daily_abs_diff"] > 0.02, "INVESTIGATE", "ok")
    return df
