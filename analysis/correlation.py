"""
analysis/correlation.py

Bull/bear conditional correlation analysis.

Answers a question the SAA thesis depends on implicitly: do the
diversifying assets in this book (gold, international equities,
FRN/cash) actually stay uncorrelated with core equities when it
matters -- during bear-market drawdowns -- or does correlation
converge toward 1 exactly when diversification would be most valuable?
That convergence is a well-documented feature of most real bear
markets, not a given, so this module checks it rather than assumes it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.regime import bull_bear_mask


def align_regime_to_returns(regime: pd.Series, returns_index: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fills a regime series (e.g. a daily bull/bear label derived
    from the benchmark) onto a returns DataFrame's date index. Needed
    because individual securities' return series have gaps (different
    exchange holidays, delistings, late listing dates) that don't line
    up 1:1 with the benchmark's calendar.
    """
    return regime.reindex(returns_index, method="ffill")


def conditional_correlation(
    returns: pd.DataFrame,
    mask: Optional[pd.Series] = None,
    condition: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Correlation matrix over `returns`, optionally filtered to only rows
    where `mask == condition`. mask=None returns the full-period matrix.
    """
    if mask is None:
        return returns.corr()

    aligned_mask = mask.reindex(returns.index).fillna(False)
    subset = returns.loc[aligned_mask == condition]
    if subset.empty:
        raise ValueError("No rows match the given regime condition over this return series.")
    return subset.corr()


def bull_bear_correlation_summary(
    returns: pd.DataFrame,
    benchmark_close: pd.Series,
    drawdown_threshold: float = 0.10,
    min_bear_days_warning: int = 60,
) -> dict[str, pd.DataFrame]:
    """
    Returns full-period, bull-only, and bear-only correlation matrices,
    plus a bear-minus-bull delta matrix highlighting which pairs
    correlate up the most specifically during drawdowns.

    Prints a sample-size warning if the bear-regime observation count is
    small -- a correlation estimated over 40 bear days should be read
    directionally, not treated as a precise number.
    """
    bear_mask = bull_bear_mask(benchmark_close, drawdown_threshold=drawdown_threshold)
    aligned_mask = align_regime_to_returns(bear_mask, returns.index)

    full_corr = conditional_correlation(returns)
    bull_corr = conditional_correlation(returns, aligned_mask, condition=False)
    bear_corr = conditional_correlation(returns, aligned_mask, condition=True)
    delta = bear_corr - bull_corr

    n_bear_days = int(aligned_mask.sum())
    n_bull_days = int((~aligned_mask.fillna(False)).sum())
    print(f"[correlation] Regime split: {n_bull_days} bull days, {n_bear_days} bear days.")
    if n_bear_days < min_bear_days_warning:
        print(
            f"[correlation] Warning: only {n_bear_days} bear-day observations "
            f"(< {min_bear_days_warning}). Bear-regime correlations will be noisy -- "
            f"treat direction, not precision."
        )

    return {
        "full": full_corr,
        "bull": bull_corr,
        "bear": bear_corr,
        "bear_minus_bull": delta,
        "n_bull_days": n_bull_days,
        "n_bear_days": n_bear_days,
    }


def largest_correlation_increases(
    summary: dict[str, pd.DataFrame],
    top_n: int = 5,
) -> pd.Series:
    """
    Flattens the bear_minus_bull delta matrix into a ranked list of
    ticker pairs with the largest correlation increase from bull to
    bear regime -- the pairs most likely to fail you exactly when
    diversification is supposed to earn its keep.
    """
    delta = summary["bear_minus_bull"]
    mask_upper = np.triu(np.ones(delta.shape), k=1).astype(bool)
    # dropna explicit: pandas >=2.1 changed stack()'s default dropna behavior,
    # so this can't be left implicit or NaNs from the masked lower triangle leak through.
    pairs = delta.where(mask_upper).stack().dropna()
    return pairs.sort_values(ascending=False).head(top_n)


def diversifier_bear_check(
    summary: dict[str, pd.DataFrame],
    diversifier_ticker: str,
    core_equity_ticker: str,
) -> None:
    """
    Prints a direct, portfolio-specific readout for a single pair --
    e.g. ("CGL.TO", "XUU.TO") for gold vs core US equity, or
    ("VOLX.TO", "XUU.TO") for the legacy vol position vs core equity.

    This is the check that actually validates or invalidates a specific
    named thesis claim (e.g. "gold is a tail hedge") rather than
    leaving it as a correlation matrix someone has to interpret.
    """
    bull_val = summary["bull"].loc[diversifier_ticker, core_equity_ticker]
    bear_val = summary["bear"].loc[diversifier_ticker, core_equity_ticker]
    print(
        f"[correlation] {diversifier_ticker} vs {core_equity_ticker}: "
        f"bull-regime corr = {bull_val:.2f}, bear-regime corr = {bear_val:.2f} "
        f"(delta = {bear_val - bull_val:+.2f})"
    )
    if bear_val > 0.3 and bull_val < bear_val:
        print(
            f"[correlation] Note: {diversifier_ticker}'s correlation to "
            f"{core_equity_ticker} rose materially in bear regimes. If this "
            f"is meant to be a tail hedge, that rise is worth explaining, "
            f"not skipping past."
        )
