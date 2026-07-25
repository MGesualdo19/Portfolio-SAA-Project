"""
core/backtester.py

Rolling performance diagnostics: Sharpe, Sortino, max drawdown, and
bootstrapped forward-return distributions. Factor regression is
deliberately NOT duplicated here -- it's imported from
analysis/factor_regression.py, which owns that logic exclusively, so
there's one place to fix a regression bug rather than two.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from analysis.factor_regression import run_ols_regression, summarize_regression

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Rolling risk-adjusted return
# ---------------------------------------------------------------------------

def rolling_sharpe(returns: pd.Series, window: int = TRADING_DAYS_PER_YEAR, rf: float = 0.0) -> pd.Series:
    """
    Rolling annualized Sharpe ratio. rf is a daily risk-free rate
    (decimal, not %) -- convert an annual rate before passing in if
    that's what you have (e.g. rf_daily = (1 + rf_annual) ** (1/252) - 1).
    """
    excess = returns - rf
    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std()
    return (rolling_mean / rolling_std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def rolling_sortino(returns: pd.Series, window: int = TRADING_DAYS_PER_YEAR, rf: float = 0.0) -> pd.Series:
    """
    Rolling annualized Sortino ratio -- like Sharpe, but the denominator
    only penalizes downside deviation, not total volatility. More
    relevant than Sharpe for anything with asymmetric return profiles
    (gold, VIX-linked positions), since Sharpe treats upside surprises
    as "risk" identically to downside ones.
    """
    excess = returns - rf

    def _downside_std(window_values: np.ndarray) -> float:
        downside = window_values[window_values < 0]
        if downside.size < 2:
            return np.nan
        return downside.std()

    rolling_mean = excess.rolling(window).mean()
    rolling_downside_std = excess.rolling(window).apply(_downside_std, raw=True)
    return (rolling_mean / rolling_downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def max_drawdown(price_series: pd.Series) -> dict:
    """
    Max drawdown over the full series, plus the dates of the peak and
    trough so a chart or writeup can point to exactly when it happened
    (e.g. does the worst drawdown coincide with a bear regime the
    portfolio's thesis explicitly claims to be positioned for?).
    """
    trailing_high = price_series.cummax()
    drawdown = (price_series / trailing_high) - 1.0

    trough_date = drawdown.idxmin()
    max_dd = drawdown.loc[trough_date]
    peak_date = price_series.loc[:trough_date].idxmax()

    return {
        "max_drawdown": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
    }


def drawdown_series(price_series: pd.Series) -> pd.Series:
    """Full drawdown-from-trailing-high series, useful for plotting."""
    trailing_high = price_series.cummax()
    return (price_series / trailing_high) - 1.0


# ---------------------------------------------------------------------------
# Bootstrapped forward-return distribution
# ---------------------------------------------------------------------------

def bootstrap_return_distribution(
    returns: pd.Series,
    n_years: float = 1.0,
    n_sims: int = 5000,
    seed: Optional[int] = None,
    block_size: int = 20,
) -> pd.Series:
    """
    Simulates n_sims possible cumulative return paths over n_years by
    resampling BLOCKS of historical daily returns (not single days) --
    block resampling preserves some of the real serial correlation /
    volatility clustering in the original series, which single-day
    resampling destroys. Returns the distribution of simulated total
    returns, not a single number, so you can look at the actual spread
    (e.g. 5th/50th/95th percentile) rather than one point estimate.
    """
    rng = np.random.default_rng(seed)
    values = returns.dropna().to_numpy()
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    n_blocks = int(np.ceil(n_days / block_size))

    sim_totals = np.empty(n_sims)
    for i in range(n_sims):
        path = []
        for _ in range(n_blocks):
            start = rng.integers(0, len(values) - block_size)
            path.extend(values[start:start + block_size])
        path = np.array(path[:n_days])
        sim_totals[i] = np.prod(1 + path) - 1

    return pd.Series(sim_totals, name=f"{n_years}yr_bootstrap_returns")


def summarize_return_distribution(sim_returns: pd.Series) -> dict:
    """5th/25th/50th/75th/95th percentile summary of a simulated return distribution."""
    return {
        "p5": sim_returns.quantile(0.05),
        "p25": sim_returns.quantile(0.25),
        "p50": sim_returns.quantile(0.50),
        "p75": sim_returns.quantile(0.75),
        "p95": sim_returns.quantile(0.95),
        "mean": sim_returns.mean(),
        "prob_loss": (sim_returns < 0).mean(),
    }


# ---------------------------------------------------------------------------
# Factor regression (thin pass-through — logic lives in analysis/factor_regression.py)
# ---------------------------------------------------------------------------

def run_factor_regression(asset_returns: pd.Series, factors: pd.DataFrame):
    """
    Thin wrapper so notebook code can call backtester.run_factor_regression()
    without a separate import -- the actual OLS logic is owned entirely
    by analysis/factor_regression.py, this just re-exports it in context.
    """
    model = run_ols_regression(asset_returns, factors)
    return model, summarize_regression(model)
