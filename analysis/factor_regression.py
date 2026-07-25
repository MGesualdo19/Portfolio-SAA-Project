"""
analysis/factor_regression.py

OLS factor regression: explains a security's return using one or more
factor return series.

Supports two factor sources:
  1. Genuine Fama-French factors, if/when you pull them (deferred for
     now per the hypothesis-testing build-out).
  2. Portfolio-native proxy factors built directly from ETFs already
     in the book -- e.g. VTV minus XUU as a value-factor proxy, AVUV
     minus XUU as a size/value proxy. This works today with zero new
     data dependencies, which is why it's usable now rather than
     blocked on FRED/Ken-French access.

Proxy factors are a real simplification worth flagging in any write-up:
they only capture the value/size tilt *as expressed by these specific
ETFs*, not the academic factor definitions, and they're not excess-of-
risk-free returns unless you subtract that separately.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def build_proxy_factor(long_returns: pd.Series, short_returns: pd.Series, name: str) -> pd.Series:
    """
    Long-minus-short return spread used as a factor proxy, e.g.
    build_proxy_factor(vtv_returns, xuu_returns, "value_proxy") to get
    a value-factor stand-in from holdings already in the portfolio.
    """
    factor = (long_returns - short_returns).dropna()
    factor.name = name
    return factor


def run_ols_regression(
    asset_returns: pd.Series,
    factors: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Runs asset_returns ~ factors via OLS with an intercept (alpha).

    If using true Fama-French data, asset_returns and factors should
    both already be excess-of-risk-free. If using proxy factors from
    build_proxy_factor(), those are already long-minus-short spreads
    and don't need further adjustment, but asset_returns should still
    be excess-of-risk-free for the alpha term to be meaningful as
    "return unexplained by these factors" rather than just raw return.
    """
    aligned = pd.concat([asset_returns, factors], axis=1).dropna()
    if aligned.empty:
        raise ValueError("No overlapping dates between asset_returns and factors after alignment.")
    if len(aligned) < 60:
        print(
            f"[factor_regression] Warning: only {len(aligned)} overlapping observations. "
            f"Regression coefficients will be unstable -- treat directionally."
        )

    y = aligned.iloc[:, 0]
    X = sm.add_constant(aligned.iloc[:, 1:])
    return sm.OLS(y, X).fit()


def summarize_regression(model: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    """Alpha/betas, t-stats, p-values in one table, plus R^2 as a summary row."""
    summary = pd.DataFrame({
        "coef": model.params,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
    })
    summary.loc["R_squared"] = [model.rsquared, None, None]
    return summary
