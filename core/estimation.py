"""
core/estimation.py

Covariance and expected-return estimation.

This module exists because of a specific, well-documented failure mode:
a mean-variance optimiser fed raw sample means and a raw sample
covariance matrix reliably returns a two- or three-asset portfolio. That
is not the optimiser malfunctioning -- it is the optimiser correctly
maximising an objective built from inputs whose estimation error dwarfs
their signal. Michaud called it an "estimation-error maximiser": the
asset whose mean was most overstated by sampling noise is precisely the
asset the optimiser piles into.

The arithmetic is unforgiving. The standard error of a mean return
estimated over T years on a sigma-vol asset is sigma/sqrt(T). For a
19-year sample of a 19%-vol asset that is 4.4%/yr -- wider than the
entire spread of plausible expected returns across this book's equity
sleeves. Sample means simply cannot rank these assets, and any
allocation that leans on them is expressing noise.

Three corrections are applied here, all standard practice:

  1. LEDOIT-WOLF SHRINKAGE on the covariance matrix. Sample covariance
     over N=10 assets and T~4,700 days is estimable, but its smallest
     eigenvalues are still biased downward, and those are exactly the
     directions a minimum-variance optimiser loads into. Shrinking
     toward a structured target pulls them back.

  2. A BEAR-REGIME BLEND. Correlations converge in drawdowns, so the
     full-sample covariance understates exactly the risk this portfolio
     is being built to survive. The matrix actually optimised against is
     a convex blend of full-sample and bear-regime covariance, which
     makes the optimiser price diversification by whether it survives a
     crash rather than by its unconditional average.

  3. EQUILIBRIUM (REVERSE-OPTIMISED) EXPECTED RETURNS, adjusted by
     explicit Black-Litterman views. Instead of asking history what each
     asset returned, this asks what expected returns would make a
     sensible reference allocation optimal, then tilts away from that
     anchor only where the thesis states a real view -- and only as far
     as the stated confidence in that view justifies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------

def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * TRADING_DAYS


def shrunk_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Ledoit-Wolf shrinkage toward a scaled-identity target. Returns the
    annualised matrix and the fitted shrinkage intensity, which is worth
    surfacing: a high intensity means the sample matrix was poorly
    conditioned and the raw one should not have been trusted.
    """
    lw = LedoitWolf().fit(returns.to_numpy())
    cov = pd.DataFrame(lw.covariance_ * TRADING_DAYS,
                       index=returns.columns, columns=returns.columns)
    return cov, float(lw.shrinkage_)


def regime_blended_covariance(
    returns: pd.DataFrame,
    bear_mask: pd.Series,
    bear_weight: float = 0.35,
) -> tuple[pd.DataFrame, dict]:
    """
    Blend full-sample and bear-regime covariance:

        Sigma = (1 - w) * Sigma_full + w * Sigma_bear

    `bear_weight` is a risk-preference parameter, not an estimate. It says
    how much the allocation should be priced off crash behaviour rather
    than average behaviour. At w=0 this reduces to the ordinary shrunk
    covariance; at w=1 the book is optimised as though the market were
    permanently in a >10% drawdown, which over-hedges. The 0.35 default
    roughly matches the unconditional frequency of bear days in the
    sample, i.e. "weight crash behaviour by how often crashes happen,"
    and it is exposed as a slider in the dashboard rather than buried.

    Both components are shrunk before blending, since the bear subsample
    is much smaller and needs the regularisation more, not less.
    """
    if not 0.0 <= bear_weight <= 1.0:
        raise ValueError("bear_weight must be in [0, 1].")

    aligned = bear_mask.reindex(returns.index).ffill().fillna(False).astype(bool)
    bear_rows = returns.loc[aligned]
    bull_rows = returns.loc[~aligned]

    cov_full, shrink_full = shrunk_covariance(returns)
    if len(bear_rows) < 120:
        # Too few bear days to estimate a 10x10 matrix; fall back rather than
        # emit a confident-looking matrix built on nothing.
        return cov_full, {"bear_days": int(len(bear_rows)), "bear_weight_used": 0.0,
                          "shrinkage_full": shrink_full, "shrinkage_bear": None,
                          "note": "Insufficient bear observations; full-sample covariance used."}

    cov_bear, shrink_bear = shrunk_covariance(bear_rows)
    blended = (1 - bear_weight) * cov_full + bear_weight * cov_bear

    diag = {
        "bear_days": int(len(bear_rows)),
        "bull_days": int(len(bull_rows)),
        "bear_share": float(aligned.mean()),
        "bear_weight_used": bear_weight,
        "shrinkage_full": shrink_full,
        "shrinkage_bear": shrink_bear,
        "vol_full": pd.Series(np.sqrt(np.diag(cov_full)), index=returns.columns),
        "vol_bear": pd.Series(np.sqrt(np.diag(cov_bear)), index=returns.columns),
    }
    return blended, diag


def correlation_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.diag(cov))
    return pd.DataFrame(cov.to_numpy() / np.outer(d, d), index=cov.index, columns=cov.columns)


# ---------------------------------------------------------------------------
# Expected returns
# ---------------------------------------------------------------------------

def implied_equilibrium_returns(
    cov: pd.DataFrame,
    reference_weights: pd.Series,
    risk_aversion: float = 2.8,
    rf: float = 0.0275,
) -> pd.Series:
    """
    Reverse optimisation (Black-Litterman's prior): pi = delta * Sigma * w_ref.

    Rather than estimating what each asset WILL return, this asks what
    set of expected returns would make `reference_weights` the optimal
    portfolio, and takes that as the neutral starting point. Its great
    virtue is internal consistency -- pi is generated by the same
    covariance matrix the optimiser will use, so feeding pi back in
    reproduces the reference weights exactly and cannot produce a corner
    solution. All concentration in the final answer therefore has to be
    argued for by an explicit view, not smuggled in by sampling noise.

    `risk_aversion` (delta) is the market price of risk, roughly
    (E[Rm] - rf) / sigma_m^2. For a ~5% equity risk premium at ~15%
    market vol, delta is about 2.2-3.0; 2.8 is used here.
    """
    w = reference_weights.reindex(cov.index).fillna(0.0)
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"reference_weights must sum to 1.0, got {w.sum():.4f}")
    pi = risk_aversion * cov.to_numpy() @ w.to_numpy()
    return pd.Series(pi + rf, index=cov.index, name="equilibrium_return")


@dataclass
class View:
    """
    One Black-Litterman view.

    `picks` maps tickers to coefficients. A relative view uses
    coefficients summing to zero, e.g. {"VTV": 0.5, "AVUV": 0.5,
    "XUU.TO": -1.0} with q=0.015 reads "the value sleeve outperforms
    core US by 1.5%/yr". An absolute view uses coefficients summing to
    one.

    `confidence` is in (0, 1]: the fraction of the way the posterior is
    pulled from the equilibrium prior toward the view. It is deliberately
    NOT a p-value -- these are judgements from the thesis, and stating
    them as tunable confidences is the honest representation.
    """
    name: str
    picks: dict[str, float]
    q: float
    confidence: float = 0.5
    rationale: str = ""


def black_litterman(
    cov: pd.DataFrame,
    pi: pd.Series,
    views: list[View],
    tau: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Black-Litterman posterior expected returns.

    Omega (view uncertainty) is set by the Idzorek confidence method:
    each view's variance is scaled so the posterior moves the stated
    fraction of the way from prior to view. With no views this returns
    pi unchanged, which is the desired behaviour -- silence means
    "hold the reference allocation", not "hold nothing".
    """
    if not views:
        return pi.copy(), pd.DataFrame(index=cov.index)

    tickers = list(cov.index)
    P = np.zeros((len(views), len(tickers)))
    Q = np.zeros(len(views))
    for i, v in enumerate(views):
        for tk, coef in v.picks.items():
            if tk not in tickers:
                raise ValueError(f"View {v.name!r} references unknown ticker {tk!r}.")
            P[i, tickers.index(tk)] = coef
        Q[i] = v.q

    S = cov.to_numpy()
    tau_S = tau * S
    # Idzorek-style: a confidence of c implies omega_i such that the tilt is
    # c of the full-confidence tilt. c -> 1 gives omega -> 0 (view taken as
    # certain); c -> 0 gives omega -> inf (view ignored).
    omega_diag = []
    for i, v in enumerate(views):
        c = min(max(v.confidence, 1e-4), 0.9999)
        base = float(P[i] @ tau_S @ P[i].T)
        omega_diag.append(base * (1 - c) / c)
    Omega = np.diag(omega_diag)

    inv_tau_S = np.linalg.inv(tau_S)
    inv_Omega = np.linalg.inv(Omega)
    posterior_cov = np.linalg.inv(inv_tau_S + P.T @ inv_Omega @ P)
    mu = posterior_cov @ (inv_tau_S @ pi.to_numpy() + P.T @ inv_Omega @ Q)

    detail = pd.DataFrame({
        "equilibrium": pi,
        "posterior": pd.Series(mu, index=tickers),
    })
    detail["view_tilt"] = detail["posterior"] - detail["equilibrium"]
    return pd.Series(mu, index=tickers, name="bl_expected_return"), detail


def shrunk_historical_returns(
    returns: pd.DataFrame,
    shrink_to_grand_mean: float = 0.7,
) -> pd.Series:
    """
    James-Stein-flavoured fallback for when a purely historical view is
    wanted anyway: pull each asset's sample mean most of the way toward
    the cross-sectional average. Provided as a comparison row in the
    diagnostics so the effect of NOT using equilibrium returns is
    visible, rather than as the production estimator.
    """
    ann = (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1
    grand = ann.mean()
    return (1 - shrink_to_grand_mean) * ann + shrink_to_grand_mean * grand
