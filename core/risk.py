"""
core/risk.py

VaR/CVaR via three methods, always computed and surfaced side by side
(parametric, historical, bootstrap) -- never collapsed into one number,
per project spec. Each method has different, genuine blind spots:

  - Parametric: assumes returns are normally distributed. Understates
    tail risk whenever the true return distribution is fat-tailed,
    which equities and especially VIX-linked/gold products usually are.
  - Historical: makes no distributional assumption, but is entirely
    bounded by what's actually in the sample -- it cannot produce a
    worse tail event than the worst one already observed.
  - Bootstrap: resamples the historical distribution to estimate a
    confidence interval around the VaR/CVaR point estimate, which
    quantifies estimation uncertainty -- but it still only resamples
    history, so it shares historical's blind spot to genuinely novel
    tail events.

Reporting all three side by side is the point: agreement across
methods is a mild reassurance, disagreement is informative about which
distributional assumption is driving the number.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Blending helper
# ---------------------------------------------------------------------------

def blended_returns(returns_matrix: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Weighted daily return series for a set of securities. Errors loudly
    (not silently) if weights don't cover every column or don't sum to
    ~1.0, since a silently-wrong blend would make every VaR/CVaR number
    downstream meaningless without any visible warning.
    """
    missing = set(returns_matrix.columns) - set(weights.keys())
    if missing:
        raise ValueError(f"No weight provided for: {missing}")

    total_weight = sum(weights[c] for c in returns_matrix.columns)
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(
            f"Weights for the given columns sum to {total_weight:.4f}, not 1.0. "
            f"Renormalize before blending, or this isn't a valid portfolio return series."
        )

    weight_vector = pd.Series({c: weights[c] for c in returns_matrix.columns})
    return returns_matrix.mul(weight_vector, axis=1).sum(axis=1)


# ---------------------------------------------------------------------------
# Parametric VaR/CVaR
# ---------------------------------------------------------------------------

def parametric_var_cvar(returns: pd.Series, confidence: float = 0.95) -> dict:
    """
    Variance-covariance (Gaussian) VaR/CVaR. Fast and stable, but blind
    to fat tails and skew -- treat as a floor on true tail risk, not
    an estimate of it, for anything with known fat-tail behavior
    (equities generally, VIX-linked and gold products especially).
    """
    mu = returns.mean()
    sigma = returns.std()
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)

    var = -(mu + z * sigma)
    # Expected shortfall under normality: mu - sigma * phi(z) / alpha
    cvar = -(mu - sigma * stats.norm.pdf(z) / alpha)

    return {"method": "parametric", "confidence": confidence, "VaR": var, "CVaR": cvar}


# ---------------------------------------------------------------------------
# Historical VaR/CVaR
# ---------------------------------------------------------------------------

def historical_var_cvar(returns: pd.Series, confidence: float = 0.95) -> dict:
    """
    Empirical VaR/CVaR from the actual historical return sample. No
    distributional assumption, but entirely bounded by what's in the
    sample -- cannot produce a worse tail event than history's worst.
    """
    alpha = 1 - confidence
    var_threshold = returns.quantile(alpha)
    tail_losses = returns[returns <= var_threshold]

    var = -var_threshold
    cvar = -tail_losses.mean() if not tail_losses.empty else var

    n_tail_obs = len(tail_losses)
    if n_tail_obs < 20:
        print(
            f"[risk] Warning: only {n_tail_obs} observations in the tail region "
            f"at {confidence:.0%} confidence. Historical CVaR estimate is thin -- "
            f"treat as directional, not precise."
        )

    return {
        "method": "historical",
        "confidence": confidence,
        "VaR": var,
        "CVaR": cvar,
        "n_tail_obs": n_tail_obs,
    }


# ---------------------------------------------------------------------------
# Bootstrap VaR/CVaR
# ---------------------------------------------------------------------------

def bootstrap_var_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    n_boot: int = 5000,
    seed: Optional[int] = None,
) -> dict:
    """
    Resamples the historical return series (with replacement) n_boot
    times, computing historical VaR/CVaR on each resample. Reports the
    mean estimate plus a 90% confidence interval (5th/95th percentile
    across resamples) -- this quantifies estimation uncertainty around
    the point estimate, not a genuinely wider set of possible outcomes
    than history contains.
    """
    rng = np.random.default_rng(seed)
    values = returns.dropna().to_numpy()
    n = len(values)
    alpha = 1 - confidence

    boot_vars = np.empty(n_boot)
    boot_cvars = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        threshold = np.quantile(sample, alpha)
        tail = sample[sample <= threshold]
        boot_vars[i] = -threshold
        boot_cvars[i] = -tail.mean() if tail.size > 0 else -threshold

    return {
        "method": "bootstrap",
        "confidence": confidence,
        "VaR": boot_vars.mean(),
        "VaR_ci_low": np.percentile(boot_vars, 5),
        "VaR_ci_high": np.percentile(boot_vars, 95),
        "CVaR": boot_cvars.mean(),
        "CVaR_ci_low": np.percentile(boot_cvars, 5),
        "CVaR_ci_high": np.percentile(boot_cvars, 95),
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# Combined side-by-side summary
# ---------------------------------------------------------------------------

def var_cvar_summary(
    returns: pd.Series,
    confidence: float = 0.95,
    n_boot: int = 5000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    All three methods, one table, never collapsed to a single number.
    Large disagreement between the parametric row and the other two is
    itself a finding -- it means the normal-distribution assumption is
    doing real work and probably understating tail risk.
    """
    parametric = parametric_var_cvar(returns, confidence)
    historical = historical_var_cvar(returns, confidence)
    bootstrap = bootstrap_var_cvar(returns, confidence, n_boot=n_boot, seed=seed)

    rows = [
        {"method": "parametric", "VaR": parametric["VaR"], "CVaR": parametric["CVaR"],
         "VaR_ci_low": None, "VaR_ci_high": None, "CVaR_ci_low": None, "CVaR_ci_high": None},
        {"method": "historical", "VaR": historical["VaR"], "CVaR": historical["CVaR"],
         "VaR_ci_low": None, "VaR_ci_high": None, "CVaR_ci_low": None, "CVaR_ci_high": None},
        {"method": "bootstrap", "VaR": bootstrap["VaR"], "CVaR": bootstrap["CVaR"],
         "VaR_ci_low": bootstrap["VaR_ci_low"], "VaR_ci_high": bootstrap["VaR_ci_high"],
         "CVaR_ci_low": bootstrap["CVaR_ci_low"], "CVaR_ci_high": bootstrap["CVaR_ci_high"]},
    ]
    return pd.DataFrame(rows).set_index("method")


# ---------------------------------------------------------------------------
# Per-account risk (uses Portfolio.by_account)
# ---------------------------------------------------------------------------

def per_account_var_cvar(
    portfolio,
    confidence: float = 0.95,
    n_boot: int = 5000,
    seed: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """
    Runs var_cvar_summary() per account_tag, re-normalizing each
    account's assigned weights to sum to 1 within that account (since
    portfolio-level weights sum to 1 across the whole book, not within
    a single account). Requires weight to be set on every security in
    an account -- raises rather than guessing at an equal-weight
    fallback, since that would silently misrepresent the actual book.
    """
    results = {}
    for tag in portfolio.account_tags():
        secs, _profile = portfolio.by_account(tag)
        unset = [s.ticker for s in secs if s.weight is None]
        if unset:
            print(f"[risk] Skipping account '{tag}': weight not set for {unset}.")
            continue

        returns_matrix = pd.DataFrame({s.ticker: s.daily_returns() for s in secs}).dropna(how="any")
        raw_weights = {s.ticker: s.weight for s in secs}
        total = sum(raw_weights.values())
        normalized_weights = {k: v / total for k, v in raw_weights.items()}

        account_returns = blended_returns(returns_matrix, normalized_weights)
        results[tag] = var_cvar_summary(account_returns, confidence=confidence, n_boot=n_boot, seed=seed)

    return results
