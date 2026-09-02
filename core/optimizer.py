"""
core/optimizer.py

Constrained portfolio construction.

The brief for this module was explicit: the answer must not be "hold one
or two ETFs". That outcome is the default behaviour of naive mean-
variance optimisation, and it is avoided here by four independent
mechanisms, each of which would help on its own and which compound when
stacked:

  1. BETTER INPUTS (core/estimation.py). Equilibrium expected returns
     plus shrunk, regime-blended covariance. This is the load-bearing
     one -- most concentration in textbook optimiser output traces
     directly to sampling error in the mean vector.

  2. MULTIPLE OBJECTIVES, THEN A BLEND. Five genuinely different
     objectives are solved -- minimum variance, equal risk contribution,
     maximum diversification, maximum Sharpe on the Black-Litterman
     posterior, and minimum CVaR -- and the recommended allocation is
     their average. Each has a different, known bias (minimum variance
     hides in cash; maximum Sharpe chases the highest posterior return;
     ERC ignores returns entirely), and averaging over disagreeing
     estimators is a standard, cheap robustness gain. Where all five
     agree, the position is genuinely well-founded; where they disagree,
     the spread across objectives is reported rather than hidden, and
     that spread is the honest measure of how much confidence the
     weight deserves.

  3. RESAMPLING (Michaud). Every objective is additionally solved on
     hundreds of block-bootstrapped resamples of the return history and
     the resulting weights averaged. An asset that only looks good on
     the one particular path history happened to take gets averaged
     down; an asset that earns its place across resamples does not.
     Block resampling (not IID) preserves volatility clustering.

  4. POLICY CONSTRAINTS. Per-asset caps and asset-class group bands.
     These are the backstop, not the mechanism -- if the first three are
     working, most bounds should sit slack, and `binding_constraints()`
     reports which ones actually bind so it is visible when the answer
     is being produced by the constraint rather than by the data.

Constraints encode investment policy, not statistics, and they are worth
stating plainly: they are the judgement that a portfolio should hold
some fixed income, some non-North-American equity and some real assets
regardless of what a 19-year sample says about their Sharpe ratios,
because the sample is one draw and the policy has to survive draws it
has not seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Constraint specification
# ---------------------------------------------------------------------------

@dataclass
class GroupConstraint:
    name: str
    tickers: list[str]
    min_weight: float = 0.0
    max_weight: float = 1.0
    rationale: str = ""


@dataclass
class ConstraintSet:
    tickers: list[str]
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    groups: list[GroupConstraint] = field(default_factory=list)
    default_bounds: tuple[float, float] = (0.0, 0.25)

    def bounds_list(self) -> list[tuple[float, float]]:
        return [self.bounds.get(t, self.default_bounds) for t in self.tickers]

    def scipy_constraints(self) -> list[dict]:
        cons: list[dict] = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
        idx = {t: i for i, t in enumerate(self.tickers)}
        for g in self.groups:
            members = [idx[t] for t in g.tickers if t in idx]
            if not members:
                continue
            sel = np.zeros(len(self.tickers))
            sel[members] = 1.0
            # Bound-as-closure: `sel` and the limits are captured by value via
            # default args, otherwise every constraint would close over the
            # last loop iteration and silently apply the same group twice.
            cons.append({"type": "ineq", "fun": lambda w, s=sel, lo=g.min_weight: float(s @ w - lo)})
            cons.append({"type": "ineq", "fun": lambda w, s=sel, hi=g.max_weight: float(hi - s @ w)})
        return cons

    def feasible_start(self) -> np.ndarray:
        """
        A starting point that already respects the box bounds, built by
        distributing the residual above the per-asset minimums in
        proportion to headroom. SLSQP started from an infeasible point can
        terminate at one, so this is not cosmetic.
        """
        lo = np.array([b[0] for b in self.bounds_list()])
        hi = np.array([b[1] for b in self.bounds_list()])
        w = lo.copy()
        headroom = hi - lo
        residual = 1.0 - w.sum()
        if residual > 0 and headroom.sum() > 0:
            w = w + headroom * (residual / headroom.sum())
        return w / w.sum()

    def violations(self, weights: pd.Series, tol: float = 1e-4) -> list[str]:
        msgs = []
        for t in self.tickers:
            lo, hi = self.bounds.get(t, self.default_bounds)
            w = float(weights.get(t, 0.0))
            if w < lo - tol:
                msgs.append(f"{t} at {w:.2%} is below its {lo:.0%} floor")
            if w > hi + tol:
                msgs.append(f"{t} at {w:.2%} exceeds its {hi:.0%} cap")
        for g in self.groups:
            tot = float(sum(weights.get(t, 0.0) for t in g.tickers))
            if tot < g.min_weight - tol:
                msgs.append(f"group {g.name} at {tot:.2%} is below its {g.min_weight:.0%} floor")
            if tot > g.max_weight + tol:
                msgs.append(f"group {g.name} at {tot:.2%} exceeds its {g.max_weight:.0%} cap")
        return msgs

    def binding_constraints(self, weights: pd.Series, tol: float = 5e-3) -> list[str]:
        """
        Which limits the solution is actually resting against. A weight
        sitting exactly on a cap means the constraint, not the data, chose
        that number -- worth knowing before quoting it as a result.
        """
        out = []
        for t in self.tickers:
            lo, hi = self.bounds.get(t, self.default_bounds)
            w = float(weights.get(t, 0.0))
            if abs(w - hi) < tol and hi < 1.0:
                out.append(f"{t} is at its {hi:.0%} cap")
            elif abs(w - lo) < tol and lo > 0.0:
                out.append(f"{t} is at its {lo:.0%} floor")
        for g in self.groups:
            tot = float(sum(weights.get(t, 0.0) for t in g.tickers))
            if abs(tot - g.max_weight) < tol and g.max_weight < 1.0:
                out.append(f"group {g.name} is at its {g.max_weight:.0%} cap")
            elif abs(tot - g.min_weight) < tol and g.min_weight > 0.0:
                out.append(f"group {g.name} is at its {g.min_weight:.0%} floor")
        return out


# ---------------------------------------------------------------------------
# Portfolio maths
# ---------------------------------------------------------------------------

def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(w @ cov @ w, 1e-18)))


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each asset's share of total portfolio volatility (Euler decomposition)."""
    vol = portfolio_vol(w, cov)
    marginal = cov @ w
    return w * marginal / vol


def diversification_ratio(w: np.ndarray, cov: np.ndarray) -> float:
    """Weighted average standalone vol divided by portfolio vol. 1.0 = no diversification."""
    stdev = np.sqrt(np.diag(cov))
    return float((w @ stdev) / portfolio_vol(w, cov))


def effective_n(w: np.ndarray) -> float:
    """
    Inverse Herfindahl -- the number of equally-weighted positions the
    allocation is equivalent to. This is the direct, numeric answer to
    "does this collapse into one or two ETFs": a 10-holding book with an
    effective N of 2.1 is a two-ETF portfolio wearing a disguise, no
    matter how many non-zero weights it prints.
    """
    w = np.asarray(w, dtype=float)
    return float(1.0 / np.sum(w ** 2))


def effective_n_risk(w: np.ndarray, cov: np.ndarray) -> float:
    """
    Effective N measured on RISK contributions rather than capital. The
    stricter and more informative of the two: 40% in cash and 60% in
    equities has a respectable capital-based effective N while carrying
    essentially all of its risk in one place.
    """
    rc = risk_contributions(np.asarray(w, dtype=float), cov)
    share = rc / rc.sum()
    return float(1.0 / np.sum(share ** 2))


# ---------------------------------------------------------------------------
# Solver core
# ---------------------------------------------------------------------------

def _solve(objective: Callable[[np.ndarray], float], cs: ConstraintSet,
           n_starts: int = 6, seed: int = 7) -> np.ndarray:
    """
    SLSQP from several starting points, keeping the best feasible result.
    Multiple starts because these objectives (ERC and maximum
    diversification especially) are not convex in w, and a single start
    can settle in a local minimum that looks plausible and is not.
    """
    rng = np.random.default_rng(seed)
    bounds = cs.bounds_list()
    cons = cs.scipy_constraints()
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    starts = [cs.feasible_start()]
    for _ in range(max(0, n_starts - 1)):
        raw = rng.random(len(cs.tickers))
        w = lo + raw * (hi - lo)
        starts.append(w / w.sum())

    best_w, best_f = None, np.inf
    for w0 in starts:
        try:
            res = minimize(objective, w0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        except Exception:
            continue
        if not res.success:
            continue
        w = np.clip(res.x, lo, hi)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        f = objective(w)
        if np.isfinite(f) and f < best_f:
            best_w, best_f = w, f

    if best_w is None:
        # Never silently return an equal weight labelled as an optimisation
        # result -- the caller has to know the solve failed.
        raise RuntimeError("No feasible solution found from any starting point. "
                           "Check that the group bands and per-asset bounds are mutually satisfiable.")
    return best_w


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def min_variance(cov: pd.DataFrame, cs: ConstraintSet, **kw) -> pd.Series:
    C = cov.loc[cs.tickers, cs.tickers].to_numpy()
    w = _solve(lambda w: float(w @ C @ w), cs, **kw)
    return pd.Series(w, index=cs.tickers)


def max_diversification(cov: pd.DataFrame, cs: ConstraintSet, **kw) -> pd.Series:
    """
    Choueifaty's maximum-diversification portfolio: maximise the ratio of
    weighted-average standalone volatility to realised portfolio
    volatility. It has no view on returns at all and explicitly rewards
    holding assets that are individually risky but mutually uncorrelated
    -- structurally the opposite tendency to minimum variance, which is
    why both are in the blend.
    """
    C = cov.loc[cs.tickers, cs.tickers].to_numpy()
    w = _solve(lambda w: -diversification_ratio(w, C), cs, **kw)
    return pd.Series(w, index=cs.tickers)


def equal_risk_contribution(cov: pd.DataFrame, cs: ConstraintSet, **kw) -> pd.Series:
    """
    Risk parity: every holding contributes the same share of portfolio
    volatility. Uses no return estimates whatsoever, so it is completely
    immune to the estimation-error problem that drives concentration --
    at the cost of being unable to express any view.
    """
    C = cov.loc[cs.tickers, cs.tickers].to_numpy()
    n = len(cs.tickers)

    def obj(w):
        rc = risk_contributions(w, C)
        return float(np.sum((rc / rc.sum() - 1.0 / n) ** 2))

    w = _solve(obj, cs, **kw)
    return pd.Series(w, index=cs.tickers)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, cs: ConstraintSet,
               rf: float = 0.0275, **kw) -> pd.Series:
    """
    Maximum Sharpe on the Black-Litterman posterior. This is the only
    objective in the blend that uses expected returns, which is precisely
    why the posterior -- not sample means -- is what it is handed.
    """
    C = cov.loc[cs.tickers, cs.tickers].to_numpy()
    m = mu.loc[cs.tickers].to_numpy()

    def obj(w):
        return -float((w @ m - rf) / portfolio_vol(w, C))

    w = _solve(obj, cs, **kw)
    return pd.Series(w, index=cs.tickers)


def min_cvar(returns: pd.DataFrame, cs: ConstraintSet, confidence: float = 0.95,
             freq: str = "W") -> pd.Series:
    """
    Minimum conditional value-at-risk, via the Rockafellar-Uryasev linear
    programme. Unlike variance, CVaR is sensitive to the shape of the
    left tail specifically, so it penalises negative skew and fat tails
    that a covariance matrix cannot see at all.

    Solved on weekly returns by default: a strategic allocation is not
    managing daily noise, weekly data is far less contaminated by
    non-synchronous closing times across TSX/NYSE listings, and it keeps
    the LP at a few hundred rows instead of several thousand.

    Formulation -- minimise  alpha + 1/((1-c)T) * sum_t u_t
    subject to  u_t >= -r_t.w - alpha,  u_t >= 0, plus the usual weight
    constraints. Variables are ordered [w (n), alpha (1), u (T)].
    """
    R = returns[cs.tickers].resample(freq).sum().dropna() if freq else returns[cs.tickers]
    R = R.to_numpy()
    T, n = R.shape
    scale = 1.0 / ((1.0 - confidence) * T)

    c = np.concatenate([np.zeros(n), [1.0], np.full(T, scale)])

    # -R w - alpha - u <= 0
    A_ub = np.hstack([-R, -np.ones((T, 1)), -np.eye(T)])
    b_ub = np.zeros(T)

    # Rows and their right-hand sides are appended together, in lockstep --
    # accumulating them in separate lists and concatenating at the end is how
    # a group's floor silently ends up applied to another group's ceiling.
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for g in cs.groups:
        sel = np.zeros(n + 1 + T)
        for t in g.tickers:
            if t in cs.tickers:
                sel[cs.tickers.index(t)] = 1.0
        if not sel.any():
            continue
        rows.append(-sel); rhs.append(-g.min_weight)   # -sum w <= -min_weight
        rows.append(sel);  rhs.append(g.max_weight)    #  sum w <=  max_weight
    if rows:
        A_ub = np.vstack([A_ub, np.array(rows)])
        b_ub = np.concatenate([b_ub, np.array(rhs)])

    A_eq = np.zeros((1, n + 1 + T)); A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = cs.bounds_list() + [(None, None)] + [(0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"min-CVaR LP failed: {res.message}")
    w = np.clip(res.x[:n], 0, None)
    return pd.Series(w / w.sum(), index=cs.tickers)


# ---------------------------------------------------------------------------
# Michaud resampling
# ---------------------------------------------------------------------------

def resampled_weights(
    returns: pd.DataFrame,
    cs: ConstraintSet,
    objective: str,
    mu: Optional[pd.Series] = None,
    bear_mask: Optional[pd.Series] = None,
    n_resamples: int = 120,
    block_size: int = 21,
    bear_weight: float = 0.35,
    rf: float = 0.0275,
    seed: int = 11,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Michaud resampled optimisation. Each iteration draws a block bootstrap
    of the return history, re-estimates covariance (and, for the Sharpe
    objective, rescales expected returns by the resample's own realised
    means blended back toward the prior), re-solves, and stores the
    weights. The average across resamples is the resampled allocation;
    the dispersion across resamples is returned too, and is the more
    interesting output -- it says how much of a given weight is signal and
    how much is an artefact of this particular sample path.

    Blocks rather than IID draws, because IID resampling destroys
    volatility clustering and would make every resample look calmer than
    any real market.
    """
    from core.estimation import regime_blended_covariance, shrunk_covariance

    rng = np.random.default_rng(seed)
    idx = returns.index
    n_obs = len(idx)
    n_blocks = int(np.ceil(n_obs / block_size))

    draws = []
    for _ in range(n_resamples):
        starts = rng.integers(0, n_obs - block_size, size=n_blocks)
        pos = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n_obs]
        sample = returns.iloc[pos]

        if bear_mask is not None:
            sample_mask = bear_mask.reindex(idx).ffill().fillna(False).iloc[pos]
            sample_mask.index = sample.index
            cov_s, _ = regime_blended_covariance(sample, sample_mask, bear_weight=bear_weight)
        else:
            cov_s, _ = shrunk_covariance(sample)

        try:
            if objective == "min_variance":
                w = min_variance(cov_s, cs, n_starts=2)
            elif objective == "max_diversification":
                w = max_diversification(cov_s, cs, n_starts=2)
            elif objective == "equal_risk_contribution":
                w = equal_risk_contribution(cov_s, cs, n_starts=2)
            elif objective == "max_sharpe":
                if mu is None:
                    raise ValueError("max_sharpe resampling needs expected returns.")
                w = max_sharpe(mu, cov_s, cs, rf=rf, n_starts=2)
            elif objective == "min_cvar":
                w = min_cvar(sample, cs)
            else:
                raise ValueError(f"Unknown objective {objective!r}")
        except RuntimeError:
            continue
        draws.append(w)

    if not draws:
        raise RuntimeError(f"Every resample failed for objective {objective!r}.")

    frame = pd.DataFrame(draws)
    return frame.mean(), frame


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

STRATEGY_LABELS = {
    "min_variance": "Minimum variance",
    "equal_risk_contribution": "Equal risk contribution",
    "max_diversification": "Maximum diversification",
    "max_sharpe": "Max Sharpe (Black-Litterman)",
    "min_cvar": "Minimum CVaR (95%)",
}


def build_all_strategies(
    returns: pd.DataFrame,
    cov: pd.DataFrame,
    mu: pd.Series,
    cs: ConstraintSet,
    rf: float = 0.0275,
    resample: bool = True,
    n_resamples: int = 120,
    bear_mask: Optional[pd.Series] = None,
    bear_weight: float = 0.35,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Solve every objective, optionally with Michaud resampling, and return
    (weights per strategy, per-strategy resample dispersion).
    """
    weights, dispersion = {}, {}

    direct = {
        "min_variance": lambda: min_variance(cov, cs),
        "equal_risk_contribution": lambda: equal_risk_contribution(cov, cs),
        "max_diversification": lambda: max_diversification(cov, cs),
        "max_sharpe": lambda: max_sharpe(mu, cov, cs, rf=rf),
        "min_cvar": lambda: min_cvar(returns, cs),
    }

    for name, fn in direct.items():
        if resample:
            w, frame = resampled_weights(returns, cs, name, mu=mu, bear_mask=bear_mask,
                                         n_resamples=n_resamples, bear_weight=bear_weight, rf=rf)
            dispersion[name] = frame
        else:
            w = fn()
        weights[name] = w

    return pd.DataFrame(weights).reindex(cs.tickers), dispersion


def blend_strategies(strategy_weights: pd.DataFrame,
                     weights_of_strategies: Optional[dict[str, float]] = None) -> pd.Series:
    """
    The recommended allocation: a weighted average across objectives,
    renormalised. Equal weighting is the default because there is no
    defensible basis for declaring one of these objectives correct in
    advance -- and an equal blend of estimators that disagree is, in
    practice, more robust out of sample than any single one of them.
    """
    if weights_of_strategies is None:
        blended = strategy_weights.mean(axis=1)
    else:
        s = pd.Series(weights_of_strategies)
        s = s / s.sum()
        blended = (strategy_weights[s.index] * s).sum(axis=1)
    return blended / blended.sum()


def round_to_tradeable(weights: pd.Series, step: float = 0.005,
                       drop_below: float = 0.01) -> pd.Series:
    """
    Round to a tradeable grid and drop dust.

    A 0.7% target weight is not a real position: on a mid-five-figure
    book it is a handful of shares, its commission and bid-ask cost
    exceed any diversification it adds, and it will drift out of band on
    noise. Positions below `drop_below` are zeroed and the freed weight
    is redistributed proportionally across the survivors.
    """
    w = weights.copy()
    w[w < drop_below] = 0.0
    if w.sum() <= 0:
        raise ValueError("All weights were dropped as dust; check drop_below.")
    w = w / w.sum()
    w = (w / step).round() * step
    # Rounding rarely lands exactly on 1.0; put the residual on the largest
    # position, where it is proportionally least distorting.
    residual = 1.0 - w.sum()
    if abs(residual) > 1e-9:
        w.loc[w.idxmax()] += residual
    return w


def summarise_allocation(w: pd.Series, cov: pd.DataFrame, mu: pd.Series,
                         rf: float = 0.0275) -> dict:
    tickers = list(w.index)
    C = cov.loc[tickers, tickers].to_numpy()
    wv = w.to_numpy()
    vol = portfolio_vol(wv, C)
    exp_ret = float(wv @ mu.loc[tickers].to_numpy())
    rc = risk_contributions(wv, C)
    return {
        "expected_return": exp_ret,
        "volatility": vol,
        "sharpe": (exp_ret - rf) / vol,
        "diversification_ratio": diversification_ratio(wv, C),
        "effective_n": effective_n(wv),
        "effective_n_risk": effective_n_risk(wv, C),
        "n_positions": int((w > 1e-6).sum()),
        "risk_contributions": pd.Series(rc / rc.sum(), index=tickers),
    }
