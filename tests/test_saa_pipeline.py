"""
tests/test_saa_pipeline.py

Guards on the properties that, if they broke, would produce plausible and
wrong numbers rather than an error. These are not coverage tests -- each one
corresponds to a specific way this model has been or could be silently wrong.

The suite deliberately uses synthetic data wherever possible so it runs
offline and deterministically. The two tests that need real prices are marked
`network` and can be deselected with `-m "not network"`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.estimation import (
    View,
    black_litterman,
    implied_equilibrium_returns,
    regime_blended_covariance,
    shrunk_covariance,
)
from core.optimizer import (
    ConstraintSet,
    GroupConstraint,
    blend_strategies,
    effective_n,
    equal_risk_contribution,
    max_diversification,
    min_cvar,
    min_variance,
    risk_contributions,
    round_to_tradeable,
)


@pytest.fixture(scope="module")
def synthetic():
    """
    Five assets with a deliberately awkward structure: two nearly identical
    equities, one diversifier, one near-riskless sleeve, and one high-vol
    asset with the best in-sample mean. The last one is the trap -- a naive
    optimiser will pour everything into it.
    """
    rng = np.random.default_rng(0)
    n = 2000
    market = rng.normal(0.0003, 0.011, n)
    data = {
        "EQ_A": market + rng.normal(0, 0.004, n),
        "EQ_B": market + rng.normal(0, 0.0045, n),
        "DIVERSIFIER": rng.normal(0.0002, 0.009, n) - 0.25 * market,
        "CASH": rng.normal(0.00008, 0.0004, n),
        "LUCKY": rng.normal(0.0009, 0.020, n),
    }
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(data, index=idx)


@pytest.fixture(scope="module")
def cs(synthetic):
    tickers = list(synthetic.columns)
    return ConstraintSet(
        tickers=tickers,
        bounds={"EQ_A": (0.05, 0.35), "EQ_B": (0.05, 0.35), "DIVERSIFIER": (0.05, 0.30),
                "CASH": (0.05, 0.30), "LUCKY": (0.0, 0.25)},
        groups=[GroupConstraint("equity", ["EQ_A", "EQ_B"], 0.20, 0.60),
                GroupConstraint("defensive", ["CASH"], 0.05, 0.30)],
        default_bounds=(0.0, 0.30),
    )


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def test_solutions_respect_bounds_and_groups(synthetic, cs):
    cov, _ = shrunk_covariance(synthetic)
    for solver in (min_variance, equal_risk_contribution, max_diversification):
        w = solver(cov, cs)
        assert np.isclose(w.sum(), 1.0, atol=1e-6), f"{solver.__name__} weights must sum to 1"
        assert not cs.violations(w), f"{solver.__name__} violated {cs.violations(w)}"


def test_min_cvar_group_constraints_are_paired_correctly(synthetic, cs):
    """
    Regression guard. The CVaR linear programme builds group floor and ceiling
    rows in one loop; accumulating rows and right-hand sides in separate lists
    would apply one group's floor to another group's ceiling, which produces a
    feasible-looking and completely wrong allocation.
    """
    w = min_cvar(synthetic, cs)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert not cs.violations(w), cs.violations(w)
    assert 0.20 - 1e-4 <= w[["EQ_A", "EQ_B"]].sum() <= 0.60 + 1e-4


def test_infeasible_constraints_raise_rather_than_guess(synthetic):
    """A failed solve must not return an equal weight dressed as a result."""
    tickers = list(synthetic.columns)
    impossible = ConstraintSet(
        tickers=tickers,
        bounds={t: (0.0, 0.10) for t in tickers},  # five assets capped at 10% cannot reach 100%
        groups=[], default_bounds=(0.0, 0.10),
    )
    cov, _ = shrunk_covariance(synthetic)
    with pytest.raises(RuntimeError):
        min_variance(cov, impossible)


# ---------------------------------------------------------------------------
# The core claim: better inputs, not constraints, prevent concentration
# ---------------------------------------------------------------------------

def test_equilibrium_returns_reproduce_the_reference_portfolio(synthetic):
    """
    Reverse optimisation must be self-consistent: feeding pi back into an
    unconstrained maximum-Sharpe solve has to return the reference weights.
    If this breaks, every allocation is anchored to something arbitrary.
    """
    cov, _ = shrunk_covariance(synthetic)
    ref = pd.Series([0.25, 0.20, 0.20, 0.25, 0.10], index=synthetic.columns)
    pi = implied_equilibrium_returns(cov, ref, risk_aversion=2.8, rf=0.02)

    # Unconstrained tangency weights from pi should recover ref up to scale.
    excess = (pi - 0.02).to_numpy()
    raw = np.linalg.solve(cov.to_numpy(), excess)
    recovered = pd.Series(raw / raw.sum(), index=synthetic.columns)
    assert np.allclose(recovered.to_numpy(), ref.to_numpy(), atol=1e-6)


def test_reference_weights_must_sum_to_one(synthetic):
    cov, _ = shrunk_covariance(synthetic)
    bad = pd.Series([0.5, 0.2, 0.2, 0.2, 0.2], index=synthetic.columns)
    with pytest.raises(ValueError):
        implied_equilibrium_returns(cov, bad)


def test_zero_confidence_views_leave_the_prior_untouched(synthetic):
    """Silence means 'hold the policy portfolio', not 'hold nothing'."""
    cov, _ = shrunk_covariance(synthetic)
    ref = pd.Series(0.2, index=synthetic.columns)
    pi = implied_equilibrium_returns(cov, ref)

    same, _ = black_litterman(cov, pi, [])
    assert np.allclose(same.to_numpy(), pi.to_numpy())

    weak = View("negligible", {"EQ_A": 1.0, "EQ_B": -1.0}, q=0.05, confidence=1e-4)
    nearly, _ = black_litterman(cov, pi, [weak])
    assert np.allclose(nearly.to_numpy(), pi.to_numpy(), atol=5e-3)


def test_view_direction_moves_the_posterior_the_right_way(synthetic):
    cov, _ = shrunk_covariance(synthetic)
    ref = pd.Series(0.2, index=synthetic.columns)
    pi = implied_equilibrium_returns(cov, ref)
    v = View("A over B", {"EQ_A": 1.0, "EQ_B": -1.0}, q=0.05, confidence=0.8)
    post, _ = black_litterman(cov, pi, [v])
    assert (post["EQ_A"] - post["EQ_B"]) > (pi["EQ_A"] - pi["EQ_B"])


def test_unknown_ticker_in_a_view_raises(synthetic):
    cov, _ = shrunk_covariance(synthetic)
    pi = implied_equilibrium_returns(cov, pd.Series(0.2, index=synthetic.columns))
    with pytest.raises(ValueError):
        black_litterman(cov, pi, [View("bad", {"NOT_HELD": 1.0}, q=0.01)])


def test_blending_objectives_beats_any_single_one_on_concentration(synthetic, cs):
    cov, _ = shrunk_covariance(synthetic)
    weights = pd.DataFrame({
        "mv": min_variance(cov, cs),
        "erc": equal_risk_contribution(cov, cs),
        "md": max_diversification(cov, cs),
    })
    blended = blend_strategies(weights)
    assert np.isclose(blended.sum(), 1.0)
    # The blend is never more concentrated than the most concentrated input.
    worst = min(effective_n(weights[c].to_numpy()) for c in weights.columns)
    assert effective_n(blended.to_numpy()) >= worst - 1e-9


# ---------------------------------------------------------------------------
# Risk decomposition
# ---------------------------------------------------------------------------

def test_risk_contributions_sum_to_portfolio_volatility(synthetic):
    cov, _ = shrunk_covariance(synthetic)
    w = np.array([0.25, 0.20, 0.20, 0.25, 0.10])
    rc = risk_contributions(w, cov.to_numpy())
    assert np.isclose(rc.sum(), np.sqrt(w @ cov.to_numpy() @ w))


def test_equal_risk_contribution_actually_equalises_risk(synthetic):
    """A loose ERC solve would still satisfy the constraints and be wrong."""
    tickers = list(synthetic.columns)
    unbounded = ConstraintSet(tickers=tickers, bounds={t: (0.0, 1.0) for t in tickers},
                              groups=[], default_bounds=(0.0, 1.0))
    cov, _ = shrunk_covariance(synthetic)
    w = equal_risk_contribution(cov, unbounded)
    rc = risk_contributions(w.to_numpy(), cov.to_numpy())
    share = rc / rc.sum()
    assert share.max() - share.min() < 0.02, f"risk shares not equalised: {share}"


def test_effective_n_bounds():
    assert np.isclose(effective_n(np.array([1.0, 0.0, 0.0, 0.0])), 1.0)
    assert np.isclose(effective_n(np.full(4, 0.25)), 4.0)


# ---------------------------------------------------------------------------
# Regime blending
# ---------------------------------------------------------------------------

def test_bear_weight_zero_reduces_to_the_plain_shrunk_matrix(synthetic):
    mask = pd.Series(synthetic["EQ_A"] < synthetic["EQ_A"].quantile(0.3), index=synthetic.index)
    blended, _ = regime_blended_covariance(synthetic, mask, bear_weight=0.0)
    plain, _ = shrunk_covariance(synthetic)
    assert np.allclose(blended.to_numpy(), plain.to_numpy())


def test_insufficient_bear_days_falls_back_visibly(synthetic):
    """
    With too few bear observations to estimate a matrix, the function must fall
    back to the full-sample covariance AND say so -- not emit a confident-looking
    matrix built on nothing.
    """
    mask = pd.Series(False, index=synthetic.index)
    mask.iloc[:10] = True
    cov, diag = regime_blended_covariance(synthetic, mask, bear_weight=0.5)
    assert diag["bear_weight_used"] == 0.0
    assert "note" in diag
    plain, _ = shrunk_covariance(synthetic)
    assert np.allclose(cov.to_numpy(), plain.to_numpy())


def test_bear_weight_outside_zero_one_raises(synthetic):
    mask = pd.Series(True, index=synthetic.index)
    with pytest.raises(ValueError):
        regime_blended_covariance(synthetic, mask, bear_weight=1.5)


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------

def test_rounding_drops_dust_and_still_sums_to_one():
    w = pd.Series({"A": 0.40, "B": 0.30, "C": 0.257, "D": 0.004, "E": 0.039})
    out = round_to_tradeable(w, step=0.005, drop_below=0.01)
    assert np.isclose(out.sum(), 1.0)
    assert out["D"] == 0.0, "sub-1% dust must be dropped, not rounded up"
    assert (out.drop("D") > 0).all()


def test_rounding_lands_on_the_grid():
    w = pd.Series({"A": 0.3333, "B": 0.3333, "C": 0.3334})
    out = round_to_tradeable(w, step=0.005, drop_below=0.01)
    assert np.isclose(out.sum(), 1.0)
    # Every weight but the residual-absorbing largest sits on the 0.5% grid.
    off_grid = [t for t in out.index if not np.isclose((out[t] / 0.005) % 1, 0, atol=1e-6)]
    assert len(off_grid) <= 1


# ---------------------------------------------------------------------------
# Backtest mechanics
# ---------------------------------------------------------------------------

def test_backtest_is_not_a_daily_free_rebalance(synthetic):
    """
    `returns @ weights` implicitly rebalances daily at zero cost. A buy-and-hold
    backtest must differ from it -- if these agree, the engine is not drifting
    weights and every backtest number is optimistic.
    """
    from core.backtest_engine import BacktestConfig, run_backtest

    w = pd.Series(0.2, index=synthetic.columns)
    naive = float((1 + synthetic @ w).prod() - 1)
    held = run_backtest(synthetic, w, BacktestConfig(rebalance="none", cost_bps=0.0))
    drifted = held.stats()["total_return"]
    assert not np.isclose(naive, drifted, atol=1e-4)


def test_turnover_costs_reduce_return(synthetic):
    from core.backtest_engine import BacktestConfig, run_backtest

    w = pd.Series(0.2, index=synthetic.columns)
    free = run_backtest(synthetic, w, BacktestConfig(rebalance="M", cost_bps=0.0))
    costly = run_backtest(synthetic, w, BacktestConfig(rebalance="M", cost_bps=50.0))
    assert costly.stats()["total_return"] < free.stats()["total_return"]
    assert costly.total_costs > 0


def test_fee_drag_reduces_return(synthetic):
    from core.backtest_engine import BacktestConfig, run_backtest

    w = pd.Series(0.2, index=synthetic.columns)
    fees = pd.Series(0.01, index=synthetic.columns)
    gross = run_backtest(synthetic, w, BacktestConfig(rebalance="Q", cost_bps=0.0))
    net = run_backtest(synthetic, w, BacktestConfig(rebalance="Q", cost_bps=0.0, fee_drag=fees))
    assert net.stats()["ann_return"] < gross.stats()["ann_return"]


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def test_total_return_includes_distributions():
    from core.returns import total_return_series

    idx = pd.bdate_range("2024-01-01", periods=5)
    px = pd.DataFrame({
        "Close": [100.0, 100.0, 100.0, 100.0, 100.0],
        "Dividends": [0.0, 0.0, 2.0, 0.0, 0.0],
        "Capital Gains": [0.0] * 5,
    }, index=idx)
    r = total_return_series(px, currency="CAD")
    assert np.isclose(r.iloc[1], 0.02), "a flat price with a $2 distribution is a 2% return"
    assert np.isclose(r.drop(r.index[1]).sum(), 0.0)


def test_geometric_annualisation_is_below_arithmetic():
    """
    The arithmetic convention overstates compounded outcomes by roughly
    sigma^2/2 -- for a 20%-vol asset that is ~2%/yr of pure artefact, enough
    to reorder the optimiser's asset ranking on its own.
    """
    from core.returns import annualise_return

    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0004, 0.013, 2520))
    assert annualise_return(r, geometric=True) < annualise_return(r, geometric=False)


def test_total_wipeout_does_not_produce_a_nonsense_cagr():
    """VOLX.TO's cumulative return is -100%; the CAGR formula must not explode."""
    from core.returns import annualise_return

    r = pd.Series([-0.5, -0.6, -0.99, -1.0])
    assert annualise_return(r) == -1.0


# ---------------------------------------------------------------------------
# Network-dependent
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_fx_series_is_shifted_back_one_business_day():
    """
    The correction that keeps the US sleeve's modelled volatility honest. If
    this regresses, nothing raises -- the numbers just quietly get worse.
    """
    from core.data_loader import get_price_history
    from core.fx import usdcad

    raw = get_price_history("CAD=X")["Close"].dropna()
    fixed = usdcad()
    assert fixed.index.max() < raw.index.max(), "FX index must be shifted earlier than Yahoo's stamps"
    assert not fixed.index.has_duplicates


@pytest.mark.network
def test_pipeline_produces_a_diversified_allocation():
    """
    The end-to-end guard on the project's central success criterion. An
    effective N below 5, or weights resting on many constraints, means the
    estimation layer stopped working and the policy bands are propping up the
    answer -- a regression even though nothing raised.
    """
    from core.saa import run_saa

    result = run_saa(resample=False, verbose=False)
    s = result.summary
    assert s["n_positions"] >= 8
    assert s["effective_n"] > 5.0, f"allocation collapsed: effective N {s['effective_n']:.2f}"
    assert s["effective_n_risk"] > 4.0
    assert result.final_weights.max() < 0.30
    assert not result.violations
    assert np.isclose(result.final_weights.sum(), 1.0, atol=1e-6)


def test_legacy_annual_rebalance_alias_still_works(synthetic):
    """
    Regression guard. pandas 3.0 removed the "A" annual alias in favour of "Y",
    which turned the dashboard's Annual rebalancing option into a crash. The
    engine normalises legacy aliases so a stale rule cannot blow up mid-backtest.
    """
    from core.backtest_engine import BacktestConfig, run_backtest

    w = pd.Series(0.2, index=synthetic.columns)
    a = run_backtest(synthetic, w, BacktestConfig(rebalance="A", cost_bps=0.0))
    y = run_backtest(synthetic, w, BacktestConfig(rebalance="Y", cost_bps=0.0))
    assert np.allclose(a.nav.to_numpy(), y.nav.to_numpy())
    assert len(a.rebalance_dates) > 0


def test_unknown_rebalance_rule_raises_a_clear_error(synthetic):
    from core.backtest_engine import BacktestConfig, run_backtest

    w = pd.Series(0.2, index=synthetic.columns)
    with pytest.raises(ValueError, match="Unrecognised rebalance rule"):
        run_backtest(synthetic, w, BacktestConfig(rebalance="fortnightly"))


def test_every_methodology_entry_is_complete():
    """
    The dashboard promises that each displayed number explains its own
    derivation. An entry with no formula or no steps silently breaks that
    promise -- the tooltip renders, it just says nothing useful.
    """
    from dashboard.methodology import METHODS

    assert len(METHODS) >= 30
    for key, m in METHODS.items():
        assert m.title and m.summary, f"{key} is missing a title or summary"
        assert m.formula or m.formula_plain, f"{key} has no formula"
        assert m.steps, f"{key} has no derivation steps"
        assert m.caveat, f"{key} has no caveat -- every number has a limit"
        assert m.source, f"{key} does not name the code that computes it"
        assert m.tooltip(), f"{key} renders an empty tooltip"


def test_methodology_keys_referenced_by_views_all_exist():
    """A typo in a note("...") key fails silently at runtime; catch it here."""
    import glob
    import re

    from dashboard.methodology import METHODS

    referenced = set()
    for path in glob.glob("dashboard/views/*.py"):
        src = open(path, encoding="utf-8").read()
        for call in re.findall(r"\bnotes?\(([^)]*)\)", src, re.S):
            referenced.update(re.findall(r'"([a-z_]+)"', call))
        for call in re.findall(r"\btip\(\s*\"([a-z_]+)\"", src):
            referenced.add(call)
    referenced -= {"label", "expanded"}
    missing = {k for k in referenced if k not in METHODS}
    assert not missing, f"views reference undefined methodology keys: {missing}"
