"""
core/tracker.py

Two things the analysis layer cannot do on its own: turn target weights
into an actual order ticket, and track the live book forward from the
day it is funded.

Trade sizing
------------
`build_order_ticket()` converts CAD target weights into whole-share
orders at current prices, in each security's own quote currency. USD-
listed holdings are sized by converting the CAD budget at the current
rate, because that is the actual mechanic -- a Canadian investor funds a
USD position with converted dollars (via Norbert's Gambit, per the
thesis) and then holds an asset quoted in USD.

Whole shares matter more than they look. On a five-figure account a 4.5%
target in a $120 ETF is a handful of shares, and the difference between
rounding to 3 and to 4 is a third of the position. The ticket therefore
reports the achievable weight after rounding, not the target, so the
tracking error introduced by lot sizes is visible before trading rather
than discovered afterwards.

Forward tracking
----------------
`forward_performance()` measures the funded book from its inception date
using realised prices, and places that path inside a bootstrapped
distribution of paths the allocation could plausibly have taken. The
percentile readout is the useful output: knowing the book is up 3.1% is
close to meaningless on its own, while knowing that lands at the 58th
percentile of what this allocation does over four months says whether
anything has actually gone differently than expected.

The distinction from a backtest is worth keeping sharp. The backtest is
in-sample and retrospective; this is out-of-sample and live, and it is
the only evidence that ever accumulates about whether the allocation
works. It also accumulates slowly: a year of forward data is nowhere
near enough to conclude anything, and the module says so rather than
inviting over-reaction to a few months of noise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.fx import usdcad
from core.returns import total_return_series

TRACKER_PATH = Path(__file__).resolve().parent.parent / "data" / "tracker.json"
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Order ticket
# ---------------------------------------------------------------------------

def latest_prices(securities) -> pd.DataFrame:
    rows = {}
    fx = float(usdcad().iloc[-1])
    for s in securities:
        px = s.prices["Close"].dropna()
        if px.empty:
            continue
        last = float(px.iloc[-1])
        rows[s.ticker] = {
            "name": s.name,
            "currency": s.currency,
            "price_native": last,
            "price_cad": last * (fx if s.currency == "USD" else 1.0),
            "as_of": px.index[-1],
        }
    return pd.DataFrame(rows).T


def build_order_ticket(weights: pd.Series, securities, portfolio_value_cad: float,
                       allow_fractional: bool = False) -> pd.DataFrame:
    """
    Target weights -> share quantities at the latest close.

    `achieved_weight` is what the account will actually hold once whole
    shares are rounded, and `drift_vs_target` is the gap. On a small book
    that gap can be a full percentage point on the more expensive
    holdings; it is reported rather than smoothed over.
    """
    px = latest_prices(securities)
    rows = []
    for t, w in weights.items():
        if t not in px.index:
            continue
        p_cad = float(px.loc[t, "price_cad"])
        p_nat = float(px.loc[t, "price_native"])
        budget_cad = float(w) * portfolio_value_cad
        raw_shares = budget_cad / p_cad if p_cad > 0 else 0.0
        shares = raw_shares if allow_fractional else float(np.floor(raw_shares))
        cost_cad = shares * p_cad
        rows.append({
            "ticker": t,
            "name": px.loc[t, "name"],
            "currency": px.loc[t, "currency"],
            "price_native": p_nat,
            "price_cad": p_cad,
            "target_weight": float(w),
            "target_cad": budget_cad,
            "shares": shares,
            "cost_native": shares * p_nat,
            "cost_cad": cost_cad,
            "achieved_weight": cost_cad / portfolio_value_cad if portfolio_value_cad else 0.0,
        })
    df = pd.DataFrame(rows).set_index("ticker")
    df["drift_vs_target"] = df["achieved_weight"] - df["target_weight"]
    return df


def ticket_residual_cash(ticket: pd.DataFrame, portfolio_value_cad: float) -> float:
    """CAD left unallocated after whole-share rounding."""
    return float(portfolio_value_cad - ticket["cost_cad"].sum())


# ---------------------------------------------------------------------------
# Persisted tracker state
# ---------------------------------------------------------------------------

@dataclass
class TrackerState:
    inception: str                       # ISO date the allocation was funded
    initial_value_cad: float
    weights: dict[str, float]
    note: str = ""
    contributions: list[dict] = field(default_factory=list)  # [{"date":..., "amount_cad":...}]

    def weight_series(self) -> pd.Series:
        s = pd.Series(self.weights, dtype=float)
        return s / s.sum()


def save_tracker(state: TrackerState, path: Path = TRACKER_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    return path


def load_tracker(path: Path = TRACKER_PATH) -> Optional[TrackerState]:
    if not path.exists():
        return None
    try:
        return TrackerState(**json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Forward performance
# ---------------------------------------------------------------------------

def realised_since(securities, weights: pd.Series, inception: str,
                   initial_value: float = 100_000.0,
                   rebalance: str = "none") -> pd.DataFrame:
    """
    Actual CAD NAV of the allocation from `inception` to today, using each
    holding's realised total return. Defaults to no rebalancing, since
    that is what a freshly funded book does between review dates.
    """
    from core.backtest_engine import BacktestConfig, run_backtest

    cols = {}
    for s in securities:
        if s.ticker not in weights.index:
            continue
        cols[s.ticker] = total_return_series(s.prices, currency=s.currency, in_cad=True)
    rets = pd.DataFrame(cols).loc[inception:].dropna(how="any")
    if rets.empty:
        raise ValueError(f"No overlapping return data on or after {inception}.")

    cfg = BacktestConfig(rebalance=rebalance, cost_bps=0.0, initial_value=initial_value)
    res = run_backtest(rets, weights.reindex(rets.columns).fillna(0.0), cfg)
    out = pd.DataFrame({"nav": res.nav, "return": res.returns})
    out["cumulative"] = out["nav"] / initial_value - 1.0
    out["drawdown"] = res.drawdown()
    return out


def expectation_cone(history: pd.DataFrame, weights: pd.Series, n_days: int,
                     initial_value: float = 100_000.0, n_sims: int = 4000,
                     block_size: int = 21, seed: int = 5,
                     percentiles: tuple = (5, 25, 50, 75, 95)) -> pd.DataFrame:
    """
    Bootstrapped distribution of NAV paths for this allocation over
    `n_days`, built by block-resampling the historical return matrix
    ACROSS ASSETS JOINTLY -- whole rows are drawn, never each column
    independently, so the correlation structure and every crash day
    survive into the simulation. Sampling columns separately would
    manufacture diversification that has never existed.

    Returns one column per requested percentile: the fan chart the live
    NAV is plotted inside.
    """
    cols = [c for c in weights.index if c in history.columns]
    R = history[cols].dropna(how="any").to_numpy()
    w = weights[cols].to_numpy(dtype=float)
    w = w / w.sum()
    port = R @ w

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_days / block_size))
    paths = np.empty((n_sims, n_days))
    for i in range(n_sims):
        starts = rng.integers(0, len(port) - block_size, size=n_blocks)
        seq = np.concatenate([port[s:s + block_size] for s in starts])[:n_days]
        paths[i] = initial_value * np.cumprod(1.0 + seq)

    out = {f"p{p}": np.percentile(paths, p, axis=0) for p in percentiles}
    return pd.DataFrame(out, index=np.arange(1, n_days + 1))


def percentile_of_actual(history: pd.DataFrame, weights: pd.Series,
                         actual_cumulative_return: float, n_days: int,
                         n_sims: int = 4000, block_size: int = 21, seed: int = 5) -> float:
    """
    Where the realised return sits in the bootstrapped distribution for
    the same elapsed horizon. 50 is exactly as expected; below ~10 or
    above ~90 is genuinely unusual and worth explaining.
    """
    cols = [c for c in weights.index if c in history.columns]
    R = history[cols].dropna(how="any").to_numpy()
    w = weights[cols].to_numpy(dtype=float)
    w = w / w.sum()
    port = R @ w

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_days / block_size))
    totals = np.empty(n_sims)
    for i in range(n_sims):
        starts = rng.integers(0, len(port) - block_size, size=n_blocks)
        seq = np.concatenate([port[s:s + block_size] for s in starts])[:n_days]
        totals[i] = np.prod(1.0 + seq) - 1.0
    return float((totals < actual_cumulative_return).mean() * 100)


def drift_report(securities, state: TrackerState) -> pd.DataFrame:
    """
    Current weights versus target, for the funded book. Positions drift
    with performance; this is the table that says whether a rebalance is
    due and which trades it implies.

    A 20% relative band is the usual trigger -- tight enough that risk
    stays near target, loose enough that ordinary noise is not traded on.
    """
    w_target = state.weight_series()
    nav = realised_since(securities, w_target, state.inception, state.initial_value_cad)

    growth = {}
    for s in securities:
        if s.ticker not in w_target.index:
            continue
        r = total_return_series(s.prices, currency=s.currency, in_cad=True).loc[state.inception:]
        growth[s.ticker] = float((1 + r).prod())
    g = pd.Series(growth)
    current_value = w_target[g.index] * state.initial_value_cad * g
    current_weight = current_value / current_value.sum()

    df = pd.DataFrame({
        "target_weight": w_target[g.index],
        "current_weight": current_weight,
        "current_value_cad": current_value,
        "growth_since_inception": g - 1.0,
    })
    df["absolute_drift"] = df["current_weight"] - df["target_weight"]
    df["relative_drift"] = df["absolute_drift"] / df["target_weight"].replace(0, np.nan)
    df["rebalance_flag"] = df["relative_drift"].abs() > 0.20
    df.attrs["nav"] = nav
    return df.sort_values("absolute_drift", ascending=False)
