"""
core/backtest_engine.py

Rebalanced portfolio backtest.

The distinction that matters here, and that a weighted-average-of-returns
calculation silently gets wrong: a portfolio held with periodic
rebalancing is NOT the same thing as `returns @ weights`. That dot
product implicitly rebalances to target every single day, for free. Over
19 years and ten holdings that fictitious daily rebalancing manufactures
a meaningful amount of return out of nothing -- it is a free
sell-high/buy-low overlay executed at zero cost, and it is the reason
naive backtests of diversified portfolios look better than any investor
ever achieves.

This module drifts the weights between rebalance dates the way real
holdings drift, rebalances on a stated schedule or a drift band, and
charges turnover for the privilege. It also nets off management fees
daily, because a 0.55% MER on gold is not a rounding error against a
6% expected return.

Everything here operates on CAD total returns (see core/returns.py), so
the resulting NAV is what the account is actually worth in the currency
it will be spent in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestConfig:
    rebalance: str = "Q"            # pandas offset alias, or "none", or "band"
    drift_band: float = 0.25        # for "band": rebalance when any weight is >25% relative off target
    cost_bps: float = 8.0           # round-trip trading cost per unit turnover, in bps
    fee_drag: Optional[pd.Series] = None   # annual MER per ticker, charged daily
    initial_value: float = 100_000.0


@dataclass
class BacktestResult:
    nav: pd.Series
    returns: pd.Series
    weights: pd.DataFrame          # actual drifted weights, daily
    turnover: pd.Series            # one-way turnover charged at each rebalance
    rebalance_dates: pd.DatetimeIndex
    total_costs: float
    config: BacktestConfig = field(repr=False, default_factory=BacktestConfig)

    def stats(self, rf_annual: float = 0.0275) -> dict:
        from core.returns import summary_stats
        s = summary_stats(self.returns, rf_annual=rf_annual)
        idx = self.nav
        dd = idx / idx.cummax() - 1.0
        s.update({
            "total_return": float(idx.iloc[-1] / idx.iloc[0] - 1.0),
            "final_value": float(idx.iloc[-1]),
            "calmar": (s["ann_return"] / abs(s["max_drawdown"])) if s["max_drawdown"] else np.nan,
            "worst_day": float(self.returns.min()),
            "best_day": float(self.returns.max()),
            "pct_days_positive": float((self.returns > 0).mean()),
            "avg_annual_turnover": float(self.turnover.sum() / (len(self.returns) / TRADING_DAYS)),
            "total_cost_drag": self.total_costs,
            "current_drawdown": float(dd.iloc[-1]),
        })
        return s

    def drawdown(self) -> pd.Series:
        return self.nav / self.nav.cummax() - 1.0


# pandas 3.0 removed the legacy single-letter annual aliases ("A", "AS") in
# favour of "Y"/"YS", and callers reasonably still pass the old ones. Normalising
# here rather than at each call site keeps a stale alias from raising deep inside
# a backtest.
_FREQ_ALIASES = {"A": "Y", "AS": "YS", "M": "M", "Q": "Q"}


def _rebalance_flags(index: pd.DatetimeIndex, rule: str) -> np.ndarray:
    flags = np.zeros(len(index), dtype=bool)
    if rule in ("none", "band"):
        return flags
    rule = _FREQ_ALIASES.get(rule.upper(), rule)
    try:
        periods = pd.Series(index, index=index).groupby(index.to_period(rule)).max()
    except ValueError as exc:
        raise ValueError(
            f"Unrecognised rebalance rule {rule!r}. Use a pandas period alias "
            f"('M', 'Q', 'Y'), or 'none' / 'band'."
        ) from exc
    # Last trading day within each period is the realistic execution date.
    flags[index.get_indexer(pd.DatetimeIndex(periods.values))] = True
    return flags


def run_backtest(returns: pd.DataFrame, target: pd.Series,
                 config: Optional[BacktestConfig] = None) -> BacktestResult:
    """
    Daily-compounded backtest of a fixed target allocation.

    `returns` must be CAD total returns, `target` a weight vector over the
    same columns. Weights drift with realised returns between rebalances;
    at each rebalance the book is traded back to target and charged
    `cost_bps` on one-way turnover.
    """
    config = config or BacktestConfig()
    cols = [c for c in target.index if c in returns.columns]
    if not cols:
        raise ValueError("No overlap between target weights and the return matrix columns.")
    R = returns[cols].dropna(how="any")
    w_target = target[cols].to_numpy(dtype=float)
    w_target = w_target / w_target.sum()

    daily_fee = np.zeros(len(cols))
    if config.fee_drag is not None:
        daily_fee = config.fee_drag.reindex(cols).fillna(0.0).to_numpy() / TRADING_DAYS

    flags = _rebalance_flags(R.index, config.rebalance)
    use_band = config.rebalance == "band"

    n = len(R)
    w = w_target.copy()
    nav = np.empty(n)
    port_ret = np.empty(n)
    weights_hist = np.empty((n, len(cols)))
    turnover = np.zeros(n)
    value = config.initial_value
    total_cost = 0.0
    rebal_dates = []

    R_arr = R.to_numpy()
    for i in range(n):
        gross = R_arr[i] - daily_fee
        r_p = float(w @ gross)

        # Drift: each holding grows at its own return, then weights renormalise.
        grown = w * (1.0 + gross)
        total = grown.sum()
        w = grown / total if total > 0 else w_target.copy()

        do_rebal = flags[i]
        if use_band and not do_rebal:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(w - w_target) / np.where(w_target > 0, w_target, np.nan)
            do_rebal = bool(np.nanmax(rel) > config.drift_band)

        if do_rebal:
            one_way = float(np.abs(w - w_target).sum()) / 2.0
            cost = one_way * config.cost_bps / 10_000.0
            r_p -= cost
            total_cost += cost
            turnover[i] = one_way
            w = w_target.copy()
            rebal_dates.append(R.index[i])

        value *= (1.0 + r_p)
        nav[i] = value
        port_ret[i] = r_p
        weights_hist[i] = w

    return BacktestResult(
        nav=pd.Series(nav, index=R.index, name="nav"),
        returns=pd.Series(port_ret, index=R.index, name="return"),
        weights=pd.DataFrame(weights_hist, index=R.index, columns=cols),
        turnover=pd.Series(turnover, index=R.index, name="turnover"),
        rebalance_dates=pd.DatetimeIndex(rebal_dates),
        total_costs=total_cost,
        config=config,
    )


def compare_backtests(returns: pd.DataFrame, targets: dict[str, pd.Series],
                      config: Optional[BacktestConfig] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run several allocations over one window; returns (NAV frame, stats frame)."""
    navs, rows = {}, {}
    for name, w in targets.items():
        res = run_backtest(returns, w, config)
        navs[name] = res.nav
        rows[name] = res.stats()
    stats = pd.DataFrame(rows).T
    keep = ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
            "total_return", "avg_annual_turnover", "total_cost_drag"]
    return pd.DataFrame(navs), stats[[c for c in keep if c in stats.columns]]


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward(
    returns: pd.DataFrame,
    fit_fn,
    train_years: float = 5.0,
    step_months: int = 12,
    config: Optional[BacktestConfig] = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Out-of-sample test of the whole construction process.

    `fit_fn(train_returns) -> weights` is called on an expanding window
    and the resulting allocation is held for the following `step_months`,
    then refitted. Nothing from the holding period is visible at fit time.

    This is the only number in the project that says anything about
    whether the METHOD works, as opposed to whether one particular set of
    weights happened to suit the sample it was estimated on. A full-sample
    backtest of an allocation derived from that same full sample is
    circular by construction, and is reported here only as a reference.
    """
    config = config or BacktestConfig()
    idx = returns.index
    start = idx.min() + pd.DateOffset(years=int(train_years))
    if start >= idx.max():
        raise ValueError("Not enough history for the requested training window.")

    segments, all_weights = [], {}
    cursor = start
    while cursor < idx.max():
        nxt = min(cursor + pd.DateOffset(months=step_months), idx.max())
        train = returns.loc[:cursor].iloc[:-1]
        test = returns.loc[cursor:nxt]
        if len(test) < 5 or len(train) < 250:
            cursor = nxt
            continue
        try:
            w = fit_fn(train)
        except Exception:
            cursor = nxt
            continue
        all_weights[cursor] = w
        res = run_backtest(test, w, config)
        segments.append(res.returns)
        cursor = nxt

    if not segments:
        raise RuntimeError("Walk-forward produced no valid segments.")
    oos = pd.concat(segments).sort_index()
    oos = oos[~oos.index.duplicated(keep="first")]
    return oos, pd.DataFrame(all_weights).T
