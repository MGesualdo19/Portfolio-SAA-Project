"""
analysis/stress.py

Historical stress windows and downside-dependence measures.

Correlation over a full sample answers the wrong question for this
portfolio. The thesis holds gold, floating-rate notes and non-US equity
because they are expected to behave differently *in a crash*, and an
average correlation across nineteen mostly-calm years says almost
nothing about that. Two assets can correlate 0.1 unconditionally and
0.85 in every drawdown that has ever mattered; the unconditional number
is arithmetically correct and practically useless.

Three complementary views are provided:

  1. NAMED CRISIS WINDOWS -- what each holding and the whole portfolio
     actually did over specific, dated episodes. No statistics, no
     model: just the realised path through the GFC, the 2020 COVID
     crash, the 2022 rate shock and others.

  2. CONDITIONAL CORRELATION -- correlation restricted to the worst
     decile of days for a reference asset. This is the direct measure of
     "does the diversifier still diversify when it counts".

  3. DOWNSIDE BETA and TAIL DEPENDENCE -- asymmetry measures. An asset
     with downside beta well above its upside beta gives you the losses
     without the gains, which is the worst possible profile and is
     invisible to any symmetric statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Dated windows, chosen because each one stresses a DIFFERENT joint
# behaviour -- there is no point in five variations of "equities fell".
CRISIS_WINDOWS: dict[str, tuple[str, str, str]] = {
    "GFC (2007-2009)": ("2007-10-09", "2009-03-09",
                        "Systemic credit crisis. The test of whether the fixed-income sleeve "
                        "is a shock absorber or a correlated credit position in disguise."),
    "GFC recovery (2009)": ("2009-03-10", "2009-12-31",
                            "The other half of the test: a hedge that fails to give back its "
                            "gains in the recovery is a permanent drag, not a hedge."),
    "Euro crisis (2011)": ("2011-05-02", "2011-10-03",
                           "Regional shock. Tests whether international diversification "
                           "diversifies or just adds a second correlated equity beta."),
    "Taper tantrum (2013)": ("2013-05-22", "2013-09-05",
                             "A pure rate shock with no growth scare -- the closest historical "
                             "analogue to the thesis's 2027 hiking view, and notably a period "
                             "when gold fell hard alongside bonds."),
    "China/oil shock (2015-16)": ("2015-08-10", "2016-02-11",
                                  "Commodity and emerging-market stress. Directly relevant to a "
                                  "book holding both Canadian equity and EM."),
    "Q4 2018 selloff": ("2018-09-20", "2018-12-24",
                        "Rate-driven equity drawdown with no recession -- tests the equity "
                        "sleeves against tightening rather than against a credit event."),
    "COVID crash (2020)": ("2020-02-19", "2020-03-23",
                           "The fastest drawdown on record. Nearly everything correlated to 1, "
                           "which is exactly why it belongs in this table."),
    "2022 rate shock": ("2022-01-03", "2022-10-14",
                        "Stocks and bonds fell together for the first time in a generation. "
                        "The single most important window for judging a floating-rate sleeve, "
                        "since it is the scenario conventional bonds failed."),
}


def window_performance(returns: pd.DataFrame, windows: dict | None = None) -> pd.DataFrame:
    """Cumulative total return of each column over each crisis window."""
    windows = windows or CRISIS_WINDOWS
    rows = {}
    for label, (start, end, _why) in windows.items():
        seg = returns.loc[start:end]
        if seg.empty:
            continue
        rows[label] = (1 + seg).prod() - 1
    return pd.DataFrame(rows).T


def window_notes(windows: dict | None = None) -> pd.DataFrame:
    windows = windows or CRISIS_WINDOWS
    return pd.DataFrame(
        [{"window": k, "start": v[0], "end": v[1], "why_it_is_here": v[2]}
         for k, v in windows.items()]
    ).set_index("window")


def portfolio_window_performance(returns: pd.DataFrame, allocations: dict[str, pd.Series],
                                 windows: dict | None = None) -> pd.DataFrame:
    """
    Each candidate allocation's realised return through each crisis
    window, held without rebalancing inside the window (the honest
    assumption -- nobody rebalances mid-crash on a schedule).
    """
    windows = windows or CRISIS_WINDOWS
    from core.backtest_engine import BacktestConfig, run_backtest
    cfg = BacktestConfig(rebalance="none", cost_bps=0.0)
    rows = {}
    for label, (start, end, _why) in windows.items():
        seg = returns.loc[start:end]
        if len(seg) < 5:
            continue
        rows[label] = {name: run_backtest(seg, w, cfg).stats()["total_return"]
                       for name, w in allocations.items()}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Conditional dependence
# ---------------------------------------------------------------------------

def worst_decile_mask(reference: pd.Series, quantile: float = 0.10) -> pd.Series:
    """Days in the worst `quantile` of the reference asset's return distribution."""
    threshold = reference.quantile(quantile)
    return reference <= threshold


def conditional_correlation_matrix(returns: pd.DataFrame, reference_col: str,
                                   quantile: float = 0.10) -> pd.DataFrame:
    ref = returns[reference_col]
    mask = worst_decile_mask(ref, quantile)
    return returns.loc[mask].corr()


def correlation_regime_table(returns: pd.DataFrame, reference_col: str,
                             quantile: float = 0.10) -> pd.DataFrame:
    """
    Each asset's correlation to the reference in calm markets versus in
    the reference's worst decile, and the change between them.

    The `stress_uplift` column is the one to read. A large positive value
    means that holding stops diversifying at precisely the moment
    diversification is the only thing that matters, and the position
    should be justified on some other basis or resized.
    """
    ref = returns[reference_col]
    mask = worst_decile_mask(ref, quantile)
    calm = returns.loc[~mask].corr()[reference_col]
    stress = returns.loc[mask].corr()[reference_col]
    out = pd.DataFrame({"calm_corr": calm, "stress_corr": stress})
    out["stress_uplift"] = out["stress_corr"] - out["calm_corr"]
    return out.drop(index=reference_col).sort_values("stress_uplift", ascending=False)


def up_down_beta(returns: pd.DataFrame, reference_col: str) -> pd.DataFrame:
    """
    Beta to the reference asset estimated separately on its up days and
    its down days.

    `asymmetry` = downside beta - upside beta. Positive is bad and is the
    profile to hunt for: the asset participates in losses more than in
    gains. A genuine tail hedge should show a clearly negative downside
    beta; a mediocre one shows a downside beta near zero; anything with a
    positive downside beta above ~0.3 is a diversifier in name only.
    """
    ref = returns[reference_col]
    up, down = ref > 0, ref < 0
    rows = {}
    for col in returns.columns:
        if col == reference_col:
            continue
        y = returns[col]
        bu = np.polyfit(ref[up], y[up], 1)[0] if up.sum() > 30 else np.nan
        bd = np.polyfit(ref[down], y[down], 1)[0] if down.sum() > 30 else np.nan
        rows[col] = {"upside_beta": bu, "downside_beta": bd, "asymmetry": bd - bu}
    return pd.DataFrame(rows).T.sort_values("asymmetry", ascending=False)


def tail_dependence(returns: pd.DataFrame, reference_col: str,
                    quantile: float = 0.05) -> pd.Series:
    """
    Empirical lower-tail dependence: given the reference asset had a
    bottom-`quantile` day, how often did this asset also have one?

    Independence would give roughly `quantile` itself (5%). A reading of
    40% means that four times in ten, the "diversifier" was having its
    own worst-5% day at the same moment -- a joint-crash frequency no
    correlation coefficient would have revealed.
    """
    ref = returns[reference_col]
    ref_bad = ref <= ref.quantile(quantile)
    out = {}
    for col in returns.columns:
        if col == reference_col:
            continue
        bad = returns[col] <= returns[col].quantile(quantile)
        out[col] = float((bad & ref_bad).sum() / max(ref_bad.sum(), 1))
    return pd.Series(out, name=f"lower_tail_dependence_q{quantile:.0%}").sort_values(ascending=False)


def equity_drawdown_mask(equity_index: pd.Series, threshold: float = 0.10) -> pd.Series:
    """Bear mask: True where the index sits more than `threshold` below its trailing high."""
    return (equity_index / equity_index.cummax() - 1.0) <= -threshold
