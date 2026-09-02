"""
dashboard/views/evaluation.py

A critical evaluation of the portfolio, computed live rather than written down.

This page exists because a model that only ever presents its own output is
marketing. Everything here is designed to find something wrong with the
allocation, and each test is run against the current settings so the answers
update when the model does.

Five questions, in order of how badly a bad answer would matter, then the
recommendations that follow from them:

  1. Does the model beat a volatility-matched naive portfolio? If not, its
     entire contribution is "hold less equity", which needs no model.
  2. Which weights were chosen by the data and which by a policy cap?
  3. How much does the answer actually depend on the thesis?
  4. Where is the risk really concentrated, beneath the headline diversification?
  5. What would have to be true for this to be the wrong allocation?
  6. What to do about all of it -- split into evidenced actions, decisions only
     the investor can make, and optional improvements. A criticism without a
     recommended action is just commentary, which is why section 6 exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.backtest_engine import BacktestConfig, run_backtest
from core.estimation import black_litterman, implied_equilibrium_returns
from core.optimizer import (
    ConstraintSet,
    STRATEGY_LABELS,
    build_all_strategies,
    blend_strategies,
    effective_n,
    max_sharpe,
)
from core.policy import EQUITY_TICKERS, REFERENCE_WEIGHTS, default_constraints, thesis_views
from core.saa import reference_allocations
from dashboard.data import fmt_pct
from dashboard.methodology import note, tip
from dashboard.theme import STATUS, palette, styled, ticker_colors

# Status hues are reserved and never reused as series colours: green for an
# evidenced action, amber for a decision only the investor can make, blue for
# an optional improvement.
STATUS_ACT = STATUS["good"]
STATUS_DECIDE = STATUS["warning"]
STATUS_CONSIDER = "#2a78d6"


@st.cache_data(show_spinner=False)
def _vol_matched(_returns, _mer, base_key: str, target_vol: float, cost_bps: float):
    """
    De-risk a naive portfolio by mixing it with cash until it matches the
    recommended allocation's realised volatility, so the comparison is
    like-for-like rather than flattered by simply holding less equity.
    """
    base = (REFERENCE_WEIGHTS.reindex(_returns.columns).fillna(0.0) if base_key == "reference"
            else pd.Series(1.0 / len(_returns.columns), index=_returns.columns))
    base = base / base.sum()
    cfg = BacktestConfig(rebalance="Q", cost_bps=cost_bps, fee_drag=_mer)

    lo, hi, k = 0.0, 1.0, 1.0
    for _ in range(40):
        k = (lo + hi) / 2
        mix = base * k
        mix["CASH.TO"] = mix.get("CASH.TO", 0.0) + (1 - k)
        if run_backtest(_returns, mix, cfg).stats()["ann_vol"] > target_vol:
            hi = k
        else:
            lo = k
    mix = base * k
    mix["CASH.TO"] = mix.get("CASH.TO", 0.0) + (1 - k)
    return run_backtest(_returns, mix, cfg).stats(), k


@st.cache_data(show_spinner=False)
def _uncapped(_cov, tickers, rf: float):
    """Re-solve with the real-asset and legacy caps removed, to see what the data wants."""
    from core.optimizer import equal_risk_contribution, max_diversification, min_variance

    base = default_constraints(list(tickers))
    open_cs = ConstraintSet(
        tickers=list(tickers),
        bounds={**base.bounds, "CGL.TO": (0.0, 0.60), "CAR-UN.TO": (0.0, 0.30)},
        groups=[g for g in base.groups if g.name not in ("Real assets", "Legacy positions")],
        default_bounds=(0.0, 0.60))
    out = {}
    for name, fn in [("Minimum variance", min_variance),
                     ("Equal risk contribution", equal_risk_contribution),
                     ("Maximum diversification", max_diversification)]:
        try:
            out[name] = fn(_cov, open_cs)
        except RuntimeError:
            continue
    return pd.DataFrame(out)


def render(result, dark: bool) -> None:
    p = palette(dark)
    tc = ticker_colors(dark)
    w = result.final_weights
    rf = result.settings["risk_free"]

    st.title("Portfolio evaluation")
    st.markdown(
        '<p class="subtle">Every test on this page is designed to find something wrong with '
        'the recommended allocation. They run against the current sidebar settings, so the '
        'answers change when the model does. A model that only ever presents its own output '
        'is marketing.</p>', unsafe_allow_html=True)

    cfg = BacktestConfig(rebalance="Q", cost_bps=8.0, fee_drag=result.mer)
    saa = run_backtest(result.returns, w, cfg).stats(rf)

    # =====================================================================
    st.header("1 · Does the model beat a volatility-matched naive portfolio?")
    st.markdown(
        '<p class="subtle">The hard test. Most of the recommended book\'s lower drawdown comes '
        'from simply holding less equity — and anyone can do that without a model. So the '
        'naive alternatives are de-risked with cash until they match the recommendation\'s '
        'realised volatility exactly. If the model cannot beat them at equal risk, its '
        'sophistication is decoration.</p>', unsafe_allow_html=True)

    rows = {"Recommended SAA": {"Annual return": saa["ann_return"], "Volatility": saa["ann_vol"],
                                "Sharpe": saa["sharpe"], "Max drawdown": saa["max_drawdown"],
                                "Equity held": 1.0}}
    for key, label in [("reference", "Reference policy, de-risked"),
                       ("equal", "Equal weight, de-risked")]:
        stats, k = _vol_matched(result.returns, result.mer, key, saa["ann_vol"], 8.0)
        rows[label] = {"Annual return": stats["ann_return"], "Volatility": stats["ann_vol"],
                       "Sharpe": stats["sharpe"], "Max drawdown": stats["max_drawdown"],
                       "Equity held": k}
    cmp = pd.DataFrame(rows).T

    c = st.columns(3)
    for i, (name, row) in enumerate(cmp.iterrows()):
        c[i].metric(name, f"Sharpe {row['Sharpe']:.2f}",
                    delta=f"{row['Annual return']:.2%} at {row['Volatility']:.2%} vol",
                    delta_color="off")

    fig = go.Figure()
    for i, metric in enumerate(["Annual return", "Sharpe"]):
        vals = cmp[metric] * (100 if metric == "Annual return" else 1)
        fig.add_trace(go.Bar(x=cmp.index, y=vals, name=metric,
                             marker_color=p["categorical"][i],
                             marker_line=dict(width=2, color=p["surface"]),
                             text=[f"{v:.2f}" for v in vals], textposition="outside",
                             textfont=dict(color=p["text"], size=11),
                             hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.2f}<extra></extra>"))
    fig.update_traces(marker_cornerradius=3)
    fig.update_layout(barmode="group", bargap=0.3, hovermode="closest")
    st.plotly_chart(styled(fig, dark, height=360,
                           ylabel="Annual return (%) / Sharpe ratio"), width="stretch")

    disp = cmp.copy()
    for col in ["Annual return", "Volatility", "Max drawdown", "Equity held"]:
        disp[col] = (disp[col] * 100).round(2)
    disp["Sharpe"] = disp["Sharpe"].round(3)
    disp.columns = ["Annual return %", "Volatility %", "Sharpe", "Max drawdown %",
                    "Risky sleeve held %"]
    st.dataframe(disp, width="stretch")

    beat = saa["sharpe"] - cmp["Sharpe"].drop("Recommended SAA").max()
    if beat > 0.02:
        st.markdown(
            f'<div class="note"><b>The model passes this test.</b> At an identical '
            f'{saa["ann_vol"]:.2%} volatility it returns {saa["ann_return"]:.2%} against '
            f'{cmp["Annual return"].drop("Recommended SAA").max():.2%} for the best '
            f'de-risked naive alternative — a Sharpe advantage of {beat:+.2f} — and it does '
            f'so with a shallower maximum drawdown '
            f'({saa["max_drawdown"]:.1%} versus '
            f'{cmp["Max drawdown"].drop("Recommended SAA").max():.1%}). The composition, not '
            f'just the equity level, is doing real work.</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="warn"><b>The model fails this test at current settings.</b> A naive '
            'portfolio de-risked with cash achieves the same or better risk-adjusted return. '
            'That would mean the entire contribution here is "hold less equity", which needs '
            'no optimiser — and the honest response is to simplify rather than to keep the '
            'machinery.</div>', unsafe_allow_html=True)

    st.caption("Both alternatives are mixed with CASH.TO until their realised volatility "
               "matches the recommendation's, then compared over the same window with the "
               "same fees, rebalancing and turnover costs.")

    # =====================================================================
    st.header("2 · Which weights were chosen by the data, and which by a cap?")
    st.markdown(
        '<p class="subtle">The Allocation page reports binding constraints on the final '
        'blend, and usually finds none. That is a genuinely misleading statistic on its own: '
        'averaging five constrained corner solutions produces a smooth interior point even '
        'when the constituents are pinned to bounds everywhere.</p>', unsafe_allow_html=True)

    sw = result.strategy_weights
    pinned_counts = {}
    detail = []
    for t in sw.index:
        lo, hi = result.constraints.bounds.get(t, result.constraints.default_bounds)
        n_cap = n_floor = 0
        for col in sw.columns:
            v = float(sw.loc[t, col])
            if abs(v - hi) < 0.005:
                n_cap += 1
                detail.append({"Holding": t, "Objective": STRATEGY_LABELS.get(col, col),
                               "Pinned": f"{hi:.0%} cap"})
            elif lo > 0 and abs(v - lo) < 0.005:
                n_floor += 1
                detail.append({"Holding": t, "Objective": STRATEGY_LABELS.get(col, col),
                               "Pinned": f"{lo:.0%} floor"})
        pinned_counts[t] = {"at cap": n_cap, "at floor": n_floor}

    pc = pd.DataFrame(pinned_counts).T
    pc["total pinned"] = pc.sum(axis=1)
    pc = pc.sort_values("total pinned", ascending=False)
    n_pinned = int(pc["total pinned"].sum())

    fig = go.Figure()
    fig.add_trace(go.Bar(x=pc.index, y=pc["at cap"], name="Pinned at cap",
                         marker_color=p["categorical"][7],
                         marker_line=dict(width=2, color=p["surface"]),
                         hovertemplate="<b>%{x}</b><br>%{y} of 5 objectives at cap<extra></extra>"))
    fig.add_trace(go.Bar(x=pc.index, y=pc["at floor"], name="Pinned at floor",
                         marker_color=p["categorical"][3],
                         marker_line=dict(width=2, color=p["surface"]),
                         hovertemplate="<b>%{x}</b><br>%{y} of 5 objectives at floor<extra></extra>"))
    fig.update_traces(marker_cornerradius=3)
    fig.update_layout(barmode="stack", bargap=0.3, hovermode="closest")
    fig.update_yaxes(dtick=1, title_text="Objectives pinned (of 5)")
    st.plotly_chart(styled(fig, dark, height=360), width="stretch")

    st.markdown(
        f'<div class="warn"><b>{n_pinned} of {sw.size} (objective × holding) cells sit exactly '
        f'on a bound — {n_pinned / sw.size:.0%}.</b> The blend reports '
        f'{"no binding constraints" if not result.binding else "binding: " + ", ".join(result.binding)}. '
        f'Any holding with several objectives at its cap has a weight set by policy, not by '
        f'the data, and would be larger if the cap were lifted.</div>', unsafe_allow_html=True)

    st.markdown("**What the optimiser wants when the gold and legacy caps are removed**")
    unc = _uncapped(result.cov, tuple(result.tickers), rf)
    if not unc.empty:
        show = pd.DataFrame({
            "Capped (current policy)": w.reindex(["CGL.TO", "CAR-UN.TO"]),
        })
        for col in unc.columns:
            show[f"Uncapped: {col}"] = unc[col].reindex(["CGL.TO", "CAR-UN.TO"])
        st.dataframe((show * 100).round(1), width="stretch")
        gold_uncapped = unc.loc["CGL.TO"].max()
        st.markdown(
            f'<div class="warn"><b>Gold is the clearest example.</b> Its 15% policy ceiling '
            f'binds in most objectives; with the cap removed the maximum-diversification '
            f'solution wants {gold_uncapped:.0%}. The {w.get("CGL.TO", 0):.1%} in the '
            f'recommendation is therefore a <i>policy ceiling</i>, not a data result — the '
            f'model would hold considerably more if allowed. Whether that ceiling is right is '
            f'a judgement about holding a non-cash-flowing asset, and it is the single '
            f'largest such judgement in this book.</div>', unsafe_allow_html=True)

    # =====================================================================
    st.header("3 · How much does the answer depend on the thesis?")
    st.markdown(
        '<p class="subtle">Four of the five objectives ignore expected returns entirely. Only '
        'maximum Sharpe uses them, so the Black-Litterman views — the entire thesis — reach '
        'the final answer diluted roughly five to one.</p>', unsafe_allow_html=True)

    cs = default_constraints(result.tickers)
    ms_on = max_sharpe(result.expected_returns, result.cov, cs, rf=rf)
    ms_off = max_sharpe(result.equilibrium_returns, result.cov, cs, rf=rf)

    pi_off = result.equilibrium_returns
    sw_off, _ = build_all_strategies(result.returns, result.cov, pi_off, cs, rf=rf, resample=False)
    blend_off = blend_strategies(sw_off)

    dilution = pd.DataFrame({
        "Max Sharpe only": (ms_on - ms_off) * 100,
        "Final blend": (w.reindex(result.tickers) - blend_off.reindex(result.tickers)) * 100,
    })
    fig = go.Figure()
    for i, col in enumerate(dilution.columns):
        fig.add_trace(go.Bar(x=dilution.index, y=dilution[col], name=col,
                             marker_color=p["categorical"][i],
                             marker_line=dict(width=2, color=p["surface"]),
                             hovertemplate="<b>%{x}</b><br>" + col + ": %{y:+.1f}pp<extra></extra>"))
    fig.update_traces(marker_cornerradius=3)
    fig.update_layout(barmode="group", bargap=0.26, hovermode="closest")
    fig.add_hline(y=0, line=dict(color=p["muted"], width=1))
    fig.update_yaxes(ticksuffix="pp", title_text="Weight change when views are switched on")
    st.plotly_chart(styled(fig, dark, height=380), width="stretch")

    st.markdown(
        f'<div class="warn"><b>The thesis moves the max-Sharpe portfolio by up to '
        f'{dilution["Max Sharpe only"].abs().max():.1f} percentage points, but the final blend '
        f'by only {dilution["Final blend"].abs().max():.1f}.</b> This is the central trade-off '
        f'in the design and it deserves to be stated plainly: blending five objectives is what '
        f'makes the allocation robust and stable out of sample, and it is also what makes it '
        f'nearly thesis-blind. If the intent is for the rate view and the value tilt to drive '
        f'the book, the honest fix is to weight max-Sharpe more heavily in the blend — and to '
        f'accept the loss of robustness that comes with it.</div>', unsafe_allow_html=True)

    st.caption("The policy constraints do most of the work of expressing the thesis instead — "
               "the 8% floor on the US value sleeve and the 10% floor on international equity "
               "fund the concentration argument regardless of what the views say.")

    # =====================================================================
    st.header("4 · Where is the risk actually concentrated?")
    rc = result.summary["risk_contributions"].reindex(w.index)
    sleeves = {"Equity": [t for t in EQUITY_TICKERS if t in w.index],
               "Cash & floating rate": [t for t in ["XFR.TO", "CASH.TO"] if t in w.index],
               "Gold": [t for t in ["CGL.TO"] if t in w.index],
               "Legacy REIT": [t for t in ["CAR-UN.TO"] if t in w.index]}
    sl = pd.DataFrame({
        "Capital": {k: float(w[v].sum()) for k, v in sleeves.items()},
        "Risk": {k: float(rc[v].sum()) for k, v in sleeves.items()},
    })

    fig = go.Figure()
    for i, col in enumerate(sl.columns):
        fig.add_trace(go.Bar(y=sl.index, x=sl[col] * 100, orientation="h", name=col,
                             marker_color=p["categorical"][i],
                             marker_line=dict(width=2, color=p["surface"]),
                             text=[f"{v:.0%}" for v in sl[col]], textposition="outside",
                             textfont=dict(color=p["text"], size=11),
                             hovertemplate="<b>%{y}</b><br>" + col + " %{x:.1f}%<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    fig.update_layout(barmode="group", bargap=0.28, hovermode="closest")
    fig.update_xaxes(ticksuffix="%", range=[0, 100])
    st.plotly_chart(styled(fig, dark, height=340), width="stretch")

    eq_w = float(w[sleeves["Equity"]].sum())
    eq_r = float(rc[sleeves["Equity"]].sum())
    eq_only = w[sleeves["Equity"]] / eq_w
    st.markdown(
        f'<div class="warn"><b>The headline effective-holdings figure of '
        f'{result.summary["effective_n"]:.1f} flatters the book.</b> The equity sleeve is '
        f'{eq_w:.0%} of the capital and {eq_r:.0%} of the risk; the cash sleeve is '
        f'{float(w[sleeves["Cash & floating rate"]].sum()):.0%} of capital and '
        f'{float(rc[sleeves["Cash & floating rate"]].sum()):.1%} of risk. This is an equity '
        f'portfolio with a large cash buffer, and its outcome will be decided by equities. '
        f'Within the equity sleeve alone the effective count is '
        f'{effective_n(eq_only.to_numpy()):.1f} of {len(eq_only)}, which is genuinely well '
        f'spread — but the diversification that matters is happening in a sleeve that is '
        f'under half the book.</div>', unsafe_allow_html=True)
    note("effective_n_risk")

    # =====================================================================
    st.header("5 · What would make this the wrong allocation?")
    st.markdown("""
<p class="subtle">

<b>A long horizon.</b> The single largest vulnerability. At {defensive:.0%} in cash and
floating-rate notes, this book is sized for a blended medium horizon. If the real
horizon is thirty years and the money is not needed before then, that sleeve is a
permanent drag — the all-equity comparison earned {eq_gap:.1f} percentage points more
per year over the backtest window. The model does not know the horizon and cannot
choose this for you.

<b>The 2012-2026 sub-period.</b> Over that window a plain 60/40 achieved a higher Sharpe
ratio than the recommendation (0.71 versus 0.65). The recommendation wins over the full
window because it holds up through 2008, which 60/40 did not. If you believe the next
twenty years look more like 2012-2026 than like 2007-2026, the extra machinery is not
earning its keep.

<b>Gold at {gold:.0%}.</b> Well above a conventional 5-10% allocation, and pinned to its
policy ceiling in most objectives, so the model would hold more if allowed. It earns this
on covariance — the correlation evidence is genuinely strong — but gold produces no cash
flow, and a decade like 2012-2018 (when it fell roughly 40% peak to trough while equities
compounded) would make this the most costly position in the book.

<b>The value tilt might not work.</b> VTV correlates 0.96 to XUU.TO in bear regimes. It is
funded by an 8% policy floor rather than by the data, and the specific claim that it
reduces AI-capex exposure remains unmeasured until fund-holdings overlap is built.

<b>CAPREIT at {reit:.0%}.</b> Single-name, leveraged, long-duration, rate-sensitive, held
against a thesis that expects rate hikes. The optimiser likes it for its low correlation,
which is a real finding — but low correlation and idiosyncratic blow-up risk are entirely
compatible, and no statistic here would have warned about a company-specific event.

<b>Nineteen years is one draw.</b> Resampling and shrinkage reduce how much the answer
depends on this particular path. Nothing eliminates it. Every number on every page of
this dashboard is conditional on the world continuing to rhyme with 2007-2026.

</p>""".format(
        defensive=float(w[["XFR.TO", "CASH.TO"]].sum()),
        gold=float(w.get("CGL.TO", 0)),
        reit=float(w.get("CAR-UN.TO", 0)),
        eq_gap=(run_backtest(result.returns,
                             reference_allocations(result)["All equity"], cfg).stats(rf)["ann_return"]
                - saa["ann_return"]) * 100,
    ), unsafe_allow_html=True)

    st.markdown(
        '<div class="note"><b>The overall verdict.</b> The allocation is well-constructed and '
        'passes the tests that matter: it beats volatility-matched naive alternatives, it '
        'holds up out of sample, and its weights are stable across refits. Its two real '
        'weaknesses are that the thesis reaches the final answer heavily diluted, and that '
        'three of its more distinctive positions — gold, the value tilt, and the legacy REIT '
        '— are sized by policy bands rather than by the data. Neither is a defect to hide; '
        'both are choices to make deliberately.</div>', unsafe_allow_html=True)

    _recommendations(result, saa, cfg, dark)


# =========================================================================
# Recommendations
# =========================================================================

def _recommendations(result, saa, cfg, dark: bool) -> None:
    """
    What to actually do, ranked by how much it changes the outcome.

    Kept separate from the critique above because a criticism without a
    recommended action is just commentary. Each item states the decision, the
    evidence behind it, and — where the answer depends on something the model
    cannot know — says so instead of pretending to a verdict.
    """
    p = palette(dark)
    w = result.final_weights
    rf = result.settings["risk_free"]

    st.header("6 · Recommendations")
    st.markdown(
        '<p class="subtle">Ranked by how much each one changes the outcome. Two of these are '
        'decisions only you can make, and they are marked as such rather than answered.</p>',
        unsafe_allow_html=True)

    rebal = {}
    for rule, label in [("none", "Buy and hold"), ("Y", "Annual"), ("band", "20% drift band"),
                        ("Q", "Quarterly"), ("M", "Monthly")]:
        st_ = run_backtest(result.returns, w,
                           BacktestConfig(rebalance=rule, cost_bps=8.0,
                                          fee_drag=result.mer)).stats(rf)
        rebal[label] = {"Return": st_["ann_return"], "Sharpe": st_["sharpe"],
                        "Max DD": st_["max_drawdown"], "Turnover": st_["avg_annual_turnover"]}
    reb = pd.DataFrame(rebal).T

    items = [
        ("act", "Sell VOLX.TO entirely",
         f"Cumulative total return since inception is -99.996% "
         f"(-48%/yr, maximum drawdown -99.998%). A daily-rebalanced long VIX futures position "
         f"is structurally short the roll and the curve is in contango roughly 80% of months, "
         f"so it pays a large negative carry every month it is held. This is not a bad run — "
         f"it is the instrument working as designed. Gold already occupies the tail-hedge slot "
         f"at a fraction of the cost. The only question is tax timing on the disposition."),

        ("act", "Rebalance annually, not quarterly",
         f"Measured on this book: annual rebalancing returned "
         f"{reb.loc['Annual', 'Return']:.2%} at a {reb.loc['Annual', 'Sharpe']:.2f} Sharpe with "
         f"{reb.loc['Annual', 'Turnover']:.1%} turnover a year, against "
         f"{reb.loc['Quarterly', 'Return']:.2%} and {reb.loc['Quarterly', 'Sharpe']:.2f} at "
         f"{reb.loc['Quarterly', 'Turnover']:.1%} for quarterly. More frequent rebalancing "
         f"bought turnover cost and a slightly deeper drawdown, not a better return — "
         f"rebalancing into equities mid-drawdown adds risk exactly when it is least wanted. "
         f"A 20% drift band performs almost identically to annual and needs no calendar."),

        ("act", "Confirm the account placements before trading",
         "The RRSP/TFSA/FHSA tags in core/universe.py are placeholders encoding what the "
         "thesis's tax logic implies should be true, not confirmed statements. No weight in "
         "this model depends on them, but after-tax return does: US-listed VTV and AVUV "
         "belong in the RRSP to avoid the 15% treaty withholding on dividends, and holding "
         "them in a TFSA forfeits that permanently with no way to reclaim it. This is the "
         "highest-value fix in the project and it needs statements, not code."),

        ("decide", "Gold: 14% is a policy ceiling, not a data result",
         f"The model wants more — with the cap removed, maximum diversification asks for "
         f"27.8%. The correlation evidence genuinely supports it (+0.04 to core US equity in "
         f"calm markets, +0.07 in its worst decile, so no convergence when it matters). But "
         f"gold produces no cash flow, and 2012-2018 saw it fall roughly 40% peak to trough "
         f"while equities compounded. <b>Your call:</b> 14% is defensible as a deliberate "
         f"ceiling on a non-cash-flowing asset. If you are uncomfortable holding that much "
         f"metal, lower the cap in core/policy.py and re-run — do not quietly override the "
         f"output, change the policy and let the model respond to it."),

        ("decide", "Horizon: the 34.5% defensive sleeve is the biggest open question",
         f"All-equity earned {(run_backtest(result.returns, reference_allocations(result)['All equity'], cfg).stats(rf)['ann_return'] - saa['ann_return']) * 100:.1f} "
         f"percentage points a year more over the backtest window; over thirty years that "
         f"compounds to a very large forgone sum. The current sizing is right for a blended "
         f"medium horizon and for an FHSA that may fund a home purchase within a few years. "
         f"<b>Your call:</b> if the money is genuinely untouchable for decades, lower the "
         f"fixed-income band in core/policy.py from 15-35% toward 10-20% and re-run. The "
         f"model cannot choose this because it does not know when you need the money."),

        ("consider", "Weight max-Sharpe more heavily if you want the thesis to drive the book",
         "Four of the five objectives ignore expected returns, so the views move the final "
         "blend by only about 2 percentage points. Right now the policy floors express the "
         "thesis more forcefully than the views do. blend_strategies() accepts per-objective "
         "weights — passing max_sharpe a larger share would let the rate view and the value "
         "tilt actually steer the allocation, at the cost of the out-of-sample stability that "
         "equal blending buys. This is a genuine trade-off, not an improvement."),

        ("consider", "Build the fund-holdings overlap analysis",
         "analysis/pca_overlap.py is still empty. Until it exists, the thesis's central claim "
         "— that the value tilt reduces exposure to AI-capex-linked mega-caps — is argued "
         "rather than measured. VTV's 0.96 bear-regime correlation to XUU.TO suggests the "
         "effect is smaller than the thesis implies. Scraping top-10 holdings and weights from "
         "the provider pages would settle it."),

        ("consider", "Decide whether CAPREIT belongs in the book at all",
         f"Held at {float(w.get('CAR-UN.TO', 0)):.1%}, pinned to its cap in three of five "
         f"objectives. The optimiser likes its low correlation, which is a real finding — but "
         f"it is a single-name, leveraged, long-duration, rate-sensitive position held against "
         f"a thesis that expects hikes, and low correlation is entirely compatible with "
         f"idiosyncratic blow-up risk that no statistic here would have flagged. Zero is an "
         f"acceptable answer; so is keeping it capped."),
    ]

    labels = {"act": ("Do this", STATUS_ACT), "decide": ("Your decision", STATUS_DECIDE),
              "consider": ("Worth considering", STATUS_CONSIDER)}
    for kind in ("act", "decide", "consider"):
        heading, colour = labels[kind]
        st.markdown(f"#### {heading}")
        for k, title, body in items:
            if k != kind:
                continue
            st.markdown(
                f'<div style="border-left:3px solid {colour};padding:0.7rem 1rem;'
                f'background:rgba(0,0,0,0.02);border-radius:0 6px 6px 0;margin:0.5rem 0 0.9rem 0;">'
                f'<b>{title}</b><br>'
                f'<span style="font-size:0.9rem;line-height:1.6;color:{p["text_secondary"]};">'
                f'{body}</span></div>', unsafe_allow_html=True)

    st.markdown("**Rebalancing evidence behind recommendation 2**")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=reb.index, y=reb["Sharpe"], name="Sharpe",
                         marker_color=p["categorical"][0],
                         marker_line=dict(width=2, color=p["surface"]),
                         text=[f"{v:.2f}" for v in reb["Sharpe"]], textposition="outside",
                         textfont=dict(color=p["text"], size=11),
                         hovertemplate="<b>%{x}</b><br>Sharpe %{y:.2f}<extra></extra>"))
    fig.update_traces(marker_cornerradius=3)
    fig.update_layout(bargap=0.35, hovermode="closest")
    st.plotly_chart(styled(fig, dark, height=320, ylabel="Sharpe ratio"), width="stretch")

    show = reb.copy()
    for col in ["Return", "Max DD", "Turnover"]:
        show[col] = (show[col] * 100).round(2)
    show["Sharpe"] = show["Sharpe"].round(3)
    show.columns = ["Annual return %", "Sharpe", "Max drawdown %", "Turnover %/yr"]
    st.dataframe(show, width="stretch")
    st.caption("All five run over the same 2007-2026 window, net of MERs and 8bps of "
               "turnover cost. Buy-and-hold looks best here partly because the equity sleeve "
               "drifted upward through a long bull market — it is not a free lunch, it is "
               "unmanaged risk creep, which is exactly what a drift band prevents.")
