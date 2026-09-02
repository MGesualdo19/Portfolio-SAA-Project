"""Backtest view: how the allocation would have behaved, and whether the method holds up out of sample."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.stress import CRISIS_WINDOWS, portfolio_window_performance, window_notes
from core.backtest_engine import BacktestConfig, compare_backtests, run_backtest, walk_forward
from core.estimation import black_litterman, implied_equilibrium_returns, regime_blended_covariance
from core.optimizer import blend_strategies, build_all_strategies
from core.policy import REFERENCE_WEIGHTS, default_constraints, thesis_views
from core.returns import summary_stats
from core.saa import _equity_sleeve_index, reference_allocations
from analysis.stress import equity_drawdown_mask
from dashboard.data import fmt_pct
from dashboard.methodology import note, notes, tip
from dashboard.theme import palette, styled


@st.cache_data(show_spinner=False)
def _walk_forward(_returns, _mer, tickers, bear_weight, rf, risk_aversion, cost_bps):
    cs = default_constraints(list(tickers))
    views = thesis_views()

    def fit(train):
        mask = equity_drawdown_mask(_equity_sleeve_index(train), 0.10)
        cov, _ = regime_blended_covariance(train, mask, bear_weight=bear_weight)
        pi = implied_equilibrium_returns(cov, REFERENCE_WEIGHTS.reindex(cov.index),
                                         risk_aversion=risk_aversion, rf=rf)
        mu, _ = black_litterman(cov, pi, views)
        sw, _ = build_all_strategies(train, cov, mu, cs, rf=rf, resample=False)
        return blend_strategies(sw)

    cfg = BacktestConfig(rebalance="Q", cost_bps=cost_bps, fee_drag=_mer)
    return walk_forward(_returns, fit, train_years=5, step_months=12, config=cfg)


def render(result, dark: bool) -> None:
    p = palette(dark)
    st.title("Backward-looking backtest")
    st.markdown(
        '<p class="subtle">Growth of $100,000 CAD from 2007 to today, net of management fees '
        'and trading costs, with weights drifting between rebalances the way real holdings do. '
        'History before each fund existed is filled with index proxies — see Diagnostics for '
        'exactly how much of each series that is.</p>', unsafe_allow_html=True)

    c = st.columns(4)
    rebal = c[0].selectbox("Rebalancing", ["Y", "Q", "M", "band", "none"], index=0,
                           help=tip("turnover", "Historically on this book, annual and "
                                    "drift-band rebalancing both beat quarterly — more "
                                    "frequent rebalancing bought turnover cost and a "
                                    "slightly deeper drawdown, not a better return."),
                           format_func=lambda x: {"Q": "Quarterly", "Y": "Annual", "M": "Monthly",
                                                  "band": "20% drift band",
                                                  "none": "Buy and hold"}[x])
    cost_bps = c[1].slider("Trading cost (bps of turnover)", 0, 40, 8, 1,
                           help=tip("turnover"))
    initial = c[2].number_input("Starting value (CAD)", 10_000.0, 10_000_000.0,
                                100_000.0, 10_000.0, format="%.0f")
    net_fees = c[3].checkbox("Charge MERs", value=True,
                             help="Deducts each fund's management expense ratio daily.")

    cfg = BacktestConfig(rebalance=rebal, cost_bps=float(cost_bps),
                         fee_drag=result.mer if net_fees else None,
                         initial_value=float(initial), drift_band=0.20)

    allocs = reference_allocations(result)
    navs, stats = compare_backtests(result.returns, allocs, cfg)

    # --- NAV ------------------------------------------------------------
    st.subheader("Growth of the book")
    fig = go.Figure()
    for i, name in enumerate(navs.columns):
        is_main = name == "Recommended SAA"
        fig.add_trace(go.Scatter(
            x=navs.index, y=navs[name], name=name, mode="lines",
            line=dict(color=p["categorical"][i], width=2.6 if is_main else 1.6,
                      dash=None if is_main else "solid"),
            opacity=1.0 if is_main else 0.72,
            hovertemplate=name + ": $%{y:,.0f}<extra></extra>"))
    # Bear-regime shading: context, not a series, so it stays behind and grey.
    _shade_bear(fig, result, p)
    fig.update_yaxes(tickprefix="$", type="log",
                     title_text="Portfolio value (CAD, log scale)")
    st.plotly_chart(styled(fig, dark, height=470), width="stretch")
    notes("backtest", "total_return", "cad_conversion",
          label="How this backtest is constructed")
    st.caption("Log scale: equal vertical distances are equal percentage moves, so the "
               "2008 drawdown and the 2020 drawdown can be compared honestly. Shaded bands "
               "mark periods when this book's own equity sleeve was more than 10% below its "
               "trailing high.")

    # --- stats ----------------------------------------------------------
    st.subheader("Risk and return over the full window")
    disp = stats.copy()
    for col in ["ann_return", "ann_vol", "max_drawdown", "total_return", "total_cost_drag"]:
        if col in disp:
            disp[col] = (disp[col] * 100).round(2)
    disp["avg_annual_turnover"] = (disp["avg_annual_turnover"] * 100).round(1)
    for col in ["sharpe", "sortino", "calmar"]:
        if col in disp:
            disp[col] = disp[col].round(2)
    disp.columns = ["Ann. return %", "Ann. vol %", "Sharpe", "Sortino", "Max DD %",
                    "Calmar", "Total return %", "Turnover %/yr", "Cost drag %"]
    st.dataframe(disp, width="stretch")
    notes("annualisation", "sharpe", "sortino", "max_drawdown", "calmar", "turnover",
          label="How each column in this table is calculated")

    saa_stats = stats.loc["Recommended SAA"]
    ref_stats = stats.loc["Reference policy"]
    st.markdown(
        f'<div class="note">Against the reference policy portfolio, the recommended '
        f'allocation gave up {(ref_stats["ann_return"] - saa_stats["ann_return"]) * 100:.2f} '
        f'points of annual return and removed '
        f'{(abs(ref_stats["max_drawdown"]) - abs(saa_stats["max_drawdown"])) * 100:.1f} points '
        f'of maximum drawdown, lifting the Sharpe ratio from {ref_stats["sharpe"]:.2f} to '
        f'{saa_stats["sharpe"]:.2f} and the Calmar ratio from {ref_stats["calmar"]:.2f} to '
        f'{saa_stats["calmar"]:.2f}. That trade — less return, materially less pain — is the '
        f'whole design intent, and whether it is the right trade depends on a horizon the '
        f'model does not know.</div>', unsafe_allow_html=True)

    # --- drawdown -------------------------------------------------------
    st.subheader("Drawdown from trailing high")
    fig = go.Figure()
    for i, name in enumerate(navs.columns):
        dd = navs[name] / navs[name].cummax() - 1.0
        is_main = name == "Recommended SAA"
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd * 100, name=name, mode="lines",
            line=dict(color=p["categorical"][i], width=2.4 if is_main else 1.4),
            opacity=1.0 if is_main else 0.6,
            fill="tozeroy" if is_main else None,
            fillcolor="rgba(42,120,214,0.13)" if is_main else None,
            hovertemplate=name + ": %{y:.1f}%<extra></extra>"))
    fig.update_yaxes(ticksuffix="%", title_text="Drawdown")
    st.plotly_chart(styled(fig, dark, height=380), width="stretch")
    note("max_drawdown")

    # --- crisis windows --------------------------------------------------
    st.subheader("Named crisis windows")
    st.markdown(
        '<p class="subtle">Realised return through specific dated episodes, held without '
        'rebalancing inside each window. Each window is here because it stresses a different '
        'joint behaviour — there is no value in five variations of "equities fell".</p>',
        unsafe_allow_html=True)

    cw = portfolio_window_performance(result.returns, allocs)
    fig = go.Figure()
    for i, name in enumerate(cw.columns):
        fig.add_trace(go.Bar(
            x=cw.index, y=cw[name] * 100, name=name,
            marker_color=p["categorical"][i],
            marker_line=dict(width=2, color=p["surface"]),
            hovertemplate="<b>%{x}</b><br>" + name + ": %{y:.1f}%<extra></extra>"))
    fig.update_traces(marker_cornerradius=3)
    fig.update_layout(barmode="group", bargap=0.24, bargroupgap=0.05, hovermode="closest")
    fig.update_yaxes(ticksuffix="%", title_text="Total return over window")
    fig.add_hline(y=0, line=dict(color=p["muted"], width=1))
    st.plotly_chart(styled(fig, dark, height=440), width="stretch")

    gfc = cw.loc["GFC (2007-2009)"] if "GFC (2007-2009)" in cw.index else None
    covid = cw.loc["COVID crash (2020)"] if "COVID crash (2020)" in cw.index else None
    if gfc is not None and covid is not None:
        st.markdown(
            f'<div class="note">Through the GFC the recommended book lost '
            f'{abs(gfc["Recommended SAA"]) * 100:.1f}% against '
            f'{abs(gfc["All equity"]) * 100:.1f}% for the all-equity sleeve; through the COVID '
            f'crash, {abs(covid["Recommended SAA"]) * 100:.1f}% against '
            f'{abs(covid["All equity"]) * 100:.1f}%. The 2009 recovery row is the necessary '
            f'counterweight to read alongside them — protection that never gives anything back '
            f'in the rebound is just a permanently smaller portfolio.</div>',
            unsafe_allow_html=True)

    note("crisis_windows")

    with st.expander("Why each window is in the table"):
        st.dataframe(window_notes(), width="stretch")

    st.dataframe((cw * 100).round(1), width="stretch")

    # --- walk forward ----------------------------------------------------
    st.subheader("Out-of-sample validation (walk-forward)")
    st.markdown(
        '<p class="subtle">Everything above is in-sample: the weights were derived from the '
        'same history they are tested on, which flatters them by construction. This test '
        'refits the entire process — covariance, equilibrium returns, views, all five '
        'objectives — on an expanding window and holds the result for the following year, '
        'seeing nothing from the holding period at fit time. It is the only evidence here '
        'about whether the <i>method</i> works rather than whether one set of weights suited '
        'its own sample.</p>', unsafe_allow_html=True)

    note("walk_forward")

    if st.button("Run walk-forward test (about 40 seconds)"):
        with st.spinner("Refitting the model on 15 expanding windows..."):
            try:
                oos, wts = _walk_forward(result.returns, result.mer, tuple(result.tickers),
                                         result.settings["bear_weight"],
                                         result.settings["risk_free"],
                                         result.settings["risk_aversion"], float(cost_bps))
                st.session_state["wf"] = (oos, wts)
            except Exception as exc:
                st.error(f"Walk-forward failed: {exc}")

    if "wf" in st.session_state:
        oos, wts = st.session_state["wf"]
        oos_stats = summary_stats(oos, rf_annual=result.settings["risk_free"])
        seg = result.returns.loc[oos.index.min():]
        insample = run_backtest(seg, result.final_weights, cfg).stats(result.settings["risk_free"])

        c = st.columns(4)
        c[0].metric("Out-of-sample return", fmt_pct(oos_stats["ann_return"], 2))
        c[1].metric("Out-of-sample volatility", fmt_pct(oos_stats["ann_vol"], 2))
        c[2].metric("Out-of-sample Sharpe", f"{oos_stats['sharpe']:.2f}",
                    delta=f"{oos_stats['sharpe'] - insample['sharpe']:+.2f} vs in-sample")
        c[3].metric("Out-of-sample max drawdown", fmt_pct(oos_stats["max_drawdown"], 1))

        gap = oos_stats["sharpe"] - insample["sharpe"]
        verdict = ("essentially identical to the in-sample result, which is the outcome to hope "
                   "for: it means the weights are not fitted to noise"
                   if abs(gap) < 0.10 else
                   ("better out of sample than in, which usually means the refits happened to "
                    "favour the later part of the window rather than that the model improved"
                    if gap > 0 else
                    "meaningfully worse out of sample, which is the signature of overfitting "
                    "and should lower confidence in the weights"))
        st.markdown(
            f'<div class="note">Out-of-sample Sharpe of {oos_stats["sharpe"]:.2f} against '
            f'{insample["sharpe"]:.2f} for the fixed final weights over the same window — '
            f'{verdict}.</div>', unsafe_allow_html=True)

        st.markdown("**How much did the weights move across refits?**")
        st.markdown(
            '<p class="subtle">A process that reshuffles the book every year is reacting to '
            'noise and will bleed turnover costs. Flat lines here mean the allocation is '
            'genuinely strategic.</p>', unsafe_allow_html=True)
        fig = go.Figure()
        from dashboard.theme import ticker_colors
        tc = ticker_colors(dark)
        for col in wts.columns:
            fig.add_trace(go.Scatter(x=wts.index, y=wts[col] * 100, name=col, mode="lines",
                                     line=dict(width=2, color=tc.get(col)),
                                     hovertemplate=col + ": %{y:.1f}%<extra></extra>"))
        fig.update_yaxes(ticksuffix="%", title_text="Weight at each refit")
        st.plotly_chart(styled(fig, dark, height=400), width="stretch")
        st.caption(f"Largest single-holding movement across all refits: "
                   f"{(wts.max() - wts.min()).max():.1%}.")


def _shade_bear(fig: go.Figure, result, p: dict) -> None:
    """Shade bear-regime spans. Context layer: grey, behind, and not in the legend."""
    mask = result.bear_mask
    if mask is None or not mask.any():
        return
    edges = mask.astype(int).diff().fillna(0)
    starts = list(mask.index[edges == 1])
    ends = list(mask.index[edges == -1])
    if mask.iloc[0]:
        starts.insert(0, mask.index[0])
    if len(ends) < len(starts):
        ends.append(mask.index[-1])
    for s, e in zip(starts, ends):
        fig.add_vrect(x0=s, x1=e, fillcolor=p["muted"], opacity=0.10,
                      layer="below", line_width=0)
