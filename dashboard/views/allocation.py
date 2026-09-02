"""Allocation view: the recommended book, how it was reached, and the trades."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.optimizer import STRATEGY_LABELS
from core.policy import EXCLUDED, REFERENCE_RATIONALE, REFERENCE_WEIGHTS
from core.tracker import build_order_ticket, ticket_residual_cash
from dashboard.data import fmt_money, fmt_pct
from dashboard.methodology import note, notes, tip
from dashboard.theme import palette, styled, ticker_colors


def render(result, portfolio_value: float, dark: bool) -> None:
    p = palette(dark)
    colors = ticker_colors(dark)
    w = result.final_weights
    s = result.summary

    st.title("Recommended strategic allocation")
    st.markdown(
        '<p class="subtle">Ten holdings, weights derived from a regime-blended covariance '
        'matrix and equilibrium expected returns, solved under five separate objectives and '
        'blended. All portfolio figures are in CAD; individual securities are quoted in their '
        'own currency.</p>', unsafe_allow_html=True)

    # --- headline numbers ------------------------------------------------
    c = st.columns(6)
    c[0].metric("Expected return", fmt_pct(s["expected_return"]),
                help=tip("expected_return_portfolio"))
    c[1].metric("Expected volatility", fmt_pct(s["volatility"]), help=tip("portfolio_vol"))
    c[2].metric("Sharpe", f"{s['sharpe']:.2f}", help=tip("sharpe"))
    c[3].metric("Effective holdings", f"{s['effective_n']:.1f}", help=tip("effective_n"))
    c[4].metric("Effective risk sources", f"{s['effective_n_risk']:.1f}",
                help=tip("effective_n_risk"))
    c[5].metric("Blended MER", fmt_pct(result.blended_mer(), 3), help=tip("blended_mer"))

    notes("expected_return_portfolio", "portfolio_vol", "sharpe", "effective_n",
          "effective_n_risk", "blended_mer",
          label="How the six headline numbers above are calculated")

    st.markdown(
        f'<div class="note"><b>Is this a real portfolio or two ETFs in disguise?</b> '
        f'The allocation holds {s["n_positions"]} positions with an effective count of '
        f'{s["effective_n"]:.1f}, and its risk is spread across {s["effective_n_risk"]:.1f} '
        f'effective sources. Largest single weight is {w.max():.1%} ({w.idxmax()}); the top '
        f'three holdings account for {w.nlargest(3).sum():.1%} of capital. A degenerate '
        f'optimiser output would show an effective count near 2.</div>',
        unsafe_allow_html=True)

    # --- weights chart ---------------------------------------------------
    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("Target weights")
        order = w.sort_values()
        fig = go.Figure(go.Bar(
            x=order.values * 100, y=order.index, orientation="h",
            marker=dict(color=[colors.get(t, p["categorical"][0]) for t in order.index],
                        line=dict(width=0)),
            text=[f"{v:.1%}" for v in order.values],
            textposition="outside",
            textfont=dict(color=p["text"], size=12),
            hovertemplate="<b>%{y}</b><br>Weight %{x:.2f}%<extra></extra>",
            width=0.62,
        ))
        fig.update_traces(marker_cornerradius=4)
        fig.update_xaxes(range=[0, max(order.values * 100) * 1.22], ticksuffix="%")
        st.plotly_chart(styled(fig, dark, height=430, xlabel="Weight"),
                        width="stretch")

    with right:
        st.subheader("By sleeve")
        sl = result.sleeve_table()
        fig = go.Figure(go.Bar(
            x=sl["weight"].values * 100, y=sl.index, orientation="h",
            marker=dict(color=p["categorical"][0], line=dict(width=0)),
            text=[f"{v:.1%}" for v in sl["weight"].values],
            textposition="outside", textfont=dict(color=p["text"], size=12),
            hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>", width=0.6,
        ))
        fig.update_traces(marker_cornerradius=4)
        fig.update_xaxes(range=[0, max(sl["weight"].values * 100) * 1.28], ticksuffix="%")
        st.plotly_chart(styled(fig, dark, height=430, xlabel="Weight"),
                        width="stretch")

    notes("objectives", "resampling", "constraints",
          label="How these weights were derived, end to end")

    # --- capital vs risk -------------------------------------------------
    st.subheader("Where the capital sits versus where the risk sits")
    st.markdown(
        '<p class="subtle">These two bars tell different stories and the gap between them '
        'is the point. Cash-like holdings absorb a third of the capital and almost none of '
        'the risk; that is what they are for. If a holding shows a risk share far above its '
        'weight, it is quietly running the portfolio.</p>', unsafe_allow_html=True)

    rc = s["risk_contributions"].reindex(w.index)
    comp = pd.DataFrame({"Capital weight": w, "Risk contribution": rc}).sort_values(
        "Risk contribution", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=comp.index, x=comp["Capital weight"] * 100, orientation="h",
                         name="Capital weight", marker_color=p["categorical"][0],
                         marker_line=dict(width=2, color=p["surface"]),
                         hovertemplate="<b>%{y}</b><br>Capital %{x:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(y=comp.index, x=comp["Risk contribution"] * 100, orientation="h",
                         name="Risk contribution", marker_color=p["categorical"][1],
                         marker_line=dict(width=2, color=p["surface"]),
                         hovertemplate="<b>%{y}</b><br>Risk %{x:.1f}%<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_xaxes(ticksuffix="%")
    st.plotly_chart(styled(fig, dark, height=440, xlabel="Share of portfolio"),
                    width="stretch")
    note("risk_contribution")

    # --- how the objectives disagree --------------------------------------
    st.subheader("How much do the five objectives agree?")
    st.markdown(
        '<p class="subtle">Each objective is solved independently and the recommendation is '
        'their average. A holding where all five land close together is well-founded; a wide '
        'spread means the weight depends on which definition of risk you accept, and deserves '
        'less confidence than its single number suggests.</p>', unsafe_allow_html=True)

    sw = result.strategy_weights
    fig = go.Figure()
    for i, strat in enumerate(sw.columns):
        fig.add_trace(go.Bar(
            x=sw.index, y=sw[strat] * 100, name=STRATEGY_LABELS.get(strat, strat),
            marker_color=p["categorical"][i], marker_line=dict(width=2, color=p["surface"]),
            hovertemplate="<b>%{x}</b><br>" + STRATEGY_LABELS.get(strat, strat) +
                          ": %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=w.index, y=w.values * 100, mode="markers", name="Recommended (blend)",
        marker=dict(symbol="diamond", size=11, color=p["text"],
                    line=dict(width=2, color=p["surface"])),
        hovertemplate="<b>%{x}</b><br>Recommended: %{y:.1f}%<extra></extra>"))
    fig.update_traces(marker_cornerradius=3, selector=dict(type="bar"))
    fig.update_layout(barmode="group", bargap=0.22, bargroupgap=0.04, hovermode="closest")
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(styled(fig, dark, height=430, ylabel="Weight"), width="stretch")
    note("objectives")

    spread = (sw.max(axis=1) - sw.min(axis=1)).sort_values(ascending=False)
    widest = spread.index[0]
    st.markdown(
        f'<div class="note">Widest disagreement: <b>{widest}</b>, spanning '
        f'{sw.loc[widest].min():.1%} to {sw.loc[widest].max():.1%} across objectives '
        f'({spread.iloc[0]:.1%} spread). Tightest: <b>{spread.index[-1]}</b> at '
        f'{spread.iloc[-1]:.1%}.</div>', unsafe_allow_html=True)

    # --- detail table -----------------------------------------------------
    st.subheader("Holdings detail")
    tbl = result.weights_table()
    show = tbl.assign(
        weight=lambda d: (d["weight"] * 100).round(2),
        risk_share=lambda d: (d["risk_share"] * 100).round(1),
        mer=lambda d: (d["mer"] * 100).round(3),
        strategy_range=lambda d: [f"{lo:.1%} - {hi:.1%}" for lo, hi in
                                  zip(d["strategy_min"], d["strategy_max"])],
    )[["name", "sleeve", "currency", "weight", "risk_share", "strategy_range", "mer", "role"]]
    show.columns = ["Name", "Sleeve", "Ccy", "Weight %", "Risk %", "Across objectives",
                    "MER %", "Role"]
    st.dataframe(show, width="stretch")

    # --- currency & constraints ------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Currency exposure of the underlying assets")
        st.markdown(
            '<p class="subtle">By what the fund <i>holds</i>, not where it is listed. XUU.TO '
            'is a Canadian listing with unhedged US equity inside it, and CGL.TO is unhedged '
            'bullion priced in USD — reporting by listing currency would badly understate the '
            'real USD exposure.</p>', unsafe_allow_html=True)
        ccy = result.currency_exposure()
        fig = go.Figure(go.Bar(
            x=ccy.values * 100, y=ccy.index, orientation="h",
            marker=dict(color=p["categorical"][:len(ccy)], line=dict(width=0)),
            text=[f"{v:.1%}" for v in ccy.values], textposition="outside",
            textfont=dict(color=p["text"]), width=0.6,
            hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>"))
        fig.update_traces(marker_cornerradius=4)
        fig.update_xaxes(range=[0, max(ccy.values * 100) * 1.25], ticksuffix="%")
        st.plotly_chart(styled(fig, dark, height=300), width="stretch")
        note("currency_exposure")

    with c2:
        st.subheader("Is a constraint doing the work?")
        if result.binding:
            st.markdown('<div class="warn">These limits are binding on the final blend — the '
                        'number beside them was chosen by policy, not by the data:<br>' +
                        "<br>".join(f"· {b}" for b in result.binding) + "</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="note">No constraint binds on the final blend — every '
                        'weight sits strictly inside its policy band. Read that together '
                        'with the pinning table below, which tells a more complicated '
                        'story.</div>', unsafe_allow_html=True)
        if result.violations:
            st.error("Constraint violations after rounding: " + "; ".join(result.violations))

        # The blend can show zero binding constraints while its constituents are
        # pinned to bounds all over the place -- averaging corner solutions
        # produces a smooth interior point. Reporting only the blend would be
        # the single most misleading number on this page.
        pinned = []
        for t in result.strategy_weights.index:
            lo, hi = result.constraints.bounds.get(t, result.constraints.default_bounds)
            for col in result.strategy_weights.columns:
                v = float(result.strategy_weights.loc[t, col])
                if abs(v - hi) < 0.005:
                    pinned.append({"Holding": t, "Objective": STRATEGY_LABELS.get(col, col),
                                   "Pinned at": f"{hi:.0%} cap"})
                elif lo > 0 and abs(v - lo) < 0.005:
                    pinned.append({"Holding": t, "Objective": STRATEGY_LABELS.get(col, col),
                                   "Pinned at": f"{lo:.0%} floor"})
        total_cells = result.strategy_weights.size
        if pinned:
            pin_df = pd.DataFrame(pinned)
            share = len(pinned) / total_cells
            st.markdown(
                f'<div class="warn"><b>But the objectives underneath are pinned.</b> '
                f'{len(pinned)} of {total_cells} (objective × holding) cells sit exactly on '
                f'a bound — {share:.0%} of them. Averaging constrained corner solutions '
                f'produces a smooth interior blend that hides this. Any holding appearing '
                f'repeatedly below has a weight set by the policy ceiling or floor, not by '
                f'the data.</div>', unsafe_allow_html=True)
            counts = pin_df.groupby(["Holding", "Pinned at"]).size().reset_index(name="Objectives")
            st.dataframe(counts.sort_values("Objectives", ascending=False).set_index("Holding"),
                         width="stretch")

        st.markdown("**Policy bands**")
        rows = [{"Group": g.name,
                 "Current": f"{sum(w.get(t, 0) for t in g.tickers):.1%}",
                 "Band": f"{g.min_weight:.0%} - {g.max_weight:.0%}"}
                for g in result.constraints.groups]
        st.dataframe(pd.DataFrame(rows).set_index("Group"), width="stretch")
        note("constraints")

    # --- order ticket -----------------------------------------------------
    st.subheader(f"Order ticket at {fmt_money(portfolio_value)}")
    st.markdown(
        '<p class="subtle">Whole shares at the latest close, sized in each security\'s own '
        'currency. USD positions are funded by converting CAD (Norbert\'s Gambit, per the '
        'thesis). The drift column is the tracking error that whole-share rounding forces on '
        'you before a single trade is placed.</p>', unsafe_allow_html=True)

    try:
        ticket = build_order_ticket(w, result.securities, portfolio_value)
        residual = ticket_residual_cash(ticket, portfolio_value)
        disp = ticket.assign(
            target_weight=lambda d: (d["target_weight"] * 100).round(2),
            achieved_weight=lambda d: (d["achieved_weight"] * 100).round(2),
            drift_vs_target=lambda d: (d["drift_vs_target"] * 100).round(2),
            price_native=lambda d: d["price_native"].round(2),
            cost_cad=lambda d: d["cost_cad"].round(0),
            shares=lambda d: d["shares"].astype(int),
        )[["currency", "price_native", "target_weight", "shares", "cost_cad",
           "achieved_weight", "drift_vs_target"]]
        disp.columns = ["Ccy", "Price", "Target %", "Shares", "Cost CAD",
                        "Achieved %", "Drift pp"]
        st.dataframe(disp, width="stretch")
        note("order_ticket")
        st.caption(f"Unallocated after rounding: {fmt_money(residual, digits=2)} "
                   f"({residual / portfolio_value:.2%} of the book). "
                   f"Worst single-position rounding drift: "
                   f"{ticket['drift_vs_target'].abs().max():.2%}.")
    except Exception as exc:
        st.error(f"Could not build the order ticket: {exc}")

    # --- excluded ---------------------------------------------------------
    st.subheader("Excluded from the strategic allocation")
    for ticker, reason in EXCLUDED.items():
        with st.expander(f"{ticker} — excluded", expanded=True):
            st.markdown(f'<p class="subtle">{reason}</p>', unsafe_allow_html=True)

    with st.expander("The reference portfolio these weights are measured against"):
        st.markdown(f'<p class="subtle">{REFERENCE_RATIONALE}</p>', unsafe_allow_html=True)
        cmp = pd.DataFrame({
            "Reference %": (REFERENCE_WEIGHTS.reindex(w.index) * 100).round(1),
            "Recommended %": (w * 100).round(1),
        })
        cmp["Active pp"] = (cmp["Recommended %"] - cmp["Reference %"]).round(1)
        st.dataframe(cmp.sort_values("Active pp"), width="stretch")
