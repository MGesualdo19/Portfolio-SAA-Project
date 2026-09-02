"""Forward tracker: the live book from its funding date, against an expectation cone."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.tracker import (
    TrackerState,
    drift_report,
    expectation_cone,
    load_tracker,
    percentile_of_actual,
    realised_since,
    save_tracker,
)
from dashboard.data import fmt_money, fmt_pct
from dashboard.methodology import note, notes, tip
from dashboard.theme import palette, styled, ticker_colors


def render(result, portfolio_value: float, dark: bool) -> None:
    p = palette(dark)
    st.title("Forward performance tracker")
    st.markdown(
        '<p class="subtle">Everything on the Backtest page is retrospective and in-sample. '
        'This page is the opposite: it tracks the allocation forward from the day it is '
        'funded, against a distribution of paths it could plausibly take. The point is not '
        'the return number — it is knowing whether that number is normal.</p>',
        unsafe_allow_html=True)

    state = load_tracker()
    last_date = result.returns.index.max().date()

    with st.expander("Set up or edit the tracked portfolio",
                     expanded=state is None):
        c = st.columns(3)
        default_inception = (pd.Timestamp(state.inception).date() if state
                             else last_date - timedelta(days=90))
        inception = c[0].date_input(
            "Funding date", value=default_inception,
            min_value=result.returns.index.min().date(), max_value=last_date,
            help="The date the allocation was (or will be treated as) bought. Tracking "
                 "starts here.")
        initial = c[1].number_input(
            "Amount invested (CAD)", 1_000.0, 100_000_000.0,
            float(state.initial_value_cad) if state else float(portfolio_value),
            5_000.0, format="%.0f")
        source = c[2].radio(
            "Weights to track", ["Current recommendation", "Saved allocation"],
            index=0 if state is None else 1,
            help="Track today's model output, or the weights that were saved when the "
                 "book was actually funded.")

        note = st.text_input("Note", value=state.note if state else "",
                             placeholder="e.g. Initial funding across RRSP + TFSA")

        if st.button("Save tracked portfolio", type="primary"):
            weights = (result.final_weights if source == "Current recommendation"
                       else (state.weight_series() if state else result.final_weights))
            new_state = TrackerState(inception=str(inception),
                                     initial_value_cad=float(initial),
                                     weights={k: float(v) for k, v in weights.items()},
                                     note=note)
            save_tracker(new_state)
            st.success(f"Saved. Tracking {len(weights)} holdings from {inception}.")
            st.rerun()

    if state is None:
        st.info("No tracked portfolio saved yet. Set a funding date and amount above to "
                "start tracking. Until then, there is nothing forward-looking to measure — "
                "and inventing a start date would just be another backtest.")
        return

    weights = state.weight_series()
    st.caption(f"Tracking {len(weights)} holdings from {state.inception}, "
               f"funded at {fmt_money(state.initial_value_cad)}"
               + (f" · {state.note}" if state.note else ""))

    try:
        nav = realised_since(result.securities, weights, state.inception,
                             state.initial_value_cad)
    except Exception as exc:
        st.error(f"Could not compute realised performance: {exc}")
        return

    if len(nav) < 2:
        st.warning("Not enough trading days since the funding date to measure anything yet.")
        return

    elapsed = len(nav)
    cum = float(nav["cumulative"].iloc[-1])
    current_value = float(nav["nav"].iloc[-1])

    # --- headline ---------------------------------------------------------
    c = st.columns(5)
    c[0].metric("Current value", fmt_money(current_value),
                delta=fmt_money(current_value - state.initial_value_cad))
    c[1].metric("Return since funding", fmt_pct(cum, 2), help=tip("total_return"))
    c[2].metric("Trading days elapsed", f"{elapsed}")
    ann = (1 + cum) ** (252 / elapsed) - 1 if elapsed > 20 else np.nan
    c[3].metric("Annualised (if it continued)", fmt_pct(ann, 1) if elapsed > 20 else "—",
                help="Extrapolating a short window is close to meaningless; shown only "
                     "past 20 trading days, and still not a forecast.")
    c[4].metric("Current drawdown", fmt_pct(float(nav["drawdown"].iloc[-1]), 1),
                help=tip("max_drawdown"))

    # --- cone -------------------------------------------------------------
    st.subheader("Against the expected range")
    horizon = st.slider("Projection horizon (trading days beyond today)",
                        0, 756, min(252, max(60, elapsed)), 21)

    total_days = elapsed + horizon
    try:
        cone = expectation_cone(result.returns, weights, total_days,
                                initial_value=state.initial_value_cad)
    except Exception as exc:
        st.error(f"Could not build the expectation cone: {exc}")
        return

    # Map simulated day numbers onto real dates, extending on business days.
    hist_dates = list(nav.index)
    future_dates = list(pd.bdate_range(hist_dates[-1] + pd.Timedelta(days=1),
                                       periods=max(total_days - elapsed, 0)))
    cone_dates = (hist_dates + future_dates)[:total_days]
    cone.index = pd.DatetimeIndex(cone_dates)

    fig = go.Figure()
    band_fill = "rgba(42,120,214,0.10)"
    fig.add_trace(go.Scatter(x=cone.index, y=cone["p95"], name="95th percentile",
                             mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cone.index, y=cone["p5"], name="5th - 95th percentile",
                             mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=band_fill,
                             hovertemplate="5th pct: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=cone.index, y=cone["p75"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cone.index, y=cone["p25"], name="25th - 75th percentile",
                             mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(42,120,214,0.20)",
                             hovertemplate="25th pct: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=cone.index, y=cone["p50"], name="Median path",
                             mode="lines",
                             line=dict(color=p["categorical"][0], width=2, dash="dot"),
                             hovertemplate="Median: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=nav.index, y=nav["nav"], name="Actual",
                             mode="lines", line=dict(color=p["text"], width=2.8),
                             hovertemplate="<b>Actual: $%{y:,.0f}</b><extra></extra>"))
    fig.add_vline(x=hist_dates[-1], line=dict(color=p["muted"], width=1, dash="dash"))
    fig.add_annotation(x=hist_dates[-1], y=1.02, yref="paper", text="today",
                       showarrow=False, font=dict(size=11, color=p["text_secondary"]),
                       xanchor="left")
    fig.update_yaxes(tickprefix="$", title_text="Portfolio value (CAD)")
    st.plotly_chart(styled(fig, dark, height=470), width="stretch")
    notes("expectation_cone", "percentile_of_actual",
          label="How the cone and the percentile are computed")

    try:
        pct = percentile_of_actual(result.returns, weights, cum, elapsed)
        if pct < 10:
            tone, verdict = "warn", ("in the bottom decile of simulated paths — worth "
                                     "understanding which holding drove it before concluding "
                                     "anything about the model")
        elif pct > 90:
            tone, verdict = "note", ("in the top decile of simulated paths — pleasant, and "
                                     "equally uninformative about whether the allocation is right")
        else:
            tone, verdict = "note", "squarely inside the expected range"
        st.markdown(
            f'<div class="{tone}">The realised {cum:.2%} over {elapsed} trading days sits at '
            f'the <b>{pct:.0f}th percentile</b> of {4000:,} bootstrapped paths for this '
            f'allocation over the same horizon — {verdict}. '
            f'Note the honest caveat: at {elapsed} trading days '
            f'({elapsed / 252:.1f} years) the cone is wide and almost any outcome is '
            f'"normal". This measure only starts to carry information after several years.'
            f'</div>', unsafe_allow_html=True)
    except Exception:
        pass

    # --- drift ------------------------------------------------------------
    st.subheader("Drift versus target")
    st.markdown(
        '<p class="subtle">Positions drift with performance. A 20% <i>relative</i> band is the '
        'usual rebalancing trigger — tight enough to hold risk near target, loose enough not '
        'to trade on noise.</p>', unsafe_allow_html=True)

    try:
        drift = drift_report(result.securities, state)
    except Exception as exc:
        st.error(f"Could not compute drift: {exc}")
        return

    tc = ticker_colors(dark)
    d = drift.sort_values("absolute_drift")
    fig = go.Figure(go.Bar(
        x=d["absolute_drift"] * 100, y=d.index, orientation="h",
        marker=dict(color=[tc.get(t, p["categorical"][0]) for t in d.index],
                    line=dict(width=0)),
        text=[f"{v:+.2f}pp" for v in d["absolute_drift"] * 100],
        textposition="outside", textfont=dict(color=p["text"], size=11),
        hovertemplate="<b>%{y}</b><br>Drift %{x:+.2f}pp<extra></extra>", width=0.6))
    fig.update_traces(marker_cornerradius=4)
    fig.add_vline(x=0, line=dict(color=p["muted"], width=1))
    lim = max(abs(d["absolute_drift"] * 100).max() * 1.45, 0.5)
    fig.update_xaxes(range=[-lim, lim], ticksuffix="pp",
                     title_text="Current weight minus target")
    st.plotly_chart(styled(fig, dark, height=400), width="stretch")
    note("drift")

    flagged = drift[drift["rebalance_flag"]]
    if len(flagged):
        st.markdown('<div class="warn"><b>Outside the 20% relative band:</b> ' +
                    ", ".join(f"{t} ({drift.loc[t, 'relative_drift']:+.0%})"
                              for t in flagged.index) +
                    ". A rebalance would trade these back to target.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="note">Every holding is inside its 20% relative drift band. '
                    'No rebalancing trade is indicated.</div>', unsafe_allow_html=True)

    disp = drift.assign(
        target_weight=lambda x: (x["target_weight"] * 100).round(2),
        current_weight=lambda x: (x["current_weight"] * 100).round(2),
        absolute_drift=lambda x: (x["absolute_drift"] * 100).round(2),
        relative_drift=lambda x: (x["relative_drift"] * 100).round(1),
        growth_since_inception=lambda x: (x["growth_since_inception"] * 100).round(2),
        current_value_cad=lambda x: x["current_value_cad"].round(0),
    )[["target_weight", "current_weight", "absolute_drift", "relative_drift",
       "growth_since_inception", "current_value_cad", "rebalance_flag"]]
    disp.columns = ["Target %", "Current %", "Drift pp", "Rel. drift %",
                    "Growth %", "Value CAD", "Rebalance?"]
    st.dataframe(disp, width="stretch")

    # --- contribution -----------------------------------------------------
    st.subheader("What has driven the result so far")
    contrib = (drift["target_weight"] * drift["growth_since_inception"]).sort_values()
    fig = go.Figure(go.Bar(
        x=contrib.values * 100, y=contrib.index, orientation="h",
        marker=dict(color=[tc.get(t, p["categorical"][0]) for t in contrib.index],
                    line=dict(width=0)),
        text=[f"{v:+.2f}%" for v in contrib.values * 100],
        textposition="outside", textfont=dict(color=p["text"], size=11),
        hovertemplate="<b>%{y}</b><br>Contribution %{x:+.2f}%<extra></extra>", width=0.6))
    fig.update_traces(marker_cornerradius=4)
    fig.add_vline(x=0, line=dict(color=p["muted"], width=1))
    lim = max(abs(contrib * 100).max() * 1.4, 0.5)
    fig.update_xaxes(range=[-lim, lim], ticksuffix="%",
                     title_text="Contribution to total return")
    st.plotly_chart(styled(fig, dark, height=400), width="stretch")
    st.caption(f"Contributions sum to {contrib.sum():.2%}, against the realised "
               f"{cum:.2%} — the small gap is the compounding interaction between holdings, "
               f"which a simple weight-times-return decomposition cannot attribute.")
