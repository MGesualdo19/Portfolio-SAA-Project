"""Diagnostics view: the model's own working, including where it is weakest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.stress import correlation_regime_table, tail_dependence, up_down_beta
from core.estimation import correlation_from_cov, shrunk_covariance
from core.policy import REFERENCE_RATIONALE, thesis_views
from core.returns import reconciliation_report
from dashboard.data import fmt_pct
from dashboard.methodology import note, notes, tip
from dashboard.theme import diverging, palette, styled, ticker_colors


def render(result, dark: bool) -> None:
    p = palette(dark)
    st.title("Model diagnostics")
    st.markdown(
        '<p class="subtle">Everything the allocation depends on, including the parts that are '
        'weakest. A model whose assumptions are not inspectable is not a model, it is an '
        'opinion with decimal places.</p>', unsafe_allow_html=True)

    tabs = st.tabs(["Correlation & regime", "Downside behaviour", "Expected returns",
                    "Data provenance", "Assumptions & limits"])

    # =====================================================================
    with tabs[0]:
        st.subheader("Correlation: calm markets versus drawdowns")
        st.markdown(
            '<p class="subtle">The left matrix is the one most portfolio tools show. The right '
            'is the one that matters, because the entire case for holding gold, floating-rate '
            'notes and non-US equity is about behaviour <i>in a drawdown</i>. The third panel '
            'is the difference, and any strongly red cell there is a diversification promise '
            'that history says gets broken exactly when it is called on.</p>',
            unsafe_allow_html=True)

        rets = result.returns
        bear = result.bear_mask.reindex(rets.index).fillna(False)
        calm_c = rets.loc[~bear].corr()
        bear_c = rets.loc[bear].corr()
        delta = bear_c - calm_c

        cols = st.columns(3)
        for col, (mat, title, scale, zmid) in zip(cols, [
            (calm_c, "Calm markets", diverging(dark), (-1, 1)),
            (bear_c, "Bear regime", diverging(dark), (-1, 1)),
            (delta, "Bear minus calm", diverging(dark), (-0.6, 0.6)),
        ]):
            with col:
                fig = go.Figure(go.Heatmap(
                    z=mat.values, x=mat.columns, y=mat.index,
                    colorscale=scale, zmin=zmid[0], zmax=zmid[1], zmid=0,
                    text=mat.round(2).values, texttemplate="%{text}",
                    textfont=dict(size=9),
                    hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
                    showscale=False, xgap=2, ygap=2))
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(hovermode="closest")
                st.plotly_chart(styled(fig, dark, height=430, title=title),
                                width="stretch")

        st.caption(f"Regime split over the analysis window: {int((~bear).sum()):,} calm days, "
                   f"{int(bear.sum()):,} bear days ({bear.mean():.1%}). 'Bear' means this "
                   f"book's own equity sleeve was more than "
                   f"{result.settings['drawdown_threshold']:.0%} below its trailing high — not "
                   f"the TSX's drawdown, which is a poor description of the risk a globally "
                   f"diversified investor is actually carrying.")

        notes("conditional_correlation", "bear_regime",
              label="How the regime split and these matrices are computed")

        pairs = delta.where(np.triu(np.ones(delta.shape), k=1).astype(bool)).stack().dropna()
        top = pairs.sort_values(ascending=False).head(8)
        st.markdown("**Pairs that correlate up the most in a drawdown**")
        fig = go.Figure(go.Bar(
            x=top.values, y=[f"{a} / {b}" for a, b in top.index], orientation="h",
            marker=dict(color=p["categorical"][7], line=dict(width=0)),
            text=[f"{v:+.2f}" for v in top.values], textposition="outside",
            textfont=dict(color=p["text"], size=11),
            hovertemplate="<b>%{y}</b><br>%{x:+.2f}<extra></extra>", width=0.62))
        fig.update_traces(marker_cornerradius=4)
        fig.update_xaxes(range=[0, top.max() * 1.3],
                         title_text="Correlation increase (bear minus calm)")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(styled(fig, dark, height=340), width="stretch")

        st.markdown(
            f'<div class="note">This is precisely why the covariance matrix the optimiser '
            f'sees is a blend of the calm and bear matrices, weighted '
            f'{result.settings["bear_weight"]:.0%} toward bear behaviour. Optimising on the '
            f'calm matrix alone would buy diversification that history says evaporates in '
            f'exactly the conditions it was bought for.</div>', unsafe_allow_html=True)

        d = result.cov_diagnostics
        if "shrinkage_full" in d:
            c = st.columns(3)
            c[0].metric("Ledoit-Wolf shrinkage (full sample)",
                        f"{d['shrinkage_full']:.3f}", help=tip("shrinkage"))
            if d.get("shrinkage_bear") is not None:
                c[1].metric("Shrinkage (bear subsample)", f"{d['shrinkage_bear']:.3f}")
            c[2].metric("Bear days used", f"{d.get('bear_days', 0):,}")

        if "vol_full" in d:
            vol = pd.DataFrame({"Calm-blended": d["vol_full"], "Bear regime": d["vol_bear"]})
            vol = (vol * 100).round(1).sort_values("Bear regime", ascending=False)
            st.markdown("**Annualised volatility by regime (%)**")
            fig = go.Figure()
            for i, col in enumerate(vol.columns):
                fig.add_trace(go.Bar(x=vol.index, y=vol[col], name=col,
                                     marker_color=p["categorical"][i],
                                     marker_line=dict(width=2, color=p["surface"]),
                                     hovertemplate="<b>%{x}</b><br>" + col +
                                                   ": %{y:.1f}%<extra></extra>"))
            fig.update_traces(marker_cornerradius=3)
            fig.update_layout(barmode="group", bargap=0.26, bargroupgap=0.05,
                              hovermode="closest")
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(styled(fig, dark, height=380), width="stretch")
            note("covariance")

    # =====================================================================
    with tabs[1]:
        st.subheader("Does each holding hold up when core equity falls?")
        ref = "XUU.TO"
        st.markdown(
            f'<p class="subtle">All three tables below condition on <b>{ref}</b>, the core US '
            'equity sleeve, because that is where the concentration risk the thesis is built '
            'around actually lives.</p>', unsafe_allow_html=True)

        try:
            table = correlation_regime_table(result.returns, ref, quantile=0.10)
            st.markdown("**Correlation in calm markets vs. the worst decile of "
                        f"{ref} days**")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=table.index, x=table["calm_corr"], orientation="h",
                                 name="Calm", marker_color=p["categorical"][0],
                                 marker_line=dict(width=2, color=p["surface"]),
                                 hovertemplate="<b>%{y}</b><br>Calm %{x:.2f}<extra></extra>"))
            fig.add_trace(go.Bar(y=table.index, x=table["stress_corr"], orientation="h",
                                 name="Worst decile", marker_color=p["categorical"][7],
                                 marker_line=dict(width=2, color=p["surface"]),
                                 hovertemplate="<b>%{y}</b><br>Stress %{x:.2f}<extra></extra>"))
            fig.update_traces(marker_cornerradius=4)
            fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06,
                              hovermode="closest")
            fig.add_vline(x=0, line=dict(color=p["muted"], width=1))
            fig.update_xaxes(range=[-1, 1], title_text=f"Correlation to {ref}")
            st.plotly_chart(styled(fig, dark, height=420), width="stretch")
            st.dataframe(table.round(3), width="stretch")
            note("conditional_correlation")

            worst = table.index[0]
            gold_row = table.loc["CGL.TO"] if "CGL.TO" in table.index else None
            if gold_row is not None:
                verdict = ("holds its diversification through stress"
                           if gold_row["stress_corr"] < 0.3 else
                           "loses much of its diversification exactly when it is needed")
                st.markdown(
                    f'<div class="note"><b>The gold question, answered directly.</b> CGL.TO '
                    f'correlates {gold_row["calm_corr"]:.2f} to {ref} in calm markets and '
                    f'{gold_row["stress_corr"]:.2f} in {ref}\'s worst decile '
                    f'({gold_row["stress_uplift"]:+.2f}) — it {verdict}. The largest stress '
                    f'uplift anywhere in the book belongs to <b>{worst}</b> at '
                    f'{table.loc[worst, "stress_uplift"]:+.2f}.</div>',
                    unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Conditional correlation failed: {exc}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Upside vs downside beta**")
            try:
                ud = up_down_beta(result.returns, ref)
                fig = go.Figure()
                fig.add_trace(go.Bar(y=ud.index, x=ud["upside_beta"], orientation="h",
                                     name="Upside beta", marker_color=p["categorical"][2],
                                     marker_line=dict(width=2, color=p["surface"]),
                                     hovertemplate="<b>%{y}</b><br>Up %{x:.2f}<extra></extra>"))
                fig.add_trace(go.Bar(y=ud.index, x=ud["downside_beta"], orientation="h",
                                     name="Downside beta", marker_color=p["categorical"][7],
                                     marker_line=dict(width=2, color=p["surface"]),
                                     hovertemplate="<b>%{y}</b><br>Down %{x:.2f}<extra></extra>"))
                fig.update_traces(marker_cornerradius=4)
                fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06,
                                  hovermode="closest")
                fig.add_vline(x=0, line=dict(color=p["muted"], width=1))
                st.plotly_chart(styled(fig, dark, height=380,
                                       xlabel=f"Beta to {ref}"), width="stretch")
                st.dataframe(ud.round(3), width="stretch")
                note("up_down_beta")
            except Exception as exc:
                st.caption(str(exc))

        with c2:
            st.markdown("**Joint-crash frequency (lower tail dependence)**")
            st.markdown(
                '<p class="subtle">Given XUU.TO had a bottom-5% day, how often did this '
                'holding also have one? Independence would sit near 5%. This catches joint '
                'crash risk that no correlation coefficient reveals.</p>',
                unsafe_allow_html=True)
            try:
                td = tail_dependence(result.returns, ref, quantile=0.05)
                fig = go.Figure(go.Bar(
                    x=td.values * 100, y=td.index, orientation="h",
                    marker=dict(color=[p["categorical"][7] if v > 0.25
                                       else p["categorical"][0] for v in td.values],
                                line=dict(width=0)),
                    text=[f"{v:.0%}" for v in td.values], textposition="outside",
                    textfont=dict(color=p["text"], size=11),
                    hovertemplate="<b>%{y}</b><br>%{x:.0f}%<extra></extra>", width=0.62))
                fig.update_traces(marker_cornerradius=4)
                fig.add_vline(x=5, line=dict(color=p["muted"], width=1.5, dash="dash"),
                              annotation_text="independence", annotation_position="top",
                              annotation_font=dict(size=10, color=p["text_secondary"]))
                fig.update_xaxes(range=[0, max(td.values * 100) * 1.3], ticksuffix="%")
                fig.update_layout(hovermode="closest")
                st.plotly_chart(styled(fig, dark, height=380), width="stretch")
                note("tail_dependence")
            except Exception as exc:
                st.caption(str(exc))

    # =====================================================================
    with tabs[2]:
        st.subheader("Where the expected returns come from")
        st.markdown(
            '<p class="subtle">Not from historical averages. The standard error on a mean '
            'return estimated over 19 years of a 19%-volatility asset is about 4.4% a year — '
            'wider than the entire plausible spread of expected returns across these '
            'holdings. Sample means cannot rank these assets, so the model does not ask them '
            'to. It reverse-optimises a reference portfolio into the returns that would '
            'justify it, then tilts only where the thesis states a view.</p>',
            unsafe_allow_html=True)

        det = result.view_detail.copy()
        det["historical (shrunk)"] = result.historical_returns_shrunk
        show = (det * 100).round(2).sort_values("posterior", ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=show.index, y=show["equilibrium"], name="Equilibrium (prior)",
                             marker_color=p["categorical"][0],
                             marker_line=dict(width=2, color=p["surface"]),
                             hovertemplate="<b>%{x}</b><br>Prior %{y:.2f}%<extra></extra>"))
        fig.add_trace(go.Bar(x=show.index, y=show["posterior"], name="Posterior (after views)",
                             marker_color=p["categorical"][1],
                             marker_line=dict(width=2, color=p["surface"]),
                             hovertemplate="<b>%{x}</b><br>Posterior %{y:.2f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=show.index, y=show["historical (shrunk)"],
                                 name="Shrunk historical (comparison only)", mode="markers",
                                 marker=dict(symbol="diamond", size=10, color=p["text"],
                                             line=dict(width=2, color=p["surface"])),
                                 hovertemplate="<b>%{x}</b><br>Historical %{y:.2f}%<extra></extra>"))
        fig.update_traces(marker_cornerradius=3, selector=dict(type="bar"))
        fig.update_layout(barmode="group", bargap=0.24, bargroupgap=0.05, hovermode="closest")
        fig.update_yaxes(ticksuffix="%", title_text="Annualised expected return")
        st.plotly_chart(styled(fig, dark, height=430), width="stretch")
        st.caption("The diamonds are shrunk historical means, shown only so the difference is "
                   "visible. They are not used to set any weight — the gap between a diamond "
                   "and its bars is the amount of sampling noise the model declined to act on.")

        st.dataframe(show, width="stretch")
        notes("equilibrium_returns", "black_litterman",
              label="How expected returns are constructed")

        st.markdown("**The views, and what each one is claiming**")
        for v in thesis_views():
            picks = "  ".join(f"{'+' if c > 0 else ''}{c:g}·{t}" for t, c in v.picks.items())
            with st.expander(f"{v.name}  —  {v.q:+.2%}/yr, confidence {v.confidence:.0%}"):
                st.code(picks, language=None)
                st.markdown(f'<p class="subtle">{v.rationale}</p>', unsafe_allow_html=True)

        with st.expander("The reference portfolio these are all anchored to"):
            st.markdown(f'<p class="subtle">{REFERENCE_RATIONALE}</p>',
                        unsafe_allow_html=True)

    # =====================================================================
    with tabs[3]:
        st.subheader("How much of this history is real?")
        st.markdown(
            '<p class="subtle">Six of the ten holdings did not exist in 2007. Their '
            'pre-inception history is filled with index proxies so the covariance matrix can '
            'see 2008 and 2020 at all — the unspliced common window starts in November 2021 '
            'and contains no crash whatsoever. That is a real assumption and it is measured '
            'here rather than asserted.</p>', unsafe_allow_html=True)

        src = result.return_sources
        synth = (src.map(lambda x: str(x).startswith("~")).mean() * 100).sort_values(
            ascending=True)
        tc = ticker_colors(dark)
        fig = go.Figure(go.Bar(
            x=synth.values, y=synth.index, orientation="h",
            marker=dict(color=[tc.get(t, p["categorical"][0]) for t in synth.index],
                        line=dict(width=0)),
            text=[f"{v:.0f}%" for v in synth.values], textposition="outside",
            textfont=dict(color=p["text"], size=11),
            hovertemplate="<b>%{y}</b><br>%{x:.1f}% proxy-filled<extra></extra>", width=0.62))
        fig.update_traces(marker_cornerradius=4)
        fig.update_xaxes(range=[0, max(synth.values.max() * 1.25, 10)], ticksuffix="%",
                         title_text="Share of the analysis window filled by a proxy")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(styled(fig, dark, height=380), width="stretch")

        st.markdown("**Splice quality on the overlap period**")
        st.markdown(
            '<p class="subtle">Each proxy is scored where both it and the real holding exist. '
            'Correlation above 0.95 is a strong splice; the cash-like sleeves are judged on '
            'volatility ratio instead, because a near-riskless holding\'s daily co-movement '
            'with its proxy is mostly microstructure noise and is not what is being '
            'borrowed.</p>', unsafe_allow_html=True)
        if result.proxy_report is not None:
            st.dataframe(result.proxy_report.round(3), width="stretch")
        notes("proxy_splice", "proxy_quality", label="How the splice and its audit work")

        st.markdown("**Total-return reconciliation**")
        st.markdown(
            '<p class="subtle">Total returns are rebuilt from close prices plus distributions '
            'and cross-checked against Yahoo\'s own adjusted-close series. The two are '
            'computed from different inputs, so a large gap means one of them is corrupt for '
            'that ticker.</p>', unsafe_allow_html=True)
        try:
            rec = reconciliation_report(result.securities)
            st.dataframe(rec.round(4), width="stretch")
            bad = rec[rec["flag"] == "INVESTIGATE"]
            if len(bad):
                st.markdown('<div class="warn">Reconciliation gaps worth investigating: ' +
                            ", ".join(bad.index) + "</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="note">Every ticker reconciles to within 2% on the '
                            'worst single day. The return series feeding the model are '
                            'consistent across two independent constructions.</div>',
                            unsafe_allow_html=True)
            note("reconciliation")
        except Exception as exc:
            st.caption(f"Reconciliation unavailable: {exc}")

    # =====================================================================
    with tabs[4]:
        st.subheader("What this model assumes, and where it is weak")
        st.markdown("""
<p class="subtle">

<b>The FX timestamp correction.</b> Yahoo stamps its USD/CAD daily bar one day ahead of
the session it describes. Uncorrected, every USD holding's CAD return pairs an equity
move with the wrong day's currency move, inflating the modelled volatility of the US
sleeve by about 15% and destroying ten points of correlation. The correction is applied
in <code>core/fx.py</code> and is verifiable: it moves XUU.TO's correlation with
CAD-converted VTI from 0.80 to 0.91, and their volatility ratio from 1.155 to 0.991.
Without it the optimiser would underweight US equity for a reason that is purely a
data-vendor artefact.

<b>Proxy-extended history is an estimate.</b> Six holdings rely on index proxies for
part of the window — see the Data provenance tab for exactly how much. The alternative
was a covariance matrix estimated on a 4.7-year window containing no crash, which would
have been worse and less honest.

<b>Expected returns are assumptions, not forecasts.</b> Every return figure in this
dashboard comes from reverse-optimising a reference portfolio and applying stated views.
Set the view-confidence slider to zero and the model returns the policy portfolio: that
is the honest baseline, and the tilts away from it are exactly as strong as the stated
confidence in each view.

<b>Account placements are still placeholders.</b> The RRSP/TFSA/FHSA tags in
<code>core/universe.py</code> encode what the thesis's tax logic implies <i>should</i>
be true, not confirmed statements. The asset-location argument for holding US equity in
the RRSP is sound in principle and unverified in practice. Nothing in the weights depends
on it, but the after-tax return does, and that gap is real.

<b>No tax modelling.</b> Withholding tax on US dividends, capital-gains treatment in the
non-registered account, and the cost of Norbert's Gambit are all absent. For a book with
meaningful US exposure across registered and taxable accounts, after-tax and pre-tax
optimal allocations genuinely differ.

<b>The horizon is not modelled.</b> A single allocation is derived, but an FHSA funding a
home purchase in three years and an RRSP funding a retirement in thirty are not the same
problem. The defensive sleeve here is sized for a blended, medium horizon; if the real
horizon is longer, the fixed-income band should move down.

<b>One historical path.</b> Nineteen years is one draw from the distribution of possible
histories. Resampling and shrinkage reduce how much the answer depends on that specific
draw, but nothing eliminates it. The out-of-sample walk-forward on the Backtest page is
the closest thing here to independent evidence.

</p>""", unsafe_allow_html=True)

        st.markdown("**Current model settings**")
        # Values are a mix of floats, ints and a date string, so the column is
        # stringified -- Arrow cannot type a mixed object column and would
        # otherwise throw on serialisation.
        st.dataframe(
            pd.DataFrame({"value": {k: str(v) for k, v in result.settings.items()}}),
            width="stretch")
