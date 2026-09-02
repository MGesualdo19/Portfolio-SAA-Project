"""Securities view: investigate one holding on its own terms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.factor_regression import build_proxy_factor
from analysis.stress import CRISIS_WINDOWS, up_down_beta
from core.backtester import (
    bootstrap_return_distribution,
    drawdown_series,
    max_drawdown,
    rolling_sharpe,
    rolling_sortino,
    run_factor_regression,
)
from core.policy import EXCLUDED
from core.proxies import PROXY_MAP
from core.returns import summary_stats, total_return_series
from core.risk import var_cvar_summary
from core.universe import build_michael_portfolio
from dashboard.data import fmt_money, fmt_pct, security_frame
from dashboard.methodology import note, notes, tip
from dashboard.theme import palette, styled, ticker_colors


@st.cache_resource(show_spinner=False)
def _full_universe():
    """Every security including the ones excluded from the SAA — they still need watching."""
    pf = build_michael_portfolio()
    pf.fetch_all_prices()
    return pf.securities


def render(result, dark: bool) -> None:
    p = palette(dark)
    st.title("Security investigation")

    secs = _full_universe()
    by_ticker = {s.ticker: s for s in secs}
    options = list(by_ticker.keys())

    c = st.columns([2, 1, 1])
    ticker = c[0].selectbox("Security", options,
                            format_func=lambda t: f"{t} — {by_ticker[t].name[:56]}")
    ccy_mode = c[1].radio("Currency", ["Native", "CAD"], horizontal=True,
                          help="Native is how the security is quoted and how its own "
                               "volatility is meaningfully described. CAD is what it "
                               "contributes to this book.")
    lookback = c[2].selectbox("Window", ["Full history", "10 years", "5 years", "3 years", "1 year"],
                              index=0)

    sec = by_ticker[ticker]
    frame = security_frame(secs, ticker)

    if lookback != "Full history":
        years = int(lookback.split()[0])
        frame = frame.loc[frame.index.max() - pd.DateOffset(years=years):]

    native = ccy_mode == "Native"
    price_col = "close_native" if native else "close_cad"
    tr_col = "tr_native" if native else "tr_cad"
    ret_col = "ret_native" if native else "ret_cad"
    unit = sec.currency if native else "CAD"

    rets = frame[ret_col].dropna()
    if rets.empty:
        st.error("No return data available for this security over the selected window.")
        return
    stats = summary_stats(rets, rf_annual=result.settings["risk_free"])

    # --- header -----------------------------------------------------------
    st.markdown(f"### {sec.name}")
    meta = st.columns(6)
    meta[0].metric("Last price", f"{frame[price_col].iloc[-1]:,.2f} {unit}",
                   help=("Latest close in the security's own quote currency."
                         if native else tip("cad_conversion")))
    meta[1].metric("Annualised return", fmt_pct(stats["ann_return"], 2),
                   help=tip("annualisation"))
    meta[2].metric("Annualised volatility", fmt_pct(stats["ann_vol"], 2),
                   help=tip("annualisation"))
    meta[3].metric("Sharpe", f"{stats['sharpe']:.2f}", help=tip("sharpe"))
    meta[4].metric("Max drawdown", fmt_pct(stats["max_drawdown"], 1),
                   help=tip("max_drawdown"))
    meta[5].metric("MER", fmt_pct(sec.mer, 2),
                   help="Published management expense ratio from the provider fact sheet "
                        "(core/universe.py). It is NOT deducted from the return figures "
                        "on this page, which are gross of fees -- the Backtest page "
                        "charges MERs daily.")

    tags = [f"**Asset class** {sec.asset_class}", f"**Currency** {sec.currency}",
            f"**Account** {sec.account_tag}", f"**Role** {sec.role or '—'}"]
    st.caption(" · ".join(t.replace("**", "") for t in tags))

    weight = float(result.final_weights.get(ticker, 0.0))
    if ticker in EXCLUDED:
        st.markdown(f'<div class="warn"><b>Excluded from the strategic allocation.</b> '
                    f'{EXCLUDED[ticker]}</div>', unsafe_allow_html=True)
    elif weight > 0:
        st.markdown(f'<div class="note"><b>Target weight {weight:.1%}</b> — '
                    f'{sec.thesis or "no thesis recorded"}</div>', unsafe_allow_html=True)

    # --- price / total return ---------------------------------------------
    st.subheader(f"Price and total return ({unit})")
    st.markdown(
        '<p class="subtle">Two different questions. The price line is what the ticker shows; '
        'the total-return line reinvests every distribution and is what an owner actually '
        'earned. For the cash and floating-rate sleeves those two lines diverge enormously — '
        'almost all of their return is distributions, which is exactly why this project never '
        'optimises on price returns.</p>', unsafe_allow_html=True)

    fig = go.Figure()
    base_px = frame[price_col].dropna()
    fig.add_trace(go.Scatter(x=base_px.index, y=base_px / base_px.iloc[0] * 100,
                             name="Price only", mode="lines",
                             line=dict(color=p["categorical"][1], width=1.8),
                             hovertemplate="Price index: %{y:,.1f}<extra></extra>"))
    base_tr = frame[tr_col].dropna()
    fig.add_trace(go.Scatter(x=base_tr.index, y=base_tr / base_tr.iloc[0] * 100,
                             name="Total return (distributions reinvested)", mode="lines",
                             line=dict(color=p["categorical"][0], width=2.4),
                             hovertemplate="Total return index: %{y:,.1f}<extra></extra>"))
    fig.update_yaxes(title_text="Indexed to 100 at window start", type="log")
    st.plotly_chart(styled(fig, dark, height=420), width="stretch")

    total_gap = (base_tr.iloc[-1] / base_tr.iloc[0]) - (base_px.iloc[-1] / base_px.iloc[0])
    st.caption(f"Distributions added {total_gap * 100:.1f} percentage points of cumulative "
               f"return over this window, on top of price movement alone.")
    notes("total_return", "annualisation", label="How the return series is built")

    # --- drawdown & rolling -----------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Drawdown")
        tr_index = frame[tr_col].dropna()
        dd = drawdown_series(tr_index)
        stats_dd = max_drawdown(tr_index)
        fig = go.Figure(go.Scatter(
            x=dd.index, y=dd * 100, mode="lines", name="Drawdown",
            line=dict(color=p["categorical"][7], width=1.6),
            fill="tozeroy", fillcolor="rgba(227,73,72,0.16)",
            hovertemplate="%{y:.1f}%<extra></extra>"))
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(styled(fig, dark, height=330), width="stretch")
        st.caption(f"Worst: {stats_dd['max_drawdown']:.1%}, peaking "
                   f"{pd.Timestamp(stats_dd['peak_date']):%b %Y} and troughing "
                   f"{pd.Timestamp(stats_dd['trough_date']):%b %Y}.")

    with c2:
        st.subheader("Rolling 1-year risk-adjusted return")
        rs = rolling_sharpe(rets, window=252, rf=result.settings["risk_free"] / 252)
        rso = rolling_sortino(rets, window=252, rf=result.settings["risk_free"] / 252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rs.index, y=rs, name="Sharpe", mode="lines",
                                 line=dict(color=p["categorical"][0], width=2),
                                 hovertemplate="Sharpe %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=rso.index, y=rso, name="Sortino", mode="lines",
                                 line=dict(color=p["categorical"][2], width=2),
                                 hovertemplate="Sortino %{y:.2f}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=p["muted"], width=1))
        st.plotly_chart(styled(fig, dark, height=330), width="stretch")
        note("rolling_sharpe")
        st.caption("Sortino penalises only downside deviation. Where it sits far above "
                   "Sharpe, the volatility being punished by Sharpe is mostly upside.")

    # --- risk -------------------------------------------------------------
    st.subheader("Tail risk: three methods, never collapsed into one number")
    st.markdown(
        '<p class="subtle">Each method has a different, genuine blind spot. Parametric assumes '
        'normality and understates fat tails; historical cannot produce a worse day than the '
        'worst already in the sample; bootstrap quantifies estimation uncertainty but still '
        'only resamples history. Agreement is mild reassurance. Disagreement is the finding — '
        'it says the normal-distribution assumption is doing real work.</p>',
        unsafe_allow_html=True)

    conf = st.select_slider("Confidence", options=[0.90, 0.95, 0.99], value=0.95,
                            format_func=lambda x: f"{x:.0%}")
    vc = var_cvar_summary(rets, confidence=conf, n_boot=2000, seed=42)

    fig = go.Figure()
    for i, metric in enumerate(["VaR", "CVaR"]):
        fig.add_trace(go.Bar(
            x=vc.index, y=vc[metric] * 100, name=metric,
            marker_color=p["categorical"][i], marker_line=dict(width=2, color=p["surface"]),
            text=[f"{v:.2f}%" for v in vc[metric] * 100], textposition="outside",
            textfont=dict(color=p["text"], size=11),
            hovertemplate="<b>%{x}</b><br>" + metric + " %{y:.2f}%<extra></extra>"))
    fig.update_traces(marker_cornerradius=3, selector=dict(type="bar"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.06, hovermode="closest")
    fig.update_yaxes(ticksuffix="%", title_text=f"Daily loss at {conf:.0%} confidence")
    st.plotly_chart(styled(fig, dark, height=360), width="stretch")

    gap = float(vc.loc["parametric", "CVaR"] - vc.loc["historical", "CVaR"])
    st.markdown(
        f'<div class="{"warn" if gap < -0.002 else "note"}">Parametric CVaR sits '
        f'{abs(gap) * 100:.2f} percentage points '
        f'{"BELOW" if gap < 0 else "above"} the historical estimate. '
        + ("A material shortfall means this security's real left tail is fatter than a normal "
           "distribution allows, and any Gaussian risk number for it — including the "
           "volatility figure at the top of this page — understates what a bad day looks like."
           if gap < -0.002 else
           "The two are close, so the normal-distribution assumption is not distorting this "
           "security's tail risk much.") + '</div>', unsafe_allow_html=True)

    st.dataframe((vc * 100).round(3), width="stretch")
    note("var_cvar")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Return distribution**")
        fig = go.Figure(go.Histogram(
            x=rets * 100, nbinsx=90, marker_color=p["categorical"][0],
            marker_line=dict(width=0), hovertemplate="%{x:.2f}%: %{y} days<extra></extra>"))
        for q, lab in [(vc.loc["historical", "VaR"] * -100, f"{conf:.0%} VaR")]:
            fig.add_vline(x=q, line=dict(color=p["categorical"][7], width=2, dash="dash"),
                          annotation_text=lab, annotation_position="top left",
                          annotation_font=dict(size=11, color=p["text_secondary"]))
        fig.update_xaxes(ticksuffix="%", title_text="Daily return")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(styled(fig, dark, height=330), width="stretch")
        note("skew_kurtosis")
        st.caption(f"Skew {stats['skew']:.2f}, excess kurtosis {stats['excess_kurtosis']:.1f}. "
                   f"A normal distribution has both at zero; high kurtosis means extreme days "
                   f"are far more common than the volatility figure implies.")

    with c2:
        st.markdown("**Simulated 1-year forward return**")
        sim = bootstrap_return_distribution(rets, n_years=1.0, n_sims=3000, seed=42,
                                            block_size=21)
        fig = go.Figure(go.Histogram(
            x=sim * 100, nbinsx=70, marker_color=p["categorical"][2],
            marker_line=dict(width=0), hovertemplate="%{x:.1f}%: %{y} paths<extra></extra>"))
        for q, lab, col in [(sim.quantile(0.05) * 100, "5th", p["categorical"][7]),
                            (sim.quantile(0.50) * 100, "median", p["text"]),
                            (sim.quantile(0.95) * 100, "95th", p["categorical"][5])]:
            fig.add_vline(x=q, line=dict(color=col, width=2, dash="dash"),
                          annotation_text=lab, annotation_position="top",
                          annotation_font=dict(size=11, color=p["text_secondary"]))
        fig.update_xaxes(ticksuffix="%", title_text="Simulated 1-year total return")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(styled(fig, dark, height=330), width="stretch")
        note("bootstrap_forward")
        st.caption(f"Block bootstrap, 3,000 paths, preserving volatility clustering. "
                   f"Probability of a loss over one year: {(sim < 0).mean():.0%}.")

    # --- relationship to the book -----------------------------------------
    st.subheader("Relationship to the rest of the book")
    if ticker in result.returns.columns:
        core_ret = result.returns
        others = [c for c in core_ret.columns if c != ticker]

        corr_all = core_ret.corr()[ticker].drop(ticker)
        bear = result.bear_mask.reindex(core_ret.index).fillna(False)
        corr_bear = core_ret.loc[bear].corr()[ticker].drop(ticker)
        corr_bull = core_ret.loc[~bear].corr()[ticker].drop(ticker)
        cmp = pd.DataFrame({"Calm": corr_bull, "Bear": corr_bear}).sort_values("Bear")

        fig = go.Figure()
        fig.add_trace(go.Bar(y=cmp.index, x=cmp["Calm"], orientation="h", name="Calm markets",
                             marker_color=p["categorical"][0],
                             marker_line=dict(width=2, color=p["surface"]),
                             hovertemplate="<b>%{y}</b><br>Calm %{x:.2f}<extra></extra>"))
        fig.add_trace(go.Bar(y=cmp.index, x=cmp["Bear"], orientation="h", name="Bear regime",
                             marker_color=p["categorical"][7],
                             marker_line=dict(width=2, color=p["surface"]),
                             hovertemplate="<b>%{y}</b><br>Bear %{x:.2f}<extra></extra>"))
        fig.update_traces(marker_cornerradius=4)
        fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06, hovermode="closest")
        fig.add_vline(x=0, line=dict(color=p["muted"], width=1))
        fig.update_xaxes(range=[-1, 1], title_text=f"Correlation of CAD returns to {ticker}")
        st.plotly_chart(styled(fig, dark, height=420), width="stretch")

        note("conditional_correlation")

        rises = (cmp["Bear"] - cmp["Calm"]).sort_values(ascending=False)
        st.markdown(
            f'<div class="note">Correlation to <b>{rises.index[0]}</b> rises most in a '
            f'drawdown, from {cmp.loc[rises.index[0], "Calm"]:.2f} to '
            f'{cmp.loc[rises.index[0], "Bear"]:.2f} ({rises.iloc[0]:+.2f}). Correlation to '
            f'<b>{rises.index[-1]}</b> moves least ({rises.iloc[-1]:+.2f}). A diversifier that '
            f'holds its low correlation in the bear column is earning its place; one whose '
            f'bars converge is a diversifier only in calm markets, which is when you least '
            f'need one.</div>', unsafe_allow_html=True)

        st.markdown("**Upside and downside beta to core US equity (XUU.TO)**")
        st.markdown(
            '<p class="subtle">Beta measured separately on up days and down days. A genuine '
            'hedge has a clearly negative downside beta. Positive asymmetry — participating '
            'in the losses more than the gains — is the worst possible profile and is '
            'invisible to any symmetric statistic.</p>', unsafe_allow_html=True)
        try:
            ud = up_down_beta(core_ret, "XUU.TO")
            if ticker in ud.index:
                row = ud.loc[ticker]
                cc = st.columns(3)
                cc[0].metric("Upside beta", f"{row['upside_beta']:.2f}")
                cc[1].metric("Downside beta", f"{row['downside_beta']:.2f}")
                cc[2].metric("Asymmetry", f"{row['asymmetry']:+.2f}",
                             delta_color="inverse",
                             help="Downside beta minus upside beta. Negative is good.")
            st.dataframe(ud.round(3), width="stretch")
            note("up_down_beta")
        except Exception:
            pass

        st.markdown("**Behaviour through named crisis windows**")
        rows = {}
        for label, (start, end, _why) in CRISIS_WINDOWS.items():
            seg = core_ret.loc[start:end]
            if len(seg) < 5:
                continue
            rows[label] = {ticker: (1 + seg[ticker]).prod() - 1,
                           "Whole portfolio": (1 + seg @ result.final_weights.reindex(
                               seg.columns).fillna(0)).prod() - 1}
        cw = pd.DataFrame(rows).T
        fig = go.Figure()
        for i, col in enumerate(cw.columns):
            fig.add_trace(go.Bar(x=cw.index, y=cw[col] * 100, name=col,
                                 marker_color=p["categorical"][i],
                                 marker_line=dict(width=2, color=p["surface"]),
                                 hovertemplate="<b>%{x}</b><br>" + col +
                                               ": %{y:.1f}%<extra></extra>"))
        fig.update_traces(marker_cornerradius=3)
        fig.update_layout(barmode="group", bargap=0.26, bargroupgap=0.06, hovermode="closest")
        fig.add_hline(y=0, line=dict(color=p["muted"], width=1))
        fig.update_yaxes(ticksuffix="%", title_text="Total return over window")
        st.plotly_chart(styled(fig, dark, height=400), width="stretch")

        st.markdown("**Factor exposure**")
        st.markdown(
            '<p class="subtle">Regressed on value and size-value spreads built from holdings '
            'already in the book (VTV minus XUU, AVUV minus XUU). These are not academic '
            'Fama-French factors — read the coefficients directionally.</p>',
            unsafe_allow_html=True)
        try:
            value_proxy = build_proxy_factor(core_ret["VTV"], core_ret["XUU.TO"], "value_proxy")
            size_proxy = build_proxy_factor(core_ret["AVUV"], core_ret["XUU.TO"], "size_value_proxy")
            factors = pd.concat([value_proxy, size_proxy], axis=1)
            _model, summ = run_factor_regression(core_ret[ticker], factors)
            st.dataframe(summ.round(4), width="stretch")
            note("factor_regression")
        except Exception as exc:
            st.caption(f"Factor regression unavailable: {exc}")
    else:
        st.info(f"{ticker} is outside the strategic universe, so no cross-holding "
                f"comparison is computed here. Its standalone statistics above are "
                f"complete.")

    # --- data provenance ---------------------------------------------------
    with st.expander("Data provenance for this security"):
        st.write(f"**Own price history:** {frame.index.min():%Y-%m-%d} to "
                 f"{frame.index.max():%Y-%m-%d} ({len(frame):,} trading days)")
        chain = PROXY_MAP.get(ticker)
        if chain:
            st.write(f"**Proxy backfill:** {chain.rationale}")
            for link in chain.links:
                st.write(f"- `{link.ticker}` ({link.currency}, "
                         f"{'FX-converted' if link.fx_convert else 'own currency'}"
                         f"{', volatility-matched' if link.vol_match else ''}) — {link.note}")
            if result.proxy_report is not None and ticker in result.proxy_report.index.get_level_values(0):
                st.dataframe(result.proxy_report.loc[ticker].round(3),
                             width="stretch")
            note("proxy_quality")
        else:
            st.write("**Proxy backfill:** none — this security's own history already spans "
                     "the full analysis window.")
        st.write(f"**Fund page:** {sec.fund_site_url or 'n/a'}")
        if sec.thesis:
            st.write(f"**Thesis:** {sec.thesis}")
