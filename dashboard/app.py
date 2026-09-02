"""
dashboard/app.py

Local Streamlit dashboard for the SAA model.

Run it from the repo root:

    python run_dashboard.py

or equivalently:

    streamlit run dashboard/app.py

Five views:
  Allocation  -- the recommended book, how it was arrived at, and the trades
  Backtest    -- how it would have behaved, 2007 to today, plus out-of-sample
  Evaluation  -- adversarial tests: does the model actually earn its complexity?
  Forward     -- live tracking from a funding date against an expectation cone
  Securities  -- single-holding investigation
  Diagnostics -- the model's own working: inputs, assumptions, and weak points
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="SAA - Strategic Asset Allocation",
                   page_icon="\U0001F4CA", layout="wide",
                   initial_sidebar_state="expanded")

from dashboard.data import get_cached_saa, get_saa  # noqa: E402
from dashboard.views import (allocation, backtest, diagnostics, evaluation,  # noqa: E402
                             forward, security)

CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px;}
  h1, h2, h3 {letter-spacing: -0.015em;}
  h1 {font-size: 1.75rem !important; margin-bottom: 0.2rem;}
  h2 {font-size: 1.22rem !important; margin-top: 1.6rem;}
  h3 {font-size: 1.02rem !important;}
  .subtle {color: #52514e; font-size: 0.9rem; line-height: 1.55;}
  [data-testid="stMetricValue"] {font-size: 1.5rem; font-variant-numeric: tabular-nums;}
  [data-testid="stMetricLabel"] {color: #52514e;}
  div[data-testid="stDataFrame"] {font-variant-numeric: tabular-nums;}
  .note {border-left: 3px solid #2a78d6; padding: 0.6rem 0.9rem; background: #f4f8fd;
         border-radius: 0 6px 6px 0; font-size: 0.88rem; line-height: 1.55; margin: 0.5rem 0 1rem 0;}
  .warn {border-left: 3px solid #eda100; padding: 0.6rem 0.9rem; background: #fdf8ec;
         border-radius: 0 6px 6px 0; font-size: 0.88rem; line-height: 1.55; margin: 0.5rem 0 1rem 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: the model's judgement parameters, exposed rather than buried
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Strategic Asset Allocation")
    st.caption("Michael's book - CAD base currency")

    view = st.radio("View", ["Allocation", "Evaluation", "Backtest", "Forward tracker",
                             "Securities", "Diagnostics"], label_visibility="collapsed")

    st.divider()
    st.markdown("**Model settings**")
    st.caption("Each of these is a judgement, not an estimate. "
               "Changing one re-runs the whole allocation.")

    bear_weight = st.slider(
        "Bear-regime weight in covariance", 0.0, 1.0, 0.35, 0.05,
        help="How much the covariance matrix is priced off drawdown behaviour rather "
             "than average behaviour. 0 = full-sample only. 0.35 matches the historical "
             "frequency of bear days in the sample.")
    drawdown_threshold = st.slider(
        "Bear threshold (equity drawdown)", 0.05, 0.25, 0.10, 0.01,
        help="How far the book's own equity sleeve must be below its trailing high "
             "before a day counts as 'bear'.")
    risk_aversion = st.slider(
        "Risk aversion (delta)", 1.0, 6.0, 2.8, 0.1,
        help="Market price of risk used to reverse-optimise equilibrium returns. "
             "Higher = the model assumes investors demand more return per unit of risk.")
    view_scale = st.slider(
        "Thesis view confidence", 0.0, 2.0, 1.0, 0.1,
        help="Scales every Black-Litterman view's confidence. 0 ignores the thesis "
             "entirely and returns the policy portfolio; 2 doubles conviction.")
    n_resamples = st.select_slider(
        "Michaud resamples", options=[0, 30, 60, 120, 250], value=60,
        help="Bootstrap resamples per objective. 0 solves once on the full sample "
             "(fast, less robust). Higher is more stable and slower.")
    rf = st.number_input("Risk-free rate (annual)", 0.0, 0.10, 0.0275, 0.0025,
                         format="%.4f")

    st.divider()
    portfolio_value = st.number_input(
        "Portfolio value (CAD)", min_value=1_000.0, value=100_000.0,
        step=5_000.0, format="%.0f",
        help="Used for position sizing and the order ticket. Nothing else depends on it.")

    st.divider()
    rerun = st.button("Recompute", width="stretch")
    st.caption("Results are cached per setting combination. "
               "A cold run with 120 resamples takes 1-2 minutes.")

settings = dict(bear_weight=bear_weight, drawdown_threshold=drawdown_threshold,
                n_resamples=n_resamples, risk_aversion=risk_aversion,
                view_confidence_scale=view_scale, rf=rf)

if rerun:
    get_saa.clear()

with st.spinner("Running the allocation model..."):
    result = get_saa(**settings)

DARK = False  # charts are built and colour-validated against the light surface

if view == "Allocation":
    allocation.render(result, portfolio_value, DARK)
elif view == "Evaluation":
    evaluation.render(result, DARK)
elif view == "Backtest":
    backtest.render(result, DARK)
elif view == "Forward tracker":
    forward.render(result, portfolio_value, DARK)
elif view == "Securities":
    security.render(result, DARK)
else:
    diagnostics.render(result, DARK)

st.divider()
st.caption(
    f"Model run {result.generated_at:%Y-%m-%d %H:%M} · "
    f"data {result.returns.index.min():%Y-%m-%d} to {result.returns.index.max():%Y-%m-%d} · "
    "Educational analysis of a personal portfolio. Not investment advice, and every "
    "expected-return figure is a modelling assumption rather than a forecast."
)
