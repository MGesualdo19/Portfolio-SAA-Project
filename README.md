# Strategic Asset Allocation

A strategic asset allocation model for a single Canadian personal portfolio,
with a local dashboard for allocation, backtesting, forward tracking and
per-security investigation.

Base currency CAD. Ten holdings across cash, floating-rate notes, US, Canadian,
international developed and emerging equity, gold, and one legacy REIT.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python run_dashboard.py           # http://localhost:8501
```

The first run fetches price history from Yahoo and caches it under `data/cache/`.
A cold allocation run with resampling takes one to two minutes; after that,
results are cached per settings combination.

To see the derivation and the reasoning behind every step:

```bash
jupyter lab notebooks/01_saa_derivation.ipynb
```

---

## The problem this model solves

Hand a mean-variance optimiser sample means and a sample covariance matrix and
it returns two or three holdings. That is not a bug — it is the optimiser
correctly maximising an objective built from inputs whose estimation error
swamps their signal. The standard error of a mean return estimated over 19 years
of a 19%-volatility asset is 4.4% a year, wider than the entire plausible spread
of expected returns across this book's equity sleeves.

Four mechanisms address it, and the contribution of each is measured rather than
assumed:

| Configuration | Effective holdings | Positions | Largest weight |
|---|---|---|---|
| Textbook: sample means + sample covariance | 2.47 | 3 | 46.9% |
| + Ledoit-Wolf shrinkage | 2.47 | 3 | 46.8% |
| + equilibrium / Black-Litterman returns | 6.15 | 9 | 31.2% |
| + policy constraints | 7.40 | 10 | 25.0% |
| + objective blending and Michaud resampling | **8.07** | **10** | **21.0%** |

*Effective holdings* is the inverse Herfindahl of the weights — the number of
equally weighted positions the allocation is equivalent to. The second row is as
informative as the last: shrinkage alone changes nothing, because the problem
was never mainly the covariance matrix. Replacing sample means with equilibrium
returns does most of the work, which is why the policy constraints are a
backstop rather than the mechanism.

---

## What it does

**Data foundation.** Total returns including all distributions, never price
returns. Every portfolio-level figure converted to CAD; per-security figures kept
in native currency. Yahoo's USD/CAD series is stamped one business day ahead of
the session it describes, and correcting that moves XUU.TO's correlation with
CAD-converted VTI from 0.80 to 0.91 and its modelled/true volatility ratio from
1.155 to 0.991 — uncorrected, the optimiser underweights US equity for a reason
that is purely a data-vendor artefact.

**19 years of history, not 4.7.** The common window across all holdings starts in
November 2021 and contains no crash at all. Long-lived index proxies are spliced
onto the front of six holdings so the covariance matrix can see 2008 and 2020.
Every splice is audited for overlap correlation and volatility ratio, and the
synthetic share of each series is reported.

**Risk estimated on drawdowns, not averages.** The covariance matrix is a blend
of full-sample and bear-regime estimates, both Ledoit-Wolf shrunk. Bear is
defined off the book's own equity sleeve rather than the TSX, which is 3% of
world market capitalisation and 60% financials plus energy.

**Expected returns without forecasting.** Reverse optimisation turns a reference
policy allocation into the returns that would justify it; the thesis is then
applied as Black-Litterman views with explicit, tunable confidences. Set every
confidence to zero and the model returns the policy portfolio.

**Five objectives, blended.** Minimum variance, equal risk contribution, maximum
diversification, maximum Sharpe on the posterior, and minimum CVaR — each solved
under policy constraints and Michaud-resampled across 120 block bootstraps. The
spread across objectives is reported, and is the honest measure of how much
confidence each weight deserves.

**Validation.** A walk-forward test refits the entire process on an expanding
window and holds each result forward for a year. Out-of-sample Sharpe of 0.66
against 0.65 in-sample, with refit weights that barely move across 15 refits.

---

## Dashboard

| View | What it shows |
|---|---|
| **Allocation** | Target weights, sleeve and currency breakdown, capital vs risk contribution, disagreement across the five objectives, binding constraints, and a whole-share order ticket |
| **Backtest** | Growth of the book 2007–today net of fees and turnover, drawdown, eight named crisis windows, and walk-forward out-of-sample validation |
| **Forward tracker** | Live NAV from a funding date inside a bootstrapped expectation cone, realised percentile, drift versus target, and rebalancing signals |
| **Securities** | Any holding on its own terms: price vs total return, drawdown, rolling Sharpe/Sortino, VaR and CVaR by three methods, correlation to the rest of the book in calm vs bear regimes, factor exposure, data provenance |
| **Evaluation** | Adversarial tests: does the model beat a volatility-matched naive portfolio, which weights were set by a cap rather than the data, how diluted the thesis is, and where the risk really sits |
| **Diagnostics** | Regime correlation matrices, downside beta, tail dependence, the expected-return construction, proxy quality, return reconciliation, and a plain statement of the model's limits |

Model parameters that represent judgements rather than estimates — bear weight,
drawdown threshold, risk aversion, view confidence, resample count — are sidebar
sliders, so the effect of changing one is visible rather than buried.

**Every number explains itself.** Each metric carries a tooltip with its formula,
its inputs, the derivation steps and the caveat that stops it being over-read;
each chart and table has a "How this is calculated" expander with the same
content in full, rendered maths included, naming the module and function that
produced it. The registry behind this lives in `dashboard/methodology.py`, so the
hover text and the long-form explanation cannot drift apart — and a test asserts
that every entry has a formula, steps, a caveat and a source.

---

## Layout

```
core/
  data_loader.py     yfinance wrapper with local pickle cache
  fx.py              USD/CAD, including the date-stamp correction
  returns.py         total returns, CAD conversion, summary statistics
  proxies.py         index-proxy backfill and its quality audit
  estimation.py      shrinkage, regime blending, equilibrium returns, Black-Litterman
  optimizer.py       constraints, five objectives, Michaud resampling
  policy.py          universe, reference allocation, policy bands, thesis views
  backtest_engine.py drifting/rebalancing backtest and walk-forward
  tracker.py         order tickets, forward tracking, drift reporting
  saa.py             the orchestrator — one entry point for everything
analysis/
  regime.py  correlation.py  stress.py  factor_regression.py
dashboard/
  app.py             sidebar, navigation, six views
  methodology.py     the formula/inputs/steps/caveat registry behind every tooltip
  theme.py  data.py  views/
notebooks/
  01_saa_derivation.ipynb    the derivation and the reasoning
docs/
  OBJECTIVES.md              scope, success criteria, verdicts, critical evaluation, gaps
```

---

## Results summary

Backtest 2007–2026, quarterly rebalanced, net of MERs and 8bps turnover cost:

| | Recommended | Reference policy | Equal weight | 60/40 | All equity |
|---|---|---|---|---|---|
| Annual return | 5.48% | 5.79% | 6.03% | 5.30% | 6.33% |
| Volatility | 9.21% | 12.06% | 12.15% | 9.99% | 18.45% |
| Sharpe | 0.30 | 0.25 | 0.27 | 0.26 | 0.19 |
| Max drawdown | -24.2% | -35.4% | -35.3% | -33.3% | -55.0% |
| GFC 2007–09 | -16.7% | -28.5% | -28.5% | -28.1% | -53.1% |
| COVID 2020 | -17.2% | -21.5% | -22.8% | -18.6% | -31.1% |
| 2022 rate shock | -9.8% | -12.6% | -13.5% | -11.2% | -15.8% |

Less return, materially less pain. Whether that is the right trade depends on a
horizon the model does not know — which is why the horizon gap is listed
explicitly in `docs/OBJECTIVES.md` rather than papered over.

---

## Where it is weak

Found by deliberately trying to break it; all of these run live in the
**Evaluation** tab, and `docs/OBJECTIVES.md` carries the full write-up.

**It passes the test that matters.** De-risking the naive alternatives with cash
to an identical 9.21% volatility, the recommendation still returns 5.48% against
4.79%/4.95%, at a shallower drawdown. The composition is doing real work, not
just the lower equity level.

**But the thesis is diluted roughly five to one.** Only one of the five
objectives uses expected returns, so switching every view on moves that objective
by up to 10 percentage points and the final blend by 2. Blending is what makes
the allocation robust and what makes it nearly thesis-blind. The policy floors
express the thesis more forcefully than the views do.

**"No binding constraints" is misleading.** The blend rests on no constraint, yet
24% of the individual (objective × holding) cells sit exactly on a bound. Gold's
15% ceiling binds in four of five objectives — uncapped, maximum diversification
wants 27.8%, so the 14% is a policy ceiling rather than a data result.

**The headline diversification flatters.** The equity sleeve is 47% of capital
and 84% of the risk; the cash sleeve is 34.5% of capital and 0.6% of risk. This
is an equity portfolio with a large cash buffer.

**Plus the standing gaps:** no tax modelling, no horizon modelling, account
placements in `core/universe.py` are still placeholders, expected returns are
assumptions rather than forecasts, nineteen years is one draw, and fund-holdings
overlap is unmeasured so the AI-capex claim is argued rather than quantified.

---

*Educational analysis of a personal portfolio. Not investment advice.*
