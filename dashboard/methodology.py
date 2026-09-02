"""
dashboard/methodology.py

Every number in the dashboard, and exactly how it was derived.

A dashboard that shows a weight of 14.0% without saying where 14.0% came from
is asking to be trusted rather than checked. This module is the antidote: a
single registry mapping each displayed quantity to its formula, its inputs
(naming the module that produces each one), the derivation steps, and the
caveat that stops it being over-read.

Two rendering paths:

  * `tip(key)` returns a compact string for Streamlit's `help=` parameter --
    the "?" that appears beside metrics and controls.
  * `note(key)` renders a full expander beneath a chart: formula, inputs,
    steps, and caveat.

Keeping both in one registry means the hover text and the long-form
explanation cannot drift apart, and a formula is written once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st


@dataclass
class Method:
    title: str
    summary: str                       # one sentence: what the number is
    formula: str = ""                  # LaTeX, rendered with st.latex
    formula_plain: str = ""            # ASCII fallback shown in the tooltip
    inputs: list[tuple[str, str]] = field(default_factory=list)   # (symbol, where it comes from)
    steps: list[str] = field(default_factory=list)
    caveat: str = ""
    source: str = ""                   # the module/function that computes it

    def tooltip(self) -> str:
        parts = [self.summary]
        if self.formula_plain:
            parts.append(f"\n\n`{self.formula_plain}`")
        if self.steps:
            parts.append("\n\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(self.steps, 1)))
        if self.caveat:
            parts.append(f"\n\n⚠️ {self.caveat}")
        if self.source:
            parts.append(f"\n\nComputed in `{self.source}`")
        return "".join(parts)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

METHODS: dict[str, Method] = {}


def _add(key: str, **kw) -> None:
    METHODS[key] = Method(**kw)


# --- foundational ----------------------------------------------------------

_add("total_return",
     title="Total return",
     summary="Daily return including every distribution, not just price movement.",
     formula=r"r_t = \frac{P_t + D_t}{P_{t-1}} - 1",
     formula_plain="r_t = (P_t + D_t) / P_{t-1} - 1",
     inputs=[("P_t", "Yahoo `Close`, already split-adjusted (core/data_loader.py)"),
             ("D_t", "Yahoo `Dividends` + `Capital Gains` paid on day t")],
     steps=["Take the split-adjusted close and the distributions paid that day.",
            "Add the distribution to the closing price before differencing — this is "
            "equivalent to reinvesting it at the close.",
            "Cross-check the whole series against Yahoo's own `Adj Close`; the two are "
            "built from different inputs, so a large gap means one is corrupt."],
     caveat="Price return alone would be a serious error here: CASH.TO and XFR.TO deliver "
            "essentially all of their return as distributions and almost none as price "
            "appreciation, so a price-return model refuses to hold them at all.",
     source="core/returns.py::total_return_series")

_add("cad_conversion",
     title="CAD conversion",
     summary="USD holdings converted to CAD before any portfolio-level statistic.",
     formula=r"r^{CAD}_t = \frac{P_t X_t + D_t X_t}{P_{t-1} X_{t-1}} - 1",
     formula_plain="r_CAD = (P_t·X_t + D_t·X_t) / (P_{t-1}·X_{t-1}) - 1",
     inputs=[("X_t", "USD/CAD (CAD per 1 USD) from Yahoo `CAD=X`, date-corrected"),
             ("P, D", "the security's own price and distributions in USD")],
     steps=["Shift Yahoo's FX series back one business day. Yahoo stamps the CAD=X bar "
            "one day AHEAD of the session it describes — the feed carries a bar dated "
            "tomorrow while every equity feed ends today.",
            "Convert the price and the distribution at the same day's rate.",
            "Difference the converted series."],
     caveat="Without the date shift the modelled volatility of the US sleeve inflates by "
            "~15% (measured: XUU.TO vs CAD-converted VTI, correlation 0.80 → 0.91, "
            "vol ratio 1.155 → 0.991). The optimiser would then underweight US equity "
            "for a reason that is purely a data-vendor artefact.",
     source="core/fx.py")

_add("proxy_splice",
     title="Proxy-extended history",
     summary="Pre-inception history filled with a long-lived index proxy so the model can see 2008 and 2020.",
     formula_plain="r_t = own_return_t  if available,  else proxy_return_t",
     inputs=[("own", "the holding's real total return, always preferred where it exists"),
             ("proxy", "VTI, VBR, VEA, VWO, GLD, FLOT, BIL — see the Data provenance tab")],
     steps=["Take the holding's own returns wherever it has them.",
            "Fill only the gap before its inception, nearest proxy first.",
            "FX-convert the proxy only where the real holding carries that currency "
            "exposure — a CAD cash sleeve is proxied in the proxy's own currency.",
            "Audit each splice on the overlap period where both series exist."],
     caveat="The unspliced common window across all ten holdings starts in November 2021 "
            "and contains no crash at all. Splicing is an estimate; the alternative was a "
            "covariance matrix with no drawdown in it. Six of ten series are partly "
            "synthetic — the Data provenance tab shows exactly how much of each.",
     source="core/proxies.py")

# --- risk estimation -------------------------------------------------------

_add("covariance",
     title="Covariance matrix",
     summary="Ledoit-Wolf shrunk covariance, blended between full-sample and bear-regime estimates.",
     formula=r"\Sigma = (1-w)\,\Sigma_{\text{full}} + w\,\Sigma_{\text{bear}}",
     formula_plain="Σ = (1-w)·Σ_full + w·Σ_bear,   w = bear weight (default 0.35)",
     inputs=[("Σ_full", "Ledoit-Wolf shrunk covariance of all daily CAD total returns × 252"),
             ("Σ_bear", "the same estimator applied only to bear-regime days"),
             ("w", "the bear-weight slider — a risk preference, not an estimate")],
     steps=["Compute daily CAD total returns over the spliced 19-year window.",
            "Apply Ledoit-Wolf shrinkage toward a scaled-identity target, separately to "
            "the full sample and to the bear subsample.",
            "Annualise by multiplying by 252.",
            "Blend the two matrices with weight w."],
     caveat="The bear weight is a judgement about how much the allocation should be priced "
            "off crash behaviour rather than average behaviour. At w=0 this is the ordinary "
            "shrunk matrix; at w=1 the book is optimised as if permanently in a drawdown, "
            "which over-hedges. Shrinkage is applied before blending because the bear "
            "subsample is smaller and needs regularisation more, not less.",
     source="core/estimation.py::regime_blended_covariance")

_add("shrinkage",
     title="Ledoit-Wolf shrinkage intensity",
     summary="How far the sample covariance was pulled toward a structured target.",
     formula=r"\Sigma_{\text{shrunk}} = (1-\lambda)\,S + \lambda\,F",
     formula_plain="Σ = (1-λ)·S + λ·F,   S = sample covariance, F = scaled-identity target",
     inputs=[("λ", "chosen analytically by the Ledoit-Wolf estimator, not tuned by hand"),
             ("S", "the raw sample covariance matrix"),
             ("F", "a scaled identity matrix — the structured target")],
     steps=["Estimate the optimal λ that minimises expected squared error against the "
            "true covariance.",
            "Blend the sample matrix toward the target by that amount."],
     caveat="A high λ means the raw sample matrix was poorly conditioned and should not "
            "have been trusted. Shrinkage mainly corrects the smallest eigenvalues, which "
            "are biased downward and are exactly the directions a minimum-variance "
            "optimiser loads into.",
     source="core/estimation.py::shrunk_covariance")

_add("bear_regime",
     title="Bear regime definition",
     summary="Days when the book's own equity sleeve sits more than 10% below its trailing high.",
     formula=r"\text{bear}_t = \left[\frac{I_t}{\max_{s \le t} I_s} - 1 \le -\theta\right]",
     formula_plain="bear_t = (I_t / running_max(I) - 1) <= -threshold",
     inputs=[("I_t", "equal-weighted total-return index of the six equity sleeves, in CAD"),
             ("θ", "the drawdown-threshold slider, default 10%")],
     steps=["Build an equal-weighted index of the book's own equity holdings.",
            "Compute its drawdown from the running maximum.",
            "Flag every day below the threshold."],
     caveat="Deliberately NOT the TSX. ^GSPTSE is ~3% of world market capitalisation and "
            "roughly 60% financials plus energy; it has had drawdowns this globally "
            "diversified book barely felt and missed some it felt sharply. 'Bear' should "
            "mean the equity risk this investor actually holds is in a drawdown.",
     source="core/saa.py::_equity_sleeve_index + analysis/stress.py::equity_drawdown_mask")

# --- expected returns ------------------------------------------------------

_add("equilibrium_returns",
     title="Equilibrium (reverse-optimised) returns",
     summary="The expected returns that would make the reference policy portfolio optimal.",
     formula=r"\pi = \delta\,\Sigma\,w_{\text{ref}} + r_f",
     formula_plain="π = δ · Σ · w_ref + r_f",
     inputs=[("δ", "risk aversion, the market price of risk ≈ (E[Rm]−rf)/σ²m; default 2.8"),
             ("Σ", "the regime-blended covariance matrix above"),
             ("w_ref", "the reference policy allocation in core/policy.py"),
             ("r_f", "the risk-free rate input, default 2.75%")],
     steps=["Take a defensible reference allocation as the neutral starting point.",
            "Multiply the covariance matrix by those weights and by risk aversion.",
            "Add the risk-free rate to get total rather than excess expected returns."],
     caveat="This is the single most important design choice in the model. Sample means "
            "carry a standard error of σ/√T — about 4.4%/yr for a 19%-vol asset over 19 "
            "years, wider than the entire plausible spread of expected returns across "
            "these holdings. Because π is generated by the same Σ the optimiser uses, "
            "feeding it back reproduces w_ref exactly and CANNOT produce a corner "
            "solution. Every deviation must then be argued for by an explicit view.",
     source="core/estimation.py::implied_equilibrium_returns")

_add("black_litterman",
     title="Black-Litterman posterior",
     summary="Equilibrium returns tilted by the thesis views, each weighted by its stated confidence.",
     formula=r"\mu = \left[(\tau\Sigma)^{-1} + P^{\top}\Omega^{-1}P\right]^{-1}"
             r"\left[(\tau\Sigma)^{-1}\pi + P^{\top}\Omega^{-1}Q\right]",
     formula_plain="μ = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹π + PᵀΩ⁻¹Q]",
     inputs=[("π", "the equilibrium returns above — the prior"),
             ("P", "view matrix: one row per view, coefficients over holdings"),
             ("Q", "the magnitude claimed by each view, in %/yr"),
             ("Ω", "view uncertainty, set from each view's confidence (Idzorek method)"),
             ("τ", "scales prior uncertainty, 0.05")],
     steps=["Express each thesis claim as a relative view, e.g. 0.5·VTV + 0.5·AVUV − "
            "1.0·XUU.TO = +1.25%/yr.",
            "Convert each stated confidence c into a view variance: Ω_ii = "
            "(P_i τΣ P_iᵀ)·(1−c)/c, so c→1 takes the view as certain and c→0 ignores it.",
            "Combine prior and views by precision weighting."],
     caveat="Confidence is NOT a probability. A 0.35 confidence means the posterior moves "
            "35% of the way from the policy portfolio's implied returns toward the view. "
            "Set every confidence to zero and the model returns the policy portfolio — "
            "that is the honest baseline.",
     source="core/estimation.py::black_litterman")

_add("expected_return_portfolio",
     title="Portfolio expected return",
     summary="Weighted average of the Black-Litterman posterior returns.",
     formula=r"E[R_p] = w^{\top}\mu",
     formula_plain="E[Rp] = Σ_i w_i · μ_i",
     inputs=[("w", "the recommended weights"), ("μ", "the Black-Litterman posterior")],
     steps=["Multiply each holding's weight by its posterior expected return and sum."],
     caveat="This is a modelling assumption, not a forecast. It inherits everything from "
            "the reference portfolio and the view confidences, and would change if either "
            "did. Do not treat it as a prediction of next year's return.",
     source="core/optimizer.py::summarise_allocation")

# --- optimisation ----------------------------------------------------------

_add("portfolio_vol",
     title="Portfolio volatility",
     summary="Annualised standard deviation implied by the weights and the covariance matrix.",
     formula=r"\sigma_p = \sqrt{w^{\top}\Sigma w}",
     formula_plain="σp = sqrt(wᵀ Σ w)",
     inputs=[("w", "the recommended weights"), ("Σ", "the regime-blended covariance matrix")],
     steps=["Form the quadratic form wᵀΣw — this is the portfolio variance.",
            "Take the square root. Σ is already annualised, so σp is annual."],
     caveat="Forward-looking, computed from Σ rather than from a realised return series. "
            "The Backtest page's volatility figure is the realised one and will differ. "
            "Because Σ is regime-blended, this sits above a pure full-sample estimate — "
            "deliberately.",
     source="core/optimizer.py::portfolio_vol")

_add("sharpe",
     title="Sharpe ratio",
     summary="Excess return per unit of total volatility.",
     formula=r"S = \frac{E[R_p] - r_f}{\sigma_p}",
     formula_plain="S = (E[Rp] − rf) / σp",
     inputs=[("E[Rp]", "expected or realised annual return depending on the page"),
             ("r_f", "the risk-free input, default 2.75%"), ("σp", "annualised volatility")],
     steps=["Subtract the risk-free rate from the annual return.",
            "Divide by annualised volatility."],
     caveat="Penalises upside and downside volatility identically, which misprices assets "
            "with asymmetric profiles — gold especially. Read it alongside Sortino, which "
            "only penalises downside.",
     source="core/returns.py::summary_stats")

_add("risk_contribution",
     title="Risk contribution",
     summary="Each holding's share of total portfolio volatility, via Euler decomposition.",
     formula=r"RC_i = \frac{w_i (\Sigma w)_i}{\sqrt{w^{\top}\Sigma w}}, \quad \sum_i RC_i = \sigma_p",
     formula_plain="RC_i = w_i · (Σw)_i / σp    (these sum exactly to σp)",
     inputs=[("w", "the recommended weights"), ("Σ", "the regime-blended covariance matrix")],
     steps=["Compute the marginal contribution to risk, (Σw)_i, for each holding.",
            "Multiply by that holding's weight and divide by portfolio volatility.",
            "The results sum exactly to σp, which is what makes this a genuine "
            "decomposition rather than an approximation."],
     caveat="This is where a capital-weight pie chart misleads most. In this book the cash "
            "sleeve holds 34.5% of the capital and contributes 0.6% of the risk, while the "
            "equity sleeve holds 47% of capital and carries 84% of the risk.",
     source="core/optimizer.py::risk_contributions")

_add("effective_n",
     title="Effective number of holdings",
     summary="Inverse Herfindahl of the weights — the number of equally weighted positions this is equivalent to.",
     formula=r"N_{\text{eff}} = \frac{1}{\sum_i w_i^2}",
     formula_plain="N_eff = 1 / Σ w_i²",
     inputs=[("w", "the recommended weights")],
     steps=["Square each weight and sum.", "Take the reciprocal."],
     caveat="The direct test of whether an optimiser produced a real portfolio. Ten equal "
            "weights give 10.0; a book with ten non-zero weights but N_eff near 2 is a "
            "two-ETF portfolio in disguise. For reference, the textbook recipe (sample "
            "means + sample covariance) on this same universe gives 2.47.",
     source="core/optimizer.py::effective_n")

_add("effective_n_risk",
     title="Effective risk sources",
     summary="The same inverse-Herfindahl measure applied to risk contributions rather than capital.",
     formula=r"N^{\text{risk}}_{\text{eff}} = \frac{1}{\sum_i (RC_i / \sigma_p)^2}",
     formula_plain="N_eff_risk = 1 / Σ (RC_i / σp)²",
     inputs=[("RC", "risk contributions from the Euler decomposition above")],
     steps=["Normalise risk contributions to shares of total volatility.",
            "Square, sum, take the reciprocal."],
     caveat="The stricter and more honest of the two measures. A portfolio of 60% equity "
            "and 40% cash has a respectable capital-based effective N while carrying "
            "essentially all of its risk in one place.",
     source="core/optimizer.py::effective_n_risk")

_add("diversification_ratio",
     title="Diversification ratio",
     summary="Weighted average standalone volatility divided by realised portfolio volatility.",
     formula=r"DR = \frac{\sum_i w_i \sigma_i}{\sqrt{w^{\top}\Sigma w}}",
     formula_plain="DR = (Σ w_i·σ_i) / σp",
     inputs=[("σ_i", "each holding's standalone annualised volatility"),
             ("σp", "portfolio volatility from the full covariance matrix")],
     steps=["Compute the weighted average of individual volatilities — what the portfolio "
            "would be if every holding were perfectly correlated.",
            "Divide by actual portfolio volatility."],
     caveat="1.0 means no diversification benefit at all. Higher is better, but it is "
            "measured against the regime-blended covariance matrix, so it is a more "
            "conservative figure than a full-sample calculation would give.",
     source="core/optimizer.py::diversification_ratio")

_add("objectives",
     title="The five objectives",
     summary="Five different definitions of 'optimal', solved separately and averaged.",
     formula_plain="w_final = mean(w_minvar, w_erc, w_maxdiv, w_maxsharpe, w_mincvar)",
     inputs=[("Minimum variance", "minimise wᵀΣw — uses no return estimates"),
             ("Equal risk contribution", "make every RC_i equal — uses no return estimates"),
             ("Maximum diversification", "maximise DR — uses no return estimates"),
             ("Max Sharpe", "maximise (wᵀμ − rf)/σp — the ONLY one using expected returns"),
             ("Minimum CVaR", "minimise average loss in the worst 5% of weeks")],
     steps=["Solve each objective independently under the same policy constraints.",
            "Repeat each solve on 120 block-bootstrap resamples (Michaud resampling) and "
            "average the weights.",
            "Average across the five objectives, then round to a 0.5% grid and drop "
            "positions under 1% as untradeable dust."],
     caveat="Because four of the five ignore expected returns entirely, the thesis views "
            "influence roughly one fifth of the final answer. Views move the max-Sharpe "
            "solution by up to 10 percentage points but the blend by only about 2. That "
            "is what makes the allocation robust, and it also means it is closer to a "
            "risk-driven portfolio than a thesis-driven one.",
     source="core/optimizer.py::build_all_strategies + blend_strategies")

_add("resampling",
     title="Michaud resampling",
     summary="Each objective re-solved on 120 bootstrapped histories and the weights averaged.",
     formula_plain="w = (1/B) · Σ_b argmin f(w; Σ_b),   B = 120 resamples",
     inputs=[("Σ_b", "covariance re-estimated on each block-bootstrap resample"),
             ("block size", "21 trading days, so volatility clustering survives resampling")],
     steps=["Draw whole 21-day blocks of history with replacement until the sample length "
            "is reproduced.",
            "Re-estimate covariance on that resample and re-solve the objective.",
            "Average the resulting weights; the spread across resamples measures how much "
            "of a weight is signal rather than an artefact of this one sample path."],
     caveat="Blocks rather than independent daily draws, because IID resampling destroys "
            "volatility clustering and makes every resample look calmer than any real "
            "market.",
     source="core/optimizer.py::resampled_weights")

_add("constraints",
     title="Policy constraints",
     summary="Per-holding caps and floors plus asset-class group bands.",
     formula_plain="lo_i <= w_i <= hi_i,   and   lo_g <= Σ_{i∈g} w_i <= hi_g,   Σ w = 1",
     inputs=[("bounds", "per-holding, in core/policy.py"),
             ("groups", "eight asset-class bands with stated rationales")],
     steps=["Solve every objective subject to these bounds.",
            "Check afterwards which limits the solution rests against."],
     caveat="These encode investment policy, not statistics — the judgement that the book "
            "holds some fixed income, some non-North-American equity and some real assets "
            "regardless of what one 19-year sample says. Important nuance: the BLEND shows "
            "no binding constraints, but 24% of the individual (objective, holding) cells "
            "sit exactly on a bound. Averaging constrained corner solutions produces a "
            "smooth-looking result that hides the pinning underneath.",
     source="core/optimizer.py::ConstraintSet")

# --- backtest --------------------------------------------------------------

_add("backtest",
     title="Backtest mechanics",
     summary="Weights drift with returns between rebalances; turnover and fees are charged.",
     formula=r"w_{i,t+1} = \frac{w_{i,t}(1+r_{i,t})}{\sum_j w_{j,t}(1+r_{j,t})}",
     formula_plain="w drifts with returns; on a rebalance date, trade back to target and charge cost",
     inputs=[("r", "daily CAD total returns"), ("MER", "each fund's expense ratio, charged daily"),
             ("cost", "basis points on one-way turnover at each rebalance")],
     steps=["Each day, grow every holding at its own net-of-fee return and renormalise "
            "the weights — this is the drift a real account experiences.",
            "On a rebalance date, compute one-way turnover as ½·Σ|w − w_target|, charge "
            "the cost, and reset to target.",
            "Compound the resulting portfolio return into the NAV."],
     caveat="`returns @ weights` is NOT a backtest — that dot product implicitly rebalances "
            "to target every single day for free, a fictitious sell-high/buy-low overlay at "
            "zero cost. It is why naive backtests of diversified portfolios look better "
            "than any investor achieves.",
     source="core/backtest_engine.py::run_backtest")

_add("max_drawdown",
     title="Maximum drawdown",
     summary="The worst peak-to-trough decline over the period.",
     formula=r"MDD = \min_t \left(\frac{V_t}{\max_{s \le t} V_s} - 1\right)",
     formula_plain="MDD = min(V_t / running_max(V) − 1)",
     inputs=[("V", "the backtest NAV series, net of fees and costs")],
     steps=["Track the running maximum of the NAV.",
            "Express each day as a percentage below that running maximum.",
            "Take the minimum."],
     caveat="Path-dependent and driven by a single historical episode, so it is a weaker "
            "statistic than its prominence suggests. It also says nothing about how long "
            "recovery took, which is often what actually matters to an investor.",
     source="core/backtest_engine.py::BacktestResult.drawdown")

_add("sortino",
     title="Sortino ratio",
     summary="Excess return per unit of DOWNSIDE volatility only.",
     formula=r"\text{Sortino} = \frac{E[R_p] - r_f}{\sigma_{\text{down}}}, \quad "
             r"\sigma_{\text{down}} = \sqrt{252}\cdot\text{std}(r_t \mid r_t < 0)",
     formula_plain="Sortino = (R − rf) / σ_downside,  where σ_downside uses only negative days",
     inputs=[("σ_down", "annualised standard deviation of negative daily returns only")],
     steps=["Filter to negative return days.", "Take their standard deviation and "
            "annualise by √252.", "Divide excess return by that."],
     caveat="More appropriate than Sharpe for anything with an asymmetric return profile — "
            "gold and the legacy volatility position especially — since Sharpe treats an "
            "upside surprise as identically bad to a downside one.",
     source="core/returns.py::summary_stats")

_add("calmar",
     title="Calmar ratio",
     summary="Annualised return divided by the absolute maximum drawdown.",
     formula=r"\text{Calmar} = \frac{R_{\text{ann}}}{|MDD|}",
     formula_plain="Calmar = annual return / |max drawdown|",
     inputs=[("R_ann", "annualised return"), ("MDD", "maximum drawdown")],
     steps=["Divide the annualised return by the absolute maximum drawdown."],
     caveat="Inherits max drawdown's dependence on a single historical episode. Useful as "
            "a return-per-unit-of-pain measure, not as a precise ranking.",
     source="core/backtest_engine.py::BacktestResult.stats")

_add("turnover",
     title="Turnover and cost drag",
     summary="One-way turnover traded at each rebalance, and its cumulative cost.",
     formula=r"\text{turnover} = \tfrac{1}{2}\sum_i |w_i - w_i^{\text{target}}|",
     formula_plain="turnover = ½ · Σ|w_drifted − w_target|;  cost = turnover × bps/10000",
     inputs=[("w drifted", "weights just before the rebalance"),
             ("bps", "the trading-cost slider, default 8 bps")],
     steps=["At each rebalance date, sum the absolute weight differences and halve it — "
            "halving converts a two-way sum into the one-way amount actually traded.",
            "Multiply by the cost in basis points and deduct from that day's return."],
     caveat="8 bps is a reasonable retail estimate for liquid ETFs including spread, but "
            "it ignores the cost of Norbert's Gambit on the USD holdings, which is a real "
            "and lumpy expense this model does not capture.",
     source="core/backtest_engine.py::run_backtest")

_add("crisis_windows",
     title="Crisis-window returns",
     summary="Cumulative return over specific dated episodes, held without rebalancing inside the window.",
     formula=r"R_{\text{window}} = \prod_{t \in \text{window}} (1 + r_{p,t}) - 1",
     formula_plain="R = Π(1 + r_t) − 1 over the window, no rebalancing inside it",
     inputs=[("windows", "eight hand-picked episodes, each stressing a different joint behaviour"),
             ("r_p", "portfolio return from the buy-and-hold backtest over that window")],
     steps=["Slice the return matrix to the window's dates.",
            "Run the allocation buy-and-hold through it — nobody rebalances mid-crash on "
            "a schedule.",
            "Compound to a total return."],
     caveat="Window boundaries are chosen with hindsight, which flatters any allocation "
            "that happened to suit those specific dates. Read the 2009 recovery row "
            "alongside the crash rows: protection that never gives anything back in the "
            "rebound is just a permanently smaller portfolio.",
     source="analysis/stress.py::portfolio_window_performance")

_add("walk_forward",
     title="Walk-forward validation",
     summary="The entire model refitted on an expanding window and held forward for a year at a time.",
     formula_plain="for each year: fit on data < T, hold weights over [T, T+1yr], record returns",
     inputs=[("training window", "expanding, minimum 5 years"),
             ("holding period", "12 months, then refit")],
     steps=["Fit covariance, equilibrium returns, views and all five objectives on data "
            "strictly before the test date.",
            "Hold the resulting weights for the next twelve months, charging fees and "
            "turnover.",
            "Refit and repeat, then stitch the out-of-sample segments together."],
     caveat="The only number in the project that says anything about whether the METHOD "
            "works, as opposed to whether one set of weights suited the sample it was "
            "estimated on. Everything else on the Backtest page is in-sample and "
            "circular by construction.",
     source="core/backtest_engine.py::walk_forward")

# --- forward tracking ------------------------------------------------------

_add("expectation_cone",
     title="Expectation cone",
     summary="Percentile bands of NAV paths this allocation could plausibly follow.",
     formula_plain="draw 21-day blocks of JOINT return rows; V_t = V_0·Π(1 + w·r); take percentiles",
     inputs=[("history", "the 19-year CAD return matrix"),
             ("w", "the tracked portfolio's weights"),
             ("blocks", "21 trading days, 4,000 simulated paths")],
     steps=["Draw whole rows of the return matrix in 21-day blocks — never each column "
            "independently, which would manufacture diversification that has never "
            "existed and destroy every joint crash day.",
            "Apply the weights to build a portfolio return path.",
            "Compound to a NAV path, repeat 4,000 times, take percentiles at each horizon."],
     caveat="Resamples history, so it can only produce crises resembling ones that have "
            "already happened. The width of the cone is the honest output: over one year "
            "the 5th-to-95th range is wide enough that almost any outcome is 'normal'.",
     source="core/tracker.py::expectation_cone")

_add("percentile_of_actual",
     title="Realised percentile",
     summary="Where the actual return sits within the simulated distribution for the same elapsed horizon.",
     formula=r"p = 100 \cdot \frac{1}{B}\sum_b \mathbb{1}\!\left[R_b < R_{\text{actual}}\right]",
     formula_plain="p = % of simulated paths that returned less than the actual",
     inputs=[("R_actual", "realised cumulative return since the funding date"),
             ("R_b", "4,000 bootstrapped totals over the same number of trading days")],
     steps=["Simulate the same elapsed horizon 4,000 times from history.",
            "Count what fraction of those paths did worse than the actual result."],
     caveat="50 is exactly as expected. Below 10 or above 90 is genuinely unusual. Over "
            "short horizons the distribution is so wide that this number carries almost no "
            "information — it only starts to mean something after several years.",
     source="core/tracker.py::percentile_of_actual")

_add("drift",
     title="Drift versus target",
     summary="How far each holding has moved from its target weight through performance alone.",
     formula=r"\text{relative drift}_i = \frac{w^{\text{current}}_i - w^{\text{target}}_i}"
             r"{w^{\text{target}}_i}",
     formula_plain="absolute drift = w_current − w_target;  relative = absolute / w_target",
     inputs=[("w_current", "target weight grown by each holding's realised return since funding"),
             ("w_target", "the tracked allocation")],
     steps=["Grow each holding's initial value by its realised total return in CAD.",
            "Renormalise to get current weights.",
            "Compare against target in both absolute and relative terms."],
     caveat="The 20% RELATIVE band is the trigger, not the absolute one — a 2% target "
            "drifting to 2.5% is a 25% relative move worth correcting, while a 20% target "
            "drifting to 20.5% is noise. Rebalancing frequency matters: on this book, "
            "annual and drift-band rebalancing both beat quarterly historically.",
     source="core/tracker.py::drift_report")

_add("order_ticket",
     title="Order ticket",
     summary="Target weights converted to whole shares at the latest close.",
     formula_plain="shares_i = floor(w_i × book_value_CAD / price_CAD_i)",
     inputs=[("price_CAD", "latest close, converted at the current USD/CAD for USD listings"),
             ("book value", "the portfolio-value input in the sidebar")],
     steps=["Convert each holding's latest close into CAD.",
            "Divide the CAD budget for that holding by the CAD price.",
            "Round DOWN to whole shares, so the ticket is never over-funded.",
            "Report the achieved weight after rounding and the resulting drift."],
     caveat="Whole-share rounding is not cosmetic on a small book: a 4.5% target in a $124 "
            "ETF is a handful of shares, and the difference between 28 and 29 is a third of "
            "a percentage point of the portfolio. USD positions also require Norbert's "
            "Gambit to fund, whose cost is not modelled here.",
     source="core/tracker.py::build_order_ticket")

# --- security-level --------------------------------------------------------

_add("var_cvar",
     title="VaR and CVaR, three ways",
     summary="Value-at-risk and expected shortfall computed by three methods that are never collapsed into one.",
     formula=r"\text{VaR}_\alpha = -Q_\alpha(r), \quad "
             r"\text{CVaR}_\alpha = -E[\,r \mid r \le Q_\alpha(r)\,]",
     formula_plain="VaR = −quantile(r, 1−c);  CVaR = −mean of returns below that quantile",
     inputs=[("Parametric", "assumes normality: VaR = −(μ + z·σ)"),
             ("Historical", "the empirical quantile of the actual return sample"),
             ("Bootstrap", "5,000 resamples of the sample, giving a confidence interval")],
     steps=["Parametric: fit mean and standard deviation, read off the normal quantile.",
            "Historical: take the empirical quantile and average everything below it.",
            "Bootstrap: resample the return series with replacement and repeat the "
            "historical calculation, reporting the mean and a 90% interval."],
     caveat="Each has a genuine blind spot. Parametric understates fat tails; historical "
            "cannot produce a worse day than the worst already observed; bootstrap "
            "quantifies estimation uncertainty but still only resamples history. "
            "Agreement is mild reassurance; DISAGREEMENT is the finding — it means the "
            "normal-distribution assumption is doing real work.",
     source="core/risk.py::var_cvar_summary")

_add("rolling_sharpe",
     title="Rolling Sharpe and Sortino",
     summary="Risk-adjusted return computed over a moving 252-day window.",
     formula=r"S_t = \frac{\text{mean}(r_{t-251..t}) - r_f}{\text{std}(r_{t-251..t})}\sqrt{252}",
     formula_plain="rolling 252-day mean excess return / rolling std, annualised by √252",
     inputs=[("window", "252 trading days ≈ one year"),
             ("r_f", "the risk-free rate, converted to a daily figure")],
     steps=["Compute the rolling mean and standard deviation of excess returns.",
            "Divide and annualise by √252.",
            "Sortino replaces the denominator with downside deviation only."],
     caveat="A one-year window is short: these lines are noisy and should be read for "
            "regime shifts, not levels. Where Sortino sits far above Sharpe, the "
            "volatility Sharpe is punishing is mostly upside.",
     source="core/backtester.py::rolling_sharpe / rolling_sortino")

_add("bootstrap_forward",
     title="Simulated forward return",
     summary="Distribution of one-year outcomes from block-resampling this security's own history.",
     formula_plain="draw 21-day blocks until 252 days; total = Π(1+r) − 1; repeat 3,000×",
     inputs=[("history", "the security's own daily total returns"),
             ("block size", "21 days, preserving volatility clustering")],
     steps=["Draw 21-day blocks of consecutive historical returns with replacement.",
            "Concatenate to a 252-day path and compound.",
            "Repeat 3,000 times and take percentiles."],
     caveat="Only resamples what already happened, so it cannot generate a crisis unlike "
            "any in the sample. For a holding with a short history this is a distribution "
            "over very few distinct market conditions.",
     source="core/backtester.py::bootstrap_return_distribution")

_add("conditional_correlation",
     title="Calm vs bear correlation",
     summary="Correlation recomputed separately on calm days and drawdown days.",
     formula=r"\rho_{\text{bear}} = \text{corr}(r_i, r_j \mid \text{bear}_t)",
     formula_plain="corr computed on bear-regime rows only, vs on calm rows only",
     inputs=[("bear mask", "the equity-sleeve drawdown definition above"),
             ("returns", "daily CAD total returns")],
     steps=["Split the return matrix by regime.", "Compute a correlation matrix on each "
            "subset.", "Difference them."],
     caveat="This is the question the whole thesis rests on. Two assets can correlate 0.1 "
            "unconditionally and 0.85 in every drawdown that ever mattered — the "
            "unconditional number is arithmetically correct and practically useless. Bear "
            "subsamples are smaller, so these estimates are noisier; read direction, not "
            "precision.",
     source="analysis/stress.py + core/estimation.py")

_add("up_down_beta",
     title="Upside and downside beta",
     summary="Beta to core US equity, estimated separately on its up days and its down days.",
     formula=r"\beta^{\pm} = \frac{\text{cov}(r_i, r_m \mid r_m \gtrless 0)}"
             r"{\text{var}(r_m \mid r_m \gtrless 0)}",
     formula_plain="β_up = slope on days XUU.TO rose;  β_down = slope on days it fell;  "
                   "asymmetry = β_down − β_up",
     inputs=[("r_m", "XUU.TO daily CAD returns — the core US equity sleeve"),
             ("r_i", "the holding's daily CAD returns")],
     steps=["Split days by the sign of the market return.",
            "Fit a separate univariate regression slope on each subset.",
            "Subtract to get the asymmetry."],
     caveat="Positive asymmetry is the worst possible profile: the asset participates in "
            "losses more than in gains. It is completely invisible to correlation, which "
            "is symmetric by construction. A genuine tail hedge shows a clearly negative "
            "downside beta.",
     source="analysis/stress.py::up_down_beta")

_add("tail_dependence",
     title="Lower tail dependence",
     summary="How often this holding has its own worst-5% day at the same time as core US equity.",
     formula=r"\lambda_L = P\!\left(r_i \le Q_{0.05}(r_i) \mid r_m \le Q_{0.05}(r_m)\right)",
     formula_plain="P(asset in its worst 5% | XUU.TO in its worst 5%)",
     inputs=[("quantile", "5% of each series' own distribution")],
     steps=["Flag each series' worst 5% of days independently.",
            "Count the fraction of the market's worst days on which the holding was also "
            "having one of its own worst days."],
     caveat="Independence would give roughly 5%. A reading of 40% means four times in ten "
            "the 'diversifier' was crashing simultaneously — a joint-crash frequency no "
            "correlation coefficient would reveal.",
     source="analysis/stress.py::tail_dependence")

_add("factor_regression",
     title="Factor regression",
     summary="Return regressed on value and size-value spreads built from holdings already in the book.",
     formula=r"r_{i,t} = \alpha + \beta_1 F^{\text{value}}_t + \beta_2 F^{\text{size}}_t + \varepsilon_t",
     formula_plain="value factor = VTV − XUU.TO;  size-value factor = AVUV − XUU.TO",
     inputs=[("F_value", "VTV return minus XUU.TO return, a long-short spread"),
             ("F_size", "AVUV return minus XUU.TO return")],
     steps=["Build the two long-minus-short spreads from holdings already owned.",
            "Run OLS with an intercept over the aligned dates."],
     caveat="These are NOT academic Fama-French factors. They capture the value and size "
            "tilts only as expressed by these specific ETFs, and they are not excess of "
            "the risk-free rate, so the intercept is not a clean alpha. Read the "
            "coefficients directionally.",
     source="analysis/factor_regression.py")

_add("skew_kurtosis",
     title="Skew and excess kurtosis",
     summary="How far the return distribution departs from normal.",
     formula=r"\text{skew} = E\!\left[\left(\tfrac{r-\mu}{\sigma}\right)^3\right], \quad "
             r"\text{ex.kurt} = E\!\left[\left(\tfrac{r-\mu}{\sigma}\right)^4\right] - 3",
     formula_plain="skew = third standardised moment;  excess kurtosis = fourth − 3",
     inputs=[("r", "daily returns over the selected window")],
     steps=["Standardise returns by their mean and standard deviation.",
            "Take the third and fourth moments; subtract 3 from the fourth so a normal "
            "distribution reads zero."],
     caveat="A normal distribution has both at zero. High excess kurtosis means extreme "
            "days are far more common than the volatility figure implies, so any Gaussian "
            "risk number for that security — including its Sharpe ratio — understates what "
            "a bad day looks like.",
     source="core/returns.py::summary_stats")

_add("annualisation",
     title="Annualisation",
     summary="Daily figures scaled to annual, geometrically for returns and by √252 for volatility.",
     formula=r"R_{\text{ann}} = \left(\prod_t (1+r_t)\right)^{252/T} - 1, \quad "
             r"\sigma_{\text{ann}} = \sigma_{\text{daily}}\sqrt{252}",
     formula_plain="R_ann = (Π(1+r))^(252/T) − 1;   σ_ann = σ_daily × √252",
     inputs=[("T", "number of trading days in the sample"), ("252", "trading days per year")],
     steps=["Compound the full daily series, then take the 252/T root.",
            "Scale daily standard deviation by the square root of 252."],
     caveat="Geometric, not arithmetic. The common `mean × 252` convention overstates "
            "compounded outcomes by roughly σ²/2 — for a 20%-vol asset that is about "
            "2%/yr of pure artefact, enough to reorder the optimiser's ranking of assets "
            "on its own.",
     source="core/returns.py::annualise_return")

_add("currency_exposure",
     title="Currency exposure",
     summary="Weight grouped by the currency of the UNDERLYING assets, not the listing venue.",
     formula_plain="group weights by what the fund holds, not where it trades",
     inputs=[("mapping", "XUU.TO → USD, CGL.TO → USD (bullion), VIU.TO → other developed, etc.")],
     steps=["Map each holding to the currency of its underlying assets.",
            "Sum weights within each group."],
     caveat="Reporting by listing venue would show this book as roughly 85% Canadian, "
            "because XUU.TO and CGL.TO are TSX-listed. Both hold unhedged USD-denominated "
            "assets, so the real USD exposure is far higher — and it is the real exposure "
            "that determines what happens when the Canadian dollar moves.",
     source="core/saa.py::SAAResult.currency_exposure")

_add("blended_mer",
     title="Blended MER",
     summary="Weighted average management expense ratio of the recommended book.",
     formula=r"\text{MER}_p = \sum_i w_i \cdot \text{MER}_i",
     formula_plain="MER_p = Σ w_i × MER_i",
     inputs=[("MER_i", "each fund's published expense ratio, from core/universe.py")],
     steps=["Multiply each weight by that fund's MER and sum."],
     caveat="MERs are entered from provider fact sheets and do change — spot-check them. "
            "This excludes trading costs, bid-ask spread, and the cost of Norbert's Gambit "
            "on the USD holdings.",
     source="core/saa.py::SAAResult.blended_mer")

_add("proxy_quality",
     title="Splice quality metrics",
     summary="How closely each proxy tracked the real holding on the period where both existed.",
     formula=r"TE = \sqrt{252}\cdot\text{std}(r_{\text{own}} - r_{\text{proxy}}), \quad "
             r"\text{vol ratio} = \sigma_{\text{proxy}} / \sigma_{\text{own}}",
     formula_plain="overlap correlation; tracking error = √252 × std(difference); vol ratio",
     inputs=[("overlap", "dates where both the real holding and the proxy have returns")],
     steps=["Align the two series on their overlap.",
            "Compute correlation, annualised tracking error, and the volatility ratio."],
     caveat="Correlation above 0.95 is a strong splice. The two cash-like sleeves are "
            "judged on volatility ratio instead — a near-riskless holding's daily "
            "co-movement with a proxy is mostly microstructure noise, and what is being "
            "borrowed from BIL/FLOT is the short-rate path, not a correlation.",
     source="core/proxies.py::proxy_quality_report")

_add("reconciliation",
     title="Total-return reconciliation",
     summary="Cross-check of the reconstructed total return against Yahoo's own adjusted close.",
     formula_plain="compare Π(1 + (P+D)/P₋₁ − 1) against Π(1 + AdjClose.pct_change())",
     inputs=[("reconstructed", "close plus distributions, built in core/returns.py"),
             ("Adj Close", "Yahoo's own dividend-adjusted series")],
     steps=["Compute both series over their overlap.",
            "Compare cumulative returns and the worst single-day difference.",
            "Flag anything where the worst day differs by more than 2%."],
     caveat="The two are computed from genuinely different inputs, so agreement is real "
            "evidence the data is sound. A large gap means one of the two columns is "
            "corrupt for that ticker and the number should not be used until explained.",
     source="core/returns.py::reconciliation_report")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def tip(key: str, extra: str = "") -> str:
    """Tooltip string for a Streamlit `help=` parameter."""
    m = METHODS.get(key)
    if m is None:
        return extra
    text = m.tooltip()
    return f"{extra}\n\n{text}" if extra else text


def note(key: str, expanded: bool = False, label: str | None = None) -> None:
    """Full methodology expander: formula, inputs, steps, caveat, source."""
    m = METHODS.get(key)
    if m is None:
        return
    with st.expander(label or f"How this is calculated — {m.title}", expanded=expanded):
        st.markdown(f"**{m.summary}**")
        if m.formula:
            st.latex(m.formula)
        elif m.formula_plain:
            st.code(m.formula_plain, language=None)
        if m.inputs:
            st.markdown("**Inputs**")
            st.markdown("\n".join(f"- `{sym}` — {src}" for sym, src in m.inputs))
        if m.steps:
            st.markdown("**Steps**")
            st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(m.steps, 1)))
        if m.caveat:
            st.markdown(f'<div class="warn"><b>What this does not tell you.</b> {m.caveat}</div>',
                        unsafe_allow_html=True)
        if m.source:
            st.caption(f"Source: `{m.source}`")


def notes(*keys: str, label: str = "How these numbers are calculated") -> None:
    """One expander covering several related quantities."""
    ms = [METHODS[k] for k in keys if k in METHODS]
    if not ms:
        return
    with st.expander(label):
        for i, m in enumerate(ms):
            if i:
                st.divider()
            st.markdown(f"### {m.title}")
            st.markdown(f"**{m.summary}**")
            if m.formula:
                st.latex(m.formula)
            elif m.formula_plain:
                st.code(m.formula_plain, language=None)
            if m.inputs:
                st.markdown("**Inputs**")
                st.markdown("\n".join(f"- `{sym}` — {src}" for sym, src in m.inputs))
            if m.steps:
                st.markdown("**Steps**")
                st.markdown("\n".join(f"{j}. {s}" for j, s in enumerate(m.steps, 1)))
            if m.caveat:
                st.markdown(f'<div class="warn"><b>What this does not tell you.</b> '
                            f'{m.caveat}</div>', unsafe_allow_html=True)
            if m.source:
                st.caption(f"Source: `{m.source}`")
