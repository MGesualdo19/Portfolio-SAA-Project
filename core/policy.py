"""
core/policy.py

Investment policy: the universe admitted to the optimisation, the
reference allocation that anchors equilibrium returns, the policy bands,
and the Black-Litterman views that encode the thesis.

Everything here is a judgement, not an estimate, and each one is stated
with its reasoning so it can be argued with. Statistics live in
core/estimation.py; this file is where a human decides what the
portfolio is for.
"""

from __future__ import annotations

import pandas as pd

from core.estimation import View
from core.optimizer import ConstraintSet, GroupConstraint

# ---------------------------------------------------------------------------
# Universe admission
# ---------------------------------------------------------------------------

EXCLUDED: dict[str, str] = {
    "VOLX.TO": (
        "Excluded from the strategic allocation on the evidence, not on preference. "
        "Measured total return since inception is -100.0% (annualised -48%/yr, maximum "
        "drawdown -99.998%). This is not a bad run: a daily-rebalanced long VIX futures "
        "position is structurally short the roll, and the VIX futures curve is in contango "
        "in roughly 80% of months, so the position pays a large negative carry every month "
        "it is held and only wins in the few weeks around a volatility spike. It is a "
        "tactical trade with an expiry date, and the SAA is the wrong instrument for it -- "
        "any strategic weight above zero is a decision to fund a permanent negative carry. "
        "Gold already occupies the tail-hedge slot at a fraction of the cost. Retained in "
        "the universe for individual analysis in the dashboard so the position can be "
        "monitored and sized down deliberately, but given no strategic weight."
    ),
}

STRATEGIC_UNIVERSE = [
    "XFR.TO", "CASH.TO", "XUU.TO", "VTV", "AVUV",
    "XIC.TO", "VIU.TO", "VEE.TO", "CGL.TO", "CAR-UN.TO",
]

SLEEVES: dict[str, list[str]] = {
    "Cash & floating rate": ["CASH.TO", "XFR.TO"],
    "US equity": ["XUU.TO", "VTV", "AVUV"],
    "Canadian equity": ["XIC.TO"],
    "International developed": ["VIU.TO"],
    "Emerging markets": ["VEE.TO"],
    "Real assets": ["CGL.TO"],
    "Legacy single name": ["CAR-UN.TO"],
}

EQUITY_TICKERS = ["XUU.TO", "VTV", "AVUV", "XIC.TO", "VIU.TO", "VEE.TO"]


# ---------------------------------------------------------------------------
# Reference allocation (the equilibrium anchor)
# ---------------------------------------------------------------------------

REFERENCE_WEIGHTS = pd.Series({
    "XFR.TO": 0.13,
    "CASH.TO": 0.09,
    "XUU.TO": 0.13,
    "VTV": 0.09,
    "AVUV": 0.06,
    "XIC.TO": 0.15,
    "VIU.TO": 0.15,
    "VEE.TO": 0.08,
    "CGL.TO": 0.10,
    "CAR-UN.TO": 0.02,
})

REFERENCE_RATIONALE = """
This is the neutral portfolio the optimiser is measured against, not a
recommendation. Its only job is to be a defensible starting point whose
implied expected returns are sane, because reverse optimisation turns
whatever is put here into the prior that every subsequent weight is a
deviation from.

It is built as a plain policy allocation for a Canadian investor: 22%
short-duration fixed income and cash, 61% equity split roughly by a
blend of global market capitalisation and home-country practicality
(28% US, 15% Canada, 15% developed ex-North-America, 8% emerging), 10%
gold, and the 2% legacy REIT position marked to roughly its current
size. The Canadian weight is far above Canada's ~3% share of global
market capitalisation; that is a deliberate home-bias allowance
reflecting CAD liabilities and the dividend tax treatment the thesis
already relies on, and it is stated here rather than left implicit.

Because pi = delta * Sigma * w_ref reproduces w_ref exactly when fed
back into an unconstrained optimiser, this allocation is also the
answer the model returns if every view is set to zero confidence. That
is the intended behaviour: with no view, hold the policy portfolio.
"""


# ---------------------------------------------------------------------------
# Policy bands
# ---------------------------------------------------------------------------

def default_constraints(tickers: list[str] | None = None) -> ConstraintSet:
    """
    Per-asset bounds and asset-class bands.

    These are the backstop against a degenerate answer, not the mechanism
    -- shrinkage, equilibrium returns, resampling and objective-blending
    are what should be producing a diversified allocation. Bands here
    exist to encode policy that no amount of historical data should be
    allowed to override: that the book holds some dry powder, some
    non-North-American equity, and some real assets across every
    conceivable sample, because one 19-year path is not enough evidence
    to abandon any of those.

    Floors are set at the smallest size worth holding after trading
    costs; caps at the point where a single fund becomes a single point
    of failure.
    """
    tickers = tickers or list(STRATEGIC_UNIVERSE)

    bounds = {
        "XFR.TO":    (0.05, 0.25),
        "CASH.TO":   (0.03, 0.20),
        "XUU.TO":    (0.05, 0.25),
        "VTV":       (0.03, 0.20),
        "AVUV":      (0.02, 0.15),
        "XIC.TO":    (0.05, 0.22),
        "VIU.TO":    (0.05, 0.22),
        "VEE.TO":    (0.02, 0.12),
        "CGL.TO":    (0.03, 0.15),
        "CAR-UN.TO": (0.00, 0.05),
    }

    groups = [
        GroupConstraint(
            "Fixed income & cash", ["XFR.TO", "CASH.TO"], 0.15, 0.35,
            "A floor because the thesis expects hikes through 2027 and because "
            "rebalancing into a drawdown requires something to rebalance from. A cap "
            "because a 30-year horizon cannot be funded out of 3% real yields."),
        GroupConstraint(
            "Total equity", EQUITY_TICKERS, 0.45, 0.72,
            "The growth engine. Floored so the portfolio is not de-risked into "
            "irrelevance by an optimiser reacting to one bad sample; capped so a "
            "single equity drawdown cannot define the whole outcome."),
        GroupConstraint(
            "US equity", ["XUU.TO", "VTV", "AVUV"], 0.18, 0.40,
            "Capped at 40%: the US is ~60% of global market cap, and the thesis is "
            "explicitly that this concentration -- and its AI-capex composition -- is "
            "the risk being managed. Floored at 18% because underweighting the world's "
            "deepest, most profitable equity market is itself an active bet."),
        GroupConstraint(
            "Canadian equity", ["XIC.TO"], 0.05, 0.22,
            "Home bias, bounded. Canada is ~3% of global market cap and the index is "
            "roughly 60% financials plus energy, so a large weight is a concentrated "
            "sector bet wearing a diversification label."),
        GroupConstraint(
            "International equity", ["VIU.TO", "VEE.TO"], 0.10, 0.30,
            "The most direct expression of the concentration thesis: developed ex-North-"
            "America and emerging markets have materially less AI-capex exposure than "
            "the US market. Floored so that claim is actually funded."),
        GroupConstraint(
            "US value sleeve", ["VTV", "AVUV"], 0.08, 0.30,
            "The thesis's central active position. Floored at 8% so the value tilt is a "
            "real allocation rather than a rounding error next to XUU.TO's cap-weighted "
            "exposure to the same mega-caps the tilt is meant to offset."),
        GroupConstraint(
            "Real assets", ["CGL.TO"], 0.03, 0.15,
            "Gold as tail hedge. Floored because a hedge sized below 3% cannot move the "
            "portfolio when it is needed; capped at 15% because bullion produces no cash "
            "flow and a larger position is a macro trade, not an allocation."),
        GroupConstraint(
            "Legacy positions", ["CAR-UN.TO"], 0.00, 0.05,
            "Single-name equity risk with no diversification mandate. Capped hard, and "
            "zero is an acceptable answer."),
    ]

    return ConstraintSet(tickers=tickers, bounds=bounds, groups=groups,
                         default_bounds=(0.0, 0.25))


# ---------------------------------------------------------------------------
# Black-Litterman views
# ---------------------------------------------------------------------------

def thesis_views() -> list[View]:
    """
    The thesis translated into views. Each is relative (coefficients sum
    to zero) rather than absolute, because a relative statement -- "this
    beats that" -- is what the thesis actually claims, and relative views
    tilt the allocation without requiring a forecast of the market's
    overall level.

    Confidences are deliberately modest. A 0.35 confidence does not mean
    "35% likely to be right"; it means the posterior moves 35% of the way
    from the policy portfolio's implied returns toward this view. Even
    the strongest conviction here is a tilt, not a bet.
    """
    return [
        View(
            name="Value tilt over cap-weighted US core",
            picks={"VTV": 0.5, "AVUV": 0.5, "XUU.TO": -1.0},
            q=0.0125,
            confidence=0.35,
            rationale=(
                "The book's central active claim. Value spreads against growth remain "
                "near the wide end of their historical range, and a cap-weighted total-"
                "market fund now carries roughly a third of its weight in ten AI-capex-"
                "linked names. 1.25%/yr is a modest number by the standards of historical "
                "value premia precisely because the premium's timing is unreliable -- the "
                "position is justified more by what it avoids than by what it earns."),
        ),
        View(
            name="Ex-North-America equity over US core",
            picks={"VIU.TO": 0.6, "VEE.TO": 0.4, "XUU.TO": -1.0},
            q=0.010,
            confidence=0.25,
            rationale=(
                "Same concentration argument, expressed geographically. Developed ex-NA "
                "and emerging markets trade at a wide and persistent earnings-multiple "
                "discount to the US. Low confidence because that discount has been wide "
                "and persistent for over a decade without closing -- it is a valuation "
                "argument, and valuation is a poor short-horizon timing tool."),
        ),
        View(
            name="Floating rate over cash",
            picks={"XFR.TO": 1.0, "CASH.TO": -1.0},
            q=0.004,
            confidence=0.30,
            rationale=(
                "Direct expression of the rate view. Floating-rate notes reset with "
                "short rates and pick up a modest credit spread over a HISA, so if hikes "
                "arrive through 2027 the FRN sleeve captures them with a spread on top and "
                "essentially no duration risk either way. Small q because the spread is "
                "small and the two instruments are close substitutes."),
        ),
        View(
            name="Gold below equity over the long run",
            picks={"CGL.TO": 1.0, "XIC.TO": -1.0},
            q=-0.015,
            confidence=0.20,
            rationale=(
                "A deliberate brake, not a bearish call. Gold produces no cash flow, so "
                "its long-run expected return should sit below equity's; without this view "
                "the optimiser sees gold's strong realised run and its attractive "
                "covariance and would happily push it to its cap. Stating the return "
                "penalty explicitly forces gold to earn its weight on hedging value alone, "
                "which is the only reason the thesis wants it."),
        ),
        View(
            name="CAPREIT below broad Canadian equity",
            picks={"CAR-UN.TO": 1.0, "XIC.TO": -1.0},
            q=-0.010,
            confidence=0.25,
            rationale=(
                "Consistency with the rate view: a residential REIT is a long-duration, "
                "rate-sensitive, leveraged asset, and the thesis expects hikes. It also "
                "carries idiosyncratic single-name risk that a diversified index does not, "
                "with no compensating diversification role in this book."),
        ),
    ]
