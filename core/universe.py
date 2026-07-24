"""
core/universe.py

Builds the actual Portfolio object for Michael's book, from the
holdings defined in the portfolio thesis doc.

IMPORTANT - PLACEHOLDER DATA:
  - `account_tag` assignments below are PLACEHOLDERS. The thesis names
    the RRSP/TFSA/FHSA tax-treatment logic in principle (US equities in
    RRSP to avoid withholding drag; CAD equities elsewhere) but doesn't
    specify which real account currently holds which ticker. Every
    Security below is tagged provisionally by what the *thesis logic*
    implies it *should* be, not by confirmed actual placement. Correct
    these against real account statements before trusting by_account()
    output, and before drawing any conclusion from the "Home bias / tax
    treatment" assumption in the thesis.
  - `weight` is left as None for every security - weights are pending
    mean-variance optimization per the thesis doc, not yet assigned.
  - `mer` values are entered from provider fact sheets at time of writing
    and should be spot-checked, since MERs do change.
"""

from __future__ import annotations

from core.account_profile import AccountProfile
from core.portfolio import Portfolio
from core.security import Security

# ---------------------------------------------------------------------------
# Account profiles (PLACEHOLDER structure - confirm against real accounts)
# ---------------------------------------------------------------------------

ACCOUNT_PROFILES = [
    AccountProfile(
        tag="RRSP",
        name="Michael - RRSP",
        account_type="RRSP",
        us_withholding_exempt=True,
        notes="US-listed equity holdings should sit here per thesis tax-treatment logic.",
    ),
    AccountProfile(
        tag="TFSA",
        name="Michael - TFSA",
        account_type="TFSA",
        us_withholding_exempt=False,
        notes="No treaty withholding exemption - US equities held here incur drag per thesis.",
    ),
    AccountProfile(
        tag="FHSA",
        name="Michael - FHSA",
        account_type="FHSA",
        us_withholding_exempt=False,
        notes="Same withholding treatment as TFSA.",
    ),
    AccountProfile(
        tag="NON_REG",
        name="Michael - Non-Registered",
        account_type="Non-Registered",
        us_withholding_exempt=False,
        notes="Taxable account - foreign tax credit applies annually instead of treaty exemption.",
    ),
]


def build_michael_portfolio() -> Portfolio:
    """Instantiate Portfolio with every Security from the current thesis doc."""
    portfolio = Portfolio()
    for profile in ACCOUNT_PROFILES:
        portfolio.add_account_profile(profile)

    securities = [
        # ------------------------------------------------------------------
        # Fixed Income
        # ------------------------------------------------------------------
        Security(
            ticker="XFR.TO",
            name="iShares Floating Rate Index ETF",
            asset_class="Fixed Income - Floating Rate",
            currency="CAD",
            mer=0.0015,
            fund_site_url="https://www.blackrock.com/ca/investors/en/products/239487/ishares-floating-rate-index-etf",
            account_tag="NON_REG",  # placeholder - confirm actual account
            role="Rate hedge",
            thesis="Coupon resets with short-term rates; keeps duration near zero given the hikes-through-2027 view.",
        ),
        Security(
            ticker="CASH.TO",
            name="Purpose High Interest Savings ETF (proxy for Scotiabank ISA)",
            asset_class="Fixed Income - Cash Proxy",
            currency="CAD",
            mer=0.0015,
            fund_site_url="https://ads.scotiabank.com/investment-savings-account",
            account_tag="NON_REG",  # placeholder - confirm actual account
            role="Liquidity / modeling proxy",
            thesis=(
                "Modeling proxy only - has a tradeable NAV history for mean-variance work. "
                "Live cash should be routed to whichever real HISA rate is actually best at "
                "time of funding (currently Scotiabank ISA at 2.2%), not assumed to equal "
                "CASH.TO's yield."
            ),
        ),
        # ------------------------------------------------------------------
        # Equities - US
        # ------------------------------------------------------------------
        Security(
            ticker="XUU.TO",
            name="iShares Core S&P U.S. Total Market Index ETF",
            asset_class="Equity - US Core",
            currency="CAD",
            mer=0.0006,
            fund_site_url="https://www.blackrock.com/ca/investors/en/products/272104/ishares-core-sp-us-total-market-index-etf",
            account_tag="RRSP",  # placeholder - US equity, belongs here per tax-treatment logic
            role="Core US beta",
            thesis="~3,500 names, more diluted than S&P 500 alone; baseline exposure the value tilts below are offsetting.",
        ),
        Security(
            ticker="VTV",
            name="Vanguard Value ETF",
            asset_class="Equity - US Value",
            currency="USD",
            mer=0.0004,
            fund_site_url="https://investor.vanguard.com/investment-products/etfs/profile/vtv",
            account_tag="RRSP",  # placeholder - US-listed, requires Norbert's Gambit
            role="Concentration offset",
            thesis="Screens for statistically cheap large caps; structurally underweight mega-cap AI-capex names.",
        ),
        Security(
            ticker="AVUV",
            name="Avantis U.S. Small Cap Value ETF",
            asset_class="Equity - US Small Cap Value",
            currency="USD",
            mer=0.0025,
            fund_site_url="https://www.avantisinvestors.com/avantis-investments/avantis-us-small-cap-value-etf/",
            account_tag="RRSP",  # placeholder - US-listed, requires Norbert's Gambit
            role="Concentration offset",
            thesis="Excludes trillion-dollar names entirely by definition - cleaner expression of the concentration-offset thesis than VTV alone.",
        ),
        # ------------------------------------------------------------------
        # Equities - Canada
        # ------------------------------------------------------------------
        Security(
            ticker="XIC.TO",
            name="iShares Core S&P/TSX Capped Composite Index ETF",
            asset_class="Equity - Canada",
            currency="CAD",
            mer=0.0006,
            fund_site_url="https://www.blackrock.com/ca/investors/en/products/239837/ishares-sptsx-capped-composite-index-etf",
            account_tag="TFSA",  # placeholder - CAD equity, no withholding concern either way
            role="Tax-efficient CAD equity",
            thesis="Held for asset-location efficiency (CAD dividends avoid US withholding drag), not a standalone view on Canadian equities.",
        ),
        # ------------------------------------------------------------------
        # Equities - International
        # ------------------------------------------------------------------
        Security(
            ticker="VIU.TO",
            name="Vanguard FTSE Developed All Cap ex North America Index ETF",
            asset_class="Equity - International Developed",
            currency="CAD",
            mer=0.0022,
            fund_site_url="https://www.vanguard.ca/en/product/etf/equity/9569/vanguard-ftse-developed-all-cap-ex-north-america-index-etf",
            account_tag="FHSA",  # placeholder - confirm actual account
            role="Geographic diversification",
            thesis="International developed exposure with materially less AI-capex concentration than the US market.",
        ),
        Security(
            ticker="VEE.TO",
            name="Vanguard FTSE Emerging Markets All Cap Index ETF",
            asset_class="Equity - Emerging Markets",
            currency="CAD",
            mer=0.0024,
            fund_site_url="https://www.vanguard.ca/en/product/etf/equity/9556/vanguard-ftse-emerging-markets-all-cap-index-etf",
            account_tag="FHSA",  # placeholder - confirm actual account
            role="Geographic diversification",
            thesis="Same diversification logic as VIU, extended to a growth profile largely independent of US mega-cap tech capex cycles.",
        ),
        # ------------------------------------------------------------------
        # Real Assets
        # ------------------------------------------------------------------
        Security(
            ticker="CGL.TO",
            name="iShares Gold Bullion ETF",
            asset_class="Real Assets - Gold",
            currency="CAD",
            mer=0.0055,
            fund_site_url="https://www.blackrock.com/ca/investors/en/products/272269/ishares-gold-bullion-etf",
            account_tag="NON_REG",  # placeholder - confirm actual account
            role="Tail hedge",
            thesis="Sized as a small position for an inflationary or supply-shock-driven drawdown (AI-capex unwind, Taiwan/geopolitical shock) - not held as a general diversifier.",
        ),
        # ------------------------------------------------------------------
        # Legacy positions - flagged, not yet reconciled with thesis
        # ------------------------------------------------------------------
        Security(
            ticker="CAR-UN.TO",  # CAPREIT
            name="Canadian Apartment Properties REIT (CAPREIT)",
            asset_class="Legacy - Single Name Equity",
            currency="CAD",
            mer=0.0,
            fund_site_url="",
            account_tag="NON_REG",  # placeholder - confirm actual account
            role="Under review",
            thesis="Legacy position, single-name real estate equity. No thesis yet stated consistent with the rest of the book - flagged for separate review, not folded in here.",
        ),
        Security(
            ticker="VOLX.TO",
            name="BetaPro S&P 500 VIX Short-Term Futures ETF (VOLX)",
            asset_class="Legacy - Volatility",
            currency="CAD",
            mer=0.0118,  # ~1.18% expense ratio per fund site, confirm current figure
            fund_site_url="https://betapro.ca/product/volx",
            account_tag="NON_REG",  # placeholder - confirm actual account
            role="Under review",
            thesis=(
                "Legacy VIX-linked position - tracks the S&P 500 VIX Short-Term Futures Index "
                "via daily-rebalanced futures. Decays in contango (normal) markets, making it "
                "expensive to hold continuously as a passive hedge rather than a tactical trade. "
                "May be duplicating gold's tail-hedge role - worth resolving which one (if "
                "either) earns a place in the book before sizing either."
            ),
        ),
    ]

    for security in securities:
        portfolio.add_security(security)

    return portfolio
