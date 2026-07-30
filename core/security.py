"""
core/security.py

Security is the atomic holding object. Every ETF/stock/legacy position
in the portfolio is one Security instance. It carries an account_tag
(plain string) pointing to an AccountProfile — Security itself has no
reference to AccountProfile or Portfolio, keeping the dependency
direction one-way (Portfolio depends on Security, not vice versa).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.data_loader import get_price_history

VALID_ASSET_CLASSES = {
    "Fixed Income - Floating Rate",
    "Fixed Income - Cash Proxy",
    "Fixed Income - Fixed Rate",
    "Equity - US Core",
    "Equity - US Value",
    "Equity - US Small Cap Value",
    "Equity - Canada",
    "Equity - International Developed",
    "Equity - Emerging Markets",
    "Real Assets - Gold",
    "Volatility",
    # Backward-compatible aliases used by legacy thesis entries.
    "Legacy - Single Name Equity",
    "Legacy - Volatility",
}


@dataclass
class Security:
    ticker: str
    name: str
    asset_class: str  # one of VALID_ASSET_CLASSES
    currency: str  # "CAD" | "USD"
    mer: float  # management expense ratio, decimal (e.g. 0.0011 for 0.11%)
    fund_site_url: str
    account_tag: str  # points to an AccountProfile.tag, not validated here
    weight: Optional[float] = None  # target weight; None until mean-variance optimization is run
    role: Optional[str] = None  # short label, e.g. "Concentration offset"
    thesis: Optional[str] = None  # one-line rationale, mirrors the portfolio thesis doc
    holdings: Optional[pd.DataFrame] = field(default=None, repr=False)  # None until scraped from fund site
    _prices: Optional[pd.DataFrame] = field(default=None, repr=False, compare=False)

    # __post_init__ runs automatically after the dataclass-generated __init__. <= after object is created, but before it is returned to the caller.
    # It is a good place to validate fields that depend on the class invariants.
    def __post_init__(self) -> None:
        if self.asset_class not in VALID_ASSET_CLASSES:
            raise ValueError(
                f"asset_class '{self.asset_class}' not recognized for {self.ticker}. "
                f"Expected one of {sorted(VALID_ASSET_CLASSES)}."
            )
        if self.currency not in {"CAD", "USD"}:
            raise ValueError(f"currency must be 'CAD' or 'USD', got {self.currency!r}")

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    # fetch_prices is a normal method that explicitly loads price history.
    def fetch_prices(self, force_refresh: bool = False) -> pd.DataFrame:
        """Pull (or refresh) full price history via core.data_loader."""
        self._prices = get_price_history(self.ticker, force_refresh=force_refresh)
        return self._prices

    # The @property decorator makes prices behave like an attribute.
    # Accessing security.prices loads the data lazily on first use.
    # Output is a pandas DataFrame with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits.
    @property
    def prices(self) -> pd.DataFrame:
        """Full OHLCV history. Fetches on first access if not already loaded."""
        if self._prices is None:
            self.fetch_prices()
        return self._prices

    # close is also a computed attribute, returning the series of closing prices.
    @property
    def close(self) -> pd.Series:
        return self.prices["Close"]

    # Compute daily percent returns from close prices.
    def daily_returns(self) -> pd.Series:
        return self.close.pct_change().dropna()

    # ------------------------------------------------------------------
    # Fund holdings (scrape target — placeholder until pca_overlap.py phase)
    # ------------------------------------------------------------------

    # set_holdings lets external code attach a holdings DataFrame later.
    def set_holdings(self, holdings_df: pd.DataFrame) -> None:
        """Attach a scraped fund holdings table (ticker/weight at minimum)."""
        self.holdings = holdings_df

    # __repr__ defines how the object is shown in interactive output.
    def __repr__(self) -> str:
        weight_str = f"{self.weight:.1%}" if self.weight is not None else "unset"
        return (
            f"Security({self.ticker}, class={self.asset_class!r}, "
            f"account={self.account_tag!r}, weight={weight_str})"
        )
