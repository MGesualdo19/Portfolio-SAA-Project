"""
core/portfolio.py

Portfolio is the top-of-house object: a flat list of Security objects
plus a lookup of AccountProfile objects keyed by tag. It does not nest
securities under accounts — by_account() filters and joins on demand,
so a security's account can be corrected without restructuring anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.account_profile import AccountProfile
from core.security import Security


@dataclass
class Portfolio:
    # Create list of Security objects and a dictionary of AccountProfile objects
    # The list of securities is initialized as an empty list and the dictionary of account profiles is initialized as an empty dictionary.
    
    # in the string representation of the Portfolio object.
    securities: list[Security] = field(default_factory=list)
    # Single underscore is a convention to indicate that this attribute is intended for internal use within the class and should not be accessed directly from outside the class.
    # The repr=False argument in the field() function for _account_profiles means that this attribute will not be included in printable representations of the Portfolio object, which can be useful for keeping the output concise and focused on the most relevant information.
    _account_profiles: dict[str, AccountProfile] = field(default_factory=dict, repr=False) 
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    # Type hints for each add method, specifying the expected argument types and return type (None).
    # Add a Security object to the portfolio.
    def add_security(self, security: Security) -> None:
        self.securities.append(security)

    # Add or update an AccountProfile by its tag.
    def add_account_profile(self, profile: AccountProfile) -> None:
        self._account_profiles[profile.tag] = profile

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    # Will return a tuple containing a list of Security objects filtered by the specified account_tag
    # and the corresponding AccountProfile object (or None if no matching profile is found).
    # Return securities and the matching account profile for a given account tag.
    def by_account(self, tag: str) -> tuple[list[Security], Optional[AccountProfile]]:
        """
        Returns (securities filtered to this account_tag, the matching
        AccountProfile or None if the tag has no registered profile).
        """
        secs = [s for s in self.securities if s.account_tag == tag]
        profile = self._account_profiles.get(tag)
        if profile is None:
            print(f"[Portfolio] Warning: no AccountProfile registered for tag '{tag}'.")
        return secs, profile

    # Return every account_tag currently used by the portfolio holdings.
    def account_tags(self) -> set[str]:
        """All account_tag values currently in use across securities."""
        return {s.account_tag for s in self.securities}

    # Identify account tags used by securities that have no registered profile.
    def unregistered_account_tags(self) -> set[str]:
        """
        account_tag values used by securities but with no matching
        AccountProfile registered — a data-integrity check worth running
        before by_account() is trusted anywhere downstream.
        """
        return self.account_tags() - set(self._account_profiles.keys())

    # Return the list of all tickers held in the portfolio.
    def tickers(self) -> list[str]:
        return [s.ticker for s in self.securities]

    # Lookup a security by ticker symbol, or return None if not found.
    def get(self, ticker: str) -> Optional[Security]:
        for s in self.securities:
            if s.ticker == ticker:
                return s
        return None

    # Filter securities by asset class.
    def by_asset_class(self, asset_class: str) -> list[Security]:
        return [s for s in self.securities if s.asset_class == asset_class]

    # ------------------------------------------------------------------
    # Weights / summary
    # ------------------------------------------------------------------

    # Compute the total portfolio weight, returning None if any weight is unset.
    def total_weight(self) -> Optional[float]:
        """Sum of assigned weights. Returns None if any security has weight=None."""
        weights = [s.weight for s in self.securities]
        if any(w is None for w in weights):
            return None
        return sum(weights)

    # Validate that weights are set and sum roughly to 1.0, printing warnings if not.
    def weight_check(self, tolerance: float = 0.001) -> None:
        """Prints a warning if assigned weights don't sum to ~1.0."""
        total = self.total_weight()
        if total is None:
            unset = [s.ticker for s in self.securities if s.weight is None]
            print(f"[Portfolio] {len(unset)} security/securities have no weight set: {unset}")
            return
        if abs(total - 1.0) > tolerance:
            print(f"[Portfolio] Warning: weights sum to {total:.4f}, not 1.0.")

    # Convert the portfolio holdings into a pandas DataFrame summary.
    def as_dataframe(self) -> pd.DataFrame:
        """Flat summary table — one row per security, useful for notebook display."""
        rows = [
            {
                "ticker": s.ticker,
                "name": s.name,
                "asset_class": s.asset_class,
                "currency": s.currency,
                "mer": s.mer,
                "account_tag": s.account_tag,
                "weight": s.weight,
                "role": s.role,
            }
            for s in self.securities
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Bulk price fetching
    # ------------------------------------------------------------------

    # Fetch price history for every security in the portfolio.
    def fetch_all_prices(self, force_refresh: bool = False) -> None:
        failures: dict[str, str] = {}
        for s in self.securities:
            try:
                s.fetch_prices(force_refresh=force_refresh)
            except Exception as exc:  # noqa: BLE001
                failures[s.ticker] = str(exc)
        if failures:
            print(f"[Portfolio] Failed to fetch prices for {len(failures)} ticker(s):")
            for ticker, msg in failures.items():
                print(f"  {ticker}: {msg}")

    # Build a DataFrame of aligned daily returns for all securities with loaded prices.
    def returns_matrix(self) -> pd.DataFrame:
        """
        Daily returns for every security, aligned on a common date index
        (inner join — dates where every security has a price). Securities
        without loaded prices are skipped with a warning, not silently
        dropped.
        """
        series = {}
        for s in self.securities:
            if s._prices is None:
                print(f"[Portfolio] Skipping {s.ticker}: prices not fetched yet.")
                continue
            series[s.ticker] = s.daily_returns()
        return pd.DataFrame(series).dropna(how="any")
