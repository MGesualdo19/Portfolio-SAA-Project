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
    securities: list[Security] = field(default_factory=list)
    _account_profiles: dict[str, AccountProfile] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_security(self, security: Security) -> None:
        self.securities.append(security)

    def add_account_profile(self, profile: AccountProfile) -> None:
        self._account_profiles[profile.tag] = profile

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

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

    def account_tags(self) -> set[str]:
        """All account_tag values currently in use across securities."""
        return {s.account_tag for s in self.securities}

    def unregistered_account_tags(self) -> set[str]:
        """
        account_tag values used by securities but with no matching
        AccountProfile registered — a data-integrity check worth running
        before by_account() is trusted anywhere downstream.
        """
        return self.account_tags() - set(self._account_profiles.keys())

    def tickers(self) -> list[str]:
        return [s.ticker for s in self.securities]

    def get(self, ticker: str) -> Optional[Security]:
        for s in self.securities:
            if s.ticker == ticker:
                return s
        return None

    def by_asset_class(self, asset_class: str) -> list[Security]:
        return [s for s in self.securities if s.asset_class == asset_class]

    # ------------------------------------------------------------------
    # Weights / summary
    # ------------------------------------------------------------------

    def total_weight(self) -> Optional[float]:
        """Sum of assigned weights. Returns None if any security has weight=None."""
        weights = [s.weight for s in self.securities]
        if any(w is None for w in weights):
            return None
        return sum(weights)

    def weight_check(self, tolerance: float = 0.001) -> None:
        """Prints a warning if assigned weights don't sum to ~1.0."""
        total = self.total_weight()
        if total is None:
            unset = [s.ticker for s in self.securities if s.weight is None]
            print(f"[Portfolio] {len(unset)} security/securities have no weight set: {unset}")
            return
        if abs(total - 1.0) > tolerance:
            print(f"[Portfolio] Warning: weights sum to {total:.4f}, not 1.0.")

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
