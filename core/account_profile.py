"""
core/account_profile.py

AccountProfile is metadata only — it never holds securities. Security
objects point to an AccountProfile via `account_tag` (a plain string),
and Portfolio.by_account() joins the two together on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass #Prevents needing to write __init__ and __repr__ methods manually, automatically generates them based on the class attributes.
class AccountProfile:
    tag: str                       # unique key, matched against Security.account_tag
    name: str                      # human-readable label, e.g. "Michael - RRSP"
    account_type: str              # "RRSP", "TFSA", "FHSA", "Non-Registered"
    horizon_years: Optional[int] = None
    us_withholding_exempt: bool = False
    # True only for RRSP/RRIF under the Canada-US tax treaty. Relevant to
    # the thesis's asset-location logic: US-listed equities held here avoid
    # withholding drag; the same tickers in a TFSA/FHSA do not.
    notes: str = ""

    def __post_init__(self) -> None:
        valid_types = {"RRSP", "TFSA", "FHSA", "Non-Registered"}
        if self.account_type not in valid_types:
            raise ValueError(
                f"account_type '{self.account_type}' not recognized. "
                f"Expected one of {valid_types}."
            )
        if self.account_type == "RRSP" and not self.us_withholding_exempt:
            # Not an error — just a nudge, since this is the exact mechanism
            # the thesis's home-bias/tax-treatment assumption depends on.
            pass
