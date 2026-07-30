"""Compatibility shim for regime logic.

The project originally referenced the regime helper under `core.regime`, while the
implemented code lives in `analysis.regime`. Keep both import paths working so
notebooks and analysis modules can load without path churn.
"""

from __future__ import annotations

from analysis.regime import bull_bear_mask

__all__ = ["bull_bear_mask"]
