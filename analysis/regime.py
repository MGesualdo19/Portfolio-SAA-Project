"""
core/regime.py

Regime classification used by the core risk framework.

Only bull/bear drawdown classification lives here for now. The
rate-cycle / inflation-direction / concentration classifiers built to
test the portfolio's specific assumptions are deferred to the
hypothesis-testing phase and will live in analysis/ instead, since
they're diagnostic tools rather than core infrastructure.
"""

from __future__ import annotations

import pandas as pd


def bull_bear_mask(
    benchmark_close: pd.Series,
    drawdown_threshold: float = 0.10,
) -> pd.Series:
    """
    Drawdown-threshold regime classification off the TSX Composite
    (^GSPTSE), per project spec.

    Returns a boolean Series aligned to benchmark_close's index:
    True = 'bear' (more than `drawdown_threshold` below trailing
    all-time high as of that date), False = 'bull'.
    """
    trailing_high = benchmark_close.cummax()
    drawdown = (benchmark_close / trailing_high) - 1.0
    return drawdown <= -drawdown_threshold
