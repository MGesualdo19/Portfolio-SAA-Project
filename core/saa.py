"""
core/saa.py

The orchestrator: one function that runs the whole strategic asset
allocation end to end and returns every intermediate object.

This module exists so that the notebook and the dashboard cannot drift
apart. Both call `run_saa()`; neither reimplements any step. If the
notebook shows a weight, it is the same weight the dashboard shows,
because it came from the same call.

Pipeline
--------
  1. Load the universe and fetch prices.
  2. Build CAD total returns, proxy-extended back to 2007 (core/proxies.py).
  3. Define the equity-sleeve drawdown regime that "bear" means here.
  4. Estimate a shrunk, bear-blended covariance (core/estimation.py).
  5. Reverse-optimise the reference allocation into equilibrium returns,
     then apply the thesis views via Black-Litterman.
  6. Solve five objectives under policy constraints, each Michaud-
     resampled, and blend them (core/optimizer.py).
  7. Round to a tradeable grid and report diagnostics, including whether
     any constraint is doing the work.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from analysis.stress import equity_drawdown_mask
from core.estimation import (
    black_litterman,
    correlation_from_cov,
    implied_equilibrium_returns,
    regime_blended_covariance,
    shrunk_historical_returns,
)
from core.optimizer import (
    ConstraintSet,
    STRATEGY_LABELS,
    blend_strategies,
    build_all_strategies,
    round_to_tradeable,
    summarise_allocation,
)
from core.policy import (
    EQUITY_TICKERS,
    EXCLUDED,
    REFERENCE_WEIGHTS,
    SLEEVES,
    STRATEGIC_UNIVERSE,
    default_constraints,
    thesis_views,
)
from core.proxies import extended_returns_frame, proxy_quality_report
from core.returns import returns_frame, summary_stats
from core.universe import build_michael_portfolio

CACHE_PATH = Path(__file__).resolve().parent.parent / "outputs" / "saa_result.pkl"
HISTORY_START = "2007-05-31"   # earliest date on which every proxy chain is populated
RISK_FREE = 0.0275             # ~current Canadian 3-month bill; a policy input, not an estimate


@dataclass
class SAAResult:
    """Everything the notebook or dashboard needs, computed once."""
    generated_at: datetime
    tickers: list[str]
    securities: list = field(repr=False, default_factory=list)

    returns: pd.DataFrame = field(repr=False, default=None)          # spliced, CAD, aligned
    return_sources: pd.DataFrame = field(repr=False, default=None)
    native_returns: pd.DataFrame = field(repr=False, default=None)   # own history, native ccy
    proxy_report: pd.DataFrame = field(repr=False, default=None)

    bear_mask: pd.Series = field(repr=False, default=None)
    equity_index: pd.Series = field(repr=False, default=None)

    cov: pd.DataFrame = field(repr=False, default=None)
    cov_diagnostics: dict = field(repr=False, default_factory=dict)
    corr: pd.DataFrame = field(repr=False, default=None)

    equilibrium_returns: pd.Series = field(repr=False, default=None)
    expected_returns: pd.Series = field(repr=False, default=None)
    view_detail: pd.DataFrame = field(repr=False, default=None)
    historical_returns_shrunk: pd.Series = field(repr=False, default=None)

    strategy_weights: pd.DataFrame = field(repr=False, default=None)
    dispersion: dict = field(repr=False, default_factory=dict)
    blended_weights: pd.Series = field(repr=False, default=None)
    final_weights: pd.Series = field(repr=False, default=None)

    constraints: ConstraintSet = field(repr=False, default=None)
    binding: list = field(repr=False, default_factory=list)
    violations: list = field(repr=False, default_factory=list)
    summary: dict = field(repr=False, default_factory=dict)
    mer: pd.Series = field(repr=False, default=None)
    settings: dict = field(repr=False, default_factory=dict)

    # ------------------------------------------------------------------
    def weights_table(self) -> pd.DataFrame:
        """Presentation table: final weight, per-strategy spread, sleeve, risk share."""
        sec = {s.ticker: s for s in self.securities}
        sleeve_of = {t: name for name, ts in SLEEVES.items() for t in ts}
        rc = self.summary.get("risk_contributions", pd.Series(dtype=float))
        rows = []
        for t in self.final_weights.index:
            s = sec.get(t)
            spread = self.strategy_weights.loc[t] if t in self.strategy_weights.index else pd.Series(dtype=float)
            rows.append({
                "ticker": t,
                "name": s.name if s else t,
                "sleeve": sleeve_of.get(t, "Other"),
                "currency": s.currency if s else "",
                "mer": s.mer if s else np.nan,
                "weight": float(self.final_weights[t]),
                "risk_share": float(rc.get(t, np.nan)),
                "strategy_min": float(spread.min()) if len(spread) else np.nan,
                "strategy_max": float(spread.max()) if len(spread) else np.nan,
                "role": s.role if s else "",
            })
        df = pd.DataFrame(rows).set_index("ticker")
        return df.sort_values("weight", ascending=False)

    def sleeve_table(self) -> pd.DataFrame:
        w = self.final_weights
        rows = []
        for name, ts in SLEEVES.items():
            members = [t for t in ts if t in w.index]
            rows.append({"sleeve": name,
                         "weight": float(w[members].sum()),
                         "holdings": ", ".join(members)})
        return pd.DataFrame(rows).set_index("sleeve").sort_values("weight", ascending=False)

    def currency_exposure(self) -> pd.Series:
        """
        Weight by the currency of the UNDERLYING assets, not the listing.

        This distinction is routinely missed and it matters here: XUU.TO
        is CAD-listed but holds US stocks unhedged, so it carries exactly
        the same USD exposure as VTV does. Reporting by listing currency
        would show this book as ~85% CAD when its true USD exposure is
        far higher.
        """
        underlying = {
            "XFR.TO": "CAD", "CASH.TO": "CAD", "XIC.TO": "CAD", "CAR-UN.TO": "CAD",
            "XUU.TO": "USD", "VTV": "USD", "AVUV": "USD",
            "CGL.TO": "USD",           # bullion is priced in USD; CGL.TO is unhedged
            "VIU.TO": "Other developed", "VEE.TO": "Emerging",
        }
        w = self.final_weights
        out: dict[str, float] = {}
        for t, v in w.items():
            k = underlying.get(t, "CAD")
            out[k] = out.get(k, 0.0) + float(v)
        return pd.Series(out).sort_values(ascending=False)

    def blended_mer(self) -> float:
        return float((self.final_weights * self.mer.reindex(self.final_weights.index).fillna(0)).sum())


def _equity_sleeve_index(returns: pd.DataFrame) -> pd.Series:
    """
    The regime benchmark: an equal-weighted total-return index of this
    book's own equity sleeves, in CAD.

    Using the portfolio's actual equity exposure rather than ^GSPTSE is
    the correction that makes the regime definition mean something here.
    The TSX is ~3% of world market capitalisation and roughly 60%
    financials plus energy; it has had drawdowns this globally-diversified
    book barely felt, and missed some it felt sharply. "Bear" should mean
    "the equity risk this investor actually holds is in a drawdown."
    """
    cols = [t for t in EQUITY_TICKERS if t in returns.columns]
    sleeve = returns[cols].mean(axis=1)
    return (1 + sleeve).cumprod()


def run_saa(
    bear_weight: float = 0.35,
    drawdown_threshold: float = 0.10,
    n_resamples: int = 120,
    resample: bool = True,
    rf: float = RISK_FREE,
    risk_aversion: float = 2.8,
    view_confidence_scale: float = 1.0,
    constraints: Optional[ConstraintSet] = None,
    history_start: str = HISTORY_START,
    verbose: bool = True,
) -> SAAResult:
    """
    Run the full pipeline. Every tunable that represents a judgement
    rather than an estimate is a named argument here, so the dashboard can
    expose it and the effect of changing it is visible rather than buried.
    """
    log = print if verbose else (lambda *a, **k: None)

    log("[saa] Building universe and fetching prices...")
    portfolio = build_michael_portfolio()
    portfolio.fetch_all_prices()
    securities = [s for s in portfolio.securities if s.ticker not in EXCLUDED]
    tickers = [s.ticker for s in securities]
    missing = set(STRATEGIC_UNIVERSE) - set(tickers)
    if missing:
        raise RuntimeError(f"Strategic universe tickers missing from the portfolio: {missing}")

    log("[saa] Splicing proxy history and building CAD total returns...")
    rets, sources = extended_returns_frame(securities, min_start=history_start, align=True)
    rets = rets[STRATEGIC_UNIVERSE]
    sources = sources[STRATEGIC_UNIVERSE]
    native = returns_frame(securities, in_cad=False, align=False)
    proxy_rpt = proxy_quality_report(securities)

    log(f"[saa] Window: {rets.index.min().date()} -> {rets.index.max().date()} ({len(rets)} days)")

    equity_index = _equity_sleeve_index(rets)
    bear_mask = equity_drawdown_mask(equity_index, threshold=drawdown_threshold)
    log(f"[saa] Bear regime: {int(bear_mask.sum())} of {len(bear_mask)} days "
        f"({bear_mask.mean():.1%}) at a {drawdown_threshold:.0%} drawdown threshold.")

    log("[saa] Estimating regime-blended shrunk covariance...")
    cov, cov_diag = regime_blended_covariance(rets, bear_mask, bear_weight=bear_weight)
    corr = correlation_from_cov(cov)

    log("[saa] Reverse-optimising equilibrium returns and applying thesis views...")
    ref = REFERENCE_WEIGHTS.reindex(cov.index)
    if ref.isna().any():
        raise RuntimeError(f"Reference weights missing for {list(ref[ref.isna()].index)}")
    pi = implied_equilibrium_returns(cov, ref, risk_aversion=risk_aversion, rf=rf)

    views = thesis_views()
    if view_confidence_scale != 1.0:
        for v in views:
            v.confidence = float(np.clip(v.confidence * view_confidence_scale, 1e-4, 0.99))
    mu, view_detail = black_litterman(cov, pi, views)

    cs = constraints or default_constraints(list(cov.index))

    log(f"[saa] Solving {len(STRATEGY_LABELS)} objectives"
        f"{f' x {n_resamples} resamples' if resample else ''}...")
    strat_w, dispersion = build_all_strategies(
        rets, cov, mu, cs, rf=rf, resample=resample, n_resamples=n_resamples,
        bear_mask=bear_mask, bear_weight=bear_weight,
    )

    blended = blend_strategies(strat_w)
    final = round_to_tradeable(blended, step=0.005, drop_below=0.01)

    summary = summarise_allocation(final, cov, mu, rf=rf)
    binding = cs.binding_constraints(final)
    violations = cs.violations(final)

    mer = pd.Series({s.ticker: s.mer for s in securities})

    log(f"[saa] Done. {summary['n_positions']} positions, "
        f"effective N = {summary['effective_n']:.1f} (risk-based {summary['effective_n_risk']:.1f}), "
        f"expected {summary['expected_return']:.2%} at {summary['volatility']:.2%} vol.")
    if violations:
        log(f"[saa] WARNING -- constraint violations after rounding: {violations}")

    return SAAResult(
        generated_at=datetime.now(),
        tickers=list(cov.index),
        securities=securities,
        returns=rets,
        return_sources=sources,
        native_returns=native,
        proxy_report=proxy_rpt,
        bear_mask=bear_mask,
        equity_index=equity_index,
        cov=cov,
        cov_diagnostics=cov_diag,
        corr=corr,
        equilibrium_returns=pi,
        expected_returns=mu,
        view_detail=view_detail,
        historical_returns_shrunk=shrunk_historical_returns(rets),
        strategy_weights=strat_w,
        dispersion=dispersion,
        blended_weights=blended,
        final_weights=final,
        constraints=cs,
        binding=binding,
        violations=violations,
        summary=summary,
        mer=mer,
        settings={
            "bear_weight": bear_weight,
            "drawdown_threshold": drawdown_threshold,
            "n_resamples": n_resamples if resample else 0,
            "risk_free": rf,
            "risk_aversion": risk_aversion,
            "view_confidence_scale": view_confidence_scale,
            "history_start": history_start,
        },
    )


# ---------------------------------------------------------------------------
# Caching
#
# Pickle is used deliberately here and is safe in this context: the only
# writer is save_result() on this machine, the file lives inside the
# repo's gitignored outputs/ directory, and it is never fetched from or
# shared with anywhere. It stores live pandas objects and a ConstraintSet,
# which JSON cannot round-trip without a bespoke schema for every field.
# If this cache ever becomes something downloaded or shared, replace it
# with a schema-validated format before doing so.
# ---------------------------------------------------------------------------

def save_result(result: SAAResult, path: Path = CACHE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return path


def load_result(path: Path = CACHE_PATH) -> Optional[SAAResult]:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        # A stale pickle from an older class definition should force a clean
        # recompute, not crash the dashboard on import.
        return None


def reference_allocations(result: SAAResult) -> dict[str, pd.Series]:
    """
    Benchmarks the recommended allocation has to beat to justify itself.

    Equal weight is the classic hard-to-beat naive rule; the reference
    portfolio is the policy anchor; 60/40 is what the allocation is
    implicitly being chosen over; all-equity is the opportunity cost of
    holding anything defensive at all.
    """
    tickers = result.tickers
    eq = pd.Series(1.0 / len(tickers), index=tickers)

    sixty_forty = pd.Series(0.0, index=tickers)
    for t, w in {"XUU.TO": 0.30, "XIC.TO": 0.18, "VIU.TO": 0.12,
                 "XFR.TO": 0.25, "CASH.TO": 0.15}.items():
        if t in sixty_forty.index:
            sixty_forty[t] = w

    all_equity = pd.Series(0.0, index=tickers)
    eq_members = [t for t in EQUITY_TICKERS if t in tickers]
    all_equity[eq_members] = 1.0 / len(eq_members)

    return {
        "Recommended SAA": result.final_weights,
        "Reference policy": REFERENCE_WEIGHTS.reindex(tickers).fillna(0.0),
        "Equal weight": eq,
        "60/40 style": sixty_forty / sixty_forty.sum(),
        "All equity": all_equity,
    }
