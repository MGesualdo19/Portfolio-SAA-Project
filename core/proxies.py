"""
core/proxies.py

Proxy-backfilled return history.

The problem this solves: inner-joining the eleven holdings on dates
where all of them traded leaves a common window starting 2021-11 --
about 4.7 years. A covariance matrix over 11 assets estimated on 4.7
years of data is badly conditioned, but the more damaging issue is
WHICH 4.7 years: the window contains no 2008 and no 2020. An SAA built
on it would be calibrated entirely to a single post-COVID rate-hiking
regime, and every "does diversification survive a crash" test would be
answered using a sample with no real crash in it.

The fix is standard practice: splice a long-lived index proxy onto the
front of each short-lived holding's own history, using the holding's
real returns wherever they exist and the proxy's only before inception.

Two rules make this defensible rather than hand-wavy:

  1. The proxy must track the same underlying exposure -- VTI for a US
     total-market ETF, VBR for a small-cap-value ETF, GLD for bullion.
     proxy_quality_report() measures overlap-period correlation and
     tracking error for every splice so the assumption is audited, not
     asserted. Anything below ~0.95 overlap correlation is a weak splice.

  2. Currency is handled per-exposure, not per-listing. A CAD-listed
     unhedged fund on US assets (XUU.TO) has the same economics as VTI
     converted to CAD, so its proxy IS FX-converted. A CAD cash or
     floating-rate sleeve is proxied by a USD short-rate fund used in
     its OWN currency (fx_convert=False) -- what is being borrowed
     there is the short-rate path, and converting it would inject a
     currency exposure the real holding does not have and would make
     the cash sleeve look roughly 8x more volatile than it is.

Spliced history is an estimate. It is used for covariance, regime and
stress work, where a longer, regime-diverse sample dominates. It is
never presented as the holding's own track record.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.data_loader import get_price_history
from core.returns import total_return_series


@dataclass
class ProxyLink:
    ticker: str                 # proxy ticker on Yahoo
    currency: str               # proxy's own quote currency
    fx_convert: bool = True     # convert proxy into CAD? see rule 2 above
    vol_match: bool = False     # rescale residual volatility? see below
    note: str = ""


@dataclass
class ProxyChain:
    """Proxies in priority order: first is closest, later ones extend further back."""
    target: str
    links: list[ProxyLink] = field(default_factory=list)
    rationale: str = ""


# ---------------------------------------------------------------------------
# The map. Every entry is a judgement call and is stated as one.
# ---------------------------------------------------------------------------

PROXY_MAP: dict[str, ProxyChain] = {
    "XUU.TO": ProxyChain("XUU.TO", [
        ProxyLink("VTI", "USD", fx_convert=True,
                  note="Vanguard Total Stock Market -- same exposure, back to 2001."),
    ], "XUU.TO is unhedged, so VTI converted to CAD is the same economics, not an approximation."),

    "AVUV": ProxyChain("AVUV", [
        ProxyLink("VBR", "USD", fx_convert=False, note="Vanguard Small-Cap Value, back to 2004."),
        ProxyLink("IWN", "USD", fx_convert=False, note="Russell 2000 Value, back to 2000."),
    ], "AVUV is USD-denominated so the proxy stays in USD. VBR is index small-value; AVUV "
       "additionally screens on profitability, so the splice understates AVUV's quality "
       "tilt across the pre-2019 segment."),

    "VIU.TO": ProxyChain("VIU.TO", [
        ProxyLink("VEA", "USD", fx_convert=True, note="Developed ex-US, back to 2007."),
        ProxyLink("EFA", "USD", fx_convert=True, note="EAFE, back to 2001."),
    ], "VEA includes Canada and EFA excludes small caps, where VIU is developed ex-North-America "
       "all-cap. Regional overlap is high enough for covariance work."),

    "VEE.TO": ProxyChain("VEE.TO", [
        ProxyLink("VWO", "USD", fx_convert=True, note="FTSE Emerging, back to 2005."),
        ProxyLink("EEM", "USD", fx_convert=True, note="MSCI Emerging, back to 2003."),
    ], "VEE.TO holds VWO directly, so this is close to an exact splice once FX-converted."),

    "CGL.TO": ProxyChain("CGL.TO", [
        ProxyLink("GLD", "USD", fx_convert=True, note="Spot gold trust, back to 2004."),
    ], "CGL.TO is unhedged bullion; GLD in CAD is the same exposure."),

    "XFR.TO": ProxyChain("XFR.TO", [
        ProxyLink("FLOT", "USD", fx_convert=False, vol_match=True,
                  note="IG floating-rate notes, back to 2011."),
        ProxyLink("BIL", "USD", fx_convert=False, vol_match=True,
                  note="1-3 month T-bills, back to 2007."),
    ], "Short-rate path only, deliberately NOT FX-converted -- XFR.TO carries no currency risk. "
       "Vol-matched because FLOT carries US IG credit-spread risk (2.2x XFR's raw volatility) "
       "that XFR's Canadian bank/government floaters do not."),

    "CASH.TO": ProxyChain("CASH.TO", [
        ProxyLink("BIL", "USD", fx_convert=False, vol_match=True,
                  note="1-3 month T-bills, back to 2007."),
    ], "HISA/ISA return tracks the overnight rate. Not FX-converted, same reason as XFR.TO. "
       "US and Canadian overnight paths differ in level but move together in direction, "
       "which is what a covariance estimate needs."),
}

# Deliberately unproxied: XIC.TO (2001), VTV (2004) and CAR-UN.TO (1997) already
# span 2008. VOLX.TO is excluded from the SAA entirely (see docs/OBJECTIVES.md) --
# no index proxy for a daily-rebalanced VIX futures roll would be honest.


CARRY_WINDOW = 63  # ~one quarter, the smoothing window for the rate-carry level


def _proxy_returns(link: ProxyLink) -> pd.Series:
    px = get_price_history(link.ticker)
    return total_return_series(px, currency=link.currency, in_cad=link.fx_convert)


def _vol_match(proxy: pd.Series, own: pd.Series) -> pd.Series:
    """
    Rescale a cash-like proxy's RESIDUAL volatility to the target's, while
    leaving its rate-carry level path untouched.

    Naively rescaling the whole series would rescale the carry too -- BIL
    is ~6x calmer than XFR.TO, so a flat 6x scaling would also multiply the
    modelled cash yield by six. Splitting the proxy into a rolling-mean
    carry component and a residual, then scaling only the residual, keeps
    the policy-rate path that makes these proxies worth using (near zero
    2009-2015, ~2% in 2018, near zero in 2020-21, ~5% in 2023) while making
    the risk contribution match the holding actually owned.

    Falls back to the unscaled proxy if the overlap is too short to
    estimate a ratio, rather than inventing one.
    """
    joined = pd.concat([own, proxy], axis=1, keys=["own", "proxy"]).dropna()
    if len(joined) < 120:
        return proxy
    carry_overlap = joined["proxy"].rolling(CARRY_WINDOW, min_periods=20).mean()
    resid_overlap = (joined["proxy"] - carry_overlap).dropna()
    own_resid = (joined["own"] - joined["own"].rolling(CARRY_WINDOW, min_periods=20).mean()).dropna()
    if resid_overlap.std() == 0 or own_resid.empty:
        return proxy
    scale = float(own_resid.std() / resid_overlap.std())

    carry = proxy.rolling(CARRY_WINDOW, min_periods=1).mean()
    return carry + (proxy - carry) * scale


def extended_returns(security, min_start: str | None = None) -> tuple[pd.Series, pd.Series]:
    """
    Returns (spliced_return_series, source_labels) for one security, in CAD.

    The security's own returns always win on any date it has one; proxies
    fill only the gap before its inception, nearest proxy first. The
    second series labels each date's source so a chart can shade the
    synthetic segment and no reader mistakes it for a live track record.
    """
    own = total_return_series(security.prices, currency=security.currency, in_cad=True)
    own.name = security.ticker
    source = pd.Series(security.ticker, index=own.index)

    chain = PROXY_MAP.get(security.ticker)
    if chain is not None:
        base = own.copy()  # match against the holding's real returns, never a prior splice
        for link in chain.links:
            proxy = _proxy_returns(link)
            if link.vol_match:
                proxy = _vol_match(proxy, base)
            gap = proxy.index.difference(own.index)
            gap = gap[gap < own.index.min()]
            if len(gap):
                own = pd.concat([proxy.reindex(gap), own]).sort_index()
                source = pd.concat([pd.Series("~" + link.ticker, index=gap), source]).sort_index()

    if min_start is not None:
        cutoff = pd.Timestamp(min_start)
        own = own.loc[own.index >= cutoff]
        source = source.loc[source.index >= cutoff]
    own.name = security.ticker
    return own, source


def extended_returns_frame(securities, min_start: str | None = None,
                           align: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spliced CAD total-return matrix plus the matching source-label matrix."""
    rets, srcs = {}, {}
    for s in securities:
        r, src = extended_returns(s, min_start=min_start)
        rets[s.ticker], srcs[s.ticker] = r, src
    frame = pd.DataFrame(rets).sort_index()
    source = pd.DataFrame(srcs).sort_index()
    if align:
        frame = frame.dropna(how="any")
        source = source.reindex(frame.index)
    return frame, source


def proxy_quality_report(securities) -> pd.DataFrame:
    """
    Audits every splice on the overlap period where BOTH the real holding
    and its proxy exist. An overlap_corr below ~0.95, or a large annualised
    tracking error, means the spliced pre-inception history carries exposure
    the real holding does not have, and any conclusion depending on that
    segment should be re-checked against the unspliced window.
    """
    rows = []
    for s in securities:
        chain = PROXY_MAP.get(s.ticker)
        if chain is None:
            continue
        own = total_return_series(s.prices, currency=s.currency, in_cad=True)
        for link in chain.links:
            proxy = _proxy_returns(link)
            if link.vol_match:
                proxy = _vol_match(proxy, own)
            joined = pd.concat([own, proxy], axis=1, keys=["own", "proxy"]).dropna()
            if len(joined) < 60:
                rows.append({"ticker": s.ticker, "proxy": link.ticker, "overlap_days": len(joined),
                             "overlap_corr": np.nan, "tracking_error_ann": np.nan,
                             "vol_ratio": np.nan, "verdict": "INSUFFICIENT OVERLAP"})
                continue
            corr = float(joined["own"].corr(joined["proxy"]))
            te = float((joined["own"] - joined["proxy"]).std() * np.sqrt(252))
            vol_ratio = float(joined["proxy"].std() / joined["own"].std())
            if link.vol_match:
                # A near-riskless sleeve's daily co-movement with its proxy is
                # mostly microstructure noise and is not the thing being
                # borrowed -- the carry path and the risk scale are. Judge it
                # on vol_ratio and on tracking error being small in absolute
                # terms, not on correlation.
                verdict = "rate-proxy ok" if (0.7 <= vol_ratio <= 1.4 and te < 0.02) else "RATE-PROXY OFF"
            elif corr >= 0.95:
                verdict = "strong"
            elif corr >= 0.85:
                verdict = "acceptable"
            else:
                verdict = "WEAK"
            rows.append({
                "ticker": s.ticker, "proxy": link.ticker, "overlap_days": len(joined),
                "overlap_corr": corr, "tracking_error_ann": te, "vol_ratio": vol_ratio,
                "verdict": verdict,
            })
    return pd.DataFrame(rows).set_index(["ticker", "proxy"])
