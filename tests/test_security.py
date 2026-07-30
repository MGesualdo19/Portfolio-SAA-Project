"""
tests/test_security.py

Basic construction/validation tests. Doesn't hit the network (no price
fetching tested here) - just checks the OOP skeleton holds together.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.correlation import bull_bear_correlation_summary
from core.account_profile import AccountProfile
from core.portfolio import Portfolio
from core.security import Security


def test_security_construction_valid():
    s = Security(
        ticker="XIC.TO",
        name="iShares Core S&P/TSX Capped Composite Index ETF",
        asset_class="Equity - Canada",
        currency="CAD",
        mer=0.0006,
        fund_site_url="https://example.com",
        account_tag="TFSA",
    )
    assert s.ticker == "XIC.TO"
    assert s.weight is None


def test_security_rejects_invalid_asset_class():
    with pytest.raises(ValueError):
        Security(
            ticker="FAKE",
            name="Fake",
            asset_class="Not A Real Class",
            currency="CAD",
            mer=0.001,
            fund_site_url="",
            account_tag="TFSA",
        )


def test_legacy_asset_class_names_are_accepted_for_thesis_compatibility():
    legacy_equity = Security(
        ticker="CAR-UN.TO",
        name="CAPREIT",
        asset_class="Legacy - Single Name Equity",
        currency="CAD",
        mer=0.0,
        fund_site_url="",
        account_tag="NON_REG",
    )
    legacy_vol = Security(
        ticker="VOLX.TO",
        name="BetaPro VIX",
        asset_class="Legacy - Volatility",
        currency="CAD",
        mer=0.0118,
        fund_site_url="",
        account_tag="NON_REG",
    )
    assert legacy_equity.asset_class == "Legacy - Single Name Equity"
    assert legacy_vol.asset_class == "Legacy - Volatility"


def test_security_rejects_invalid_currency():
    with pytest.raises(ValueError):
        Security(
            ticker="FAKE",
            name="Fake",
            asset_class="Equity - Canada",
            currency="EUR",
            mer=0.001,
            fund_site_url="",
            account_tag="TFSA",
        )


def test_account_profile_rejects_invalid_type():
    with pytest.raises(ValueError):
        AccountProfile(tag="X", name="Bad", account_type="NOT_REAL")


def test_portfolio_by_account_joins_profile():
    portfolio = Portfolio()
    profile = AccountProfile(tag="RRSP", name="Test RRSP", account_type="RRSP", us_withholding_exempt=True)
    portfolio.add_account_profile(profile)

    sec = Security(
        ticker="VTV",
        name="Vanguard Value ETF",
        asset_class="Equity - US Value",
        currency="USD",
        mer=0.0004,
        fund_site_url="",
        account_tag="RRSP",
    )
    portfolio.add_security(sec)

    secs, matched_profile = portfolio.by_account("RRSP")
    assert secs == [sec]
    assert matched_profile is profile


def test_portfolio_flags_unregistered_account_tags():
    portfolio = Portfolio()
    sec = Security(
        ticker="VTV",
        name="Vanguard Value ETF",
        asset_class="Equity - US Value",
        currency="USD",
        mer=0.0004,
        fund_site_url="",
        account_tag="RRSP",
    )
    portfolio.add_security(sec)
    assert portfolio.unregistered_account_tags() == {"RRSP"}


def test_weight_check_reports_unset_weights(capsys):
    portfolio = Portfolio()
    sec = Security(
        ticker="VTV",
        name="Vanguard Value ETF",
        asset_class="Equity - US Value",
        currency="USD",
        mer=0.0004,
        fund_site_url="",
        account_tag="RRSP",
    )
    portfolio.add_security(sec)
    portfolio.weight_check()
    captured = capsys.readouterr()
    assert "no weight set" in captured.out


def test_correlation_module_imports_and_runs_with_regime_helper():
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.00, -0.03, 0.02, 0.01],
            "B": [0.00, -0.01, 0.02, 0.00, -0.02, 0.01, -0.01, -0.02, 0.03, 0.02],
        },
        index=idx,
    )
    benchmark = pd.Series([100, 101, 102, 101, 99, 95, 90, 85, 82, 80], index=idx)

    summary = bull_bear_correlation_summary(returns, benchmark, drawdown_threshold=0.10)
    assert set(summary.keys()) >= {"full", "bull", "bear", "bear_minus_bull"}
    assert summary["full"].shape == (2, 2)
