"""
tests/test_security.py

Basic construction/validation tests. Doesn't hit the network (no price
fetching tested here) - just checks the OOP skeleton holds together.
"""

import pytest

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
