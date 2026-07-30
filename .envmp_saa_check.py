import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.universe import build_michael_portfolio
from core.data_loader import get_price_history
from core.risk import var_cvar_summary
from analysis.correlation import bull_bear_correlation_summary, largest_correlation_increases

portfolio = build_michael_portfolio()
for security in portfolio.securities:
    security.fetch_prices(force_refresh=False)

available = [s for s in portfolio.securities if s._prices is not None]
close_matrix = pd.concat({s.ticker: s.close for s in available}, axis=1).dropna()
returns = close_matrix.pct_change().dropna()
benchmark_close = get_price_history('^GSPTSE')['Close'].dropna()

annualized_return = returns.mean() * 252
annualized_vol = returns.std() * np.sqrt(252)
annualized_rf = 0.02
sharpe = (annualized_return - annualized_rf) / annualized_vol

print('ASSET_COUNT', len(annualized_return))
print('MEAN_RET_1', annualized_return.head().to_dict())
print('SHARPE_1', sharpe.head().to_dict())

corr = bull_bear_correlation_summary(returns, benchmark_close, drawdown_threshold=0.10, min_bear_days_warning=60)
print('REGIME', corr['n_bull_days'], corr['n_bear_days'])
print('TOP_RISES\n', largest_correlation_increases(corr, top_n=3))

mean_vec = annualized_return
cov_mat = returns.cov() * 252
n_assets = len(mean_vec)
weights0 = np.ones(n_assets) / n_assets

def portfolio_stats(w):
    w = np.asarray(w, dtype=float)
    exp_ret = float(w @ mean_vec)
    variance = float(w @ cov_mat @ w)
    volatility = float(np.sqrt(variance))
    sharpe = (exp_ret - annualized_rf) / volatility if volatility > 0 else 0.0
    return {'return': exp_ret, 'volatility': volatility, 'variance': variance, 'sharpe': sharpe}

cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
bounds = [(0.0, 1.0)] * n_assets
res = minimize(lambda w: -portfolio_stats(w)['sharpe'], weights0, method='SLSQP', bounds=bounds, constraints=cons)
print('SUCCESS', res.success)
print('MESSAGE', res.message)
print('OPT_WEIGHTS_SUM', res.x.sum())
print('PORT_METRICS', portfolio_stats(res.x))
