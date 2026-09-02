# Project objectives

## Purpose

Derive and maintain a defensible strategic asset allocation for one real
Canadian personal portfolio, and make the reasoning inspectable enough that any
individual weight can be argued with.

The portfolio thesis (`core/Thesis.txt`) states a view: rate hikes through 2027,
a value tilt to avoid concentration in AI-linked mega-caps, gold as protection
against a geopolitical or AI-capex-driven drawdown. The objective is not to
prove that thesis right. It is to translate it into an allocation, measure which
parts of it the data supports, and say plainly which parts it cannot test.

## Success criteria

The model is doing its job when all of these hold.

**1. The output is a portfolio, not a corner solution.**
The specific failure this project was built to avoid: a mean-variance optimiser
handed sample means and a sample covariance matrix returns two or three
holdings. Measured by *effective N* — the inverse Herfindahl of the weights, the
number of equally weighted positions the allocation is equivalent to.

| Configuration | Effective N | Positions | Largest weight |
|---|---|---|---|
| Textbook: sample means + sample covariance | 2.47 | 3 | 46.9% |
| + Ledoit-Wolf shrinkage | 2.47 | 3 | 46.8% |
| + equilibrium/Black-Litterman returns | 6.15 | 9 | 31.2% |
| + policy constraints | 7.40 | 10 | 25.0% |
| + objective blend and resampling | **8.07** | **10** | **21.0%** |

Target: effective N above 7 with no more than two binding constraints. The
second row matters as much as the last — it shows the problem was the mean
vector, not the covariance matrix, which is why constraints alone would have
been the wrong fix.

**2. Diversification is judged on drawdown behaviour, not averages.**
Correlations converge in crashes. An allocation optimised on full-sample
covariance buys diversification that evaporates when it is needed. The
covariance matrix actually optimised against is a blend of full-sample and
bear-regime estimates, and every diversification claim in the thesis is tested
conditionally (`analysis/stress.py`).

**3. Portfolio-level figures are in CAD; per-security figures are native.**
USD holdings stay quoted in USD when examined individually. Everything that
aggregates — covariance, VaR, backtest NAV, drawdown — is CAD, because that is
the currency this book is spent in.

**4. Every assumption is stated where it is made and surfaced in the output.**
Proxy-filled history, view confidences, policy bands, the FX correction, the
placeholder account tags. A number without its assumption attached is not a
result.

**5. The method survives out of sample.**
The whole pipeline refits on an expanding window and is held forward for a year
at a time. Out-of-sample Sharpe should track in-sample, and the refit weights
should be stable — a process that reshuffles the book annually is fitting noise.

## Scope

### In scope

- Strategic weights across the ten admitted holdings, CAD-denominated.
- Regime-conditional risk analysis: bear correlations, downside beta, tail
  dependence, dated crisis windows.
- Backward-looking backtest with realistic drift, rebalancing, turnover cost and
  fee drag; plus walk-forward out-of-sample validation.
- Forward tracking of the funded book against a bootstrapped expectation cone,
  with drift-band rebalancing signals.
- Per-security investigation: total return, tail risk by three methods, rolling
  risk-adjusted return, factor exposure, behaviour in each crisis window.
- Position sizing to whole shares, in each security's own currency.
- A desktop application: native window, no browser, engine supervised as a child
  process and killed with the window. Shortcuts on the Desktop and Start menu.
- Self-documenting output: every displayed number carries its formula, inputs,
  derivation steps, caveat and source module, in a tooltip and in a full
  expander beside the chart.
- A live critical evaluation of the model's own output, with ranked
  recommendations separating evidenced actions from decisions only the investor
  can make.

### Explicitly out of scope

- **Tactical allocation and market timing.** This is a strategic model. It has
  no view on what to do this month.
- **Security selection.** The universe comes from the thesis. The model sizes
  it; it does not search for replacements.
- **Order execution.** The dashboard produces a ticket. Nothing places trades.
- **Tax optimisation.** See gaps below.

## Current allocation

Ten holdings. Effective N 8.07 on capital, 7.29 on risk. Blended MER 0.185%.

| Sleeve | Weight | Holdings |
|---|---|---|
| Cash & floating rate | 34.5% | XFR.TO, CASH.TO |
| US equity | 23.0% | XUU.TO, VTV, AVUV |
| Real assets | 14.0% | CGL.TO |
| Canadian equity | 11.5% | XIC.TO |
| International developed | 8.0% | VIU.TO |
| Emerging markets | 4.5% | VEE.TO |
| Legacy single name | 4.5% | CAR-UN.TO |

Backtest 2007–2026, quarterly rebalanced, net of fees and 8bps turnover cost:
5.48%/yr at 9.21% volatility, maximum drawdown -24.2%, against -35.4% for the
reference policy portfolio and -55.0% for an all-equity book. GFC loss -16.7%
versus -53.1% all-equity; COVID -17.2% versus -31.1%.

## Verdicts on the thesis

| Claim | Verdict |
|---|---|
| Rate hikes through 2027 → hold floating rate and cash | **Instrument supported.** XFR.TO returned +1.0% through the 2022 rate shock while Canadian equity returned -15.0%. The rate *forecast* itself is untested and the allocation does not depend on it. |
| Value tilt offsets AI-capex concentration | **Partially supported.** VTV correlates 0.79 to XUU.TO in calm markets and 0.96 in bear regimes. It diversifies at the margin, not structurally. Quantifying the AI-capex claim needs fund-holdings overlap, which is not yet built. |
| Gold as tail hedge | **Supported.** +0.04 correlation to core US equity in calm markets, +0.07 in its worst decile — no convergence in drawdowns. Caveat: gold fell with everything else in the 2013 taper tantrum, so it hedges growth and credit shocks better than pure rate shocks. |
| VOLX as volatility hedge | **Remove.** -48%/yr, -99.998% maximum drawdown, -99.996% cumulative. Structural contango decay, not a bad run. Gold occupies this slot at a fraction of the cost. |
| CAPREIT | **Cap at 5%.** Genuinely low correlation to the rest of the book, but single-name, leveraged, rate-sensitive against a hiking thesis. Zero is an acceptable answer. |
| Home bias / tax treatment | **Untestable as configured.** Account tags are placeholders. |

## Critical evaluation

These are the results of deliberately trying to break the model. All of them run
live in the dashboard's **Evaluation** tab against the current settings.

### It passes the test that matters

Most of the recommended book's lower drawdown could be explained by simply
holding less equity — which needs no model. So the naive alternatives were
de-risked with cash until their realised volatility matched the recommendation's
exactly:

| At an identical 9.21% volatility | Annual return | Sharpe | Max drawdown |
|---|---|---|---|
| **Recommended SAA** | **5.48%** | **0.30** | **-24.2%** |
| Reference policy, de-risked (77% risky + 23% cash) | 4.79% | 0.22 | -27.7% |
| Equal weight, de-risked (77% risky + 23% cash) | 4.95% | 0.24 | -27.4% |

The composition, not just the equity level, is doing real work — roughly 60bps a
year and 3 points of drawdown at equal risk.

### Weakness 1 — the thesis is diluted roughly five to one

Only one of the five objectives (maximum Sharpe) uses expected returns at all.
Switching every Black-Litterman view on moves that objective's weights by up to
**10 percentage points**, but the final blend by only **2**.

This is the central trade-off in the design, and it cuts both ways: blending is
what makes the allocation stable out of sample, and it is also what makes it
nearly thesis-blind. In practice the *policy floors* express the thesis more
forcefully than the views do — the 8% floor on the US value sleeve and the 10%
floor on international equity fund the concentration argument regardless of what
the views say. If the intent is for the rate view and the value tilt to drive
the book, the fix is to weight maximum Sharpe more heavily and accept the loss
of robustness.

### Weakness 2 — "no binding constraints" is a misleading statistic

The blend rests on no constraint. But **12 of 50 (objective × holding) cells sit
exactly on a bound — 24%.** Averaging five constrained corner solutions produces
a smooth interior point that hides the pinning underneath.

The clearest case is gold. Its 15% ceiling binds in four of the five objectives;
with the cap removed, maximum diversification wants **27.8%**. The 14% in the
recommendation is a *policy ceiling*, not a data result. CAPREIT is the same
story at a smaller scale — uncapped, two objectives want 10-12% against its 5%
cap.

### Weakness 3 — the headline diversification figure flatters

Effective holdings of 8.07 sounds well spread. Underneath:

| Sleeve | Capital | Risk |
|---|---|---|
| Equity | 47.0% | **84.0%** |
| Cash & floating rate | 34.5% | 0.6% |
| Gold | 14.0% | 10.7% |
| Legacy REIT | 4.5% | 4.7% |

This is an equity portfolio with a large cash buffer, and its outcome will be
decided by equities. Within the equity sleeve the effective count is 5.45 of 6,
which is genuinely good — but the diversification that matters is happening in a
sleeve that is under half the book.

### Weakness 4 — the parameters barely matter

No sidebar setting moves any weight by more than 2.5pp, and swapping the
reference portfolio for a wildly different one (equity-heavy vs defensive) moves
nothing by more than 1.7pp. The allocation is therefore determined almost
entirely by the covariance matrix and the constraint set. Robust, but it means
the judgement inputs are doing less than their prominence in the code suggests.

### Weakness 5 — a plausible world where this is wrong

- **A long horizon.** At 34.5% defensive this is sized for a blended medium
  horizon. All-equity earned 0.85pp/yr more over the backtest window; over thirty
  years that compounds to a large forgone sum.
- **The 2012-2026 sub-period.** Plain 60/40 beat the recommendation on Sharpe
  (0.71 vs 0.65) over that window. The recommendation wins over the full window
  because it survives 2008. If the next twenty years look more like 2012-2026,
  the machinery is not earning its keep.
- **Gold at 14%** is well above a conventional 5-10%. A repeat of 2012-2018
  (gold roughly -40% peak to trough while equities compounded) would make it the
  costliest position in the book.
- **CAPREIT** carries idiosyncratic single-name risk that no statistic here
  would have warned about.

### Incidental finding: quarterly rebalancing is the wrong default

| Rebalancing | Return | Sharpe | Max DD | Turnover | Total cost |
|---|---|---|---|---|---|
| Buy and hold | 5.69% | 0.31 | -21.4% | 0% | 0.00% |
| **Annual** | **5.61%** | **0.32** | -22.9% | 4.8%/yr | 0.07% |
| 20% drift band | 5.64% | 0.31 | -24.3% | 5.0%/yr | 0.07% |
| Quarterly | 5.48% | 0.30 | -24.2% | 8.0%/yr | 0.12% |
| Monthly | 5.46% | 0.29 | -24.9% | 13.6%/yr | 0.20% |

More frequent rebalancing bought turnover cost and a slightly *deeper* drawdown,
not a better return — rebalancing into equities mid-drawdown adds risk exactly
when it is least wanted. The dashboard default is now annual.

## Recommendations

These live in the app's **Evaluation** tab, section 6, computed against the
current settings. Summarised here for the record.

**Evidenced actions**
1. **Sell VOLX.TO.** -99.996% cumulative, structural contango decay. Gold holds
   the tail-hedge slot at a fraction of the cost. Only open question is tax
   timing.
2. **Rebalance annually, not quarterly.** Annual returned 5.61% at a 0.32 Sharpe
   on 4.8%/yr turnover; quarterly returned 5.48% at 0.30 on 8.0%. A 20% drift
   band performs almost identically to annual without a calendar.
3. **Confirm the account placements before trading.** US-listed VTV and AVUV
   belong in the RRSP; holding them in a TFSA forfeits the 15% treaty
   withholding relief permanently.

**Decisions only the investor can make**
4. **Gold at 14% is a policy ceiling, not a data result** — uncapped, maximum
   diversification wants 27.8%. If that is uncomfortable, lower the cap in
   `core/policy.py` and re-run rather than overriding the output.
5. **The 34.5% defensive sleeve depends entirely on horizon.** All-equity earned
   0.85pp/yr more over the backtest window. If the money is untouchable for
   decades, move the fixed-income band from 15-35% toward 10-20%.

**Worth considering**
6. Weight max-Sharpe more heavily in `blend_strategies()` if the thesis should
   actually steer the book — at the cost of out-of-sample stability.
7. Build `analysis/pca_overlap.py` to settle the AI-capex concentration claim.
8. Decide whether CAPREIT belongs in the book at all; zero is acceptable.

## Known gaps, in priority order

1. **Confirm account placements.** The RRSP/TFSA/FHSA tags are placeholders
   encoding what the thesis implies rather than actual statements. No weight
   depends on them, but after-tax return does. Needs statements, not code.
2. **Fund-holdings overlap** (`analysis/pca_overlap.py`, still empty). Until it
   exists, the concentration claim at the centre of the thesis is argued rather
   than measured.
3. **Tax modelling.** US dividend withholding, capital-gains treatment in the
   non-registered account, Norbert's Gambit costs. Pre-tax and after-tax optimal
   allocations genuinely differ for a book split across registered and taxable
   accounts.
4. **Horizon modelling.** One allocation is produced, but an FHSA funding a home
   purchase in three years and an RRSP funding retirement in thirty are not the
   same problem. The 34% defensive sleeve is sized for a blended medium horizon.
5. **Contribution scheduling.** The tracker records a single funding event.
   Regular contributions are a natural extension and would change the
   rebalancing logic.

## Standing constraints on any change

- One calculation, one home — see `CLAUDE.md`.
- Judgement lives in `core/policy.py`; statistics live in `core/estimation.py`
  and `core/optimizer.py`. Keeping them apart is what makes the model arguable.
- No silent fallbacks in the optimiser. A failed solve raises.
- Assumptions ship with the number that depends on them.
