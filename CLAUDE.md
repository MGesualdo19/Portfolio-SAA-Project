# CLAUDE.md

Working notes for Claude Code in this repository. Read this before changing
anything in `core/` or `analysis/`.

## What this project is

A strategic asset allocation (SAA) model for a single real personal portfolio,
plus a local dashboard over it. Not a library, not a backtesting framework, not
a trading system. Everything serves one question: **what should this specific
book hold, and why.**

The portfolio is Canadian. Base currency is CAD. Two holdings are USD-listed.

## Run it

```bash
python run_dashboard.py                  # dashboard on http://localhost:8501
jupyter lab notebooks/01_saa_derivation.ipynb
pytest -q                                # tests
```

Dependencies live in `requirements.txt`; the venv is `.venv/`. On Windows use
`.venv/Scripts/python.exe`. Scripts run from the repo root need `PYTHONPATH=.`
because `core` and `analysis` are top-level packages, not installed.

## Architecture, and the one rule that matters

```
core/           analysis/          dashboard/         notebooks/
  data_loader     regime             app.py             01_saa_derivation
  fx              correlation        methodology.py
  returns         stress             theme.py
  proxies         factor_regression  data.py
  estimation                         views/*.py
  optimizer
  policy
  backtest_engine
  tracker
  saa       <-- the orchestrator
```

**One calculation, one home.** `core/saa.py::run_saa()` is the single entry
point that produces an allocation. The notebook calls it. The dashboard calls
it. Neither reimplements any step. If you find yourself writing a covariance
estimate or a weight blend inside `dashboard/` or a notebook cell, it belongs in
`core/` instead — the whole point is that the notebook and the dashboard cannot
disagree about a number.

Dependency direction is one-way: `Security` knows nothing about `Portfolio`;
`analysis/` may import from `core/`; `core/` must not import from `analysis/`
except `core/saa.py`, which is the orchestrator and sits above both.

## Non-negotiables

These are load-bearing. Breaking one silently produces plausible, wrong numbers.

1. **Total returns, never price returns.** `Close.pct_change()` is a bug in this
   repo. `CASH.TO` and `XFR.TO` deliver essentially all of their return as
   distributions; a price-return optimiser refuses to hold them. Use
   `core.returns.total_return_series`.

2. **The FX date-stamp correction stays.** Yahoo stamps `CAD=X` one business day
   ahead of the session it describes. `core/fx.py` shifts it back. Removing that
   line inflates the US sleeve's modelled volatility by ~15% and drops its
   correlation with CAD-converted VTI from 0.91 to 0.80. The evidence is in the
   `core/fx.py` docstring and reproduced in notebook section 1.3.

3. **Portfolio statistics in CAD; per-security statistics in native currency.**
   A security's own volatility is meaningfully described in the currency it is
   quoted in. Its contribution to this book is not.

4. **Never optimise on sample means.** Expected returns come from reverse
   optimisation plus stated Black-Litterman views (`core/estimation.py`,
   `core/policy.py`). `shrunk_historical_returns()` exists only as a
   diagnostic comparison and must not feed a weight.

5. **`returns @ weights` is not a backtest.** It rebalances daily for free.
   Use `core/backtest_engine.py`, which drifts weights, rebalances on a
   schedule, and charges turnover.

6. **Proxy history is an estimate and must stay labelled.** `core/proxies.py`
   returns a source-label frame alongside the returns. Anything that displays
   spliced history shows how much of it is synthetic.

## Where judgement lives vs. where statistics live

`core/policy.py` holds every judgement: the admitted universe, the reference
allocation, the policy bands, the views and their confidences. `core/estimation.py`
and `core/optimizer.py` hold statistics and solvers and contain no opinions about
this portfolio.

If a change is "I think gold should be capped lower", it goes in `policy.py`. If
it is "the covariance estimator should shrink differently", it goes in
`estimation.py`. Keeping these apart is what makes the model arguable.

## Style

- Explain **why**, not what. The existing docstrings carry the reasoning for
  each design choice; match that density. A comment restating the code is noise.
- State assumptions in the docstring of the function that makes them.
- Fail loudly. `core/optimizer.py::_solve` raises rather than returning an equal
  weight labelled as an optimisation result. Preserve that behaviour — a silent
  fallback here becomes a recommendation someone acts on.
- British/American spelling is mixed in the existing prose; do not churn it.
- **Every displayed number needs a methodology entry.** If you add a metric,
  chart or table to the dashboard, add its derivation to
  `dashboard/methodology.py` and wire it with `tip(key)` on the widget and
  `note(key)`/`notes(...)` beneath the chart. A test asserts every entry has a
  formula, steps, a caveat and a named source, and that no view references a key
  that does not exist. The point is that a reader can check any number rather
  than having to trust it.
- Charts follow `dashboard/theme.py`: fixed categorical order never cycled, one
  hue for magnitude, blue-red diverging through neutral grey for polarity,
  status colours reserved. The palette is CVD-validated; do not substitute hues
  without re-validating.

## Testing changes to the model

A change to estimation or optimisation is not verified by "it runs". Check:

```bash
PYTHONPATH=. .venv/Scripts/python.exe -c "
from core.saa import run_saa
r = run_saa(resample=False, verbose=True)
print('effective N', r.summary['effective_n'], 'risk', r.summary['effective_n_risk'])
print('binding', r.binding); print('violations', r.violations)
print((r.final_weights*100).round(1).to_string())"
```

Effective N below ~5, or a long list of binding constraints, means the
estimation layer stopped doing its job and the policy bands are propping the
answer up. That is a regression even if nothing raised.

For the dashboard, `streamlit.testing.v1.AppTest` actually executes the script
and surfaces exceptions; loading the URL does not.

## Known gaps

Do not present these as solved:

- Account tags in `core/universe.py` are placeholders, not confirmed statements.
- No tax modelling (US withholding, capital gains, Norbert's Gambit costs).
- No horizon modelling; one blended-horizon allocation is produced.
- `analysis/pca_overlap.py` is empty. Until it exists, the claim that the value
  tilt reduces AI-capex concentration is argued, not measured.

See `docs/OBJECTIVES.md` for the full scope and roadmap.
