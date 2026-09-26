# quant-risk-monitor

Point-in-time cross-asset risk policy and portfolio analytics for PureSaber research,
backtesting, and paper trading. The package cannot send live orders:
`CrossAssetRiskPolicy.sends_live_orders` is always `False`.

Version `0.4.0` implements the `quant_execution.PortfolioRiskPolicy` protocol frozen in the
Cross-Asset & Multi-Frequency v2 RFC. Internal runtime dependencies are pinned to released tags:

- `quant-data-kit v0.8.1` (`8f258f11be8e4d8edddcd41b79b817bd6c925970`)
- `quant-execution v0.5.1` (`15e4e5c9dbaf2fe9b438732b2e94db295d5ea58c`)

## Cross-asset policy

`CrossAssetRiskPolicy` is immutable and composes through the real QExec gate:

```python
from quant_execution import RuleBookRiskGate
from quant_risk_monitor import (
    CrossAssetRiskLimits,
    CrossAssetRiskPolicy,
    PITRiskInputs,
)

policy = CrossAssetRiskPolicy(
    instruments=instrument_registry,
    limits=CrossAssetRiskLimits(
        max_gross_leverage="2.0",
        max_abs_net_leverage="1.0",
        max_instrument_concentration="0.20",
        max_margin_utilization="0.50",
        max_adv_participation="0.10",
        max_days_to_liquidate="5",
    ),
    inputs=PITRiskInputs(prices=price_observations),
)
gate = RuleBookRiskGate(instruments=instrument_registry, ledger=ledger, policies=[policy])
```

Order checks combine QExec's signed `projected_notional_base` with the current
`PortfolioRiskSnapshot`, so every decision is based on the proposed post-order portfolio.
`reduce_only` projections can only reduce the current absolute position and cannot be counted as
new exposure.

Supported controls:

- gross and absolute net leverage;
- instrument, asset-class, settlement-currency, venue, and current-strategy concentration;
- initial margin, maintenance margin, and initial-margin/NAV utilization;
- order ADV participation and position days-to-liquidate;
- additive instrument, asset-class, currency, and venue stress scenarios;
- historical/parametric VaR-CVaR and factor-exposure drift thresholds.

All money, quantity, price, margin, ADV, FX, and analytics inputs crossing the QExec boundary use
`FixedPoint`. Limits and stress shocks are normalized to exact `Decimal` values.

## Target-portfolio decision gate

`DecisionPortfolioLimits` and `check_decision_portfolio` validate a long-only target before a
strategy emits order intents. The pure check covers single-name and total invested weight, minimum
cash, number of positions, traded-value turnover, industry concentration, and estimated cost/NAV.
When industry risk is enabled, every target must have an explicit classification. Results include
machine-readable metrics and stable critical alert codes for dashboards and unattended workflows.

## PIT input contract

`PriceObservation`, `FxRateObservation`, `LiquidityObservation`,
`StrategyExposureSnapshot`, and `AnalyticsRiskSnapshot` carry both `observed_at` and
`available_at`. The policy selects the latest causally available value and never reads the network
or substitutes a future/latest value.

Enabled rules fail closed with stable codes. Examples include `MISSING_PIT_PRICE`,
`PIT_PRICE_NOT_AVAILABLE`, `MISSING_PIT_FX`, `PIT_FX_MISMATCH`, `MISSING_PIT_ADV`,
`PIT_ADV_NOT_AVAILABLE`, `MISSING_CLASSIFICATION`, and `MISSING_STRATEGY_EXPOSURE`.
QExec remains the source of truth for positions, marks, FX conversion, margin, Funding,
Settlement, and NAV; the policy verifies supplied PIT values against the QExec snapshot.

QExec v0.4 does not place strategy attribution in `PortfolioRiskSnapshot`. Therefore current
strategy concentration is supplied as an immutable `StrategyExposureSnapshot`, produced at or
before the decision time. Missing strategy attribution is rejected instead of treated as zero.

## Existing analytics and CLI

The original CSV-based CLI remains compatible:

```bash
python -m pip install --no-deps -r requirements.lock
python -m pip check
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
quant-risk check --config configs/default.yaml --out state/alerts.json
```

Existing drawdown, daily-loss, single-name, stress, liquidity, covariance/risk-contribution, and
factor-exposure outputs are preserved. Tail-risk output now also includes parametric VaR-CVaR;
optional `parametric_var_limit` and `parametric_cvar_limit` rules add critical alerts without
changing the historical `var_limit` and `cvar_limit` semantics. Advanced CSV inputs remain
documented in `configs/advanced.example.yaml`. The CLI exits with code 1 for any critical alert.

Incomplete non-zero holdings now fail closed in both covariance and factor-exposure analytics.
Successful advanced checks include explicit input coverage; missing or non-finite exposure/return
history produces a critical `*_not_evaluable` alert rather than a zero-risk estimate. A configured
decision cost limit likewise requires a finite cost estimate.

## Barra-style statistical risk model

The public `fit_barra_style_risk_model` API estimates point-in-time cross-sectional factor returns,
a shrunk PSD factor covariance `F`, shrunk specific variance `D`, and asset covariance
`Σ=XFXᵀ+D`. It reports absolute and benchmark-relative factor exposures, annualized portfolio risk,
tracking error, and factor/specific variance attribution. Inputs carry separate effective,
observation, and availability times; incomplete coverage, immature returns, rank deficiency, and
ill-conditioned exposures stop estimation.

This is an independently implemented Barra-style linear risk model, not an MSCI Barra model.
Return-derived or statistical exposures must use `model_kind="statistical_proxy"`; this label is
preserved in model snapshots, diagnostics, and portfolio reports. The library never invents a
market factor or other missing descriptor. See
[`docs/BARRA_STYLE_FACTOR_MODEL.md`](docs/BARRA_STYLE_FACTOR_MODEL.md) and run
`python examples/barra_style_proxy.py` for the complete contract and example.

## Quality gates

```bash
ruff check src tests
ruff format --check src tests
pytest -q --cov=quant_risk_monitor --cov-branch --cov-fail-under=80
```

CI runs Python 3.10, 3.11, and 3.12. Full-project coverage must be at least 80%, and
`cross_asset.py` branch coverage must be at least 90%. Integration tests use the real
`RuleBookRiskGate` and `DeterministicRunEngine` to verify rejected order events, runtime risk
events after Funding/Settlement/mark changes, boundary acceptance, and three-run determinism.

## M6 governance and rollback

`[tool.quant-workspace]` declares the real QDK `puresaber.instrument-spec` and QExec
`puresaber.execution.account-snapshot`/`puresaber.execution.order-intent` contracts consumed by
the policy. The risk monitor is strictly a QExec gate plugin: it can reject an intent but cannot
amend the ledger, position, mark, FX, margin, Funding, Settlement, or NAV source of truth.

`requirements.lock` is the only audited Python3.10-3.12 lock for runtime, development, and
editable-build dependencies. Rebuild it from Python3.10 only:

```bash
python -m piptools compile --extra dev --build-deps-for editable --allow-unsafe --strip-extras \
  --resolver backtracking --index-url https://pypi.org/simple \
  --constraint requirements-constraints.txt --output-file requirements.lock pyproject.toml
```

After a rebuild, run the locked install commands above, both Ruff commands, the full test suite,
and the QExec integration/determinism tests on Python3.10,3.11,and3.12. Roll back this governance
change with `git revert` so `pyproject.toml`, constraints, and the lock return together. Existing
tags and immutable run artifacts must never be moved, deleted, or recreated.
