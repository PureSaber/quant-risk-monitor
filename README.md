# quant-risk-monitor

Point-in-time cross-asset risk policy and portfolio analytics for PureSaber research,
backtesting, and paper trading. The package cannot send live orders:
`CrossAssetRiskPolicy.sends_live_orders` is always `False`.

Version `0.3.0` implements the `quant_execution.PortfolioRiskPolicy` protocol frozen in the
Cross-Asset & Multi-Frequency v2 RFC. Internal runtime dependencies are pinned to released tags:

- `quant-data-kit v0.6.0` (`960db6d7f30eae942efd46a0dab7596585277823`)
- `quant-execution v0.4.0` (`9d8b3b8a9dfd04873af5ae8c16338f1eac5b492a`)

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
pip install -e ".[dev]"
quant-risk check --config configs/default.yaml --out state/alerts.json
```

Existing drawdown, daily-loss, single-name, stress, liquidity, covariance/risk-contribution, and
factor-exposure outputs are preserved. Tail-risk output now also includes parametric VaR-CVaR;
optional `parametric_var_limit` and `parametric_cvar_limit` rules add critical alerts without
changing the historical `var_limit` and `cvar_limit` semantics. Advanced CSV inputs remain
documented in `configs/advanced.example.yaml`. The CLI exits with code 1 for any critical alert.

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
