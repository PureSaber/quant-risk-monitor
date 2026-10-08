# quant-risk-monitor

多策略持仓重叠报告已提供`quant-risk portfolio-overlap`：复用 QDK 的 PIT ETF 穿透，显示
直接持仓与 ETF 共同底层证券、净/总敞口、已覆盖/未知/过期/未来/循环/深度限制比例及来源
哈希。只有完整资料才输出有效 overlap ratio；known-only 只作不可用诊断。输入 schema、
合成手算例和风险预测核验摘要见[使用说明](docs/PORTFOLIO_EXPOSURE.md)。

逐期风险预测校验已提供`quant-risk validate-forecasts`和`verify-forecasts`：
按预测期初可得输入估计风险，再对精确匹配的成熟收益评价组合、主动和单资产总风险。
可运行合成示例、公式与证据边界见[使用说明](docs/RISK_FORECAST_VALIDATION.md)。
评分完成不代表校准通过，不改变风险限制或前向账户。

Advanced liquidity checks require non-empty positions, finite market values and positive finite
ADV for every position, with unique symbols. Participation must be in `(0, 1]`; the optional
`max_days_to_exit` must be finite and non-negative. Invalid or incomplete liquidity inputs
replace the report with `evaluation_status: unavailable` and exit code 2. A valid horizon
breach exits 1; a valid passing check exits 0. JSON reports never emit NaN or Infinity.

Point-in-time cross-asset risk policy and portfolio analytics for PureSaber research,
backtesting, and paper trading. The package cannot send live orders:
`CrossAssetRiskPolicy.sends_live_orders` is always `False`.

Version `0.4.0` implements the `quant_execution.PortfolioRiskPolicy` protocol frozen in the
Cross-Asset & Multi-Frequency v2 RFC. Current internal runtime dependencies use immutable commits:

- `quant-data-kit` (`ba136c2fa2eea121bfb2ad7887b536c3952586f7`)
- `quant-execution` (`21aace45d2dde6458db8fcda4829f252f8c640f1`)

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
new exposure. Existing position-derived limit breaches may decline monotonically during an unwind.
Portfolio-level analytics snapshots are not recomputed from an individual proposed order, so an
active analytics breach continues to block `reduce_only` orders until the caller supplies a fresh,
compliant snapshot or applies an explicit recovery policy outside this library.

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
or substitutes a future/latest value. Recency is determined first by `observed_at`; `available_at`
selects the latest revision of the same observation. A late-arriving older observation cannot mask
a fresher value.

Enabled rules fail closed with stable codes. Examples include `MISSING_PIT_PRICE`,
`PIT_PRICE_NOT_AVAILABLE`, `MISSING_PIT_FX`, `PIT_FX_MISMATCH`, `MISSING_PIT_ADV`,
`PIT_ADV_NOT_AVAILABLE`, `MISSING_CLASSIFICATION`, `MISSING_STRATEGY_EXPOSURE`, and
`MISSING_FACTOR_DRIFT`. When a factor-drift threshold is enabled, an empty drift map is rejected.
`required_factor_drift_factors` can declare the required factor universe; every declared factor
must be present, while every supplied factor is checked against the threshold.
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
documented in `configs/advanced.example.yaml`. The CLI exits with code 1 for a critical risk alert.
Invalid or unavailable input produces an `input_data_invalid` critical alert,
`evaluation_status: unavailable`, and exit code 2, replacing any previous output report.
NAV checks require at least two finite, non-negative observations with unique valid dates and
a positive opening value. Invalid NAV rows are never dropped; invalid holdings weights are
never filled with zero. The original holdings CSV contract is long-only: finite non-negative
weights, nonempty symbols, and at least one row. Rows sharing a symbol are aggregated before
concentration checks (leading zeroes are preserved). Existing fraction/relative-weight
normalization is unchanged; explicit zero weights remain valid for a cash-only portfolio.

Incomplete non-zero holdings now fail closed in both covariance and factor-exposure analytics.
Successful advanced checks include explicit input coverage; missing or non-finite exposure/return
history produces a critical `*_not_evaluable` alert rather than a zero-risk estimate. A configured
decision cost limit likewise requires a finite cost estimate. Stress scenarios reject missing or
non-finite returns for every non-zero holding, tail-risk estimators reject infinite or non-numeric
observations, and risk attribution rejects non-PSD covariance matrices.

## Barra-style statistical risk model

The public `fit_barra_style_risk_model` API estimates point-in-time cross-sectional factor returns,
a shrunk PSD factor covariance `F`, shrunk specific variance `D`, and asset covariance
`Σ=XFXᵀ+D`. It reports absolute and benchmark-relative factor exposures, annualized portfolio risk,
tracking error, and factor/specific variance attribution. Inputs carry separate effective,
observation, and availability times; incomplete coverage, immature returns, rank deficiency, and
ill-conditioned exposures stop estimation.

Model matrices are stored privately and public accessors return defensive copies. Before analysis
or serialization, the model verifies label alignment, finiteness, PSD covariance, and the exact
`Σ=XFXᵀ+D` identity; inconsistent state is rejected rather than repaired silently.

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
