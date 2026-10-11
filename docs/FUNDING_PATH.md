# Continuous portfolio funding stress

```bash
python -m quant_risk_monitor.funding_path --input examples/funding_path.json --out state/funding-path.json
```

The output is exclusively created and includes the exact input SHA-256. Python APIs
`evaluate_funding_path` and `evaluate_joint_funding` return the same dictionaries.
This is a hypothetical scenario engine, not an execution ledger or live risk gate.

## External native-account interface

`quant-risk-monitor.funding-path/v1` requires a base currency, named `collateral_pool`,
`as_of`, `snapshot_known_at<=as_of`, `source_ref`, opening cash (including posted cash
collateral), cash reserve, unique original positions and an explicit liquidation order.
Each original position has `opening_mark_value`. Negative derivative liabilities are
allowed. All monetary quantities and fractions are finite decimal **strings**.

Strictly increasing scenario steps provide external and financing cash flows, and an
explicit valuation for every original position:

| Field | Meaning, per 100% of the original position |
|---|---|
| `mark_value` | Remaining asset/liability fair value at this step |
| `cash_pnl` | Native cash P&L settled since the preceding step, excluding external flows |
| `initial_margin`, `maintenance_margin` | Required collateral; maintenance cannot exceed initial |
| `eligible_collateral` | Haircut-adjusted noncash collateral; cannot exceed positive mark |
| `liquidation_notional`, `liquidation_cost_rate` | Additional cash liquidation cost = notional × rate |
| `max_liquidation_fraction` | Per-step quantity capacity, fraction of the original position |
| `settlement_lag_steps` | Integer delay until fair-mark liquidation proceeds become cash |

The native producer handles instrument pricing, contract events, futures VM, FX,
margin schedules and liquidity estimates. These must describe one internally consistent
scenario path and one declared fungible collateral pool. Closing a position scales its
later scenario P&L, margins and marks by the remaining fraction. A futures producer
normally supplies zero post-settlement mark and realized VM in `cash_pnl`; supplying the
same P&L in both fields double-counts it and is a producer contract error.

Cash and margin accounting are separate: margin is an encumbrance, not a payment or
expense. Available collateral is cash plus eligible noncash collateral. Maintenance or
cash-reserve breach triggers the explicit liquidation waterfall, aiming to restore
initial margin plus reserve. Each liquidation exchanges removed mark for equal proceeds
and pays additional costs immediately. Removed noncash collateral is subtracted when
calculating margin relief. Capacity applies to original position fractions, capped by
the actual remaining quantity. Receivables/payables pending settlement count in equity,
but cannot cure today's cash or collateral deficit. Unsettled proceeds beyond the
scenario horizon remain explicitly listed.

Every step reports before/after cash and margin gaps, actual simulated liquidations,
remaining fractions, settlement balances, insolvency, and the conservation identity:

`closing equity = previous equity + external flows + financing cash flow + settled native P&L + remaining mark changes - liquidation fees`.

Decimal calculations use 40-digit precision. Required fractional sale sizes round upward
by at most one arithmetic ULP before applying the actual liquidity capacity; this avoids
a rounded-down 1/3 sale falsely leaving a numerical funding gap. Only the conservation check allows numerical
residual `1e-28 * max(abs(expected equity), abs(actual equity), 1)`; reported funding gaps
are not rounded away. The bundled two-step case independently yields equity 90 then 74:
first-day cash 18 plus remaining securities 72, followed by VM -6, security loss -9 and
financing -1. Tests also verify that margin movements are not a second expense.

`quant-risk-monitor.joint-funding/v1` wraps a list `pools` of these paths. Pools must have
unique identities, identical scenario times, and the same reporting currency. It sums
equity and each pool's `max(cash gap, maintenance gap)` independently. Excess liquidity
in one pool cannot offset a shortfall in another. Transfers, if explicitly modeled, must
be supplied as external flows in both pools by the producer; this engine does not infer
or authorize them.

## Limits and evidence

The fixture is synthetic software validation. It does not calibrate real margin or
liquidation schedules. Fractional, proportional liquidation does not model discrete
lots, nonlinear portfolio-margin recomputation, auction priority, unsettled credit risk,
intrastep margin calls, or emergency financing. Use smaller scenario steps and externally
valued consistent paths when those effects matter; do not call this exchange-certified
liquidation. Native options/FX/bonds/swaps lifecycle engines remain external, unchanged.
