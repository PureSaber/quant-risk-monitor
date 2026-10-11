# Chronological model diagnostics

```sh
python examples/model_diagnostics_demo.py state/diagnostic-inputs
quant-risk diagnose-model --input state/diagnostic-inputs/change.json --out state/change-report
quant-risk diagnose-model --input state/diagnostic-inputs/null.json --out state/null-report
```

The API is `quant_risk_monitor.model_diagnostics.diagnose_models(payload)`. CLI
outputs include the immutable-by-exclusive-create input/result files and a readable
domain status summary. The result binds canonical plan/input SHA256 and actual
implementation source SHA256. No network or model call is made.

`quant.model-diagnostics-input/v1` declares evidence kind, as-of, a closed
`quant.model-diagnostic-plan/v1`, and chronological nonoverlapping observations.
Each row has an ID, source SHA256, forecast/start/end/outcome-known timestamps and
regime ID. See the executable example for the full schema. Currency, decimal
returns and cost basis points are explicit. A score report is not a portfolio
return series and cannot be ranked under an economic measurement contract.

The plan fixes a reference window, subsequent disjoint window sizes, block size,
minimum complete blocks, total number of monitoring windows, bounds, thresholds
and family alpha. It must be declared before the reference forecasts. Source
timestamps and preregistration are producer assertions, not independent proof.
Unobserved future outcomes are withheld. Availability gaps, horizon exhaustion,
nonfinite/out-of-bound observations and unknown fields fail closed.

Seven separate, explicitly labeled diagnostics:

| Domain | Statistic (positive generally indicates deterioration) |
| --- | --- |
| Data drift | Single bounded feature mean shift, either direction |
| Forecast error | Absolute regression error / (2 × forecast bound) |
| Signal decay | Increase in direction-classification error; zero is its own class |
| Risk miscalibration | Absolute-return threshold exceedance rate minus declared probability |
| Cost miscalibration | (actual − predicted cost bps) / fixed error bound |
| Strategy performance | (baseline − net return) / fixed active-return bound |
| Operational failures | Supplied non-ok execution/operation rate |

These are limited diagnostics, not a generic distribution-shift detector, a
correlation/IC test, strategy invalidation, or broker execution certification.
Classification mistakes can alert while strategy performance remains unchanged.
`strategy_invalid` is always unknown and `execution_authorized` is always false.

Block means receive bounded Hoeffding intervals. Bonferroni allocates family alpha
over every planned domain/window and reference/current interval. Coverage requires
independent bounded blocks; dependence within a block is allowed. Accepting that
assumption is explicit. Without it, output is descriptive only. No automatic
choice of block size or data-driven threshold fitting is performed. Wide intervals
yield `no_clear_breach`, which is not evidence that a model is healthy.

Any missing observation in a domain yields `partial_observation` and suppresses
full-window inference, even if many complete blocks remain. Partial windows or
too few blocks are `insufficient_sample`. A declared regime change invalidates
comparison to the old reference and yields `structural_change`; it is not a
statistical failure attribution. Bounds are never silently clipped. A new plan is
a new study, not a retroactive repair of the old result.

The deterministic null and change fixtures have independently specified means and
interval bounds. Tests cover selective deletion of bad observations, dependence,
regime changes, future-outcome perturbation and classification/performance
separation. These demonstrate software behavior, not real-market false-positive
frequency or economic effectiveness. Reports never authorize trading, retraining,
account recovery or revised thresholds.
