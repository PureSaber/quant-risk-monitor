# Integer repair, held-risk coverage and estimator validation

This research extends the recovered-input study on an explicit opt-in branch.
Default v1, raw inputs, benchmarks, investment pools, costs and constraints are
unchanged. It depends on the data-recovery work in PR #34 and the separate
quant-portfolio constraint-directed repair PR.

`repair_risk_coverage.py --held-exposure-carry-sessions 5 --lifecycle-events ...`
creates a new immutable cache. Existing complete exposures, factor covariance
and specific risk are preserved. Known exits receive a specific unavailable
reason; no post-exit price, sellability or risk is fabricated. The evidence
availability timestamp is respected separately from its effective date.

For a currently observed trading asset whose momentum endpoint was a confirmed
suspension, a last published full exposure at most five sessions old may be
used **only as a labelled research estimate outside the investment pool**.
All other required price/fundamental fields must be finite, the industry must
match and every missing quote in the descriptor window must be an evidenced
suspension. Unknown states, unexplained gaps and expired estimates are rejected.
This does not impute prices/returns or change residual-calendar estimation.
The policy is not a promotion of a new default risk model.

The cache binds its input files, lifecycle configuration and policy code.
`study_v3_portfolios.py` verifies those bindings before using it. Rebuild into a
new output directory after policy changes; never relabel a stale cache.

`robustness_risk_validation.py --source <recovered-account-inputs> --out <new-dir>`
refits the original twelve estimator configurations and existing ablations on
identical dated portfolios. It includes independent active, concentrated and
industry directions and omits the scalar-duplicate active-capitalization case.
Only labels matured by the end of 2023 select a candidate. All 2024–2026
results are retrospective; block resampling uses 5,000 draws, ten dates per
block and Bonferroni intervals. There are no new independent forward windows.

`robustness_carry_sensitivity.py --source ... --coverage ... --out ...`
artificially masks five observed assets per date with a fixed seed. The 1/3/5
session stale-exposure comparisons use unchanged returns and covariance. This
is sensitivity evidence, not observed truth for genuinely missing exposures.

`robustness_field_estimator.py --baseline <original-panel.pkl> --source ...
--out ...` crosses original/restored/four leave-one-field-out panels with OLS
and the previously frozen enhanced estimator. It uses a shared causal daily
regression sample and common complete residual histories. It does not select
new parameters or replace the 22-estimator study. Active-cap portfolios are
omitted because they are scalar duplicates of the capitalization direction.

Acceptance separates coverage gains, constraint feasibility, integer search
success, exact ledger validity and risk forecast quality. A lower loss without
adequate calibration/coverage and independent evidence does not authorize model
promotion. The original 80 new mature independent-window requirement remains.

Rollback: disable the optional carry and lifecycle experiment, select the old
frozen package/runtime and input cache, and write a new comparison run. Preserve
all old and new account records; no simulator deployment is part of this PR.
