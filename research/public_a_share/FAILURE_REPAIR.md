# Barra 200 failure-repair campaign

This campaign preserves the frozen 200-company sample, original capitalization
benchmark, 5% cap, 95%/80% investment, subsequent 20% turnover and 3%/6% annual TE.
Existing `runs/main` is read-only. No default v1 model, frozen account or strategy
promotion changes. The original 80 independent matured-window gate still applies.

## Coverage contract

`repair_risk_coverage.py --source OLD --out NEW_COVERAGE` keeps the original entry
assets (investment pool), benchmark, complete-asset risk estimates and factor
covariances. It extends only the risk universe for potential held assets. A
calendar-aware 252-session specific-risk window requires 126 real observations
and effective sample size 20. Decay uses actual calendar gaps; HAC uses only real
calendar pairs. Original shrinkage and regime parameters are retained.

Current reliable exposure is required. A confirmed suspension may carry its last
observed exposure as an explicitly labelled stale estimate, with last valuation
and exposure dates. Unknown trading state/missing exposure is not a zero return
or an industry proxy. No industry-proxy sensitivity is enabled in this campaign.
Reliable coverage and predictive quality are separate acceptance questions.

The cache binds frozen inputs, producer/estimator code and output digests. It is
a separate `calendar-partial-v1` research model, never a default-model replacement.

## Independent failure evidence

`repair_failure_audit.py --source OLD --out AUDIT` reconstructs all 102 previous
repair failures from exact native account snapshots, holdings and fills, and
calls the production execution adapter. The output is a counterfactual target
audit of each frozen account, not a newly evolved strategy. It saves the entire
problem, constraints, continuous target, fees and search trace. Source/input
fingerprints make the cases independently reproducible.

`repair_feasibility_audit.py --source OLD --out CERTIFICATES` independently solves
2,184 budget/model/date TE floors. Convex supporting planes minimized analytically
over the capped simplex give actual lower-bound certificates, rather than treating
a feasible candidate as proof of the optimum. The same construction with the
turnover objective certifies the 14 account-dependent conflicts. No full asset
covariance is constructed. These relaxed diagnostics cannot publish order targets.

## Corporate actions

`failure_repair_configs` contains strict and separate floor/ceiling sensitivity
configurations. For 600763, the source ratio is 0.400595 added share and 0.43565
cash per eligible share; public fractional ordering cannot identify the virtual
account's exact registered allocation. For 603007, the final reorganisation plan
(`1221743709`, pages 13 and 26) explicitly drops fractional shares for ordinary
holders, whose disclosed approximate ratio is 0.155346 added share. The issuer's
total capital expansion is not this holder entitlement. The ceiling branch is
therefore never applied to 603007.

Strict runs preserve unresolved rights. Sensitivity runs explicitly assume the
announced delivery/listing day and permitted fractional result. Later annual-report
confirmation is audit evidence only. Exact registered amounts are not invented.

## Replays and restoration

Use `study_v3_batch.py --source OLD --out NEW --coverage NEW_COVERAGE
--share-rules CONFIG --workers 4 --batch-id VERSION`. Each scenario owns one native
ledger and immutable identity. Numerical modules are hashed and cannot change
during a run. Retain failure logs and use a new version after any correction.

Report full-horizon completion separately from effective rebalance rate, longest
no-target streak, held/frozen intervals, coverage, residuals and costs. Cross-predict
every model's actual held account even on failed allocation dates. Cash stays in
active-risk calculations. Strict and sensitivity results must never be pooled.

Restore by selecting the original software lock and immutable source run. Do not
overwrite new ledgers or replay corrected actions into an already credited account;
restore a checkpoint before the event and create a separately identified replay.
Keep raw input hashes and a same-machine cross-drive recovery copy. Such a copy
is not an independent-hardware backup. D/F/C storage budgets remain 25/20/5 GiB.
