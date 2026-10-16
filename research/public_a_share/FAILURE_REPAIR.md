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

Longer sensitivity replays exposed another fractional distribution: 001210 on
2026-06-05. Its 2025 and 2026 implementation notices confirm additional ratios
0.2 and 0.4, with largest-tail allocation and random ties, credited and tradable
on 2025-06-09 and 2026-06-05 respectively. The separately identified `extended`
configurations retain every earlier rule and add only these evidenced endpoints.
They do not overwrite the base branch's safe stop or infer registrar facts.
Document hashes, availability times and collection time are in
`failure_repair_configs/001210-evidence-v1.json`. Final report acceptance requires
an extended receipt for every base sensitivity account stopped on this asset.

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

## Repaired ordinary environment

Install `requirements-failure-repair.lock` in a new Python 3.10–3.12 environment;
never modify the original v3 environment. Verify `pip check`, installed Git commit
IDs, and that imports come from that environment's `site-packages` (no editables).
Set `QUANT_RISK_SOURCE` to the installed `quant_risk_monitor` directory when using
the standalone research scripts: their legacy bootstrap otherwise seeks an old
checkout. Set OPENBLAS/OMP/MKL thread counts to one before launching Python.

The fixed 102 historical accounts now yield 65 independently verified executable
targets and 37 bounded-search failures. Both 2024-01-05 witnesses (41/46) pass.
These are isolated original-account cases, not a new strategy performance path.
The unchanged input certificates confirm 5,551 structural and 14 turnover
conflicts; 26 scenarios remain infeasible across every original decision date.

`repair_evidence.py` checks all 2,893 original missing-union cases and the 19
corporate-action stops. New partial histories fully cover 1,549 original unions;
1,344 still lack reliable current exposures. Missing fields and unknown trading
states are retained by asset/date; coverage improvement is not model promotion.
All 19 original corporate-action accounts still have unresolved exact rights in
the strict evidence policy. Public final-plan rounding rules inform only the
labelled sensitivity assumptions when actual registry quantities are absent.

`repair_report.py` verifies native exact journals, independently recalculates
fee quotes, shares, T+1, cash, post-fee weights and factor-form risk. It produces
per-branch full account comparisons and cross predictions; it refuses to mark
delivery complete until all 62 strict scenarios and the required sensitivity
scenarios have receipts. Use a new output directory for every report version.
The current source contains the report pipeline; final full replay acceptance
must be read from its generated `acceptance.json`, not inferred from unit tests.

Large decision artifacts use deterministic gzip JSON in new sensitivity runs;
readers accept both old plain JSON and compressed JSON, rejecting ambiguous or
corrupt copies. This preserves every failure input/trace without consuming the
whole storage budget. A six-session historical native-account parity run checks
identical weights, integer targets, fees and exact economic ledger fields.
The strict runtime remains immutable; the compression-only runtime is identified
separately. Recovery verifies every decoded archived payload and materializes
representative accounts plus their inputs, with explicit counts rather than
claiming all files were restored to disk.

After the restored six-account audit passes, `repair_finalize_recovery.py` seals
its receipts, final release record and report QA in a separate immutable
`final-evidence/manifest.json`. It checks the base-manifest and audit bindings and
does not rewrite the original recovery manifest to manufacture a completed audit.

## Completed historical acceptance

The immutable `failure-repair-03/reports-v2` run contains 132 account runs and
5,431 published targets, all independently checked. Maximum normalized constraint
residual is 9.94326844749871e-08. Published-target P95 is at most 4.257 seconds
across branches; maximum observed successful rebalance is 29.954 seconds. These
timings do not impute elapsed time for missing-input failures.

| Branch | Accounts | Reached 2026-09-30 | Published targets |
| --- | ---: | ---: | ---: |
| Strict | 62 | 31 | 1,548 |
| Floor sensitivity | 33 | 31 | 1,879 |
| Ceiling sensitivity | 33 | 31 | 1,880 |
| Extended floor | 2 | 2 | 62 |
| Extended ceiling | 2 | 2 | 62 |

The strict full-horizon count includes 26 accounts with no targets. Both base
sensitivity branches preserve the 001210 stops for scenarios 0/31; four separate
extended accounts complete them under the explicitly allowed integer endpoints.
Full-horizon completion is not continuous successful rebalancing. Every retained
input, continuous and integer-search failure remains in the failure inventory.

Model replacement is not accepted. In the floor branch, frozen-enhanced versus
legacy-OLS integer-target absolute-risk loss difference is -0.1692 with corrected
95% interval [-0.3418, 0.0152]; active-risk difference is +0.0243 with interval
[-0.1663, 0.2774]. Both cross zero. Actual-held portfolios have only 14/15 common
complete dates (absolute/active), below the block-inference gate. All reviewed
history remains retrospective and contributes zero new independent windows.
Four pre-entitlement branch comparisons differ (scenarios 29/30 in both branches);
timed-search differences prevent attributing all later return changes to rights.

The compact `failure_repair_configs/final-research-receipt.json` includes per-branch
failure counts, cross-risk conclusions and SHA-256s for the full local evidence.
It complements the raw reproducible account artifacts without uploading them to Git.
