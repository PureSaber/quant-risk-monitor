# Public A-share historical PIT research

This independent study compares the original `54fcf4d` OLS implementation, an
OLS control with the enhanced model's industry constraint, and twelve enhanced
risk specifications on the same historical exposures, stocks, weights and dates.
It excludes Huisheng/Dataway and does not change production defaults.

The fixed cohort contains 200 Shanghai/Shenzhen main-board companies drawn from
the CSRC list publicly available in November 2021. The dataset spans 2019–2025;
2023 selects parameters and 2024–2025 is the historical holdout. This is an
honestly dated retrospective study, not a claim of historical preregistration,
all-market coverage, or replay of a live vendor feed.

## Inputs and chronology

- CNINFO original disclosures supply as-reported financial values. Publication
  plus one calendar day determines availability; revisions affect only later
  snapshots. The original cover corrects mislabelled directory periods.
- CSRC/CAPCO dated original industry publications control classifications. Carry
  the last known release through publication gaps, including the 2021–2023 gap.
- Dated CNINFO capital histories, supplemented by nine Sina historical capital
  tables with announcement and effective dates, determine total shares. Unknown
  `19000101` announcement placeholders are rejected.
- Eastmoney unadjusted prices determine capitalization and turnover. BaoStock
  multiplicative-adjusted prices determine returns, momentum and outcomes.
  Eastmoney additive HFQ returns were rejected after an independent cross-check.
- Present-day Sina financial statements are diagnostic references only: they
  cannot supply a historical value or decide whether it is missing.

Price information is lagged one session. Announcement/effective times, filing
periods, source receipts and hashes remain inspectable. Suspended/unobserved
quotes do not become synthetic zero returns. No stock is replaced because it
subsequently exits. Eligibility is based on preceding observations; all controls
use the same eligible assets. A full year of observed residuals is required,
which favors continuously quoted stocks and limits the scope of inference.

## Reproduce using an existing archive

Use Python 3.12 in a dedicated environment and install `requirements.lock`.
The lock records the actual research environment, independently of the package's
production dependency lock. Set `PUBLIC_PIT_ROOT` to the existing study archive;
`QUANT_RISK_SOURCE` can explicitly select the project's `src/quant_risk_monitor`.
Do not load untrusted pickle files.

The archive already contains the frozen `protocol.json`, its dated
clarifications, original files/receipts, clean matrices and run directory.
`select` rejects an existing selection; `holdout` rejects a repeated holdout and
checks the frozen input/code hashes. Preserve the original run. To repeat an
experiment, use a separate root containing the same verified raw archive and
protocol, never delete the first result or silently reselect after holdout.

```text
python original_financials.py
python prepare_pit.py industries
python build_panel.py prepare
python build_panel.py regress
python audit_sources.py
python validate_pipeline.py
python run_study.py select
python run_study.py ablate
python run_study.py holdout
python produce_report.py
python execute_notebook.py
```

For canonical validation, copy `reference/original_factor_model.py` to
`$PUBLIC_PIT_ROOT/raw/code/original_factor_model.py`. This is an unmodified MIT
licensed source snapshot from commit
`54fcf4d9dffd4f98ca173f26ee80e96832ea9ee1`, with SHA-256
`f8c141f80d0afdd8eb9ea18f5466ed7ca7c168caf025bcf22ded32f58da9dbd7`.
Validation compares cached forecasts with both real model entry points and
poisons future financial disclosures to check past-feature invariance.

Hashes bind exact file bytes. Line-ending conversion or any code/data change
requires a new run and explanation; it must not masquerade as the original run.
`frozen/` preserves the original design and cohort for review. A fresh download
may contain later corrections, so it is not automatically identical to the
archived reconstruction.

## Fresh public acquisition

Use a new empty data root and these bounded, resumable download commands before
the computation above. No authenticated commercial feed is needed.

```text
python public_sources.py cohort
python public_sources.py industries
python public_sources.py download
python filings.py pdfs
python collect_shares.py
python collect_missing_capital.py
python collect_returns.py
python crosscheck_prices.py
python freeze_protocol.py
```

Preserve the accompanying dated clarification policy from `frozen/` and record
the actual new execution date before any scores. The original 2026-10-13 run's
PDF extraction used isolated processes, never concurrent PyMuPDF threads.
Acquisition requires internet access, public endpoint availability and about
14 GB for raw documents; do not commit those archives or generated runs.

## Evaluation and acceptance

Forecast five-session buy-and-hold variance on a nonoverlapping calendar grid.
Mean QLIKE is `log(v) + r²/v`; lower is better. Each date gives equal total weight
to the equal-weight portfolio, capitalization-weight portfolio, and industry
portfolio family. Missing endpoint holdings are never renormalized. Publish
common complete dates and a separate stale-mark sensitivity; stale marks do not
establish executable delisting recovery prices.

The frozen requirements are at least 40 complete selection dates and 80 complete
holdout dates, a paired moving-block 95% interval entirely below zero against
both OLS controls, and no material volatility-bias deterioration. A coverage
shortfall is reported and prevents promotion even when point estimates improve.
Research completion and model promotion are separate outcomes. Ablations use
2023 only and never cause reselection. No performance guarantee is implied.

Run deterministic offline parser/PIT tests separately from the service suite:

```text
python -m pytest research/public_a_share/test_public_pit.py -q
python -m ruff check --config research/public_a_share/ruff.toml research/public_a_share/*.py
```

The HTML report, executed audit notebook, per-forecast predictions, paired scores,
source manifest, frozen configuration and acceptance record are retained in the
local archive. The report independently recalculates both scores and bootstrap
intervals. See `docs/PUBLIC_A_SHARE_VALIDATION.md` for the completed run's results.

## Exposure definition v2

Use the existing v1 archive read-only and a new output root. The commands below
run from the repository root with this directory's research environment:

```text
python research/public_a_share/collect_exposure_inputs.py --old-root OLD --out NEW
python research/public_a_share/exposure_v2_study.py --old-root OLD --out NEW
python research/public_a_share/collect_confirmation_inputs.py --old-root OLD --out NEW/confirmation-inputs
python research/public_a_share/confirmation_context.py --old-root OLD --out NEW/confirmation-inputs
python research/public_a_share/confirm_exposure_v2.py --old-root OLD --study-root NEW
python research/public_a_share/audit_exposure_v2.py --old-root OLD --out NEW
python research/public_a_share/validate_exposure_v2.py --old-root OLD --out NEW
python research/public_a_share/benchmark_exposure_v2.py --out NEW/capacity
python -m pytest research/public_a_share -q
```

The v2 protocol is created before development scores. Seven exposure variants
share assets, dates, portfolio definitions and fixed risk estimators; the 2024–25
outcomes are explicitly reused diagnostics. 2026 confirmation reads a frozen
selection binding, never reselects, and retains the 80-forecast requirement.
Public acquisition is retrospective reconstruction, not authenticated historical
vendor vintages. No Huisheng/Dataway feed is used.

`--resume-histories` verifies `history-manifest.json` before unpickling the local
regressions. Only use trusted local archives. Source/configuration changes need a
new run; report-only edits may reuse a mathematically identical history producer.
The full daily snapshot cache preserves imputation/source audit records. The
smaller history pickle omits their repeated provenance to reduce memory use.

Current local evidence roots: `barra-public-pit-20261013` (original) and
`barra-exposure-v2-20261014` (new), under the user's ChatGPT SCRATCH directory.
The first v2 run's producer code, selection and diagnostics were retained before
adding manifest checks; independently rescoring its forecast CSVs produced
identical metrics and selection. This is recorded explicitly in its result JSON.
Do not overwrite the original archive or describe seen diagnostics as a new holdout.

The completed 2026 confirmation retained the development-selected residual-volatility
candidate. Of 36 scheduled five-session forecasts, 26 have complete common outcomes;
ten are excluded for missing held-stock endpoint prices without renormalizing weights.
Neither fixed estimator establishes an advantage against the original definitions,
and the frozen 80-forecast promotion minimum is unmet. v1 remains the default.
See `docs/EXPOSURE_V2.md` for scores, intervals, source limitations and capacity results.

The new archive contains `confirmation/result.json`, `forecasts.csv`, source and
selection bindings, producer-code snapshots, and an independent forecast audit in
`confirmation/independent-forecast-validation.json`. Its arithmetic/pairing/source
validator is retained as `verify_confirmation.py` in the same archive. The original
study's producer hashes remain intact: later parser-validation hardening did not
change its actual descriptor values, as recorded in `independent-model-validation.json`.

## Fixed 200-company study v3

The v3 experiment integrates the v2 exposure candidate with the current optimizer
and native A-share execution bridge. It does not change the default exposure
definition, promote a strategy, or modify frozen accounts. The cutoff is
2026-09-30; 2023 is development and every observed 2024–2026 outcome is explicitly
retrospective. Eighty genuinely independent five-session windows remain required
for any later promotion decision.

Use an ordinary installed environment, the exact repository revisions and the
frozen dependency manifest. `OLD` and `V2` below are read-only input archives;
`DATA`, `BACKUP` and `RUN` must be separate dedicated directories. The approved
local budgets are 25 GiB for data, 20 GiB for the same-machine recovery copy and
5 GiB for code/environment, retaining 40 GiB free on the system drive.

```text
python research/public_a_share/study_v3_archive.py --old OLD --exposure V2 --out DATA --backup BACKUP
python research/public_a_share/study_v3_inputs.py --old OLD --exposure V2 --out RUN
python research/public_a_share/study_v3_quality.py --old OLD --exposure V2 --out RUN
python research/public_a_share/study_v3_risk.py --out RUN
python research/public_a_share/study_v3_portfolio_models.py --out RUN
python research/public_a_share/study_v3_execution_inputs.py --old OLD --exposure V2 --root DATA/inputs --out RUN
python research/public_a_share/study_v3_batch.py --out RUN --workers 4 --batch-id unique-run-id
python research/public_a_share/study_v3_recompute.py --old DATA/inputs/old --exposure DATA/inputs/exposure --reference RUN --out EMPTY_RECOMPUTATION
python research/public_a_share/study_v3_recovery.py --data DATA --backup BACKUP --code CODE_ENVIRONMENT_ROOT
python research/public_a_share/study_v3_report.py --out RUN --independent EMPTY_RECOMPUTATION --batch-id unique-run-id
```

Install `requirements-v3.lock` with `python -m pip install --no-deps -r`
and run `python -m pip check`. It pins all distributions and the eight related
repositories to immutable revisions. The frozen-runtime workflow validates
ordinary installs on Windows/Linux and Python 3.10–3.12, including native solvers
and the public-source parser runtime. Record the workflow result separately from
study completion. `study_v3_provenance.py --out RUN/producer-source` preserves the
exact installed source bytes; later implementation-only speed improvements are
explicit producer versions, not changes to frozen scientific parameters.

The optional `study_v3_pdf.py --root RUN --pdf OUTPUT.pdf --font FONT.ttf`
renders the report directly from its structured evidence using ReportLab.
The renderer is a separate presentation dependency; it is not required to
recompute the research. Generate the final report only after all 62 processes
finish. `--allow-partial` is for clearly marked interim reports only.

Run `--help` on the evidence collectors and execution-input builder for their
explicit archive paths. Public responses, issuer PDFs, their actual capture
times and source hashes are retained; failed captures require a new capture root
when retried. Financial disclosure availability and actual collection time are
separate. Revised financial facts are not overwritten. The ten old incomplete
2026 windows are audited against original suspension records, without inserting
zero returns or renormalizing surviving positions.

The risk experiment fixes three exposure variants, the original twelve estimator
configurations and five ablations. It uses joint date-block resampling (5,000
draws, ten forecast dates per block), family-level aggregation and Bonferroni
simultaneous intervals. Selection uses 2023 only. Independent recomputation uses
an empty exposure cache and compares every forecast; dense covariance arithmetic
is confined to explicit validation checks.

Confidence intervals require at least two complete blocks at the frozen length
of ten dates. Smaller samples retain their point estimates and an explicit
`insufficient_date_blocks` status, with no confidence interval or historical
improvement flag. This descriptive-inference safeguard does not relax the separate
80 genuinely independent-window promotion gate. Actual-close portfolio common
samples must be checked separately from continuous and integer target samples.

There are 62 native-account scenarios: each investment budget (95% and 80%) has
one equal-weight reference and six frozen model combinations with minimum
variance, minimum active risk at 3%/6% TE, and IC-vol cost-aware momentum at
3%/6% TE. All 22 risk configurations cross-predict the same continuous, integer
and actual-close portfolios. Each account starts with CNY 1,000,000, a 5%
single-name cap and a 20% subsequent L1 turnover cap. The full initial build pays
actual native fees. The benchmark is the research-sample capitalization portfolio,
not a market index. Prediction windows use linear five-session returns for risk
loss; actual account returns and sample-benchmark returns compound independently.

The account calendar contains 182 decisions, including 2023-12-28 and
2026-09-28. It is independent of the 180 mature risk-evaluation windows:
development-boundary exclusions and an immature final outcome must not remove
trades. The additional causal risk snapshots live in a separate bound cache;
the original risk experiment and its selection stay unchanged. Final immature
cross predictions retain predicted risk and an explicit `outcome_pending`
status. The superseded `utc-02` account batch is retained; use a new batch id
for the corrected cadence.

`complete.json` means the scenario process produced verified artifacts; inspect
`full_horizon_completed`, `published`, failure statuses and the optional
`full-horizon-stop.json` before interpreting performance. An unsupported exact
entitlement rolls back the full replay. A separately rebuilt safe prefix is only
diagnostic and never supplies post-failure NAV. An infeasible target retains the
existing account and publishes no fallback. A pilot's `limit_dates` is part of
its identity and must never be reported as a full-period result. Raw failed
attempts and superseded run identities remain available.

The optional `study_v3_feasibility.py --out RUN` explains tight TE constraints by
solving the continuous minimum active risk at the fixed budget and name cap,
before turnover, fees and integer restrictions. It is a post-study diagnostic,
never parameter selection or a substitute published allocation. Analytic
cash-budget and concentrated-benchmark projections validate its arithmetic.
Reports include this input-bound diagnosis when present.

During local execution, expanding to extra account workers caused Windows
commitment pressure despite available physical memory. The helper batches were
cancelled and the four-worker primary batch retained. The resource incident and
failed attempts are preserved; incomplete scenarios are retried from initial
state. Shared-machine timings include temporary resource competition. Check
commit headroom as well as physical memory before any future concurrency change.

Direct portfolio-worker entry points now set OpenBLAS, OpenMP and MKL thread
limits to one before importing native numerical libraries. An isolated 60-session
native-account comparison reproduced all 12 weights exactly and the same 430
fills, fees, positions, NAV and economic journal postings; its measured peak
private commitment was 486,379,520 bytes. Run-dependent journal hashes change
export ordering, so economic postings were compared in canonical order. This is
a prefix-equivalence check, not a full-study speed guarantee. Each new account
identity records the declared environment and observed native thread pools.
Earlier accounts retain their original producer identities and evidence. Calls
that import the runner as a library retain their caller's threading environment.

Scenario processes hold an operating-system lock for their output root, scenario
and full/pilot extent. A duplicate waits, then verifies and reuses the completed
receipt; it never writes a second account concurrently. Crashed owners release
their OS lock automatically. Lock address files remain in place and must not be
deleted while a batch is active. Separate batch logs can therefore help finish
pending scenarios on a machine with verified spare CPU and memory; the primary
batch still requires its own complete set of 62 receipt references. Each batch
remains bounded to at most four workers. The lock changes no mathematical or
execution behavior; producer hashes retain the wrapper change explicitly.

Use `tests/test_study_v3.py` and `tests/test_price_availability_index.py` in
addition to the existing exposure/PIT/cache regressions. The execution bridge
also tests exact daily/event parity, cached/uncached account parity, distinct
exchange references and shareholder entitlements, fractional-share failure and
guard rollback. These synthetic edge tests are separate from evidence actually
encountered in the fixed historical sample.

Recovery is a read-only source-to-object-store copy followed by a separate
restore and SHA-256 verification. Never replace an old manifest or a changed
output; start a new run. Restore the fixed environment and archived inputs into
fresh paths and rerun the commands with the recorded specification. No service
or account rollback is needed for this isolated research workflow.

After all scenario and engineering checks have passed, freeze and restore the
final report/source evidence with `study_v3_delivery.py --data DATA --backup
BACKUP --code CODE_ENVIRONMENT_ROOT --version delivery-v1`. The immutable
manifest excludes only the reproducible daily exposure cache. Original raw
inputs have their separate original/supplemental manifests. Restoring the
environment from the lock still needs package sources; the same-machine copy
does not claim an offline environment or an independent hardware backup.
