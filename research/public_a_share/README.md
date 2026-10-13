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
