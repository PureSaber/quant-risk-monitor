# Point-in-time annual-data recovery

This opt-in research pipeline fixes missed report-title variants, ambiguous table
columns, wrapped amounts and missing comparative-year vintages. The frozen
200-company cohort, original files, default v1 model and execution accounts are
unchanged. Recovered numeric observations and estimated descriptors are different
artifacts; neither is automatically promoted to a production model.

## Evidence and admission

* `recovery_inventory.py` binds the original panel, financial vintages, cohort,
  quality table and audited blocker list by SHA-256. It inventories 2018–2025
  annual versions, including revisions and reports with yearless titles.
* `recovery_collect.py` archives missing post-delisting reports incrementally.
  Exchange originals and issuer-document mirrors are labelled separately. A
  mirror without an exchange cross-check remains quarantined. A failed request
  is not evidence that a report was never published.
* `financial_table_recovery.py` uses physical table coordinates, column years,
  adjusted/comparative labels, consolidated scope and explicit units.
  `recovery_extract.py` independently verifies PDF text and restricts its text
  fallback to actual current-year consolidated statements. Opening-balance
  reconciliations, quarterly summaries, revenue after deductions and supplementary
  alternative accounting standards are not silently substituted for the target.
* `recovery_admission.py` preserves every vintage. Same-time conflicts require an
  explicit reviewed correction bound to the source hash and old/new values.
  `data_recovery_configs/reviewed-corrections.json` records 18 reviewed vintage
  corrections, including 13 liability-unit corrections. Each additionally loaded
  original is checked with both PDF readers. `reviewed-comparatives.json` binds
  the China Life restated comparative figures to the later report and its actual
  availability date. Nonstandard financial audit opinions are not automatically admitted;
  historical audit discussions and internal-control opinions are distinguished.
* `recovery_rebuild.py` writes an independent fact table and six descriptor
  matrices. It checks frozen inputs before deserializing the trusted local panel.
  Historical comparative figures are available only from the actual source
  publication date, conservatively on the next calendar day.
* `recovery_lifecycle.py` verifies nine issuer delisting notices. Its overlay
  starts after both publication availability and the effective delisting date.
  It changes neither prices, return labels, holdings nor account cash.
* `interim_table_recovery.py` separately verifies quarterly/YTD columns and
  insurance equity labels. Issuer-only statements require dated evidence that
  consolidation is not applicable; this is not a general parent-company fallback.
  `recovery_supplemental.py` binds manual comparative-column reviews to original
  hashes, both PDF readers and source availability. Cover-header exceptions and
  an issuer's disputed assurance remain explicit. Later comparatives never use
  the older report's availability. Unaudited interim reports are not described
  as carrying an annual audit opinion.
* `recovery_gap_reasons.py` explains every remaining descriptor cell using the
  underlying formula and information available on its date. A missing component
  is not proof that the original announcement does not exist.

An audit-opinion classification is a gate, not a new audit of the issuer. Existing
unreviewed financial observations retain their legacy status. Full-report PDF
identity, successful text extraction and financial reliability are separate checks.

## Remaining-gap experiments

`recovery_imputation.py` compares industry medians and a fixed ridge regression on
industry plus contemporaneous size. It withholds issuer blocks of 63 trading
sessions and a whole year, with fixed seed 20261017 and 5%/10%/20% masking. A
prediction uses only that day's unmasked peers, requires at least five industry
peers and refuses unsupported size extrapolation. Missing core financial/market
inputs and known adverse current annual opinions block estimation.

2023 remains development data; 2024–2026 remain retrospective diagnostics. The
fraction grid is not an approved safe filling threshold. Real gaps have no hidden
truth and receive only labelled sensitivity estimates. No annual revenue, profit,
cash, price, entitlement or sellable quantity is imputed.

`recovery_risk_validation.py` separately compares old inputs, restored inputs and
the two growth-composite sensitivity candidates. It refits daily factor returns
with lagged exposures, uses 252 actual calendar sessions, reports missing outcomes
without renormalizing portfolio weights, and keeps factor/specific risk separate.
Matched fixed-portfolio forecasts are not a full account replay or independent
forward model confirmation. A failed research gate does not change default v1.

## Reproduction

Use the ordinary-install versions in `requirements-failure-repair.lock`; this
campaign reused that frozen Python 3.12 environment read-only. The research CI
also checks Windows/Ubuntu and Python 3.10–3.12. Run from this directory, with
explicit paths to the preserved archive and prior audit:

```text
python recovery_inventory.py --base BASE --audit AUDIT --out NEW_RUN
python recovery_collect.py --base BASE --out NEW_RUN
python recovery_extract.py --inventory NEW_RUN/recovery-inventory.csv --cache NEW_RUN/extract
python recovery_extract.py --inventory ANNUAL_FOLLOWUP_INVENTORY --cache NEW_RUN/annual-followup
python interim_table_recovery.py --inventory data_recovery_configs/interim-inventory.json --out NEW_RUN/interim-geometry.json
python recovery_supplemental.py --geometry NEW_RUN/interim-geometry.json --reviews data_recovery_configs/reviewed-interim-v3.json --target NEW_RUN/supplemental.json
python recovery_rebuild.py --base BASE --run NEW_RUN --cache NEW_RUN/extract --additional-cache NEW_RUN/annual-followup --supplemental NEW_RUN/supplemental.json --audit AUDIT --target NEW_RUN/rebuilt --corrections data_recovery_configs/reviewed-corrections.json --comparatives data_recovery_configs/reviewed-comparatives.json
python recovery_lifecycle.py --base BASE --out NEW_RUN/lifecycle
python recovery_lifecycle.py --base BASE --out NEW_RUN/lifecycle --events data_recovery_configs/lifecycle-events.json
python recovery_imputation.py --base BASE --recovered NEW_RUN/rebuilt --out NEW_RUN/imputation
python recovery_risk_validation.py --base BASE --recovered NEW_RUN/rebuilt --out NEW_RUN/risk-validation
```

`ANNUAL_FOLLOWUP_INVENTORY` is the archived `annual-followup-inventory.csv`
covering the reviewed 002255 annual reports. Inventory and reviewed evidence paths
refer to preserved originals; relocate those paths explicitly when restoring on
another machine, and require their recorded hashes to match. Do not substitute
newly downloaded revisions under an old observation date.

`recovery_account_inputs.py` rebuilds the six frozen generator combinations with
restored observations. Every original decision date retains the exact original
candidate set and benchmark weights. Run `repair_risk_coverage.py` on that new
source, then the existing `study_v3_batch.py` with the strict share rules.
`recovery_account_report.py` verifies all 62 receipts, exact-money journals,
fee/cash arithmetic, risk unions, lot rules and continuous/integer constraints
before producing a final account acceptance. It compares against the prior strict
replay and does not calculate a return improvement between unequal stopping dates.
Imputation candidates are excluded from account decisions because they did not
pass development validation.

## Final supplemental recovery

Final data and imputation evidence is in `data_recovery_results/README.md`, with an executed independent audit notebook. The checkpoint below remains preserved for provenance and is not the final result.

## Initial annual recovery checkpoint (superseded by supplemental validation)

For the unchanged 200 companies and 908 evaluation sessions (181,600 company-days
per field), intermediate `recovered-v3` coverage is:

| Descriptor | Before | Restored | Newly observed cells |
|---|---:|---:|---:|
| Three-year revenue growth | 88.9950% | 97.8904% | 16,154 |
| Three-year profit growth | 90.4752% | 97.7572% | 13,224 |
| Trailing earnings yield | 92.8684% | 95.4857% | 4,753 |
| Year-on-year profit growth | 96.9180% | 97.0551% | 249 |

The merged fact table has 944 new unique vintages and 18 reviewed corrections.
No previously observed descriptor cell was lost. Of 160 company-year-field
blockers, 137 are restored before their first affected decision and 23 are not
admitted: 10 post-delisting source-verification cases, eight disclosure-timeline
cases and five audit-opinion review cases. A source retrieval failure, including
the recorded HTTP 403, is not evidence of nonpublication. The full inventory and
remaining-case evidence are in the delivery artifacts.

Both imputation candidates fail to demonstrate stable rank recovery and preserve
too little of the observed cross-sectional variation. Neither 5%, 10% nor 20%
masking establishes a safe automatic filling threshold. On the fixed-portfolio
risk check, restored inputs improve development loss but worsen retrospective
2024–2026 loss against the old inputs (old minus restored mean −0.003665,
simultaneous 95% interval [−0.006218, −0.000657]). Data correctness and predictive
model quality are separate acceptance decisions; default v1 remains unchanged.

These counts describe reviewed recovery overlays, not a new audit of all legacy
observations or independent forward confirmation. The run preserves exact producer
hashes; subsequent formatting-only commits do not relabel older artifacts as
having been produced by the newer source hash.

Use a new run/cache/output version when code or inputs change; do not overwrite
completed experiments. Downloads stop at the agreed 25 GiB data-root budget and
preserve at least 40 GiB system-disk free space. Original inputs are referenced by
hash instead of copied into each experiment. Archive new evidence incrementally
within the separate 20 GiB archive budget. A second drive in the same machine is
not claimed as an independent hardware backup.

## Restore and boundaries

Restoring the prior research path means selecting its original panel, protocol,
fixed commit and frozen environment. No account, service or default configuration
was migrated, so there is no live rollback step. Retain the new recovery artifacts
and rejected intermediate versions as evidence. Verify their content manifests
before reuse; never restore by overwriting a newer ledger or deleting an old run.

The delivery report records final coverage, field-level unresolved reasons,
masked-validation results, risk diagnostics, storage receipts and exact source
hashes. Report-only revisions do not authorize strategy or model promotion.
