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
  `data_recovery_configs/reviewed-corrections.json` records five source-page
  checks. Nonstandard financial audit opinions are not automatically admitted;
  historical audit discussions and internal-control opinions are distinguished.
* `recovery_rebuild.py` writes an independent fact table and six descriptor
  matrices. It checks frozen inputs before deserializing the trusted local panel.
  Historical comparative figures are available only from the actual source
  publication date, conservatively on the next calendar day.
* `recovery_lifecycle.py` verifies nine issuer delisting notices. Its overlay
  starts after both publication availability and the effective delisting date.
  It changes neither prices, return labels, holdings nor account cash.

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
python recovery_rebuild.py --base BASE --run NEW_RUN --cache NEW_RUN/extract --audit AUDIT --target NEW_RUN/rebuilt --corrections data_recovery_configs/reviewed-corrections.json
python recovery_lifecycle.py --base BASE --out NEW_RUN/lifecycle
python recovery_lifecycle.py --base BASE --out NEW_RUN/lifecycle --events data_recovery_configs/lifecycle-events.json
python recovery_imputation.py --base BASE --recovered NEW_RUN/rebuilt --out NEW_RUN/imputation
python recovery_risk_validation.py --base BASE --recovered NEW_RUN/rebuilt --out NEW_RUN/risk-validation
```

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
