# 200-company data recovery and imputation validation

Final numeric overlay: `data-recovery-01/recovered-v5`. The unchanged cohort has
200 historical issuers and 908 evaluation dates (2023-01-03–2026-09-30), giving
181,600 company-days per descriptor. Delisted companies remain in the denominator.
Neither estimated values nor synthetic prices/returns count as observations.

| Descriptor | Original coverage | Restored coverage | Newly observed cells |
|---|---:|---:|---:|
| Three-year revenue growth | 88.9950% | 97.9262% | 16,219 |
| Three-year profit growth | 90.4752% | 97.7930% | 13,289 |
| Trailing earnings yield | 92.8684% | 96.7093% | 6,975 |
| Year-on-year profit growth | 96.9180% | 98.2682% | 2,452 |
| Book to price | 96.2847% | 96.7847% | 908 |
| Book leverage | 95.9416% | 96.8822% | 1,708 |

No previously observed cell in these matrices was lost. The complete 17-field
company-days rose from 141,465 to 161,212. Issuers with at least one gap anywhere
in the evaluation period decreased from 103/200 (51.5%) to 59/200 (29.5%). This
is an ever-missing cohort statistic, not the fraction unusable on any given day.

## Recovery evidence

The new fact store contains 32,193 unique vintages: 1,044 new versions and 18
reviewed corrections relative to the immutable original 31,149 rows. Duplicated
evidence from multiple reports is not counted as another unique vintage.

Of 160 original company-year-field blockers, 139 are restored before the first
affected decision date: 40 report-collection gaps, 61 extraction gaps and 38
missing comparative-history loads. Later reports restore older periods only
from the later report's actual public availability, never retroactively.

The 21 remaining annual blockers are fully listed in
`unresolved-annual-fields.csv`:

- 10 post-delisting field cases: mirrors exist, but exchange-original, revision
  and/or numeric cross-checks are not complete. An HTTP 403 is access failure,
  not evidence that an announcement never existed.
- Eight disclosure-timeline cases: six have dated non-disclosure evidence and
  two only an expectation of non-disclosure. Neither establishes permanent
  absence for every subsequent date.
- Three audit-opinion cases: 600190's 2024 profit and 603359's 2025 profit/revenue
  have originals but are not automatically admissible as reliable observations.

The supplemental recovery also verifies quarterly/YTD columns, prior-year
comparatives, China Life's equity label, and 600992's issuer-only statements
with dated evidence that consolidation did not apply. Parent-company-only
financials are not a general substitute for consolidated accounts. Unaudited
interim evidence is labelled separately from annual financial audit opinions.

`remaining-reason-summary.csv` explains the surviving cells. In particular,
remaining growth-history gaps are stale annual reports; remaining year-on-year
profit gaps are stale post-delisting reports. Book leverage is undefined on
2,517 company-days with nonpositive equity. Market-input and rolling-history
gaps caused by suspension/delisting are retained. Nine original delisting
notices resolve 5,321 previously unknown states without inventing prices.

## Imputation and risk findings

All 96 fixed experiment groups are retained: two descriptors, two estimators,
three masking fractions, four years and two persistent missingness patterns.
For 2023 at 10% masking, predicted/true variance ratios are approximately
0.058–0.171; the revenue industry-size candidate has negative rank correlation
under both patterns. Medians recover some information but do not establish a
reliable automatic replacement rule. No 5%/10%/20% fraction is approved as a
safe threshold, and no financial fact is imputed.

Fixed-portfolio risk validation contains 2,912 predictions and zero numerical
failure records. Forty development and 83 retrospective dates have complete
outcomes. Restored observations have a lower development point loss, but the
old-minus-restored simultaneous interval crosses zero. In 2024–2026 the old
inputs have lower loss: mean delta −0.0028855, simultaneous 95% interval
[−0.0054105, −0.0000871]. Both growth-composite imputation candidates worsen
development loss; retrospective candidate intervals cross zero. Their style
definition also changes, so those differences are not attributable solely to
imputation. Correcting data does not establish superior predictive calibration.

Default v1, original observations, frozen accounts and services remain unchanged.
These historical results provide no new independent forward confirmation.

## Independent reproduction

`recovery-audit.ipynb` was executed top-to-bottom in the frozen Python 3.12
environment with notebook tools isolated outside that environment. Its five code
cells recheck original hashes, finite masks, axes, unique vintages, 96 masking
groups, and risk-loss arithmetic from saved per-prediction artifacts. All pass.
The exported notebook was visually inspected. To rerun with a locally available
Jupyter kernel containing the locked research dependencies, select that kernel
or use `--ExecutePreprocessor.kernel_name=python3` with `jupyter nbconvert
--execute --to notebook`; source paths are parameters in the first code cell.

The full PDFs, parsed cells, time metadata, missing-cell reasons, model inputs
and intermediate attempts remain in the local content-addressed recovery archive;
only bounded summaries and the executed audit companion are stored in Git.
See `../DATA_RECOVERY.md` and the source-bound review configurations for recovery
commands and admission rules. The production source/default config diff is empty.

## Account acceptance

The final 62-scenario replay uses `account-inputs-v5`, `account-coverage-v5` and
the strict share rules. Its acceptance is pending in this draft; no incomplete
replay is labelled a completed account validation. Superseded intermediate
replays remain preserved and are excluded from final before/after claims.
