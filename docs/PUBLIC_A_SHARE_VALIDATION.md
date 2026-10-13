# Public A-share PIT calibration — 2026-10-13

The public-source historical study is complete and independently checked.
The enhanced model has a lower mean holdout QLIKE, but **does not qualify to
replace the original default**: both improvement intervals include zero,
industry volatility bias deteriorates beyond the frozen tolerance, and 79
complete holdout dates fall short of the prespecified minimum of 80.
No production parameters were changed.

## Dataset and design

The cohort is a fixed 200-company historical Shanghai/Shenzhen main-board
sample from the CSRC list available on 2021-11-10, with no subsequent survivor
replacement. Data cover 1,699 sessions from 2019-01-02 to 2025-12-31. The archive
contains 5,711 original disclosures and 27,743 independently extracted financial
facts across all 200 companies. Final verification covers 12,255 files and
13,259,328,252 bytes, including original disclosures, source receipts, dated
capital/industry data, primary prices and the market benchmark.

No Huisheng/Dataway inputs are used. Values come from original CNINFO reports,
CSRC/CAPCO historical classifications, CNINFO/Sina dated capital histories,
Eastmoney raw prices and BaoStock multiplicative-adjusted prices. Current Sina
financial statements and Eastmoney additive HFQ prices are diagnosis-only.

2023 has 48 scheduled five-session forecasts, 44 complete dates for selecting
among twelve enhanced configurations. After freezing selection, code and data,
2024–2025 has 96 scheduled forecasts, with 79 complete dates. Missing holdings
are never redistributed. All models use the same historical exposures, eligible
stocks, portfolio weights and forecast dates. The research was created in 2026
and is explicitly retrospective; it does not claim historical preregistration.

## Holdout results

QLIKE is `log(predicted variance) + realized return² / predicted variance`.
Lower is better. Each date equally weights the equal-weight portfolio,
capitalization-weight portfolio and industry-portfolio family.

| Model | Mean QLIKE | Equal-weight volatility bias | Cap-weight bias | Industry bias |
|---|---:|---:|---:|---:|
| Original unconstrained OLS | -5.465867 | 1.091229 | 1.024692 | 0.957612 |
| OLS with common industry constraint | -5.486453 | 1.027080 | 0.961132 | 0.953301 |
| Selected enhanced model | -5.578791 | 0.910231 | 0.938612 | 0.852522 |

Volatility bias is realized RMS return / predicted volatility; ideal is one.
QLIKE uses common complete dates; bias uses each family's observed portfolio
endpoints, paired identically across models. A ratio below one means predicted
risk exceeds realized risk. These are risk-forecast metrics, not strategy P&L.

| Paired comparison | Enhanced minus control | Moving-block 95% interval |
|---|---:|---:|
| Common-basis OLS | -0.092339 | [-0.261439, 0.234030] |
| Original OLS | -0.112925 | [-0.277218, 0.198304] |

The bootstrap preserves all portfolios within a forecast date, uses blocks of
ten forecast dates, 2,000 replications and seed 20261013. The prespecified
stale-mark sensitivity gives -0.078750 relative to common-basis OLS. Each model
has 52 missing portfolio endpoints across the 96 scheduled dates. Stale marks
do not establish a realizable delisting recovery price.

## Selected configuration and ablations

Selection chose covariance shrinkage 0.10, factor half-life 90, specific half-life
60 and specific prior degrees of freedom 10. Fixed enhancements use square-root
capitalization regression objective weights, a cap-neutral industry constraint,
four Newey–West lags, a 20-day volatility regime window with half-life 10,
and industry-group Bayesian specific-risk shrinkage with a global fallback.

| 2023-only diagnostic | Mean QLIKE |
|---|---:|
| Selected complete specification | -6.571754 |
| Without cap weighting | -6.558708 |
| Without time decay | -6.551562 |
| Without serial adjustment | -6.599798 |
| Without regime adjustment | -6.506860 |
| Without group Bayesian specific risk | -6.568644 |

Serial adjustment was not helpful in this selection-period ablation. Other
enhancements individually improve this particular comparison. Ablations are
diagnostic, not new candidates; they did not trigger parameter reselection or
additional holdout searches.

## Repairs and acceptance evidence

- Rejected additive HFQ percentage returns after cross-checking eight historical
  companies, including later exits. Unadjusted closes agreed exactly.
- Corrected quarterly-vs-YTD column selection, unit scales, negative equity,
  liability subtotals, footnote numbers, share-table wrapping, and directory
  report years against original cover pages. Conflicting facts stay missing.
- Used isolated PDF worker processes, avoiding unsafe PyMuPDF threading.
- Prevented current financial restatements from determining historical values
  or missingness. Share-event availability requires both announcement and
  effective date; unknown announcement placeholders are rejected.
- Built 1,006 exposure snapshots and 1,005 regression dates for each of the
  three regression variants, with no failed regression dates.
- Poisoned 8,206 future original financial facts; all 200 companies' features
  on 21 preceding dates remained unchanged. Offline tests additionally cover
  future price poisoning and industry/capital publication boundaries.
- Compared the first real forecast with both production model entry points and
  the unmodified original OLS source. Maximum daily covariance discrepancy was
  2.28e-18. Independently recomputed losses and both bootstrap intervals agree.

The package suite passed 362 tests, with 88.82% coverage; the separate public-PIT
suite passed 24 tests. Ruff checks and formatting pass. The saved audit notebook
executes top-to-bottom, and the report was inspected in normal and narrow
browser layouts. Numerical reproducibility and research completion passed;
model-promotion acceptance failed and remains failed.

## Scope and artifacts

This is a fixed main-board cohort, not all A-shares or later IPOs. Requiring a
full observed residual year favors continuously quoted stocks. Industry release
gaps carry forward the last known public classification. Total shares times the
A-share quote form a consistent capitalization proxy. Public historical
reconstruction does not reproduce live data-arrival latency. The factors are
public Barra-style definitions, not MSCI proprietary model parameters.

Run ID: `20261013T130831Z`. The local archive contains the HTML report, executed
notebook, per-forecast results, paired date losses, source manifest, frozen
configuration and acceptance record. Raw archives and generated outputs are
intentionally outside Git; the reusable scripts, frozen design and original
reference implementation are in `research/public_a_share/`.

| Binding | SHA-256 |
|---|---|
| Protocol | `b86773d2392a9cd6d466df789c8bc0dcfdf889ccdf25b43b098183ae6c7fc57b` |
| Selection | `cbbee3cfff9b026a855f45774d970c9b72b8fa7bd36e6a036397f1b7e0c361d1` |
| Holdout result | `4a13828473410f07ecee3d5f670c0452e551ce209e1bfbfffafc377780165c72` |
| Source manifest | `18db5caa5aba07a3e11b136f66adb0adf452979735cc9c16dffbdc0529b9da03` |

Reproduction instructions: [public-source research README](../research/public_a_share/README.md).
