# Fixed 200-company Barra study v3

This study preserves the original historical company cohort, including companies
that subsequently exited the market. The cutoff is 2026-09-30. It integrates v2
exposure candidates with factor-form portfolio optimization and the existing
exact A-share execution ledger. Default v1 exposures, frozen accounts and live
trading settings are unchanged.

## Reproducible inputs and quality

The original read-only archives contain 43,699 files (18,274,682,443 logical
bytes). Content-addressed copying and restoration verified every file. The
original manifest SHA-256 is
`f005999e5b1097a881d3224cecd5b17c2d4c13cde54d869ca957fb8e0e0db68f`.
The supplemental recovery inventory contains 1,078 files, with manifest SHA-256
`e003da921b6b0e043cb91712556d514cbdec7de77c4080184cc08bd9f83cf68c`.
These are same-machine copies on separate volumes, not independent hardware
backups. Raw data and extracted facts are retained locally rather than in Git.

Daily quality artifacts cover 237,400 company-days: 673 observed suspension
days, 5,555 days with unknown trading status, 4,828 short-history days and 27
industry changes. Financial inventories preserve 31,149 facts and 1,103
revision rows, with original capture metadata available for every retained
fact. Disclosure availability and actual acquisition time are separate.
Historical public-data reconstruction is not an authenticated contemporaneous
vendor vintage. Quote absence is not used to infer delisting.

The ten incomplete windows from the prior study were checked against archived
suspension records. Missing held-stock endpoints remain missing; no zero return
or surviving-weight renormalization was introduced. V3 has its own continuous
five-session forecast grid (180 dates, including 35 in 2026). Its 27 complete
2026 dates do not mean that one of the prior ten gaps was repaired.

Issuer announcements support distinct exchange adjustment references and actual
shareholder cash/share entitlements. Reference-only restructurings do not create
cash or shares in the account. An unsupported exact shareholder entitlement
stops an exposed replay; its safe prefix is diagnostic only. Fractional-share
allocation is not invented from a company's aggregate capitalization ratio.

## Frozen risk comparison

The experiment uses five-session forecasts and 252-session history. It first
holds OLS and evaluation portfolios fixed while comparing v1, repaired v2 and
the previously selected residual-volatility candidate. It then holds repaired
v2 fixed for the original twelve estimator settings and five ablations. There
are 22 distinct configurations and 51,480 forecasts. Selection uses 2023 only;
all observed 2024–2026 outcomes are retrospective diagnostics.

Inference resamples forecast dates jointly: 5,000 draws, ten-date blocks,
portfolio-family aggregation and simultaneous Bonferroni-adjusted intervals.
Negative change in QLIKE means lower loss relative to the stated control.

| Comparison against repaired v2 OLS | Period | Complete dates | Mean delta | Simultaneous 95% interval |
|---|---|---:|---:|---|
| Residual-volatility exposure, OLS | 2023 development | 43 | -0.002181 | [-0.003621, -0.000904] |
| Residual-volatility exposure, OLS | 2024–2026 diagnostic | 100 | +0.000693 | [-0.003936, +0.003591] |
| Frozen enhanced estimator | 2023 development | 43 | -0.096055 | [-0.187781, +0.029665] |
| Frozen enhanced estimator | 2024–2026 diagnostic | 100 | -0.061576 | [-0.249906, +0.258440] |
| Frozen enhanced estimator | 2026 diagnostic | 27 | +0.003406 | [-0.071995, +0.098812] |

These results do not establish a replacement advantage. Newly independent
mature windows are zero; the unchanged promotion minimum is 80 nonoverlapping
five-session windows. Historical model selection and data reconstruction cannot
be relabelled as new confirmation. Model replacement has not passed.

Independent recomputation reparsed restored raw price sources, rebuilt exposure
caches from empty paths and reproduced all 51,480 forecast rows byte for byte:
`a28770d74b628bd60f06fceb6e023aaf4590e8a66256db0b00dfa6b82a1fadc5`.
The 242 explicit small-sample dense/factor comparisons had maximum absolute
variance error `3.2526065174565133e-19`. Normal research and optimization paths
use factor form. Financial recomputation uses archived extracted facts whose
original source hashes are retained; it does not claim renewed manual reading
of every PDF.

## Native portfolio evidence and limitations

The runner defines 62 accounts: two investment budgets (95% and 80%), one equal
reference per budget, and six frozen generators with five optimization variants
each. Accounts start at CNY 1,000,000, rebalance every five sessions, cap a name
at 5%, pay all initial costs and subsequently enforce 20% L1 turnover. Active
scenarios impose predicted annualized TE of 3% or 6%. The capitalization
benchmark is explicitly a research-sample benchmark, not a market index.

All 22 configurations cross-predict common continuous, integer and actual-close
portfolios. An independent reporting pass also includes actual holdings on dates
when allocation failed. It preserves missing risk assets and outcome coverage.
Actual account returns include native costs; frozen-weight risk outcomes use
the common five-session return definition. These are separate evaluation lines.

The frozen estimator comparison admits assets with complete 252-session residual
histories. Suspension-related missing observations can exclude an already held
asset from the risk universe. In that case no new full-account target is
published. Later returns and low turnover cannot be interpreted as continuing
successful optimization. This limitation must be read alongside loss and return
tables, not hidden by dropping failed decisions.

`study-acceptance.json` is the authoritative completion state. A final report
requires receipts for all 62 processes. A process receipt is not proof of a
full-horizon investment result: inspect `full_horizon_completed`, publication
counts, missing risk and `full-horizon-stop.json`. Do not rank safe prefixes with
different end dates, or treat a fully infeasible cash account as a successful
portfolio. P95 rebalance duration covers successful optimization and integer
repair; total native replay time, memory and failures are recorded separately.

Independent account checks verify artifact hashes, exact double-entry balance,
nonnegative cash, integer shares, continuous and integer residuals, fee/NAV
reconciliation and factor-form integer TE. Execution edge cases such as partial
fills and unavailable quantities are also covered by synthetic regressions;
they are not represented as observed historical events when absent from the
fixed sample.

## Environment and recovery

Use `research/public_a_share/requirements-v3.lock` for ordinary installation.
It pins eight related repositories and the complete dependency set; conditional
Python 3.10 requirements are explicit. CI exercises Windows/Linux and Python
3.10–3.12, native solvers, parser runtime, research regressions and the existing
coverage thresholds. Installed-source snapshots preserve exact producer bytes,
including implementation-only performance changes made during the batch.

The [research README](../research/public_a_share/README.md) documents collection,
quality, fitting, replay, independent audit, report and restoration commands.
Final outputs include structured data, HTML reports, a static PDF, per-account
native evidence, dependency/code receipts and immutable recovery manifests.
Recover into fresh paths and verify hashes before using a trusted local pickle.
Environment installation requires package sources; no offline-install claim is
made. No service switch or frozen-account rollback is part of this isolated
study. Engineering delivery and model promotion have separate acceptance states.
