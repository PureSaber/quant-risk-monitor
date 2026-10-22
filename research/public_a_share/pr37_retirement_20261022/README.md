# PR #37: preserved research, closed without merging

Decision dated 2026-10-22: stop active development of PR #37 for the current
personal-investment decision-support scope. Retain its code and evidence for
possible future single-stock research. This is a prioritization decision,
not proof that all stale-exposure methods or enhanced risk models have no value.

## Why it is not needed in the current phase

- The current implemented pilot is allocation across four ETFs. The optional
  held-exposure carry and estimator experiments live under `research/public_a_share`;
  the current ETF implementation does not call these research entry points.
- PR #34's financial-data repair is already merged. Its accepted tree matches
  commit `5d7ea37828060138a1cac0fc77374e27a3d28aa9`, the parent scope of this work.
  After that scope is excluded, #37 adds 11 files' changes (1,022 insertions,
  8 deletions). It is not necessary to merge #37 to retain #34 or Portfolio #32/#33.
- The saved 62-scenario comparison separates integer-search repair from the
  incremental held-risk policy: published targets increase from 1,607 to 1,627,
  while full-horizon accounts remain 27. On 7,768 common decision dates, 30
  previously unpublished decisions become published and 10 previously published
  decisions become unavailable. More published targets do not establish better
  investment decisions, returns or risk calibration.
- Five-session stale-exposure sensitivity has a positive mean loss change in
  all six saved model comparisons; every simultaneous interval includes zero.
  Lower QLIKE is better. These artificial masks do not recover the truth for
  genuinely missing exposures and do not establish either a reliable improvement
  or conclusive harm.
- Existing estimator and account studies are retrospective. No new independent
  forward evidence or default-model promotion is supplied by this closure.

## Evidence and verification

The original PR head is
`ea8abbea34fef37c8e1e8332a301d56d109a7755`. This archival branch descends from
that exact commit; the original PR branch is retained. Main and the frozen
account/runtime are not changed by this archive.

- `decision.json`: decision, scope, archive fingerprints and reopening conditions.
- `scenario-comparison.csv`: 62 saved scenarios, with endpoints, completion
  status and target counts retained; workstation paths and raw accounts omitted.
- `comparison-summary.json`: original saved aggregate and transition counts.
- `carry-sensitivity-summary.json`: original saved artificial-mask results.

This review independently reconciled the saved scenario and decision CSVs,
72 existing account-audit receipts covering 3,234 published targets, and the
13 original-input SHA-256 hashes. It did not rerun the research campaign or
re-certify the statistical method. Successful historical CI for the original
PR head is software evidence, not an investment promotion.

The complete local preservation package is
`F:/QuantArchive/quant-risk-monitor-pr37-20261022/`:

- `research-evidence.zip`, with a per-member SHA-256 manifest and full readback
  verification, contains final and comparison evidence, exact account outputs,
  72 audit receipts, baseline accounts, 13 source inputs, analysis scripts and
  source working-copy snapshots.
- Two verified Git bundles preserve the risk-monitor and portfolio source histories.
- Uncommitted patches and working-copy snapshots preserve the original developer's
  unfinished cross-prediction changes, documentation and untracked regression test.
  They are archived drafts, not newly accepted production changes.
- `external-input-map.json` maps archived inputs to their original paths;
  `original-worktree-status.json` records the unchanged source worktrees.

The local archive is a same-machine preservation copy, not an offsite backup or
a self-contained operating-system/dependency image. Raw research data and full
account records are kept locally. Only these compact derived results are published.

## Reopening criteria

Reopen or extract a narrowly scoped successor only when:

1. A concrete single-stock portfolio decision is blocked by an evidenced exposure
   gap that this policy addresses; an ETF allocation use case alone is insufficient.
2. A prospective comparison is registered against a simpler handling policy,
   with net decision value, calibration, uncertainty and operational cost assessed.
3. New independent evidence, or a demonstrated defect invalidating the old result,
   justifies further work. Keep existing promotion requirements and original failures.

Routine bug repairs and missing-risk stops on the active path remain useful and
must not be coupled to adopting estimated exposures or a new default estimator.

## Recovery

Fetch this archival branch to inspect the original code and compact evidence.
For the full local package, verify `manifest.json` against every ZIP member before
extracting into a new empty directory; the archive contains data files only and
should not be treated as executable setup instructions. Restore a Git bundle into
a new checkout rather than overwriting an existing working tree. Preserve the
original experiment identities, missing dates and unsuccessful outcomes.
