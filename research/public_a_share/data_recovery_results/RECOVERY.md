# Restore this recovery campaign

No model configuration, running service or frozen account was deployed or
migrated. To use the old research path, select the original panel, protocol,
fixed commits and ordinary-install environment. Do not overwrite a ledger.

The same-machine recovery copy is
`F:\QuantArchive\barra-200-v3\data-recovery-v5\manifest.json`.
Its expected SHA-256 and verification receipt are recorded at
`D:\QuantData\barra-200-v3\runs\data-recovery-01\delivery-v5\archive-receipt.json`.
This is a cross-drive copy, not an independent-hardware backup.

From the PR checkout, verify all decoded archive objects without restoring files:

```powershell
$receipt = Get-Content -LiteralPath 'D:\QuantData\barra-200-v3\runs\data-recovery-01\delivery-v5\archive-receipt.json' -Raw | ConvertFrom-Json
python research/public_a_share/recovery_restore.py --manifest $receipt.manifest --sha256 $receipt.manifest_sha256
```

Restore representative files into a new directory with enough free space:

```powershell
python research/public_a_share/recovery_restore.py --manifest $receipt.manifest --sha256 $receipt.manifest_sha256 --target 'C:\Users\51076\Documents\ChatGPT\SCRATCH\barra-restored-check' --key 'run/recovered-v5/financial-vintages.parquet' --key 'run/delivery-v5/coverage-receipt.json'
```

Every requested key must exist in the manifest. Path traversal, changed archive
objects, a changed manifest and overwriting different destination contents all
fail explicitly. Add exact `--key` values from `manifest.files` to restore other
files; the default operation verifies every unique object but writes nothing.

The Git bundle is stored under `tools/recovery-pr34.bundle` in the manifest.
Restore it and run `git bundle verify` before using it in a new checkout. Rebuild
an ordinary environment with `research/public_a_share/requirements-failure-repair.lock`,
run `pip check`, and confirm actual import paths. Installing those exact external
distributions requires the corresponding registries/remotes or preserved wheels;
the archive is not claimed to be a self-contained offline package mirror.

Original PDF paths and announcement timestamps are bound to the saved evidence.
If moving to a different root, use the manifest's source-to-restored mapping and
create a new review/config/cache version; do not relabel the original producer
hashes or change availability dates. Recompute the independent notebook and
account verifier before accepting a restored research run.

The superseded attempts remain preserved in their original directories. They
are not final account acceptance and are never silently spliced into the final
62-scenario run. The default model and the original account histories remain
unchanged regardless of the restoration outcome.
