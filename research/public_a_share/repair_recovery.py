"""Budgeted content-addressed recovery of the completed repair campaign."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from study_v3_archive import GIB, checked_copy, contained, digest, write_json
from study_v3_recovery import size


def collect_tree(entries, prefix, root):
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("recovery input cannot be a symlink")
        if path.is_file() and "__pycache__" not in path.parts:
            entries[prefix + "/" + path.relative_to(root).as_posix()] = path


def run(args):
    acceptance = json.loads((args.report / "acceptance.json").read_text())
    if acceptance["engineering_research_delivery"] != "complete_with_explicit_unresolved_cases":
        raise ValueError("cannot archive incomplete research acceptance")
    release = json.loads((args.code / "release-evidence.json").read_text())
    if release["engineering_acceptance"] != "passed":
        raise ValueError("cannot archive unverified engineering release")
    if not args.version.replace("-", "").isalnum():
        raise ValueError("version must contain only letters, numbers and hyphens")
    archive = args.backup / args.version
    restored = args.data / "restore-check" / args.version
    if (archive / "manifest.json").exists():
        raise ValueError("recovery manifest is immutable")
    entries = {}
    for label, root in (
        ("run", args.run), ("report", args.report),
        ("integer-audit", args.code / "audit-04"),
        ("certificates", args.code / "certified-feasibility-04"),
        ("data-evidence", args.code / "data-evidence-02"),
        ("corporate-evidence", args.code / "corporate-evidence"),
        ("runtime-research", args.code / "runtime-research-03"),
        ("git-bundles", args.code / "git-bundles"),
        ("report-source", args.code / "quant-risk-monitor/research/public_a_share"),
    ):
        collect_tree(entries, label, root)
    for name in ("environment-verification.json", "release-evidence.json", "runtime-pins.lock",
                 "share-rules-strict.json", "share-rules-sensitivity_floor.json",
                 "share-rules-sensitivity_ceil.json", "RECOVERY.md", "IMPLEMENTATION.md"):
        entries["environment/" + name] = args.code / name
    for path in args.code.glob("*.log"):
        entries["engineering-logs/" + path.name] = path
    source = args.data / "runs/main"
    for name in (
        "panel.pkl", "histories.pkl", "portfolio-risk-models.pkl", "risk-models.pkl",
        "execution-bars.parquet", "execution-actions.parquet", "corporate-action-events.pkl",
        "execution-catalog.csv", "protocol.json", "execution-input-guards.json",
        "execution-input-issues.json", "source-trading-status.parquet", "daily-quality.parquet",
        "portfolio-risk-models-binding.json", "history-binding-v3.json",
    ):
        entries["frozen-inputs/" + name] = source / name
    manifest = {
        key: {"sha256": digest(path), "bytes": path.stat().st_size}
        for key, path in sorted(entries.items())
    }
    # Reuse old verified objects by content digest, not by a mutable filename.
    existing = {}
    for root in (args.backup / "objects", args.backup / "delivery-v1/objects",
                 args.backup / "supplemental/objects", archive / "objects"):
        if root.exists():
            existing.update({p.name: p for p in root.iterdir() if p.is_file()})
    unique = {meta["sha256"]: meta["bytes"] for meta in manifest.values()}
    new_archive_bytes = sum(amount for sha, amount in unique.items() if sha not in existing)
    restore_bytes = sum(meta["bytes"] for meta in manifest.values())
    if size(args.backup) + new_archive_bytes > 20 * GIB:
        raise ValueError("F archive budget exceeded; no evidence deleted")
    if size(args.data) + restore_bytes > 25 * GIB:
        raise ValueError("D restore budget exceeded; no evidence deleted")
    if size(args.code) > 5 * GIB or shutil.disk_usage(args.code).free < 40 * GIB:
        raise ValueError("C workspace budget/reserve exceeded")
    for key, source_path in entries.items():
        sha = manifest[key]["sha256"]
        object_path = existing.get(sha, contained(archive, "objects/" + sha))
        checked_copy(source_path, object_path, sha)
        manifest[key]["object"] = object_path.relative_to(args.backup).as_posix()
        checked_copy(object_path, contained(restored, key), sha)
    write_json(archive / "manifest.json", {"files": manifest})
    receipt = {
        "status": "passed", "created_at": datetime.now(timezone.utc).isoformat(),
        "files_restored_and_hashed": len(manifest), "manifest": str(archive / "manifest.json"),
        "manifest_sha256": digest(archive / "manifest.json"), "restored": str(restored),
        "backup_kind": "same-machine cross-drive copy; hardware independence not asserted",
        "original_raw_inputs": "retained in earlier content-addressed archive; not redownloaded",
        "data_bytes": size(args.data), "archive_bytes": size(args.backup),
        "code_environment_bytes": size(args.code),
        "environment_recovery": "new ordinary environment from exact lock; external distributions required",
    }
    write_json(args.code / "recovery-receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("data", "backup", "code", "report", "run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--version", required=True)
    run(parser.parse_args())
