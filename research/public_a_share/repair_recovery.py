"""Budgeted content-addressed recovery of the completed repair campaign."""

import argparse
import gzip
import hashlib
import json
import os
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


def verify_object(path, metadata, target=None):
    """Read every archived byte and verify its decoded original SHA-256."""
    opener = gzip.open if metadata["encoding"] == "gzip" else open
    result, length = hashlib.sha256(), 0
    temporary = None if target is None else target.with_suffix(target.suffix + ".pending")
    if target is not None:
        if target.exists():
            if digest(target) != metadata["sha256"]:
                raise ValueError("existing restoration differs; not overwritten")
            target = temporary = None
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
    output = None if temporary is None else temporary.open("wb")
    try:
        with opener(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(chunk)
                length += len(chunk)
                if output is not None:
                    output.write(chunk)
    finally:
        if output is not None:
            output.close()
    if result.hexdigest() != metadata["sha256"] or length != metadata["bytes"]:
        raise ValueError("restored payload changed")
    if target is not None:
        os.replace(temporary, target)


def restore_selection(key):
    # Decode every object, but materialize only the inputs and representative
    # accounts used in the recovery exercise. Other objects remain restorable.
    if key.startswith("run/"):
        if key in ("run/coverage/portfolio-risk-models.pkl", "run/coverage/coverage-binding.json"):
            return True
        parts = key.split("/")
        return len(parts) >= 5 and parts[2] == "portfolios" and parts[3][:4] in ("016-", "046-")
    if key.startswith("frozen-inputs/"):
        return key.split("/")[-1] in ("execution-bars.parquet", "source-trading-status.parquet")
    return not key.startswith("engineering-logs/")


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
        ("compression-parity", args.code / "compression-parity-01"),
        ("runtime-research", args.code / "runtime-research-03"),
        ("runtime-research-compressed", args.code / "runtime-research-04"),
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
    staged = {}
    staging = args.code / "recovery-staging"
    staging.mkdir(exist_ok=True)
    code_bytes = size(args.code)
    for key, source_path in entries.items():
        meta = manifest[key]
        sha = meta["sha256"]
        if sha in existing:
            meta.update(encoding="identity", object=existing[sha].relative_to(args.backup).as_posix())
        else:
            packed = staging / (sha + ".gz")
            if sha not in staged:
                if code_bytes + meta["bytes"] > 5 * GIB:
                    raise ValueError("C compression staging could exceed budget; no evidence deleted")
                if not packed.exists():
                    with source_path.open("rb") as src, packed.open("wb") as raw:
                        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                            shutil.copyfileobj(src, stream)
                    code_bytes += packed.stat().st_size
                staged[sha] = packed
            meta.update(encoding="gzip", object=f"{args.version}/objects/{sha}.gz",
                        encoded_sha256=digest(packed), encoded_bytes=packed.stat().st_size)
    new_archive_bytes = sum(path.stat().st_size for sha, path in staged.items()
                            if not (archive / "objects" / (sha + ".gz")).exists())
    restore_bytes = sum(meta["bytes"] for key, meta in manifest.items()
                        if restore_selection(key) and not contained(restored, key).exists())
    if size(args.backup) + new_archive_bytes > 20 * GIB:
        raise ValueError("F archive budget exceeded; no evidence deleted")
    if size(args.data) + restore_bytes > 25 * GIB:
        raise ValueError("D restore budget exceeded; no evidence deleted")
    if size(args.code) > 5 * GIB or shutil.disk_usage(args.code).free < 40 * GIB:
        raise ValueError("C workspace budget/reserve exceeded")
    restored_files = []
    for key, source_path in entries.items():
        meta = manifest[key]
        sha = meta["sha256"]
        object_path = contained(args.backup, meta["object"])
        if meta["encoding"] == "gzip":
            checked_copy(staged[sha], object_path, meta["encoded_sha256"])
        else:
            checked_copy(source_path, object_path, sha)
        target = contained(restored, key) if restore_selection(key) else None
        verify_object(object_path, meta, target)
        if target is not None:
            restored_files.append(key)
    write_json(archive / "manifest.json", {"files": manifest})
    receipt = {
        "status": "passed", "created_at": datetime.now(timezone.utc).isoformat(),
        "all_files_decoded_and_hashed": len(manifest),
        "representative_files_restored": len(restored_files), "restored_files": restored_files,
        "manifest": str(archive / "manifest.json"),
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
