"""Archive supplemental evidence and verify restoration within explicit disk budgets."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from study_v3_archive import GIB, checked_copy, contained, digest, write_json


def size(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def run(data, backup, code):
    entries = []
    for name in ("corporate-actions", "action-evidence", "bao-dividends", "stock-basic.json"):
        root = data / "inputs" / name
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        if not root.exists():
            raise FileNotFoundError(root)
        for path in paths:
            if path.is_symlink():
                raise ValueError("evidence cannot follow symbolic links")
            if path.is_file():
                entries.append((path, path.relative_to(data / "inputs").as_posix()))
    manifest = {
        relative: {"sha256": digest(path), "bytes": path.stat().st_size}
        for path, relative in entries
    }
    logical = sum(row["bytes"] for row in manifest.values())
    unique = {row["sha256"]: row["bytes"] for row in manifest.values()}
    archive = backup / "supplemental"
    restore = data / "restore-check" / "supplemental"
    receipt_path = archive / "manifest.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding="utf8"))
        if previous["files"] != manifest:
            raise ValueError("supplemental evidence inventory changed; use a new version")
    else:
        previous = {"created_at": datetime.now(timezone.utc).isoformat(), "files": manifest}
    new_backup = sum(
        amount for sha, amount in unique.items() if not (archive / "objects" / sha).exists()
    )
    new_restore = sum(
        meta["bytes"] for path, meta in manifest.items() if not (restore / path).exists()
    )
    if size(backup) + new_backup > 20 * GIB or size(data) + new_restore > 25 * GIB:
        raise ValueError("approved D/F research budget would be exceeded; no deletion performed")
    if size(code) > 5 * GIB or shutil.disk_usage(code).free < 40 * GIB:
        raise ValueError("C workspace budget/reserve exceeded")
    for source, relative in entries:
        sha = manifest[relative]["sha256"]
        obj = contained(archive, "objects/" + sha)
        checked_copy(source, obj, sha)
        checked_copy(obj, contained(restore, relative), sha)
    write_json(receipt_path, previous)
    result = {
        "status": "passed",
        "files_restored_and_hashed": len(entries),
        "logical_bytes": logical,
        "unique_bytes": sum(unique.values()),
        "manifest_sha256": digest(receipt_path),
        "backup_kind": "same-machine copy; hardware independence not asserted",
        "data_bytes": size(data),
        "archive_bytes": size(backup),
        "code_environment_bytes": size(code),
        "free_gib": {
            drive: round(shutil.disk_usage(drive + ":/").free / GIB, 2)
            for drive in ("C", "D", "E", "F")
        },
    }
    write_json(data / "supplemental-recovery-receipt.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("data", "backup", "code"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.backup, args.code)
