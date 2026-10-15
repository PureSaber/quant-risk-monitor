"""Freeze final study outputs and verify an isolated same-machine restoration."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from study_v3_archive import GIB, checked_copy, contained, digest, write_json
from study_v3_recovery import size


def run(data, backup, code, version):
    if not version.replace("-", "").isalnum():
        raise ValueError("delivery version must contain letters, numbers and hyphens")
    main = data / "runs/main"
    acceptance = json.loads((main / "study-acceptance.json").read_text(encoding="utf8"))
    if acceptance["study_delivery"] != "complete_with_recorded_limitations":
        raise ValueError("cannot freeze an incomplete study delivery")
    release = json.loads((main / "release-evidence.json").read_text(encoding="utf8"))
    if release["engineering_acceptance"] != "passed":
        raise ValueError("cannot freeze pending or failed engineering checks")
    entries = {
        path.relative_to(data).as_posix(): path
        for path in main.rglob("*")
        if path.is_file() and "daily-cache" not in path.relative_to(main).parts
    }
    for relative in (
        "runs/independent-01/independent-recomputation.json",
        "supplemental-recovery-receipt.json",
        "archive-receipt.json",
    ):
        path = data / relative
        if path.exists():
            entries[relative] = path
    for path in code.glob("*.log"):
        entries["engineering-logs/" + path.name] = path
    research = code / "quant-risk-monitor/research/public_a_share"
    for path in research.iterdir():
        if path.is_file():
            entries["research-source/" + path.name] = path
    manifest = {
        name: {"sha256": digest(path), "bytes": path.stat().st_size}
        for name, path in sorted(entries.items())
    }
    archive, restored = backup / version, data / "restore-check" / version
    if (archive / "manifest.json").exists():
        raise ValueError("delivery manifest is immutable; use a new version")
    unique = {v["sha256"]: v["bytes"] for v in manifest.values()}
    if size(backup) + sum(unique.values()) > 20 * GIB:
        raise ValueError("F archive would exceed 20 GiB; no files removed")
    if size(data) + sum(v["bytes"] for v in manifest.values()) > 25 * GIB:
        raise ValueError("D data would exceed 25 GiB; no files removed")
    if size(code) > 5 * GIB or shutil.disk_usage(code).free < 40 * GIB:
        raise ValueError("C code budget or free-space reserve would be exceeded")
    for name, path in entries.items():
        sha = manifest[name]["sha256"]
        obj = contained(archive, "objects/" + sha)
        checked_copy(path, obj, sha)
        checked_copy(obj, contained(restored, name), sha)
    write_json(archive / "manifest.json", {"files": manifest})
    result = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files_restored_and_hashed": len(entries),
        "manifest_sha256": digest(archive / "manifest.json"),
        "manifest": str(archive / "manifest.json"),
        "restoration": str(restored),
        "backup_kind": "same-machine recovery copy; no independent hardware claim",
        "excluded_rebuildable_cache": "runs/main/daily-cache",
        "environment_recovery": "ordinary install from requirements-v3.lock; package sources/network required",
        "data_bytes": size(data),
        "archive_bytes": size(backup),
        "code_environment_bytes": size(code),
        "free_gib": {d: round(shutil.disk_usage(d + ":/").free / GIB, 2) for d in "CDEF"},
    }
    write_json(data / (version + "-recovery-receipt.json"), result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("data", "backup", "code"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    run(args.data, args.backup, args.code, args.version)
