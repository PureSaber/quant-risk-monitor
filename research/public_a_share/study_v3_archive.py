"""Freeze, deduplicate and restore the explicitly selected public research inputs.

No source files are modified. All destination files are content checked, including
on resume. The archive is a same-machine recovery copy, not an off-site backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024**3


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8"
    )
    os.replace(temporary, path)


def contained(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if target == root or root not in target.parents:
        raise ValueError("destination escapes dedicated research directory")
    return target


def inventory(old, exposure):
    selected = {
        "old": (old, ["raw", "clean", "cohort.json", "protocol.json", "environment.lock"]),
        "exposure": (
            exposure,
            [
                "raw",
                "confirmation-inputs",
                "annual-warmup-facts.csv",
                "protocol.json",
                "selection.json",
                "acceptance.json",
                "confirmation/selection-lock.json",
                "confirmation/source-manifest.json",
                "confirmation/price-bridges.json",
                "confirmation/result.json",
                "confirmation/merged-inputs",
            ],
        ),
    }
    result = []
    for alias, (root, names) in selected.items():
        root = root.resolve()
        for name in names:
            target = contained(root, name)
            if not target.exists():
                raise FileNotFoundError(target)
            paths = sorted(target.rglob("*")) if target.is_dir() else [target]
            for path in paths:
                if path.is_symlink():
                    raise ValueError("input archive must not follow symbolic links")
                if path.is_file():
                    result.append((alias, root, path, path.stat().st_size))
    return result


def checked_copy(source, target, expected):
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f"existing destination changed: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".pending")
    shutil.copyfile(source, temporary)
    if digest(temporary) != expected:
        raise ValueError("source changed during copy or copy failed integrity check")
    os.replace(temporary, target)


def freeze(old, exposure, out, backup):
    old, exposure, out, backup = [Path(x).resolve() for x in (old, exposure, out, backup)]
    for destination in (out, backup):
        if any(destination == p or p in destination.parents for p in (old, exposure)):
            raise ValueError("output must not be inside immutable source archive")
    entries = inventory(old, exposure)
    total = sum(x[3] for x in entries)
    if total > 20 * GIB:
        raise ValueError("input archive exceeds the approved 20 GiB recovery budget")
    for path, reserve in ((out, 25 * GIB), (backup, 20 * GIB)):
        path.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(path).free < total + reserve:
            raise ValueError("insufficient free space for snapshot and operational reserve")
    manifest_path = backup / "manifest.json"
    previous = (
        json.loads(manifest_path.read_text(encoding="utf8")) if manifest_path.exists() else None
    )
    rows = []
    for count, (alias, root, source, size) in enumerate(entries, 1):
        sha = digest(source)
        relative = alias + "/" + source.relative_to(root).as_posix()
        if previous:
            existing = previous["files"].get(relative)
            if existing != {"sha256": sha, "bytes": size}:
                raise ValueError("immutable input inventory changed on resume")
        obj = contained(backup, "objects/" + sha[:2] + "/" + sha)
        checked_copy(source, obj, sha)
        restored = contained(out, "inputs/" + relative)
        checked_copy(obj, restored, sha)
        rows.append((relative, {"sha256": sha, "bytes": size}))
        if count % 1000 == 0:
            print(json.dumps({"archived_and_restored": count, "total": len(entries)}), flush=True)
    if previous and set(previous["files"]) != {x[0] for x in rows}:
        raise ValueError("input inventory lost files")
    manifest = previous or {
        "schema": "public-barra-200-input-archive/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_roots": {"old": str(old), "exposure": str(exposure)},
        "files": dict(rows),
    }
    write_json(manifest_path, manifest)
    write_json(out / "input-manifest.json", manifest)
    unique = {v["sha256"]: v["bytes"] for _, v in rows}
    receipt = {
        "files": len(rows),
        "logical_bytes": total,
        "unique_bytes": sum(unique.values()),
        "manifest_sha256": digest(manifest_path),
        "restored_files_verified": len(rows),
        "source_mutations": False,
        "backup_kind": "same_machine_content_addressed_copy",
    }
    write_json(out / "archive-receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("old", "exposure", "out", "backup"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    freeze(args.old, args.exposure, args.out, args.backup)
