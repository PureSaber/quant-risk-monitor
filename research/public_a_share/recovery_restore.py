"""Verify the recovery archive and optionally restore selected immutable files."""

import argparse
import json
from pathlib import Path, PurePosixPath

from recovery_inventory import digest
from repair_recovery import verify_object


def contained(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or ":" in relative or "\\" in relative:
        raise ValueError("archive path must be a relative POSIX path")
    root = root.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("archive path escapes its declared root")
    return target


def run(manifest_path, expected_sha, target=None, keys=()):
    if digest(manifest_path) != expected_sha:
        raise ValueError("archive manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent.parent
    missing = set(keys) - manifest["files"].keys()
    if missing or (keys and target is None):
        raise ValueError("restore requires exact existing keys and a target directory")
    seen, restored = set(), []
    for key, metadata in manifest["files"].items():
        path = contained(root, metadata["object"])
        destination = contained(target, key) if key in keys else None
        if metadata["encoding"] not in {"identity", "gzip"}:
            raise ValueError("unknown archive encoding")
        if metadata.get("encoded_sha256") and digest(path) != metadata["encoded_sha256"]:
            raise ValueError("compressed archive object changed")
        if metadata["sha256"] not in seen or destination:
            verify_object(path, metadata, destination)
            seen.add(metadata["sha256"])
        if destination:
            restored.append({"key": key, "path": str(destination), "sha256": digest(destination)})
    result = {"manifest_sha256": expected_sha, "unique_objects_verified": len(seen),
              "files_restored": restored, "overwrites_allowed": False}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--key", action="append", default=[])
    args = parser.parse_args()
    run(args.manifest, args.sha256, args.target, args.key)
