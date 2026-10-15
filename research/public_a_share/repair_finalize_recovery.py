"""Seal post-restoration evidence without rewriting the base recovery manifest."""

import argparse
import json
from pathlib import Path

from study_v3_archive import GIB, checked_copy, contained, digest, write_json
from study_v3_recovery import size


def finalize(code, report, backup):
    receipt_path = code / "recovery-receipt.json"
    checks_path = code / "recovery-account-checks.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    checks = json.loads(checks_path.read_text(encoding="utf8"))
    acceptance = json.loads((report / "acceptance.json").read_text(encoding="utf8"))
    release = json.loads((code / "release-evidence.json").read_text(encoding="utf8"))
    if receipt["status"] != "passed" or checks["status"] != "passed":
        raise ValueError("restoration has not passed")
    expected = {f"{branch}:{scenario}" for branch in
                ("strict", "sensitivity_floor", "sensitivity_ceil") for scenario in (16, 46)}
    if set(checks["checks"]) != expected or checks["representative_accounts"] != 6:
        raise ValueError("representative restored accounts are incomplete")
    if checks["recovery_receipt_sha256"] != digest(receipt_path):
        raise ValueError("restored-account audit belongs to a different receipt")
    manifest = Path(receipt["manifest"]).resolve()
    relative = manifest.relative_to(backup.resolve())
    if contained(backup, relative).resolve() != manifest:
        raise ValueError("recovery manifest is outside the authorized archive")
    if digest(manifest) != receipt["manifest_sha256"] or (
        checks["manifest_sha256"] != receipt["manifest_sha256"]
    ):
        raise ValueError("base recovery manifest changed")
    if acceptance["engineering_research_delivery"] != "complete_with_explicit_unresolved_cases":
        raise ValueError("research acceptance is incomplete")
    if release["engineering_acceptance"] != "passed":
        raise ValueError("engineering acceptance is incomplete")
    qa = json.loads((report / "html-qa.json").read_text(encoding="utf8"))
    if qa["status"] != "passed" or qa["report_sha256"] != digest(report / "report.html"):
        raise ValueError("report QA does not match the completed report")
    sources = {name: code / name for name in (
        "recovery-receipt.json", "recovery-account-checks.json", "release-evidence.json",
        "IMPLEMENTATION.md", "RECOVERY.md",
    )}
    sources.update({name: report / name for name in ("acceptance.json", "html-qa.json")})
    target = manifest.parent / "final-evidence"
    if (target / "manifest.json").exists():
        raise ValueError("final evidence is immutable")
    if size(backup) + sum(p.stat().st_size for p in sources.values()) > 20 * GIB:
        raise ValueError("archive budget exceeded")
    files = {}
    for name, source in sources.items():
        sha = digest(source)
        checked_copy(source, target / name, sha)
        files[name] = {"sha256": sha, "bytes": source.stat().st_size}
    result = {
        "status": "passed", "files": files,
        "base_manifest": str(manifest), "base_manifest_sha256": receipt["manifest_sha256"],
        "representative_restored_accounts": 6,
        "scope": "post-restoration evidence; base manifest and archived ledgers remain immutable",
    }
    write_json(target / "manifest.json", result)
    write_json(code / "final-evidence-receipt.json", {
        "status": "passed", "manifest": str(target / "manifest.json"),
        "manifest_sha256": digest(target / "manifest.json"), "files": len(files),
    })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("code", "report", "backup"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.code, args.report, args.backup)), flush=True)
