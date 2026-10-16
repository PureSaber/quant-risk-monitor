import json

import pytest
from repair_finalize_recovery import finalize

from study_v3_archive import digest


def inputs(tmp_path):
    code, report, backup = [tmp_path / x for x in ("code", "report", "backup")]
    for folder in (code, report, backup):
        folder.mkdir()
    manifest = backup / "manifest.json"
    manifest.write_text("{}")
    receipt = code / "recovery-receipt.json"
    receipt.write_text(json.dumps({"status": "passed", "manifest": str(manifest),
                                   "manifest_sha256": digest(manifest)}))
    checks = {f"{b}:{s}": {} for b in ("strict", "sensitivity_floor", "sensitivity_ceil")
              for s in (16, 46)}
    (code / "recovery-account-checks.json").write_text(json.dumps({
        "status": "passed", "checks": checks, "representative_accounts": 6,
        "recovery_receipt_sha256": digest(receipt), "manifest_sha256": digest(manifest),
    }))
    (code / "release-evidence.json").write_text('{"engineering_acceptance":"passed"}')
    (code / "IMPLEMENTATION.md").write_text("Complete, with stated unresolved cases.")
    (code / "RECOVERY.md").write_text("Restore the immutable archived version.")
    (report / "acceptance.json").write_text(json.dumps({
        "engineering_research_delivery": "complete_with_explicit_unresolved_cases",
    }))
    (report / "report.html").write_text("<h1>Verified</h1>")
    (report / "html-qa.json").write_text(json.dumps({
        "status": "passed", "report_sha256": digest(report / "report.html"),
    }))
    return code, report, backup


def test_final_evidence_is_exact_and_does_not_rewrite_base_manifest(tmp_path):
    code, report, backup = inputs(tmp_path)
    (report / "html-qa-v2.json").write_bytes((report / "html-qa.json").read_bytes())
    before = (backup / "manifest.json").read_bytes()
    result = finalize(code, report, backup)
    assert result["status"] == "passed" and len(result["files"]) == 7
    assert "html-qa-v2.json" in result["files"] and "html-qa.json" not in result["files"]
    assert (backup / "manifest.json").read_bytes() == before
    for name, meta in result["files"].items():
        assert digest(backup / "final-evidence" / name) == meta["sha256"]
    with pytest.raises(ValueError, match="immutable"):
        finalize(code, report, backup)


def test_final_evidence_rejects_a_different_restoration_receipt(tmp_path):
    code, report, backup = inputs(tmp_path)
    with (code / "recovery-receipt.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="different receipt"):
        finalize(code, report, backup)
    assert not (backup / "final-evidence").exists()
