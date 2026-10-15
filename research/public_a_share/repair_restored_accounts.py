"""Independently recheck representative accounts from restored inputs only."""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from repair_account_audit import verify

from study_v3_archive import digest, write_json


def run(receipt_path, output):
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    if receipt["status"] != "passed":
        raise ValueError("recovery objects have not passed verification")
    manifest_path = Path(receipt["manifest"])
    if digest(manifest_path) != receipt["manifest_sha256"]:
        raise ValueError("recovery manifest changed")
    files = json.loads(manifest_path.read_text(encoding="utf8"))["files"]
    restored = Path(receipt["restored"])
    for name in receipt["restored_files"]:
        if digest(restored / name) != files[name]["sha256"]:
            raise ValueError("restored file changed: " + name)
    # The executable auditor must itself come from the verified restored tree.
    expected = restored / "report-source" / Path(__file__).name
    if Path(__file__).resolve() != expected.resolve():
        raise ValueError("run the auditor in the restored report-source directory")
    with (restored / "run/coverage/portfolio-risk-models.pkl").open("rb") as stream:
        models = pickle.load(stream)
    bars = pd.read_parquet(restored / "frozen-inputs/execution-bars.parquet")
    statuses = pd.read_parquet(restored / "frozen-inputs/source-trading-status.parquet")
    checks = {}
    for branch in ("strict", "sensitivity_floor", "sensitivity_ceil"):
        for scenario in (16, 46):
            folders = list((restored / "run" / branch / "portfolios").glob(f"{scenario:03d}-*"))
            if len(folders) != 1:
                raise ValueError("missing or ambiguous representative account")
            folder = folders[0]
            complete = json.loads((folder / "complete.json").read_text(encoding="utf8"))
            for name, sha in complete["outputs"].items():
                if digest(folder / name) != sha:
                    raise ValueError("restored account receipt mismatch")
            checks[f"{branch}:{scenario}"] = verify(folder, models, bars, statuses)
            print(json.dumps({"branch": branch, "scenario": scenario, "check": "passed"}),
                  flush=True)
    if output.exists():
        raise ValueError("restored-account evidence is immutable")
    write_json(output, {
        "status": "passed", "representative_accounts": len(checks), "checks": checks,
        "restored": str(restored), "recovery_receipt_sha256": digest(receipt_path),
        "manifest_sha256": receipt["manifest_sha256"], "auditor_sha256": digest(__file__),
        "scope": "restored exact accounts and risk inputs; not a new strategy replay",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.receipt, args.output)
