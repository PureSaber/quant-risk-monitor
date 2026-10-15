"""Finish one already-started strict replay, then isolated sensitivity branches."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from study_v3_report import batch_results


def run(args):
    strict = args.run / "strict"
    progress = strict / "portfolio-logs-full03/progress.json"
    started = time.monotonic()
    while True:
        records = json.loads(progress.read_text()) if progress.exists() else []
        if len(records) == 62:
            break
        if time.monotonic() - started > 24 * 3600:
            raise TimeoutError("strict replay not complete after 24 hours; evidence retained")
        if any(row["exit_code"] != 0 for row in records):
            raise RuntimeError("strict worker failed; do not launch dependent sensitivity runs")
        time.sleep(15)
    rows = batch_results(strict, "full03", allow_partial=False)
    original = batch_results(args.source, "cadence-03", allow_partial=False)
    needed = sorted({r["scenario"] for r in original + rows if not r.get("full_horizon_completed")})
    if any("outputs" not in row for row in rows):
        raise RuntimeError("strict outputs are incomplete")
    print(json.dumps({"phase": "strict_verified", "sensitivity_scenarios": needed}), flush=True)
    for label in ("sensitivity_floor", "sensitivity_ceil"):
        command = [
            sys.executable, str(args.runtime / "study_v3_batch.py"),
            "--source", str(args.source), "--out", str(args.run / label),
            "--coverage", str(args.run / "coverage"),
            "--share-rules", str(args.code / f"share-rules-{label}.json"),
            "--workers", "4", "--batch-id", "full03",
            "--scenarios", ",".join(map(str, needed)),
        ]
        subprocess.run(command, check=True)
        results = batch_results(args.run / label, "full03", allow_partial=True)
        if {row["scenario"] for row in results} != set(needed) or any(
            "outputs" not in row for row in results
        ):
            raise RuntimeError("sensitivity replay has a failed or missing account")
        print(json.dumps({"phase": "branch_verified", "branch": label, "accounts": len(results)}),
              flush=True)
    command = [
        sys.executable, str(Path(__file__).with_name("repair_report.py")),
        "--source", str(args.source), "--coverage", str(args.run / "coverage"),
        "--out", str(args.run / "reports-v1"),
        "--integer-audit", str(args.code / "audit-04"),
        "--evidence", str(args.code / "data-evidence-02"),
        "--certificates", str(args.code / "certified-feasibility-04"),
    ]
    for label in ("strict", "sensitivity_floor", "sensitivity_ceil"):
        command += ["--branch", label, str(args.run / label), "full03"]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "run", "code", "runtime"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
