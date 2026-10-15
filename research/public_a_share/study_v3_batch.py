"""Bounded process orchestration; each scenario owns one native account and output."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from study_v3_archive import write_json


def run(out, indices, workers, limit_dates, batch_id=""):
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between one and four")
    script = Path(__file__).with_name("study_v3_portfolios.py")
    if batch_id and (not batch_id.replace("-", "").isalnum()):
        raise ValueError("batch_id must contain only letters, numbers and hyphens")
    logs = out / (
        ("pilot-logs" if limit_dates else "portfolio-logs") + ("-" + batch_id if batch_id else "")
    )
    logs.mkdir(exist_ok=True)

    def one(index):
        args = [sys.executable, str(script), "--out", str(out), "--scenario", str(index)]
        if limit_dates:
            args += ["--limit-dates", str(limit_dates)]
        log = logs / f"scenario-{index:03d}.log"
        if log.exists():
            raise ValueError("batch logs are immutable; use a new explicit log directory/version")
        with log.open("w", encoding="utf8") as stream:
            result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, check=False)
        return {"scenario": index, "exit_code": result.returncode, "log": str(log)}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, index) for index in indices]
        for future in as_completed(futures):
            record = future.result()
            results.append(record)
            write_json(logs / "progress.json", sorted(results, key=lambda r: r["scenario"]))
            print(
                json.dumps(record | {"finished": len(results), "total": len(indices)}), flush=True
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", default=",".join(map(str, range(62))))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-dates", type=int)
    parser.add_argument("--batch-id", default="")
    args = parser.parse_args()
    run(
        args.out,
        [int(x) for x in args.scenarios.split(",")],
        args.workers,
        args.limit_dates,
        args.batch_id,
    )
