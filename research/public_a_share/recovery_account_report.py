"""Compare restored-observation accounts with the frozen strict replay.

The account verifier uses exact decimal money and an independent factor-risk
calculation. No successful or stopped account is dropped from the comparison.
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd
from repair_account_audit import verify

from study_v3_archive import digest, write_json
from study_v3_report import batch_results, read


def run(source, coverage, replay, baseline, out, batch_id):
    if (out / "account-acceptance.json").exists():
        raise ValueError("completed account audit is immutable")
    binding = read(coverage / "coverage-binding.json")
    if digest(coverage / "portfolio-risk-models.pkl") != binding["output"]:
        raise ValueError("risk coverage cache changed")
    for name, sha in binding["inputs"].items():
        if digest(source / name) != sha:
            raise ValueError("account input changed: " + name)
    rows = batch_results(replay, batch_id, allow_partial=False)
    if any("outputs" not in row for row in rows):
        raise ValueError("all account workers must produce verifiable artifacts")
    old = {}
    for folder in sorted((baseline / "portfolios").iterdir()):
        if not (folder / "complete.json").exists():
            continue
        row = read(folder / "complete.json")
        for name, sha in row["outputs"].items():
            if digest(folder / name) != sha:
                raise ValueError("baseline account artifact changed")
        if row["scenario"] in old:
            raise ValueError("ambiguous frozen scenario")
        old[row["scenario"]] = (row, read(folder / "specification.json"))
    if set(old) != set(range(62)):
        raise ValueError("all 62 frozen baseline accounts are required")
    # This pickle is a trusted local artifact, checked against its binding above.
    with (coverage / "portfolio-risk-models.pkl").open("rb") as stream:
        models = pickle.load(stream)
    bars = pd.read_parquet(source / "execution-bars.parquet")
    statuses = pd.read_parquet(source / "source-trading-status.parquet")
    out.mkdir(parents=True, exist_ok=True)
    checks, records, failures, stops = {}, [], [], []
    metrics = (
        "status", "full_horizon_completed", "evaluated_through", "published", "decisions",
        "net_return", "maximum_drawdown", "actual_annualized_tracking_error",
        "actual_te_complete_windows", "total_fees", "gross_traded_notional_over_initial_capital",
        "rebalance_p95_seconds", "wall_seconds", "process_peak_bytes", "filled_orders",
        "partially_filled_orders", "longest_no_target_rebalances", "effective_rebalance_rate",
    )
    for row in rows:
        index, folder, spec = row["scenario"], Path(row["folder"]), row["specification"]
        original, old_spec = old[index]
        for key in ("model", "method", "budget", "te", "benchmark", "share_branch", "limit_dates"):
            if spec.get(key) != old_spec.get(key):
                raise ValueError("frozen scenario specification changed: " + key)
        print(json.dumps({"scenario": index, "phase": "independent_account_audit"}), flush=True)
        checks[str(index)] = verify(folder, models, bars, statuses)
        record = {"scenario": index, **{k: spec.get(k) for k in ("model", "method", "budget", "te")}}
        record.update({k: row.get(k) for k in metrics})
        record.update({"old_" + k: original.get(k) for k in metrics})
        record["return_endpoints_match"] = row.get("evaluated_through") == original.get("evaluated_through")
        # Different stopping dates are not a comparable return improvement.
        record["comparable_return_difference"] = (
            row["net_return"] - original["net_return"] if record["return_endpoints_match"] else None
        )
        records.append(record)
        for decision in read(folder / "allocation-decisions.json"):
            if "weights" not in decision:
                failures.append({
                    "scenario": index, "date": decision["session"],
                    "status": decision.get("status"),
                    "reason": decision.get("error", decision.get("reason")),
                })
        stop_path = folder / "full-horizon-stop.json"
        if stop_path.exists():
            stops.append({"scenario": index, **read(stop_path)})
    frame = pd.DataFrame(records)
    frame.to_csv(out / "account-comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(out / "account-failures.csv", index=False, encoding="utf-8-sig")
    write_json(out / "account-stops.json", stops)
    write_json(out / "account-independent-checks.json", checks)
    acceptance = {
        "schema": "restored-observation-account-audit/1",
        "producer_sha256": digest(__file__), "coverage_binding": binding,
        "scenarios": len(rows), "independently_verified_accounts": len(checks),
        "published_targets": int(frame.published.sum()),
        "old_published_targets": int(frame.old_published.sum()),
        "complete_horizon": int(frame.full_horizon_completed.sum()),
        "old_complete_horizon": int(frame.old_full_horizon_completed.sum()),
        "zero_target_scenarios": int(frame.published.eq(0).sum()),
        "old_zero_target_scenarios": int(frame.old_published.eq(0).sum()),
        "status_counts": dict(Counter(frame.status)),
        "failure_counts": dict(Counter(row["status"] for row in failures)),
        "maximum_independent_residual": max(
            row["independent_maximum_residual"] for row in checks.values()
        ),
        "original_constraints_preserved": True,
        "imputation_used_for_account_decisions": False,
        "default_model_changed": False, "model_promotion": "not_passed",
        "outputs": {name: digest(out / name) for name in (
            "account-comparison.csv", "account-failures.csv", "account-stops.json",
            "account-independent-checks.json",
        )},
    }
    write_json(out / "account-acceptance.json", acceptance)
    print(json.dumps({k: v for k, v in acceptance.items() if k != "coverage_binding"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "coverage", "replay", "baseline", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    run(args.source, args.coverage, args.replay, args.baseline, args.out, args.batch_id)
