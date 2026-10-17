"""Recompute recovered-input account risks with the original pools/benchmarks.

The output is a new experiment, never a replacement for a frozen run. Only
observed financial descriptor overlays are applied; imputation candidates are
not admitted here. The original execution data and constraints are copied.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from dataclasses import asdict
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import pandas as pd  # noqa: E402

import exposure_v2_study as study  # noqa: E402
from recovery_inventory import digest, write_json  # noqa: E402
from run_study import risk_components  # noqa: E402
from study_v3_portfolios import GENERATORS  # noqa: E402
from study_v3_risk import model_definitions  # noqa: E402

FIELDS = (
    "book_to_price", "earnings_yield_ttm", "earnings_growth_yoy", "book_leverage",
    "revenue_growth_3y", "profit_growth_3y",
)
EXECUTION_FILES = (
    "execution-bars.parquet", "execution-actions.parquet", "corporate-action-events.pkl",
    "execution-catalog.csv", "execution-input-guards.json", "execution-input-issues.json",
    "source-trading-status.parquet",
)


def apply_overlay(bundle, recovered):
    for key in ("panel", "v2"):
        bundle[key] = {**bundle[key], "descriptors": dict(bundle[key]["descriptors"])}
        for name in FIELDS:
            if name in bundle[key]["descriptors"]:
                values = pd.read_parquet(recovered / (name + ".parquet"))
                old = bundle[key]["descriptors"][name]
                if not values.index.equals(old.index) or not values.columns.equals(old.columns):
                    raise ValueError("descriptor_axis_changed")
                bundle[key]["descriptors"][name] = values
    return bundle


def run(source, recovered, out):
    summary = json.loads((recovered / "summary.json").read_text(encoding="utf-8"))
    if digest(source / "panel.pkl") != summary["baseline_sha256"]:
        raise ValueError("baseline_digest_changed")
    if out.exists():
        raise ValueError("select_new_account_input_directory")
    out.mkdir(parents=True)
    with (source / "panel.pkl").open("rb") as stream:
        bundle = apply_overlay(pickle.load(stream), recovered)
    with (out / "panel.pkl").open("wb") as stream:
        pickle.dump(bundle, stream, protocol=5)
    for name in EXECUTION_FILES:
        shutil.copy2(source / name, out / name)
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    protocol["data_recovery"] = {
        "source": str(source), "summary": digest(recovered / "summary.json"),
        "inputs": {name: digest(recovered / (name + ".parquet")) for name in FIELDS},
        "benchmark": "exact original dated weights and investment pool",
        "imputation_candidates_applied": False,
    }
    protocol["inputs"] = {**protocol["inputs"], **protocol["data_recovery"]["inputs"]}
    write_json(out / "protocol.json", protocol)
    study.VARIANTS = {name: study.VARIANTS[name] for name in protocol["variants"]}
    study.build_histories(bundle["panel"], bundle["v2"], out,
                          {**protocol, "maximum_exposure_failure_fraction": 0.05})
    write_json(out / "history-binding-v3.json", {
        name: digest(out / (name + ".pkl")) for name in ("panel", "histories")
    } | {"protocol": digest(out / "protocol.json")})
    with (out / "histories.pkl").open("rb") as stream:
        histories = pickle.load(stream)
    with (source / "portfolio-risk-models.pkl").open("rb") as stream:
        original = pickle.load(stream)
    definitions = model_definitions(protocol)
    models = {}
    panel = bundle["panel"]
    for day, entry in original.items():
        assets = entry["assets"]
        i = panel["dates"].get_loc(day)
        window = panel["dates"][i - 252:i]
        models[day] = {"assets": assets, "benchmark": entry["benchmark"].copy(), "models": {}}
        for name in GENERATORS:
            variant, cfg, _ = definitions[name]
            snapshot = study.subset(histories["snapshots"][variant][day], assets)
            history = histories["histories"][variant][cfg.regression_weighting]
            residuals = history["residuals"].loc[window, assets]
            if residuals.isna().any().any():
                raise ValueError("original_pool_lost_observed_residual_coverage")
            f, d, _ = risk_components(snapshot, history["factors"].loc[window], residuals, cfg)
            models[day]["models"][name] = {
                "exposures": snapshot.values, "F_daily_rate": f, "D_daily_rate": d,
                "configuration": asdict(cfg),
            }
        print(json.dumps({"phase": "account_models", "date": str(day.date())}), flush=True)
    for name in ("risk-models.pkl", "portfolio-risk-models.pkl"):
        with (out / name).open("wb") as stream:
            pickle.dump(models, stream, protocol=5)
    write_json(out / "portfolio-risk-models-binding.json", {
        "models": digest(out / "portfolio-risk-models.pkl"),
        "panel": digest(out / "panel.pkl"), "protocol": digest(out / "protocol.json"),
        "producer": digest(__file__), "original_models": digest(source / "portfolio-risk-models.pkl"),
        "decision_dates": len(models), "benchmark_and_pool_preserved": True,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "recovered", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.recovered, args.out)
