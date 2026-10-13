"""Independent full-model and real-financial replay checks for v2 study artifacts."""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from exposure_v2_study import digest, write_json
from model_bootstrap import FactorModelConfig
from run_study import date_grid
from validate_pipeline import canonical_check

# isort: split

from quant_risk_monitor.style_descriptors import annual_growth_descriptors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.out / "protocol.json").read_text(encoding="utf8"))
    manifest = json.loads((args.out / "history-manifest.json").read_text(encoding="utf8"))
    if manifest["histories_sha256"] != digest(args.out / "histories.pkl"):
        raise ValueError("historical model cache changed")
    if digest(args.old_root / "clean/panel.pkl") != protocol["inputs"]["panel"]:
        raise ValueError("historical input panel changed")
    with (args.old_root / "clean/panel.pkl").open("rb") as stream:
        panel = pickle.load(stream)
    with (args.out / "histories.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    common = {
        "model_kind": "fundamental_style",
        "risk_horizon": 5,
        "min_periods": 252,
        "min_assets_per_period": 100,
        "min_asset_observations": 252,
    }
    configs = {
        "common_ols": FactorModelConfig(**common, **protocol["old_protocol"]["baseline"]),
        "frozen_enhanced": FactorModelConfig(
            **common,
            **protocol["old_protocol"]["enhanced_fixed"],
            covariance_shrinkage=0.1,
            factor_half_life=90,
            specific_half_life=60,
            specific_prior_dof=10,
        ),
    }
    selection = json.loads((args.out / "selection.json").read_text(encoding="utf8"))["selected"]
    row = date_grid(panel)[0][0]
    results = []
    for name in dict.fromkeys(["legacy", "repaired", selection]):
        data = {"snapshots": saved["snapshots"][name], "histories": saved["histories"][name]}
        for estimator, config in configs.items():
            result = canonical_check(panel, data, row, config)
            results.append({"variant": name, "estimator": estimator, **result})
            print(json.dumps(results[-1]), flush=True)
    frames = [
        pd.read_csv(path, dtype={"asset": str, "report_period": str})
        for path in (
            args.old_root / "clean/financial_facts.csv",
            args.out / "annual-warmup-facts.csv",
        )
    ]
    facts = pd.concat(frames, ignore_index=True)
    descriptors, quality = annual_growth_descriptors(facts, panel["dates"], panel["assets"])
    for name, values in descriptors.items():
        pd.testing.assert_frame_equal(values, pd.read_parquet(args.out / (name + ".parquet")))
        pd.testing.assert_frame_equal(
            quality[name], pd.read_parquet(args.out / (name + "-quality.parquet"))
        )
    dates = panel["dates"][(panel["dates"] >= "2023-12-01") & (panel["dates"] <= "2023-12-29")]
    future = pd.to_datetime(facts.available_at, utc=True) > pd.Timestamp(
        "2023-12-29", tz="Asia/Shanghai"
    )
    poisoned = facts.copy()
    poisoned["value"] = poisoned.value.astype(object)
    poisoned.loc[future, "value"] = "unavailable-future-value"
    poisoned = pd.concat([poisoned, poisoned.loc[future]], ignore_index=True)
    past, _ = annual_growth_descriptors(poisoned, dates, panel["assets"])
    for name, values in past.items():
        pd.testing.assert_frame_equal(values, descriptors[name].loc[dates])
    write_json(
        args.out / "independent-model-validation.json",
        {
            "status": "passed",
            "canonical_model_checks": results,
            "real_annual_descriptors_and_quality_unchanged_after_hardening": True,
            "future_financial_rows_poisoned_and_duplicated": int(future.sum()),
            "past_dates_checked": len(dates),
            "assets_checked": len(panel["assets"]),
            "history_manifest_sha256": digest(args.out / "history-manifest.json"),
            "notes": "Original history producer hashes are preserved; additional availability validation does not relabel or overwrite that run.",
        },
    )


if __name__ == "__main__":
    main()
