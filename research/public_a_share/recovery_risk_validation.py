"""Matched retrospective risk validation of restored and estimated descriptors.

Daily factor fits use yesterday's exposures. Forecasts use 252 past sessions;
future returns enter only the outcome columns. No dense asset covariance or
synthetic returns are constructed. This is a research gate, not deployment.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from time import perf_counter

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from recovery_imputation import (  # noqa: E402
    FIELDS,
    METHODS,
    audit_eligibility,
    estimate,
    foundation_eligibility,
)
from recovery_inventory import digest, write_json  # noqa: E402

from build_panel import snapshot  # noqa: E402
from exposure_v2_study import subset  # noqa: E402
from model_bootstrap import (  # noqa: E402
    FactorModelConfig,
    public_equity_style_config,
    solve_factor_returns,
)
from study_v3_risk import factor_variance, paired_summary  # noqa: E402


def run(base, recovered, out):
    started = perf_counter()
    receipt = json.loads((recovered / "summary.json").read_text(encoding="utf-8"))
    source = base / "runs/main/panel.pkl"
    if digest(source) != receipt["baseline_sha256"]:
        raise ValueError("baseline_digest_changed")
    with source.open("rb") as stream:
        original = pickle.load(stream)["v2"]
    repaired = {**original, "descriptors": dict(original["descriptors"])}
    for item in receipt["descriptor_files"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("recovered_inputs_changed")
        repaired["descriptors"][item["field"]] = pd.read_parquet(item["path"])
    days = original["dates"][original["dates"] >= "2021-11-11"]
    panels = {"old_input": original, "restored_input": repaired}
    foundation = foundation_eligibility(repaired) & audit_eligibility(
        recovered, original["dates"], original["assets"]
    )
    for method in METHODS:
        descriptors = dict(repaired["descriptors"])
        eligible = pd.DataFrame(True, index=original["dates"], columns=original["assets"])
        for field in FIELDS:
            frame = descriptors[field].copy()
            for day in days:
                requested = frame.columns[frame.loc[day].isna()]
                values, _ = estimate(
                    frame.loc[day],
                    original["groups"].loc[day].where(foundation.loc[day]),
                    original["descriptors"]["size_log_mcap"].loc[day],
                    requested,
                    method,
                )
                frame.loc[day, requested] = values
            descriptors[field] = frame
            eligible &= frame.notna()
        panels[method] = {
            **repaired,
            "descriptors": descriptors,
            "caps": original["caps"].where(eligible),
        }
    industries = sorted(
        x for x in pd.unique(original["groups"].to_numpy().ravel()) if isinstance(x, str)
    )
    configs = {
        name: public_equity_style_config(
            industries,
            definition_version="v2",
            enhancements=("growth",) if name in METHODS else (),
            min_assets=100,
            missing_policy="industry_median",
            missing_group_fallback="global",
            max_missing_fraction=0.4,
        )
        for name in panels
    }
    cfg = FactorModelConfig(
        model_kind="fundamental_style", regression_weighting="equal", industry_constraint=True
    )
    out.mkdir(parents=True, exist_ok=True)
    if (out / "forecasts.parquet").exists():
        raise ValueError("completed_risk_validation_requires_new_version")
    write_json(
        out / "protocol.json",
        {
            "source_summary_sha256": digest(recovered / "summary.json"),
            "producer_sha256": digest(__file__),
            "variants": list(panels),
            "window": 252,
            "horizon": 5,
            "regression": "equal-weight, industry constraint; common causal daily estimation sample",
            "factor_covariance": "sample, 20% diagonal shrinkage",
            "specific_variance": "at least 126 real calendar observations, 20% global shrinkage",
            "development": 2023,
            "retrospective": [2024, 2025, 2026],
            "default_changed": False,
        },
    )
    histories = {name: {"dates": [], "f": [], "d": []} for name in panels}
    previous = {}
    forecasts, failures, exposures = [], [], []
    evaluation = original["dates"][original["dates"] >= "2023-01-03"][::5]
    snapshots = {}
    for count, day in enumerate(days, 1):
        returns = original["prices"]["returns"].loc[day].dropna()
        if len(previous) == len(panels):
            assets = returns.index
            for snap in previous.values():
                assets = assets.intersection(snap.values.index, sort=False)
            if len(assets) >= 100:
                for name, snap in previous.items():
                    current = subset(snap, assets)
                    factors, residuals, *_ = solve_factor_returns(current, returns.loc[assets], cfg)
                    histories[name]["dates"].append(day)
                    histories[name]["f"].append(pd.Series(factors, index=current.values.columns))
                    histories[name]["d"].append(pd.Series(residuals, index=assets))
        snapshots = {}
        for name in panels:
            try:
                snapshots[name] = snapshot(panels[name], day, configs[name])
            except ValueError as exc:
                failures.append(
                    {
                        "date": str(day.date()),
                        "model": name,
                        "stage": "snapshot",
                        "reason": str(exc),
                    }
                )
        previous = snapshots
        if day not in evaluation or len(snapshots) != len(panels):
            continue
        i = original["dates"].get_loc(day)
        window = original["dates"][i - 252 : i]
        common = pd.Index(original["assets"])
        data = {}
        for name, history in histories.items():
            f = pd.DataFrame(history["f"], index=history["dates"]).reindex(window)
            d = pd.DataFrame(history["d"], index=history["dates"]).reindex(window)
            if len(f) != 252 or f.isna().any().any():
                data = {}
                break
            common = common.intersection(snapshots[name].values.index, sort=False).intersection(
                d.columns[d.notna().sum() >= 126], sort=False
            )
            data[name] = (f, d)
        if len(data) != len(panels) or len(common) < 100:
            failures.append(
                {
                    "date": str(day.date()),
                    "stage": "risk",
                    "reason": "insufficient_matched_real_history",
                }
            )
            continue
        # The same sample benchmark/weights go to every model, without silently
        # deleting assets after observing the next five days' availability.
        caps = original["caps"].loc[day, common]
        benchmark = caps / caps.sum()
        equal = pd.Series(0.95 / len(common), index=common)
        weights = {
            "equal95": equal,
            "cap95": 0.95 * benchmark,
            "active_equal95": equal - benchmark,
            "active_cap80": -0.2 * benchmark,
        }
        future = original["prices"]["returns"].iloc[i + 1 : i + 6].reindex(columns=common)
        matured = len(future) == 5 and future.notna().all().all()
        realized = future.sum(min_count=5) if matured else None
        for name, (f, d) in data.items():
            covariance = f.cov().to_numpy()
            covariance = 0.8 * covariance + 0.2 * np.diag(np.diag(covariance))
            raw = d[common].var(ddof=1)
            specific = 0.8 * raw + 0.2 * raw.mean()
            x = snapshots[name].values.loc[common].to_numpy()
            exposures.append({"date": str(day.date()), "model": name, "assets": len(common)})
            for family, w in weights.items():
                variance, factor, spec = factor_variance(
                    x, covariance, specific.to_numpy(), w.to_numpy()
                )
                outcome = float(w @ realized) if matured else np.nan
                predicted = 5 * variance
                forecasts.append(
                    {
                        "date": str(day.date()),
                        "year": day.year,
                        "model": name,
                        "family": family,
                        "variance_5d": predicted,
                        "factor_variance_5d": 5 * factor,
                        "specific_variance_5d": 5 * spec,
                        "realized_5d": outcome,
                        "qlike": outcome**2 / predicted + np.log(predicted)
                        if matured and predicted > 0
                        else np.nan,
                        "assets": len(common),
                        "outcome_status": "mature" if matured else "missing_endpoint_or_unmatured",
                    }
                )
        if count % 50 == 0:
            print(
                json.dumps(
                    {"processed": count, "total": len(days), "forecast_rows": len(forecasts)}
                ),
                flush=True,
            )
    frame = pd.DataFrame(forecasts)
    frame.to_parquet(out / "forecasts.parquet", index=False)
    write_json(out / "failures.json", failures)
    pd.DataFrame(exposures).to_csv(out / "matched-coverage.csv", index=False)
    summaries = {}
    for years, label in [([2023], "development"), ([2024, 2025, 2026], "retrospective")]:
        part = frame[frame.year.isin(years)] if len(frame) else frame
        summaries[label] = paired_summary(
            part, list(panels), "restored_input", replications=5000, block=10, seed=20261017
        )
    calibration = []
    if len(frame):
        for key, g in frame.dropna(subset=["realized_5d"]).groupby(["model", "year", "family"]):
            calibration.append(
                {
                    "model": key[0],
                    "year": key[1],
                    "family": key[2],
                    "windows": len(g),
                    "realized_predicted_variance_ratio": float(
                        (g.realized_5d**2 / g.variance_5d).mean()
                    ),
                }
            )
    pd.DataFrame(calibration).to_csv(out / "calibration.csv", index=False)
    write_json(
        out / "summary.json",
        {
            "comparisons": summaries,
            "wall_seconds": perf_counter() - started,
            "forecast_rows": len(frame),
            "failure_records": len(failures),
            "promoted": False,
            "limitations": [
                "Matched estimation/forecast sample; coverage loss is reported separately.",
                "Growth-composite candidates change a style definition and remain sensitivity experiments.",
                "These are fixed-portfolio risk forecasts, not a full account/strategy replay.",
            ],
        },
    )
    print(
        json.dumps({"forecast_rows": len(frame), "wall_seconds": perf_counter() - started}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("base", "recovered", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.recovered, args.out)
