"""Frozen finite estimator grid, recovered facts, no evaluation-period tuning."""

import argparse
import json
import os
import pickle
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from exposure_v2_study import subset  # noqa: E402
from run_study import risk_components  # noqa: E402
from study_v3_archive import digest, write_json  # noqa: E402
from study_v3_risk import (  # noqa: E402
    factor_variance,
    model_definitions,
    paired_summary,
    portfolios,
)


def run(source, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf8"))
    definitions = model_definitions(protocol)
    write_json(
        out / "protocol.json",
        {
            "selection_year": 2023,
            "retrospective_years": [2024, 2025, 2026],
            "forward_independent_windows": 0,
            "horizon": 5,
            "history": 252,
            "selection_candidates": [k for k, (_, _, e) in definitions.items() if e == "estimator"],
            "models": {
                k: {"variant": v, "config": asdict(c), "experiment": e}
                for k, (v, c, e) in definitions.items()
            },
            "input_hashes": {
                name: digest(source / name)
                for name in (
                    "panel.pkl",
                    "histories.pkl",
                    "portfolio-risk-models.pkl",
                    "protocol.json",
                )
            },
            "producer": digest(__file__),
            "default_model_changed": False,
            "evaluation": "same dated portfolios for every model; no scalar-duplicate active-cap portfolio",
            "bootstrap": {"replications": 5000, "block_dates": 10, "multiplicity": "Bonferroni"},
        },
    )
    with (source / "panel.pkl").open("rb") as stream:
        panel = pickle.load(stream)["panel"]
    with (source / "histories.pkl").open("rb") as stream:
        histories = pickle.load(stream)
    with (source / "portfolio-risk-models.pkl").open("rb") as stream:
        original = pickle.load(stream)
    rows, timings = [], []
    for day, entry in original.items():
        start = perf_counter()
        i = panel["dates"].get_loc(day)
        if i + 5 >= len(panel["dates"]):
            continue
        assets = entry["assets"]
        window = panel["dates"][i - 252 : i]
        base = subset(histories["snapshots"]["repaired"][day], assets)
        weights = portfolios(base)
        weights = {k: (family, 0.95 * w) for k, (family, w) in weights.items()}
        weights["active_equal"] = (
            "active",
            pd.Series(0.95 / len(assets), index=assets) - entry["benchmark"],
        )
        future = panel["prices"]["returns"].iloc[i + 1 : i + 6].reindex(columns=assets)
        realized = future.sum(min_count=5)
        for model, (variant, cfg, experiment) in definitions.items():
            snap = subset(histories["snapshots"][variant][day], assets)
            hist = histories["histories"][variant][cfg.regression_weighting]
            f, d, _ = risk_components(
                snap, hist["factors"].loc[window], hist["residuals"].loc[window, assets], cfg
            )
            for name, (family, w) in weights.items():
                w = w.reindex(assets, fill_value=0)
                observed = realized[w.ne(0)].notna().all()
                outcome = float(w[w.ne(0)] @ realized[w.ne(0)]) if observed else np.nan
                var, fv, dv = factor_variance(snap.values.to_numpy(), f, d, w.to_numpy())
                variance = 5 * var
                rows.append(
                    {
                        "date": day,
                        "end": panel["dates"][i + 5],
                        "year": day.year,
                        "model": model,
                        "portfolio": name,
                        "family": family,
                        "experiment": experiment,
                        "variance": variance,
                        "factor_variance": 5 * fv,
                        "specific_variance": 5 * dv,
                        "realized": outcome,
                        "complete": bool(observed),
                        "qlike": np.log(variance) + outcome**2 / variance,
                        "calibration": outcome**2 / variance,
                    }
                )
        timings.append({"date": str(day.date()), "seconds": perf_counter() - start})
        if len(timings) % 20 == 0:
            print(json.dumps({"dates": len(timings)}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "forecasts.parquet", index=False)
    candidates = [k for k, (_, _, e) in definitions.items() if e == "estimator"]
    selection = frame[frame.year.eq(2023) & frame.end.lt(pd.Timestamp("2024-01-01"))]
    evaluation = frame[frame.year.ge(2024)]
    selection_summary = paired_summary(selection, list(definitions), "repaired_ols")
    selected = min(candidates, key=lambda k: (selection_summary["means"][k], k))
    write_json(
        out / "summary.json",
        {
            "selection": selection_summary,
            "selected_using_2023_only": selected,
            "retrospective": paired_summary(evaluation, list(definitions), "repaired_ols"),
            "promotion_accepted": False,
            "reason": "no new independent mature forward windows",
        },
    )
    frame.groupby(["year", "model", "family"]).agg(
        windows=("qlike", "count"), loss=("qlike", "mean"), calibration=("calibration", "mean")
    ).reset_index().to_csv(out / "calibration.csv", index=False)
    pd.DataFrame(timings).to_csv(out / "timings.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
