"""Cross-model risk on identical generated portfolios, separate from net NAV."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from study_v3_archive import digest, write_json
from study_v3_risk import paired_summary


def run(portfolios: Path, out: Path, expected_scenarios: int) -> None:
    out.mkdir(parents=True, exist_ok=False)
    frames, sources = [], []
    folders = sorted(p for p in portfolios.iterdir() if p.is_dir())
    assert len(folders) == expected_scenarios, "partial scenario campaign"
    for folder in folders:
        assert (folder / "complete.json").exists(), f"unfinished account: {folder.name}"
        spec = json.loads((folder / "specification.json").read_text())
        path = folder / "cross-predictions.parquet"
        frame = pd.read_parquet(path)
        frame["scenario"] = int(folder.name[:3])
        frame["generator"] = spec["model"]
        frame["method"] = spec["method"]
        frame["objective"] = (
            "active" if spec["method"] in {"minimum_active_risk", "cost_aware"} else "absolute"
        )
        frames.append(frame)
        sources.append(
            {"scenario": int(folder.name[:3]), "path": str(path), "sha256": digest(path)}
        )
    frame = pd.concat(frames, ignore_index=True)
    assert frame.predictor.nunique() == 6, "these accounts cross-predict six frozen generators"
    frame["loss_variance"] = frame.variance
    frame["outcome"] = frame.actual_frozen_weight_return
    active = frame.objective.eq("active")
    frame.loc[active, "loss_variance"] = frame.loc[active, "annualized_predicted_te"] ** 2 * 5 / 252
    frame.loc[active, "outcome"] = frame.loc[active, "active_frozen_weight_return"]
    usable = frame.loss_variance.gt(0) & frame.outcome.notna()
    frame["objective_qlike"] = np.where(
        usable,
        np.log(frame.loss_variance.where(usable))
        + frame.outcome**2 / frame.loss_variance.where(usable),
        np.nan,
    )
    frame["calibration"] = frame.outcome**2 / frame.loss_variance.where(usable)
    frame.to_parquet(out / "cross-forecasts.parquet", index=False)
    models = sorted(frame.predictor.unique())
    # Preserve missing/zero-risk rows in the source. Compare a portfolio only
    # when every predictor has the same observed outcome and a usable forecast.
    keys = ["scenario", "date", "target_kind", "portfolio_id", "objective"]
    valid = frame.groupby(keys).agg(
        rows=("predictor", "nunique"), usable=("objective_qlike", "count")
    )
    valid = valid[valid.rows.eq(len(models)) & valid.usable.eq(len(models))].index
    common = frame.set_index(keys).loc[valid].reset_index()
    consistency = common.groupby(keys).outcome.nunique()
    assert consistency.le(1).all(), "cross-predictors use different realized returns"
    matrix = (
        common.groupby(["objective", "target_kind", "generator", "predictor"])
        .agg(
            forecasts=("objective_qlike", "count"),
            dates=("date", "nunique"),
            mean_loss=("objective_qlike", "mean"),
            calibration=("calibration", "mean"),
        )
        .reset_index()
    )
    matrix["volatility_calibration"] = np.sqrt(matrix.calibration)
    matrix.to_csv(out / "generator-predictor-matrix.csv", index=False)
    # Deduplicate identical generated weights, then give each date one block.
    unique = common.drop_duplicates(
        ["objective", "target_kind", "date", "portfolio_id", "predictor"]
    )
    summaries = {}
    for (objective, kind), part in unique.groupby(["objective", "target_kind"]):
        part = part.copy()
        part["model"] = part.predictor
        part["family"] = "generated_portfolio"
        part["qlike"] = part.objective_qlike
        part = part[pd.to_datetime(part.date).dt.year.ge(2024)]
        summaries[objective + ":" + kind] = paired_summary(part, models, "repaired_ols")
    write_json(
        out / "summary.json",
        {
            "predictors": models,
            "scenario_count": len(folders),
            "rows": len(frame),
            "common_comparable_rows": len(common),
            "distinct_portfolios": int(unique.portfolio_id.nunique()),
            "status_counts": frame.status.value_counts().to_dict(),
            "paired_retrospective": summaries,
            "sources": sources,
            "producer": digest(__file__),
            "scope": "same frozen-weight outcome; net account NAV is evaluated separately; no optimization-bias correction promoted",
            "independent_forward_windows": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolios", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-scenarios", type=int, default=36)
    args = parser.parse_args()
    run(args.portfolios, args.out, args.expected_scenarios)
