"""Artificial exposure masking, without pretending missing observations exist."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from study_v3_archive import digest
from study_v3_risk import factor_variance, model_definitions, paired_summary


def run(source: Path, coverage: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    with (source / "panel.pkl").open("rb") as stream:
        panel = pickle.load(stream)["panel"]
    with (source / "histories.pkl").open("rb") as stream:
        histories = pickle.load(stream)
    with (coverage / "portfolio-risk-models.pkl").open("rb") as stream:
        models = pickle.load(stream)
    definitions = model_definitions(json.loads((source / "protocol.json").read_text()))
    rows = []
    for day, entry in models.items():
        i = panel["dates"].get_loc(day)
        if i + 5 >= len(panel["dates"]):
            continue
        assets = sorted(entry["assets"])
        chosen = np.random.default_rng(20261019 + i).choice(assets, 5, replace=False)
        realized = (
            panel["prices"]["returns"].iloc[i + 1 : i + 6].reindex(columns=assets).sum(min_count=5)
        )
        equal = pd.Series(0.95 / len(assets), index=assets)
        families = {"equal95": equal, "active_equal95": equal - entry["benchmark"].reindex(assets)}
        for generator in definitions:
            if generator not in entry["models"]:
                continue
            model = entry["models"][generator]
            variant = definitions[generator][0]
            x = model["exposures"].loc[assets]
            d = pd.Series(model["D_daily_rate"], index=model["exposures"].index).loc[assets]
            f = model["F_daily_rate"]
            for age in (0, 1, 3, 5):
                xp = x.copy()
                prior = histories["snapshots"][variant].get(panel["dates"][i - age])
                if age and (prior is None or not set(chosen) <= set(prior.values.index)):
                    continue
                if age:
                    xp.loc[chosen] = prior.values.reindex(index=chosen, columns=x.columns)
                for family, weights in families.items():
                    outcome = float(weights @ realized) if realized.notna().all() else np.nan
                    variance = (
                        5 * factor_variance(xp.to_numpy(), f, d.to_numpy(), weights.to_numpy())[0]
                    )
                    rows.append(
                        {
                            "date": day,
                            "year": day.year,
                            "generator": generator,
                            "model": f"carry_{age}",
                            "family": family,
                            "masked_assets": ",".join(chosen),
                            "variance": variance,
                            "realized": outcome,
                            "qlike": np.log(variance) + outcome**2 / variance,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "forecasts.parquet", index=False)
    summary = {
        name: paired_summary(
            g[g.year.ge(2024)], ["carry_0", "carry_1", "carry_3", "carry_5"], "carry_0"
        )
        for name, g in frame.groupby("generator")
    }
    (out / "summary.json").write_text(
        json.dumps(
            {
                "artificial_masking": True,
                "actual_missing_exposure_truth_recovered": False,
                "default_promotion": False,
                "five_masked_assets_per_date": True,
                "models": summary,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "coverage": str(coverage.resolve()),
                "seed": "20261019 + panel trading-session ordinal",
                "ages": [0, 1, 3, 5],
                "selection_by_future_outcome": False,
                "input_hashes": {
                    str(p): digest(p)
                    for p in [
                        source / "panel.pkl",
                        source / "histories.pkl",
                        source / "protocol.json",
                        coverage / "portfolio-risk-models.pkl",
                    ]
                },
                "producer_sha256": digest(__file__),
                "scope": "synthetic masks on observed assets; no real missing-exposure truth or default policy selection",
            },
            indent=2,
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.coverage, args.out)
