"""Fit decision-time risk on every rebalance, independently of outcome maturity."""

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from study_v3_archive import digest, write_json
from study_v3_risk import eligible_inputs, model_definitions


def decision_grid(dates):
    start = dates.searchsorted(pd.Timestamp("2023-01-01"))
    return [(i, dates[i]) for i in range(start, len(dates), 5)]


def run(out):
    from exposure_v2_study import subset
    from run_study import risk_components

    binding = json.loads((out / "history-binding-v3.json").read_text(encoding="utf8"))
    for name in ("panel", "histories"):
        if digest(out / (name + ".pkl")) != binding[name]:
            raise ValueError("changed historical producer inputs")
    risk_binding = json.loads((out / "risk-models-binding.json").read_text(encoding="utf8"))
    if digest(out / "risk-models.pkl") != risk_binding["models"]:
        raise ValueError("changed frozen risk experiment")
    with (out / "panel.pkl").open("rb") as stream:
        panel = pickle.load(stream)["panel"]
    with (out / "histories.pkl").open("rb") as stream:
        histories = pickle.load(stream)
    with (out / "risk-models.pkl").open("rb") as stream:
        original = pickle.load(stream)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf8"))
    definitions = model_definitions(protocol)
    models, added = {}, []
    for i, day in decision_grid(panel["dates"]):
        if day in original:
            models[day] = original[day]
            continue
        assets, window = eligible_inputs(panel, histories, i, day)
        base = subset(histories["snapshots"]["repaired"][day], assets)
        entry = {"assets": list(assets), "benchmark": base.market_caps / base.market_caps.sum(), "models": {}}
        for name, (variant, cfg, _) in definitions.items():
            snap = subset(histories["snapshots"][variant][day], assets)
            history = histories["histories"][variant][cfg.regression_weighting]
            f, d, _ = risk_components(snap, history["factors"].loc[window], history["residuals"].loc[window], cfg)
            entry["models"][name] = {"exposures": snap.values, "F_daily_rate": f, "D_daily_rate": d, "configuration": asdict(cfg)}
        models[day] = entry
        added.append(str(day.date()))
    target = out / "portfolio-risk-models.pkl"
    if target.exists():
        raise ValueError("portfolio model cache is immutable; use a new run for changed inputs")
    with target.open("wb") as stream:
        pickle.dump(models, stream, protocol=5)
    receipt = {
        "models": digest(target), "source_risk_models": risk_binding["models"],
        "panel": binding["panel"], "histories": binding["histories"],
        "protocol": digest(out / "protocol.json"), "producer": digest(__file__),
        "decision_dates": len(models), "added_dates": added,
        "risk_experiment_changed": False,
        "cadence": "every five sessions; no outcome-maturity or development-boundary filter",
    }
    write_json(out / "portfolio-risk-models-binding.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
