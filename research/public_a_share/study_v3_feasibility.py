"""Optimistic continuous TE floor for explanation, never a replacement target.

Hold the exact invested budget and name cap fixed, before turnover, fees and
integer restrictions. This post-study diagnosis never tunes the experiment.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from time import perf_counter

import cvxpy as cp
import numpy as np
import pandas as pd

from study_v3_archive import write_json
from study_v3_portfolios import GENERATORS


def continuous_floor(model, benchmark, budget, *, cap=.05):
    x = model["exposures"].to_numpy()
    b = benchmark.reindex(model["exposures"].index).to_numpy()
    w = cp.Variable(len(b))
    active = w - b
    risk = cp.quad_form(x.T @ active, cp.psd_wrap(model["F_daily_rate"] * 252))
    risk += cp.sum(cp.multiply(model["D_daily_rate"] * 252, cp.square(active)))
    problem = cp.Problem(cp.Minimize(risk), [w >= 0, w <= cap, cp.sum(w) == budget])
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                  tol_feas=1e-10, max_iter=500, time_limit=30)
    if problem.status != "optimal":
        raise ValueError("no accepted diagnostic optimum: " + problem.status)
    z = np.asarray(w.value).reshape(-1)
    residual = max(abs(z.sum() - budget), float(np.max(-z)), float(np.max(z - cap)), 0.)
    if residual > 1e-7:
        raise ValueError("diagnostic optimum violates constraints")
    exposure = x.T @ (z - b)
    variance = float((exposure @ model["F_daily_rate"] @ exposure +
                      model["D_daily_rate"] @ ((z-b)**2)) * 252)
    if abs(variance - problem.value) > 1e-8:
        raise ValueError("independent diagnostic risk differs from objective")
    return {"continuous_te_floor": float(np.sqrt(max(variance, 0))), "residual": residual,
            "solver_seconds": problem.solver_stats.solve_time, "status": "optimal"}


def run(out):
    raw = (out / "portfolio-risk-models.pkl").read_bytes()
    binding = json.loads((out / "portfolio-risk-models-binding.json").read_text())
    if hashlib.sha256(raw).hexdigest() != binding["models"]:
        raise ValueError("portfolio risk model binding changed")
    models = pickle.loads(raw)
    rows, started = [], perf_counter()
    for day, entry in models.items():
        for name in GENERATORS:
            model = entry["models"][name]
            for budget in (0.95, 0.80):
                row = {"date": str(day.date()), "model": name, "budget": budget,
                       "benchmark_max_weight": float(entry["benchmark"].max()),
                       "assets": len(model["exposures"])}
                try:
                    row.update(continuous_floor(model, entry["benchmark"], budget))
                except Exception as exc:
                    row.update(status="diagnostic_failed", reason=str(exc))
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "tracking-error-floor-diagnostic.csv", index=False)
    summaries = []
    for (name, budget), group in frame.groupby(["model", "budget"]):
        valid = group[group.status.eq("optimal")]
        summaries.append({"model": name, "budget": budget, "dates": len(valid),
                          "minimum_floor": float(valid.continuous_te_floor.min()) if len(valid) else None,
                          "maximum_floor": float(valid.continuous_te_floor.max()) if len(valid) else None,
                          "above_3pct": int(valid.continuous_te_floor.gt(.03+1e-7).sum()) if len(valid) else 0,
                          "above_6pct": int(valid.continuous_te_floor.gt(.06+1e-7).sum()) if len(valid) else 0})
    result = {"purpose": "post-study explanation only; no retuning or published target",
              "scope": "continuous exact invested budget and 5% cap, before turnover/cost/integer restrictions",
              "model_sha256": binding["models"], "problems": len(rows),
              "failure_count": int(frame.status.ne("optimal").sum()),
              "wall_seconds": perf_counter()-started,
              "benchmark_max_weight_range": [float(frame.benchmark_max_weight.min()),
                                             float(frame.benchmark_max_weight.max())],
              "results": summaries}
    write_json(out / "tracking-error-floor-diagnostic.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().out)), flush=True)
