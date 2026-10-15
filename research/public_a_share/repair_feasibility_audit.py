"""Independent factor-risk lower-bound certificates on the frozen accounts.

Convex supporting planes, minimized analytically on the capped simplex, give
lower bounds independently of solver status. No asset covariance is built.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import cvxpy as cp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from repair_failure_audit import date_rows, decimal_column, frozen_accounts  # noqa: E402
from study_v3_archive import digest, write_json  # noqa: E402


def linear_minimum(gradient, budget, cap):
    result = np.zeros(len(gradient))
    remaining = budget
    for index in np.argsort(gradient, kind="stable"):
        value = min(remaining, cap)
        result[index] = value
        remaining -= value
        if remaining <= 1e-14:
            break
    assert remaining <= 1e-12
    return float(gradient @ result)


def absolute_linear_minimum(gradient, current, budget, cap):
    segments = []
    for g, holding in zip(gradient, current, strict=True):
        split = float(np.clip(holding, 0, cap))
        segments.extend([(g - 1, split), (g + 1, cap - split)])
    value, remaining = float(np.abs(current).sum()), budget
    for slope, length in sorted(segments):
        delta = min(remaining, length)
        value += slope * delta
        remaining -= delta
        if remaining <= 1e-14:
            break
    assert remaining <= 1e-12
    return value


def numerical(data, benchmark):
    x = data["exposures"].to_numpy()
    f = np.asarray(data["F_daily_rate"])
    d = np.asarray(data["D_daily_rate"])
    eigenvalues, vectors = np.linalg.eigh((f + f.T) / 2)
    assert eigenvalues.min() >= -1e-12 and d.min() >= 0
    factor_root = np.sqrt(np.maximum(eigenvalues, 0))[:, None] * vectors.T
    root = factor_root @ x.T
    return root, d, benchmark.reindex(data["exposures"].index, fill_value=0).to_numpy()


def solve_bound(data, benchmark, budget, *, current=None, limit=None):
    root, d, wb = numerical(data, benchmark)
    w = cp.Variable(len(wb))
    a = w - wb
    variance = cp.sum_squares(root @ a) + cp.sum(cp.multiply(d, cp.square(a)))
    base = [w >= 0, w <= 0.05, cp.sum(w) == budget]
    if current is None:
        problem = cp.Problem(cp.Minimize(252 * variance), base)
    else:
        constraint = cp.SOC(limit, cp.hstack((root @ a, cp.multiply(np.sqrt(d), a))) * np.sqrt(252))
        problem = cp.Problem(cp.Minimize(cp.norm1(w - current)), base + [constraint])
    attempts = []
    for tolerance in (1e-10, 1e-9, 1e-8):
        problem.solve(
            solver="CLARABEL",
            tol_gap_abs=tolerance,
            tol_gap_rel=tolerance,
            tol_feas=tolerance,
            max_iter=2000,
            time_limit=30,
            warm_start=False,
        )
        attempts.append(
            {
                "status": problem.status,
                "tolerance": tolerance,
                "iterations": problem.solver_stats.num_iters,
            }
        )
        if problem.status == "optimal":
            break
    assert problem.status == "optimal", attempts
    value = np.asarray(w.value)
    active = value - wb
    sigma_active = root.T @ (root @ active) + d * active
    risk = np.sqrt(max(active @ sigma_active, 0))
    residual = max(
        abs(value.sum() - budget),
        float(np.maximum(-value, 0).max()),
        float(np.maximum(value - 0.05, 0).max()),
    )
    if current is None:
        gradient = 2 * sigma_active
        lower = float(risk**2 + linear_minimum(gradient, budget, 0.05) - gradient @ value)
        result = {
            "annual_te": float(risk * np.sqrt(252)),
            "certified_lower_te": float(np.sqrt(max(lower, 0) * 252)),
        }
    else:
        scalar, vector = constraint.dual_value
        vector = np.asarray(vector).reshape(-1)
        multiplier = max(float(np.asarray(scalar).item()), float(np.linalg.norm(vector)))
        gradient = -np.sqrt(252) * (root.T @ vector[: len(root)] + np.sqrt(d) * vector[len(root) :])
        lower = absolute_linear_minimum(gradient, current, budget, 0.05)
        lower -= gradient @ wb + multiplier * limit
        residual = max(residual, float(risk * np.sqrt(252) - limit))
        result = {
            "turnover": float(np.abs(value - current).sum()),
            "certified_lower_turnover": float(lower),
            "annual_te": float(risk * np.sqrt(252)),
        }
    assert residual <= 1e-7, residual
    return {
        **result,
        "residual": residual,
        "status": problem.status,
        "seconds": problem.solver_stats.solve_time,
        "publishable": False,
        "attempts": attempts,
    }


def run(source, out):
    from study_v3_portfolios import scenarios

    out.mkdir(parents=True, exist_ok=True)
    with (source / "portfolio-risk-models.pkl").open("rb") as stream:
        models = pickle.load(stream)
    records = []
    for day, entry in sorted(models.items()):
        for model in (
            "legacy_ols",
            "repaired_ols",
            "residual_ols",
            "legacy_enhanced",
            "enhanced_s0.1_f90_d60_p10",
            "residual_enhanced",
        ):
            for budget in (0.95, 0.8):
                bound = solve_bound(entry["models"][model], entry["benchmark"], budget)
                records.append({"date": str(day.date()), "model": model, "budget": budget, **bound})
        print(json.dumps({"phase": "floors", "date": str(day.date())}), flush=True)
    floors = pd.DataFrame(records)
    floors.to_csv(out / "certified-te-floors.csv", index=False)
    lookup = floors.set_index(["date", "model", "budget"])
    structural, turnover, scenarios_without_target = 0, [], []
    for scenario, directory in frozen_accounts(source):
        decisions = json.loads((directory / "allocation-decisions.json").read_text())
        _, model, budget, limit = list(scenarios())[scenario]
        if len(decisions) == 182 and all(x.get("status") == "infeasible" for x in decisions):
            scenarios_without_target.append(scenario)
        positions = pd.read_parquet(directory / "positions.parquet")
        snapshots = pd.read_parquet(directory / "portfolio_snapshots.parquet")
        for decision in decisions:
            if decision.get("status") != "infeasible":
                continue
            bound = lookup.loc[(decision["session"], model, budget)]
            if bound.certified_lower_te > limit:
                structural += 1
                continue
            day = pd.Timestamp(decision["session"])
            entry = models[day]
            data = entry["models"][model]
            snapshot = date_rows(snapshots, day).iloc[-1]
            nav = decimal_column(snapshot, "nav")
            current = {
                row.instrument_id: float(decimal_column(row, "market_value") / nav)
                for _, row in date_rows(positions, day).iterrows()
            }
            assert set(current) <= set(data["exposures"].index)
            current = (
                pd.Series(current, dtype=float)
                .reindex(data["exposures"].index, fill_value=0)
                .to_numpy()
            )
            result = solve_bound(data, entry["benchmark"], budget, current=current, limit=limit)
            assert result["certified_lower_turnover"] > 0.2
            assert result["turnover"] - result["certified_lower_turnover"] < 1e-6, (
                scenario,
                decision["session"],
                result,
            )
            turnover.append(
                {
                    "scenario": scenario,
                    "date": decision["session"],
                    "model": model,
                    "budget": budget,
                    "account_sha256": digest(directory / "portfolio_snapshots.parquet"),
                    **result,
                }
            )
    assert structural == 5551 and len(turnover) == 14 and len(scenarios_without_target) == 26
    write_json(
        out / "constraint-certificates.json",
        {
            "structural": structural,
            "turnover_conflicts": turnover,
            "all_period_infeasible_scenarios": scenarios_without_target,
            "model_sha256": digest(source / "portfolio-risk-models.pkl"),
            "producer_sha256": digest(__file__),
            "no_constraint_relaxation_for_trading": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
