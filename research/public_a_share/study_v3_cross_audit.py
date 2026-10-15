"""Recompute risk for every scheduled actual account, including failed decisions.

The original scenario artifacts remain immutable. This independent reporting
pass reads their native snapshots and records unavailable risk explicitly.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from study_v3_risk import factor_variance


def session_dates(frame):
    return (
        pd.to_datetime(frame.event_time, utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
        .dt.normalize()
    )


def recompute_cross(decisions, positions, snapshots, models, panel):
    records = []
    snapshot_days = session_dates(snapshots)
    position_days = session_dates(positions) if len(positions) else pd.Series(dtype="datetime64[ns]")
    last_day = snapshot_days.max()
    for decision in decisions:
        day = pd.Timestamp(decision["session"])
        end = panel["dates"][panel["dates"].get_loc(day) + 5]
        if end > last_day:
            continue
        daily_snapshot = snapshots[snapshot_days.eq(day)]
        if daily_snapshot.empty:
            raise ValueError("scheduled decision has no native closing snapshot")
        nav = daily_snapshot.iloc[-1]
        value = float(nav.nav_units) / 10 ** int(nav.nav_scale)
        actual = positions[position_days.eq(day)] if len(positions) else positions
        variants = {
            "actual_close": {
                r.instrument_id: float(r.base_market_value_units)
                / 10 ** int(r.base_market_value_scale)
                / value
                for r in actual.itertuples()
            }
        }
        if "weights" in decision:
            variants["continuous"] = decision["weights"]
            variants["integer"] = decision["diagnostics"]["execution"]["integer_weights"]
        start = panel["dates"].get_loc(day)
        returns = panel["prices"]["returns"].iloc[start + 1 : start + 6].sum(min_count=5)
        entry = models[day]
        benchmark = entry["benchmark"]
        benchmark_return = (
            float(benchmark @ returns.reindex(benchmark.index))
            if returns.reindex(benchmark.index).notna().all()
            else None
        )
        for kind, mapping in variants.items():
            weights = pd.Series(mapping, dtype=float)
            nonzero = weights[weights.abs() > 1e-12]
            outcome = returns.reindex(nonzero.index)
            actual_return = float(nonzero @ outcome) if outcome.notna().all() else None
            identity = hashlib.sha256(
                json.dumps({"date": str(day), "kind": kind, "weights": mapping}, sort_keys=True).encode()
            ).hexdigest()
            for name, data in entry["models"].items():
                exposures = data["exposures"]
                missing = sorted((set(nonzero.index) | set(benchmark.index)) - set(exposures.index))
                record = {
                    "date": day,
                    "end": end,
                    "portfolio_id": identity,
                    "target_kind": kind,
                    "predictor": name,
                    "decision_status": decision.get("status", "published"),
                    "actual_frozen_weight_return": actual_return,
                    "active_frozen_weight_return": None
                    if actual_return is None or benchmark_return is None
                    else actual_return - benchmark_return,
                    "status": "risk_union_incomplete" if missing else "complete",
                    "missing_risk_assets": ",".join(missing),
                    "variance": None,
                    "factor_variance": None,
                    "specific_variance": None,
                    "annualized_predicted_te": None,
                    "qlike": None,
                    "zero_risk_cash": not len(nonzero),
                }
                if not missing:
                    w = weights.reindex(exposures.index, fill_value=0).to_numpy()
                    wb = benchmark.reindex(exposures.index, fill_value=0).to_numpy()
                    x = exposures.to_numpy()
                    variance, factor, specific = factor_variance(x, data["F_daily_rate"], data["D_daily_rate"], w)
                    active, _, _ = factor_variance(x, data["F_daily_rate"], data["D_daily_rate"], w - wb)
                    record.update(
                        status="complete" if actual_return is not None else "outcome_missing",
                        variance=variance * 5,
                        factor_variance=factor * 5,
                        specific_variance=specific * 5,
                        annualized_predicted_te=float(np.sqrt(max(active * 252, 0))),
                        qlike=None
                        if actual_return is None or variance <= 0
                        else float(np.log(variance * 5) + actual_return**2 / (variance * 5)),
                    )
                records.append(record)
    return pd.DataFrame(records)
