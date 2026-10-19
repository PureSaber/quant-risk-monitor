"""Frozen field-by-estimator factorial; never select using evaluation outcomes."""

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_panel import snapshot
from exposure_v2_study import subset
from model_bootstrap import public_equity_style_config, solve_factor_returns
from run_study import risk_components
from study_v3_archive import digest, write_json
from study_v3_risk import factor_variance, model_definitions, paired_summary, portfolios


def run(baseline: Path, source: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    with baseline.open("rb") as stream:
        original = pickle.load(stream)["v2"]
    with (source / "panel.pkl").open("rb") as stream:
        restored = pickle.load(stream)["v2"]
    definitions = model_definitions(json.loads((source / "protocol.json").read_text()))
    configs = {
        "ols": definitions["repaired_ols"][1],
        "frozen_enhanced": definitions["enhanced_s0.1_f90_d60_p10"][1],
    }
    panels = {"old_input": original, "restored_input": restored}
    for field in ("book_to_price", "earnings_yield_ttm", "earnings_growth_yoy", "book_leverage"):
        ds = dict(restored["descriptors"])
        ds[field] = original["descriptors"][field]
        panels["without_" + field] = {**restored, "descriptors": ds}
    # No restored prices, caps or industries are silently substituted here.
    for name in ("close", "adjusted_close", "returns"):
        pd.testing.assert_frame_equal(original["prices"][name], restored["prices"][name])
    pd.testing.assert_frame_equal(original["caps"], restored["caps"])
    pd.testing.assert_frame_equal(original["groups"], restored["groups"])
    industries = sorted(
        x for x in pd.unique(original["groups"].to_numpy().ravel()) if isinstance(x, str)
    )
    style = public_equity_style_config(
        industries,
        definition_version="v2",
        enhancements=(),
        min_assets=100,
        missing_policy="industry_median",
        missing_group_fallback="global",
        max_missing_fraction=0.4,
    )
    dates = original["dates"]
    evaluation = set(dates[dates >= "2023-01-03"][::5])
    history = {
        (field, estimator): {"dates": [], "factors": [], "residuals": []}
        for field in panels
        for estimator in configs
    }
    previous = {}
    rows = []
    failures = []
    diagnostics = []
    write_json(
        out / "protocol.json",
        {
            "input_hashes": {
                str(baseline): digest(baseline),
                str(source / "panel.pkl"): digest(source / "panel.pkl"),
                str(source / "protocol.json"): digest(source / "protocol.json"),
            },
            "producer": digest(__file__),
            "fields": list(panels),
            "estimators": {name: asdict(cfg) for name, cfg in configs.items()},
            "history": 252,
            "horizon": 5,
            "select_parameters": False,
            "scope": "same causal regression sample, common complete residual history, dated portfolio and realized return; no scalar duplicate active-cap",
            "default_changed": False,
            "forward_windows": 0,
        },
    )
    for number, day in enumerate(dates[dates >= "2021-11-11"], 1):
        returns = original["prices"]["returns"].loc[day].dropna()
        if len(previous) == len(panels):
            ids = returns.index
            for snap in previous.values():
                ids = ids.intersection(snap.values.index, sort=False)
            if len(ids) >= 100:
                for field, snap in previous.items():
                    current = subset(snap, ids)
                    for estimator, cfg in configs.items():
                        factors, residuals, *_ = solve_factor_returns(
                            current, returns.loc[ids], cfg
                        )
                        h = history[field, estimator]
                        h["dates"].append(day)
                        h["factors"].append(pd.Series(factors, index=current.values.columns))
                        h["residuals"].append(pd.Series(residuals, index=ids))
        snaps = {}
        for field, panel in panels.items():
            try:
                snaps[field] = snapshot(panel, day, style)
            except ValueError as exc:
                failures.append(
                    {
                        "date": str(day.date()),
                        "field": field,
                        "stage": "snapshot",
                        "reason": str(exc),
                    }
                )
        previous = snaps
        if number % 100 == 0:
            print(
                json.dumps({"sessions": number, "date": str(day.date()), "forecasts": len(rows)}),
                flush=True,
            )
        if day not in evaluation or len(snaps) != len(panels):
            continue
        i = dates.get_loc(day)
        if i + 5 >= len(dates):
            continue
        window = dates[i - 252 : i]
        common = pd.Index(original["assets"])
        data = {}
        for (field, estimator), h in history.items():
            f = pd.DataFrame(h["factors"], index=h["dates"]).reindex(window)
            d = pd.DataFrame(h["residuals"], index=h["dates"]).reindex(window)
            if len(f) != 252 or f.isna().any().any():
                data = {}
                break
            common = common.intersection(snaps[field].values.index, sort=False).intersection(
                d.columns[d.notna().all()], sort=False
            )
            data[field, estimator] = (f, d)
        if len(data) != len(history) or len(common) < 100:
            failures.append(
                {
                    "date": str(day.date()),
                    "stage": "history",
                    "reason": "insufficient_common_complete_history",
                }
            )
            continue
        base = subset(snaps["restored_input"], common)
        weights = {name: (family, 0.95 * w) for name, (family, w) in portfolios(base).items()}
        benchmark = original["caps"].loc[day, common]
        benchmark = benchmark / benchmark.sum()
        weights["active_equal"] = (
            "active",
            pd.Series(0.95 / len(common), index=common) - benchmark,
        )
        future = (
            original["prices"]["returns"]
            .iloc[i + 1 : i + 6]
            .reindex(columns=common)
            .sum(min_count=5)
        )
        for (field, estimator), (f, d) in data.items():
            snap = subset(snaps[field], common)
            F, D, _ = risk_components(snap, f, d.loc[:, common], configs[estimator])
            x = snap.values.to_numpy()
            # Orthogonal-projector leverage describes exposure concentration;
            # it is not a correction or the constrained estimator's hat matrix.
            u, s, _ = np.linalg.svd(x, full_matrices=False)
            keep = s > s.max() * 1e-12
            leverage = (u[:, keep] ** 2).sum(axis=1)
            diagnostics.append(
                {
                    "date": str(day.date()),
                    "field": field,
                    "estimator": estimator,
                    "assets": len(common),
                    "design_rank": int(keep.sum()),
                    "max_unweighted_design_leverage": float(leverage.max()),
                    "highest_leverage_asset": str(common[np.argmax(leverage)]),
                    "max_absolute_exposure": float(abs(x).max()),
                }
            )
            for name, (family, w) in weights.items():
                w = w.reindex(common, fill_value=0)
                observed = future[w.ne(0)].notna().all()
                outcome = float(w[w.ne(0)] @ future[w.ne(0)]) if observed else np.nan
                var, fv, dv = factor_variance(x, F, D, w.to_numpy())
                variance = 5 * var
                rows.append(
                    {
                        "date": day,
                        "end": dates[i + 5],
                        "year": day.year,
                        "field": field,
                        "estimator": estimator,
                        "model": field + ":" + estimator,
                        "portfolio": name,
                        "family": family,
                        "variance": variance,
                        "factor_variance": 5 * fv,
                        "specific_variance": 5 * dv,
                        "realized": outcome,
                        "calibration": outcome**2 / variance,
                        "qlike": np.log(variance) + outcome**2 / variance,
                        "assets": len(common),
                        "complete": bool(observed),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "forecasts.parquet", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "exposure-diagnostics.csv", index=False)
    write_json(out / "failures.json", failures)
    write_json(
        out / "summary.json",
        {
            "retrospective": paired_summary(
                frame[frame.year.ge(2024)], sorted(frame.model.unique()), "restored_input:ols"
            ),
            "promotion_accepted": False,
            "reason": "factorial retrospective diagnosis; no new independent windows",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args.baseline, args.source, args.out)
