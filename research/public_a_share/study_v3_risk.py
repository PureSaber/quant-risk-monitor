"""Paired exposure/estimator experiments on matched, explicitly retrospective data."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from study_v3_archive import digest, write_json


def model_definitions(protocol):
    from run_study import configurations

    baseline, grid = configurations(protocol["old_protocol"])
    frozen = grid["enhanced_s0.1_f90_d60_p10"]
    definitions = {
        "legacy_ols": ("legacy", baseline, "exposure"),
        "repaired_ols": ("repaired", baseline, "baseline"),
        "residual_ols": ("residual_volatility", baseline, "exposure"),
        "legacy_enhanced": ("legacy", frozen, "exposure_sensitivity"),
        "residual_enhanced": ("residual_volatility", frozen, "exposure_sensitivity"),
        **{name: ("repaired", cfg, "estimator") for name, cfg in grid.items()},
    }
    ablations = {
        "without_weighting": replace(frozen, regression_weighting="equal"),
        "without_decay": replace(frozen, factor_half_life=None, specific_half_life=None),
        "without_serial": replace(frozen, newey_west_lags=0, specific_newey_west_lags=0),
        "without_regime": replace(frozen, regime_window=0),
        "without_group": replace(
            frozen, specific_target="global", specific_shrinkage_method="fixed"
        ),
    }
    definitions.update({name: ("repaired", cfg, "ablation") for name, cfg in ablations.items()})
    return definitions


def forecast_grid(dates, horizon=5):
    start = dates.searchsorted(pd.Timestamp("2023-01-01"))
    return [
        (i, dates[i], dates[i + horizon])
        for i in range(start, len(dates) - horizon, horizon)
        if dates[i].year != 2023 or dates[i + horizon].year == 2023
    ]


def eligible_inputs(panel, histories, i, day):
    window = panel["dates"][i - 252 : i]
    if len(window) != 252:
        raise ValueError("insufficient trailing sessions")
    assets = pd.Index(panel["assets"])
    for variant, snapshots in histories["snapshots"].items():
        if day not in snapshots:
            raise ValueError("exposure unavailable: " + variant)
        assets = assets.intersection(snapshots[day].values.index, sort=False)
        for data in histories["histories"][variant].values():
            if not window.isin(data["factors"].index).all():
                raise ValueError("noncontiguous factor return history")
            residuals = data["residuals"].reindex(window)
            assets = assets.intersection(residuals.columns[residuals.notna().all()], sort=False)
    if len(assets) < 100:
        raise ValueError("fewer than 100 historically eligible assets")
    return assets, window


def factor_variance(x, f, d, weights):
    a = x.T @ weights
    factor = float(a @ f @ a)
    specific = float(np.dot(weights**2, d))
    return factor + specific, factor, specific


def portfolios(snap):
    assets = snap.values.index
    result = {
        "equal": ("equal", pd.Series(1 / len(assets), index=assets)),
        "capitalization": ("capitalization", snap.market_caps / snap.market_caps.sum()),
    }
    top = snap.market_caps.sort_values(ascending=False, kind="stable").head(10).index
    result["concentrated_top10"] = (
        "concentrated",
        pd.Series(0.1, index=top).reindex(assets, fill_value=0),
    )
    for group in sorted(snap.industries.unique()):
        mask = snap.industries.eq(group).astype(float)
        result["industry:" + group] = ("industry", mask / mask.sum())
    return result


def paired_summary(frame, model_ids, control, *, replications=5000, block=10, seed=20261015):
    """Resample dates jointly; every date keeps all its portfolio observations."""
    if frame.empty:
        return {"complete_dates": 0, "comparisons": {}, "means": {}}
    grouped = frame.groupby(["date", "model", "family"], sort=True).qlike.mean()
    daily = grouped.groupby(["date", "model"]).mean().unstack("model").reindex(columns=model_ids)
    # A complete case is a date with every declared model and every portfolio.
    missing_dates = frame.loc[frame.qlike.isna(), "date"].unique()
    daily = daily.loc[~daily.index.isin(missing_dates)].dropna()
    n = len(daily)
    result = {"complete_dates": n, "means": daily.mean().to_dict(), "comparisons": {}}
    if n < 2:
        return result
    rng = np.random.default_rng(seed)
    width = min(block, n)
    starts = rng.integers(0, n - width + 1, size=(replications, int(np.ceil(n / width))))
    indices = (starts[..., None] + np.arange(width)).reshape(replications, -1)[:, :n]
    targets = [m for m in model_ids if m != control]
    alpha = 0.05 / max(len(targets), 1)
    for model in targets:
        delta = (daily[model] - daily[control]).to_numpy()
        distribution = delta[indices].mean(axis=1)
        interval = np.quantile(distribution, [alpha / 2, 1 - alpha / 2])
        result["comparisons"][model] = {
            "mean_delta": float(delta.mean()),
            "simultaneous_95_interval": interval.tolist(),
            "historical_improvement": bool(interval[1] < 0),
        }
    return result


def run(out):
    from exposure_v2_study import subset
    from run_study import risk_components

    protocol = json.loads((out / "protocol.json").read_text(encoding="utf8"))
    binding = json.loads((out / "history-binding-v3.json").read_text(encoding="utf8"))
    for name in ("panel", "histories"):
        if digest(out / (name + ".pkl")) != binding[name]:
            raise ValueError("changed study inputs")
    with (out / "panel.pkl").open("rb") as stream:
        bundle = pickle.load(stream)
    with (out / "histories.pkl").open("rb") as stream:
        histories = pickle.load(stream)
    panel = bundle["panel"]
    definitions = model_definitions(protocol)
    write_json(
        out / "model-configurations.json",
        {
            name: {"variant": v, "configuration": asdict(c), "experiment": e}
            for name, (v, c, e) in definitions.items()
        },
    )
    rows, unavailable, models, timings, endpoint_issues = [], [], {}, [], []
    market = bundle["market_returns"].shift(1)
    volatility = market.rolling(20, min_periods=20).std()
    threshold = volatility.rolling(252, min_periods=126).median()
    for count, (i, day, end) in enumerate(forecast_grid(panel["dates"]), 1):
        started = perf_counter()
        try:
            assets, window = eligible_inputs(panel, histories, i, day)
        except ValueError as exc:
            unavailable.append({"date": str(day.date()), "reason": str(exc)})
            continue
        base = subset(histories["snapshots"]["repaired"][day], assets)
        weights = portfolios(base)
        daily_returns = panel["prices"]["returns"].iloc[i + 1 : i + 6].reindex(columns=assets)
        realized_linear = daily_returns.sum(min_count=5)
        open_price = panel["prices"]["adjusted_close"].loc[day, assets]
        end_price = panel["prices"]["adjusted_close"].loc[end, assets]
        realized_compound = end_price / open_price - 1
        missing = assets[realized_compound.isna()]
        for asset in missing:
            endpoint_issues.append(
                {
                    "date": str(day.date()),
                    "end": str(end.date()),
                    "asset": asset,
                    "opening_missing": bool(pd.isna(open_price[asset])),
                    "endpoint_missing": bool(pd.isna(end_price[asset])),
                }
            )
        regime = (
            "unclassified"
            if pd.isna(threshold.loc[day])
            else (
                "high_volatility" if volatility.loc[day] > threshold.loc[day] else "low_volatility"
            )
        )
        models[day] = {
            "assets": list(assets),
            "benchmark": weights["capitalization"][1],
            "models": {},
        }
        for model, (variant, cfg, experiment) in definitions.items():
            snap = subset(histories["snapshots"][variant][day], assets)
            history = histories["histories"][variant][cfg.regression_weighting]
            f, d, _ = risk_components(
                snap, history["factors"].loc[window], history["residuals"].loc[window], cfg
            )
            x = snap.values.to_numpy()
            models[day]["models"][model] = {
                "exposures": snap.values,
                "F_daily_rate": f,
                "D_daily_rate": d,
                "configuration": asdict(cfg),
            }
            for name, (family, w) in weights.items():
                vector = w.to_numpy()
                variance, factor, specific = factor_variance(x, f, d, vector)
                variance *= 5
                nonzero = w[w > 0].index
                actual = (
                    float(w.loc[nonzero] @ realized_linear.loc[nonzero])
                    if realized_linear.loc[nonzero].notna().all()
                    else np.nan
                )
                compound = (
                    float(w.loc[nonzero] @ realized_compound.loc[nonzero])
                    if realized_compound.loc[nonzero].notna().all()
                    else np.nan
                )
                rows.append(
                    {
                        "date": str(day.date()),
                        "end": str(end.date()),
                        "model": model,
                        "experiment": experiment,
                        "portfolio": name,
                        "family": family,
                        "assets": len(assets),
                        "regime": regime,
                        "variance": variance,
                        "factor_variance": factor * 5,
                        "specific_variance": specific * 5,
                        "return_linear": actual,
                        "return_compound": compound,
                        "qlike": float(np.log(variance) + actual**2 / variance)
                        if np.isfinite(actual) and variance > 0
                        else np.nan,
                    }
                )
        timings.append(
            {"date": str(day.date()), "seconds": perf_counter() - started, "assets": len(assets)}
        )
        if count % 20 == 0:
            print(json.dumps({"risk_forecast_dates": count, "rows": len(rows)}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "risk-forecasts.parquet", index=False)
    with (out / "risk-models.pkl").open("wb") as stream:
        pickle.dump(models, stream, protocol=5)
    write_json(out / "endpoint-issues.json", endpoint_issues)
    write_json(out / "risk-unavailable.json", unavailable)
    write_json(out / "risk-timings.json", timings)
    experiments = {
        "exposures": ["repaired_ols", "legacy_ols", "residual_ols"],
        "estimators": ["repaired_ols"]
        + [n for n, (_, _, kind) in definitions.items() if kind == "estimator"],
        "ablations": ["enhanced_s0.1_f90_d60_p10"]
        + [n for n, (_, _, kind) in definitions.items() if kind == "ablation"],
    }
    summary = {}
    for label, names in experiments.items():
        summary[label] = {}
        for period, predicate in {
            "development_2023": frame.date.lt("2024-01-01"),
            "diagnostic_2024_2026": frame.date.ge("2024-01-01"),
            "diagnostic_2026": frame.date.ge("2026-01-01"),
        }.items():
            subset_frame = frame.loc[predicate & frame.model.isin(names)]
            summary[label][period] = paired_summary(subset_frame, names, names[0])
    complete = frame.dropna(subset=["return_linear"])
    calibration = []
    for (model, family, regime), group in complete.groupby(["model", "family", "regime"]):
        calibration.append(
            {
                "model": model,
                "family": family,
                "regime": regime,
                "observations": len(group),
                "volatility_bias": float(
                    np.sqrt((group.return_linear**2).mean() / group.variance.mean())
                ),
                "mean_qlike": float(group.qlike.mean()),
            }
        )
    write_json(out / "risk-calibration.json", calibration)
    selection = summary["estimators"]["development_2023"]
    candidates = experiments["estimators"][1:]
    selected = (
        min(candidates, key=lambda n: (selection["means"][n], n))
        if selection["complete_dates"] >= 40
        else None
    )
    write_json(
        out / "risk-selection.json",
        {
            "selected": selected,
            "development_dates": selection["complete_dates"],
            "protocol_sha256": digest(out / "protocol.json"),
            "created_at_after_historical_observation": True,
            "promotion": "not_passed",
            "independent_new_windows": 0,
        },
    )
    write_json(out / "risk-summary.json", summary)
    write_json(
        out / "risk-models-binding.json",
        {
            "models": digest(out / "risk-models.pkl"),
            "forecast_rows": digest(out / "risk-forecasts.parquet"),
            "protocol": digest(out / "protocol.json"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
