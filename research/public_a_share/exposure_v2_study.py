"""Frozen exposure-only comparison; reused 2024-25 outcomes are diagnostics."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_panel import snapshot
from model_bootstrap import (
    SOURCE,
    ExposureSnapshot,
    FactorModelConfig,
    public_equity_style_config,
    solve_factor_returns,
)

# isort: split

from quant_risk_monitor.exposure_cache import DailyExposureStore
from quant_risk_monitor.style_descriptors import (
    additional_market_descriptors,
    annual_growth_descriptors,
    paired_market_descriptors,
)

from run_study import date_grid, risk_components

VARIANTS = {
    "legacy": (),
    "repaired": (),
    "liquidity": ("liquidity",),
    "growth": ("growth",),
    "residual_volatility": ("residual_volatility",),
    "combined": ("liquidity", "growth", "residual_volatility"),
    "growth_available": ("growth",),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8"
    )


def history_binding(protocol, script=None):
    """Bind reusable regressions to their producer, independent of report edits."""
    tree = ast.parse(Path(script or __file__).read_text(encoding="utf8"))
    names = {"prepare", "subset", "build_histories", "VARIANTS"}
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
        or isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
    ]
    semantic = hashlib.sha256(
        ast.dump(ast.Module(body=definitions, type_ignores=[])).encode()
    ).hexdigest()
    paths = [
        SOURCE / name
        for name in (
            "exposure_builder.py",
            "style_descriptors.py",
            "risk_estimators.py",
            "factor_model.py",
            "exposure_cache.py",
        )
    ]
    paths += [
        Path(__file__).with_name(name)
        for name in ("build_panel.py", "run_study.py", "model_bootstrap.py")
    ]
    return {
        "inputs": protocol["inputs"],
        "protocol": protocol,
        "history_functions_sha256": semantic,
        "modules": {p.name: digest(p) for p in paths},
    }


def seal_histories(out, protocol):
    write_json(
        out / "history-manifest.json",
        {"binding": history_binding(protocol), "histories_sha256": digest(out / "histories.pkl")},
    )


def load_histories(out, protocol):
    manifest = json.loads((out / "history-manifest.json").read_text(encoding="utf8"))
    if manifest["binding"] != json.loads(json.dumps(history_binding(protocol))) or manifest[
        "histories_sha256"
    ] != digest(out / "histories.pkl"):
        raise ValueError("history producer/input/content binding changed; rebuild histories")
    with (out / "histories.pkl").open("rb") as stream:
        return pickle.load(stream)


def freeze(old, out):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf8"))
    old_protocol = json.loads((old / "protocol.json").read_text(encoding="utf8"))
    protocol = {
        "schema": "public-exposure-study/v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": "retrospective; 2023 development selection; 2024-25 previously inspected and not a fresh holdout",
        "variants": VARIANTS,
        "source_exclusion": ["Huisheng", "Dataway"],
        "fixed_cohort": 200,
        "lookback": 252,
        "horizon": 5,
        "min_assets": 100,
        "selection_year": 2023,
        "diagnostic_years": [2024, 2025],
        "confirmation_end": "2026-09-30",
        "minimum_selection_dates": 40,
        "minimum_confirmation_dates": 80,
        "bootstrap": {
            "replications": 5000,
            "block_dates": 10,
            "seed": 20261014,
            "correction": "Bonferroni simultaneous 95% intervals over five optional candidates",
        },
        "raw_coverage_floor": 0.8,
        "maximum_exposure_failure_fraction": 0.05,
        "maximum_log_bias_deterioration": 0.1,
        "maximum_coverage_drop": 0.05,
        "priority": "fewer added descriptors, then lower mean selection QLIKE, then identifier",
        "old_protocol": old_protocol,
        "inputs": {
            "panel": digest(old / "clean/panel.pkl"),
            "financial_facts": digest(old / "clean/financial_facts.csv"),
            "warmup": digest(out / "annual-warmup-facts.csv"),
            "source_manifest": digest(old / "clean/source-manifest.json"),
        },
    }
    write_json(path, protocol)
    return protocol


def prepare(old, out, protocol):
    with (old / "clean/panel.pkl").open("rb") as stream:
        panel = pickle.load(stream)
    facts = pd.read_csv(
        old / "clean/financial_facts.csv", dtype={"asset": str, "report_period": str}
    )
    warmup = pd.read_csv(
        out / "annual-warmup-facts.csv", dtype={"asset": str, "report_period": str}
    )
    facts = pd.concat([facts, warmup], ignore_index=True)
    data = json.loads((old / "raw/benchmark/csi300.json").read_text(encoding="utf8"))["data"]
    market = (
        pd.Series({pd.Timestamp(r.split(",")[0]): float(r.split(",")[2]) for r in data["klines"]})
        .sort_index()
        .pct_change(fill_method=None)
        .reindex(panel["dates"])
    )
    corrected, paired_counts = paired_market_descriptors(panel["prices"]["returns"], market)
    extra = additional_market_descriptors(panel["prices"]["returns"], panel["prices"]["turnover"])
    growth, quality = annual_growth_descriptors(facts, panel["dates"], panel["assets"])
    v2 = {**panel, "descriptors": {**panel["descriptors"], **corrected, **extra, **growth}}
    diagnostics = []
    for name, frame in {**extra, **growth}.items():
        frame.to_parquet(out / (name + ".parquet"))
        for year in (2022, 2023, 2024, 2025):
            sub = frame.loc[str(year)]
            diagnostics.append(
                {
                    "descriptor": name,
                    "year": year,
                    "raw_observed_fraction": float(sub.notna().mean().mean()),
                }
            )
    for name, frame in quality.items():
        frame.to_parquet(out / (name + "-quality.parquet"))
    paired_counts.to_parquet(out / "paired-observations.parquet")
    pd.DataFrame(diagnostics).to_csv(out / "descriptor-coverage.csv", index=False)
    return panel, v2


def subset(snap, assets):
    return ExposureSnapshot(
        effective_at=snap.effective_at,
        available_at=snap.available_at,
        values=snap.values.loc[assets],
        source=snap.source,
        market_caps=snap.market_caps.loc[assets],
        industries=snap.industries.loc[assets],
        industry_factors=snap.industry_factors,
    )


def build_histories(panel, v2, out, protocol):
    groups = sorted(set(protocol["old_protocol"]["sector_mapping"].values()))
    configs = {
        name: public_equity_style_config(
            groups,
            definition_version="v1" if name == "legacy" else "v2",
            enhancements=enhancements,
            available_growth=name == "growth_available",
            min_assets=100,
            missing_policy="industry_median",
            missing_group_fallback="global",
            max_missing_fraction=0.4 if name in ("legacy", "repaired") else 0.2,
        )
        for name, enhancements in VARIANTS.items()
    }
    code = {
        p.name: digest(p)
        for p in [
            SOURCE / "exposure_builder.py",
            SOURCE / "style_descriptors.py",
            SOURCE / "risk_estimators.py",
            Path(__file__),
        ]
    }
    cache = DailyExposureStore(out / "daily-cache")
    snapshots = {name: {} for name in configs}
    failures = []
    days = panel["dates"][panel["dates"] >= "2021-11-11"]
    for i, day in enumerate(days, 1):
        for name, config in configs.items():
            try:
                snap, _ = cache.get_or_build(
                    as_of=day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15),
                    configuration=asdict(config),
                    input_digests={**protocol["inputs"], **code},
                    build=lambda day=day, name=name, config=config: snapshot(
                        panel if name == "legacy" else v2, day, config
                    ),
                )
                snapshots[name][day] = subset(snap, snap.values.index)
            except ValueError as exc:
                failures.append({"variant": name, "date": str(day.date()), "reason": str(exc)})
        if i % 100 == 0:
            print(
                json.dumps(
                    {"exposure_dates": i, "total": len(days), "failed_snapshots": len(failures)}
                ),
                flush=True,
            )
    write_json(out / "snapshot-failures.json", failures)
    # Admission uses only availability and pre-2024 data, never outcomes.
    development_days = days[days < "2024-01-01"]
    admission = {
        name: {
            "development_snapshot_fraction": sum(d in snapshots[name] for d in development_days)
            / len(development_days)
        }
        for name in configs
    }
    for name, info in admission.items():
        info["admitted"] = (
            info["development_snapshot_fraction"]
            >= 1 - protocol["maximum_exposure_failure_fraction"]
        )
    admitted = [n for n in configs if admission[n]["admitted"]]
    if not {"legacy", "repaired"} <= set(admitted):
        raise ValueError("baseline exposure build failed; inspect snapshots")
    write_json(out / "candidate-admission.json", admission)
    histories = {
        n: {
            method: {"factors": [], "residuals": [], "dates": []}
            for method in ("equal", "sqrt_market_cap")
        }
        for n in admitted
    }
    common_snapshots = {n: {} for n in admitted}
    regression_failures = []
    for day in days:
        if not all(day in snapshots[n] for n in admitted):
            continue
        assets = snapshots["legacy"][day].values.index
        for name in admitted:
            assets = assets.intersection(snapshots[name][day].values.index, sort=False)
        if len(assets) < 100:
            continue
        for name in admitted:
            common_snapshots[name][day] = subset(snapshots[name][day], assets)
    for i, day in enumerate(panel["dates"]):
        if i == 0 or panel["dates"][i - 1] not in common_snapshots["legacy"]:
            continue
        prior = panel["dates"][i - 1]
        r = panel["prices"]["returns"].loc[day].dropna()
        assets = common_snapshots["legacy"][prior].values.index.intersection(r.index, sort=False)
        if len(assets) < 100:
            continue
        staged = []
        try:
            for name in admitted:
                for method in histories[name]:
                    snap = subset(common_snapshots[name][prior], assets)
                    cfg = FactorModelConfig(
                        model_kind="fundamental_style",
                        regression_weighting=method,
                        industry_constraint=True,
                    )
                    f, residual, *_ = solve_factor_returns(snap, r.loc[assets], cfg)
                    staged.append(
                        (
                            name,
                            method,
                            pd.Series(f, index=snap.values.columns),
                            pd.Series(residual, index=assets),
                        )
                    )
        except ValueError as exc:
            regression_failures.append({"date": str(day.date()), "reason": str(exc)})
            continue
        for name, method, f, residual in staged:
            histories[name][method]["factors"].append(f)
            histories[name][method]["residuals"].append(residual)
            histories[name][method]["dates"].append(day)
    for data in histories.values():
        for method in data.values():
            method["factors"] = pd.DataFrame(
                method["factors"], index=pd.DatetimeIndex(method["dates"])
            )
            method["residuals"] = pd.DataFrame(
                method["residuals"], index=pd.DatetimeIndex(method["dates"])
            ).reindex(columns=panel["assets"])
    write_json(out / "regression-failures.json", regression_failures)
    with (out / "histories.pkl").open("wb") as stream:
        pickle.dump(
            {"snapshots": common_snapshots, "histories": histories, "admission": admission},
            stream,
            protocol=5,
        )
    return common_snapshots, histories, admission


def evaluate(panel, snapshots, histories, grid, estimators):
    records, unavailable = [], []
    for i, day, end in grid:
        names = list(histories)
        if day not in snapshots[names[0]]:
            unavailable.append({"date": str(day.date()), "reason": "exposure_missing"})
            continue
        history = panel["dates"][i - 252 : i]
        assets = snapshots[names[0]][day].values.index
        baseline_assets = assets
        contiguous = True
        for name in names:
            for data in histories[name].values():
                if not history.isin(data["factors"].index).all():
                    contiguous = False
                    break
                residuals = data["residuals"].reindex(history)
                valid = residuals.columns[residuals.notna().all()]
                assets = assets.intersection(valid, sort=False)
                if name == "repaired":
                    baseline_assets = baseline_assets.intersection(valid, sort=False)
        if not contiguous or len(assets) < 100:
            unavailable.append(
                {
                    "date": str(day.date()),
                    "reason": "noncontiguous_history_or_insufficient_assets",
                    "assets": len(assets),
                }
            )
            continue
        snap = subset(snapshots[names[0]][day], assets)
        weights = {
            "equal": pd.Series(1 / len(assets), index=assets),
            "capitalization": snap.market_caps / snap.market_caps.sum(),
        }
        for industry in sorted(snap.industries.unique()):
            mask = snap.industries.eq(industry).astype(float)
            weights["industry:" + industry] = mask / mask.sum()
        px = panel["prices"]["adjusted_close"]
        actual = px.loc[end, assets] / px.loc[day, assets] - 1
        for name in names:
            current = subset(snapshots[name][day], assets)
            for estimator, cfg in estimators.items():
                data = histories[name][cfg.regression_weighting]
                f, d, _ = risk_components(
                    current, data["factors"].loc[history], data["residuals"].loc[history], cfg
                )
                for portfolio, w in weights.items():
                    e = current.values.T @ w
                    variance = float(e.to_numpy() @ f @ e.to_numpy() + (w * w * d).sum()) * 5
                    if not np.isfinite(variance) or variance <= 0:
                        raise ValueError("invalid risk forecast")
                    held = w.ne(0)
                    realized = (
                        float((w[held] * actual[held]).sum())
                        if actual[held].notna().all()
                        else np.nan
                    )
                    records.append(
                        {
                            "date": str(day.date()),
                            "end": str(end.date()),
                            "variant": name,
                            "estimator": estimator,
                            "portfolio": portfolio,
                            "family": "industry"
                            if portfolio.startswith("industry:")
                            else portfolio,
                            "assets": len(assets),
                            "baseline_assets": len(baseline_assets),
                            "variance": variance,
                            "return": realized,
                            "qlike": float(np.log(variance) + realized**2 / variance),
                        }
                    )
    return pd.DataFrame(records), unavailable


def summarize(records, protocol):
    if records.empty:
        return {
            name: {"complete_dates": 0, "mean_qlike": {}, "volatility_bias": {}, "comparisons": {}}
            for name in ("common_ols", "frozen_enhanced")
        }
    results = {}
    rng = np.random.default_rng(protocol["bootstrap"]["seed"])
    for estimator, group in records.groupby("estimator"):
        # Do not let groupby.mean silently discard missing industry endpoints.
        dates = group.groupby("date")["qlike"].apply(lambda s: s.notna().all())
        complete = group[group.date.isin(dates[dates].index)]
        scores = (
            complete.groupby(["date", "variant", "family"])
            .qlike.mean()
            .groupby(["date", "variant"])
            .mean()
            .unstack("variant")
            .dropna()
        )
        biases = {}
        for name, sub in group[group["return"].notna()].groupby("variant"):
            biases[name] = {
                family: float(np.sqrt((g["return"] ** 2).sum() / g.variance.sum()))
                for family, g in sub.groupby("family")
            }
        comparisons = {}
        n = len(scores)
        block = protocol["bootstrap"]["block_dates"]
        for name in scores.columns:
            if name == "repaired":
                continue
            delta = (scores[name] - scores.repaired).to_numpy()
            interval = None
            if n >= block:
                starts = rng.integers(
                    0,
                    n - block + 1,
                    size=(protocol["bootstrap"]["replications"], int(np.ceil(n / block))),
                )
                indices = (starts[:, :, None] + np.arange(block)).reshape(len(starts), -1)[:, :n]
                alpha = 0.05 / 5 if name != "legacy" else 0.05
                interval = np.quantile(
                    delta[indices].mean(axis=1), [alpha / 2, 1 - alpha / 2]
                ).tolist()
            bias_ok = all(
                biases[name][f] > 0
                and biases["repaired"][f] > 0
                and abs(np.log(biases[name][f]))
                <= abs(np.log(biases["repaired"][f])) + protocol["maximum_log_bias_deterioration"]
                for f in biases["repaired"]
            )
            coverage_ok = bool(
                (
                    (group.assets / group.baseline_assets) >= 1 - protocol["maximum_coverage_drop"]
                ).all()
            )
            comparisons[name] = {
                "mean_delta": float(delta.mean()) if n else None,
                "simultaneous_interval": interval,
                "bias_guardrail": bias_ok,
                "coverage_guardrail": coverage_ok,
                "passes_selection": bool(
                    n >= protocol["minimum_selection_dates"]
                    and interval is not None
                    and interval[1] < 0
                    and bias_ok
                    and coverage_ok
                ),
            }
        results[estimator] = {
            "complete_dates": n,
            "mean_qlike": scores.mean().to_dict(),
            "volatility_bias": biases,
            "comparisons": comparisons,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-histories", action="store_true")
    args = parser.parse_args()
    protocol = freeze(args.old_root, args.out)
    for key, path in {
        "panel": args.old_root / "clean/panel.pkl",
        "financial_facts": args.old_root / "clean/financial_facts.csv",
        "warmup": args.out / "annual-warmup-facts.csv",
        "source_manifest": args.old_root / "clean/source-manifest.json",
    }.items():
        if digest(path) != protocol["inputs"][key]:
            raise ValueError("frozen input changed")
    panel, v2 = prepare(args.old_root, args.out, protocol)
    if args.resume_histories:
        saved = load_histories(args.out, protocol)
        snapshots, histories, admission = saved["snapshots"], saved["histories"], saved["admission"]
    else:
        snapshots, histories, admission = build_histories(panel, v2, args.out, protocol)
        seal_histories(args.out, protocol)
    common = dict(
        model_kind="fundamental_style",
        risk_horizon=5,
        min_periods=252,
        min_assets_per_period=100,
        min_asset_observations=252,
    )
    estimators = {
        "common_ols": FactorModelConfig(**common, **protocol["old_protocol"]["baseline"]),
        "frozen_enhanced": FactorModelConfig(
            **common,
            **protocol["old_protocol"]["enhanced_fixed"],
            covariance_shrinkage=0.1,
            factor_half_life=90,
            specific_half_life=60,
            specific_prior_dof=10,
        ),
    }
    selection, diagnostic = date_grid(panel)
    records, missing = evaluate(panel, snapshots, histories, selection, estimators)
    records.to_csv(args.out / "selection-forecasts.csv", index=False)
    summary = summarize(records, protocol)
    eligible = [
        n
        for n, c in summary["common_ols"]["comparisons"].items()
        if n != "legacy" and c["passes_selection"]
    ]
    complexity = {
        "liquidity": 2,
        "growth": 1,
        "growth_available": 1,
        "residual_volatility": 2,
        "combined": 5,
    }
    selected = (
        min(eligible, key=lambda n: (complexity[n], summary["common_ols"]["mean_qlike"][n], n))
        if eligible
        else "repaired"
    )
    frozen = {
        "selected": selected,
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "development": summary,
        "admission": admission,
        "unavailable_dates": missing,
        "protocol_sha256": digest(args.out / "protocol.json"),
        "history_manifest_sha256": digest(args.out / "history-manifest.json"),
        "evaluation_code_sha256": digest(__file__),
        "optional_enhancement_passed": bool(eligible),
        "production_changed": False,
    }
    write_json(args.out / "selection.json", frozen)
    print(
        json.dumps(
            {"selection": selected, "complete_dates": summary["common_ols"]["complete_dates"]}
        ),
        flush=True,
    )
    records, missing = evaluate(panel, snapshots, histories, diagnostic, estimators)
    records.to_csv(args.out / "diagnostic-forecasts.csv", index=False)
    write_json(
        args.out / "diagnostic.json",
        {
            "status": "previously_seen_2024_2025_not_independent_confirmation",
            "metrics": summarize(records, protocol),
            "unavailable_dates": missing,
            "production_changed": False,
        },
    )


if __name__ == "__main__":
    main()
