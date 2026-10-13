"""Selection-only tuning followed by paired, frozen-parameter historical holdout."""

from __future__ import annotations

import hashlib
import itertools
import json
import pickle
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from model_bootstrap import (
    SOURCE,
    ExposureSnapshot,
    FactorModelConfig,
    estimate_covariance,
    estimate_specific_risk,
    regime_multiplier,
)
from public_sources import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numerical_code_hashes():
    files = {
        name: Path(__file__).parent / name
        for name in [
            "build_panel.py",
            "original_financials.py",
            "prepare_pit.py",
            "run_study.py",
        ]
    }
    files.update(
        {
            "quant_risk_monitor/" + name: SOURCE / name
            for name in ["factor_model.py", "exposure_builder.py", "risk_estimators.py"]
        }
    )
    return {name: digest(path) for name, path in files.items()}


def load():
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    with (ROOT / "clean/panel.pkl").open("rb") as f:
        panel = pickle.load(f)
    with (ROOT / "clean/regressions.pkl").open("rb") as f:
        regressions = pickle.load(f)
    return protocol, panel, regressions


def configurations(protocol):
    common = {
        "model_kind": "fundamental_style",
        "risk_horizon": 5,
        "min_periods": 252,
        "min_assets_per_period": 100,
        "min_asset_observations": 252,
    }
    baseline = FactorModelConfig(**common, **protocol["baseline"])
    candidates = {}
    grid = protocol["candidate_grid"]
    for shrink, (factor_hl, specific_hl), dof in itertools.product(
        grid["covariance_shrinkage"],
        grid["half_life_pairs"],
        grid["specific_prior_dof"],
    ):
        name = f"enhanced_s{shrink:g}_f{factor_hl}_d{specific_hl}_p{dof}"
        candidates[name] = FactorModelConfig(
            **common,
            **protocol["enhanced_fixed"],
            covariance_shrinkage=shrink,
            factor_half_life=factor_hl,
            specific_half_life=specific_hl,
            specific_prior_dof=dof,
        )
    return baseline, candidates


def date_grid(panel):
    dates = panel["dates"]
    first = int(dates.searchsorted(pd.Timestamp("2023-01-01")))
    selection = []
    holdout = []
    for i in range(first, len(dates) - 5, 5):
        start, end = dates[i], dates[i + 5]
        if start.year == 2023 and end.year == 2023:
            selection.append((i, start, end))
        elif start.year in (2024, 2025) and end.year <= 2025:
            holdout.append((i, start, end))
    last_selection_end = selection[-1][0] + 5
    holdout = [row for row in holdout if row[0] >= last_selection_end + 5]
    return selection, holdout


def forecast_inputs(panel, regressions, row):
    i, day, end = row
    if day not in regressions["snapshots"]:
        raise ValueError(f"Missing exposure snapshot on {day.date()}")
    current = regressions["snapshots"][day]
    expected = panel["dates"][i - 252 : i]
    if len(expected) != 252:
        raise ValueError("Insufficient historical sessions")
    eligible = current.values.index
    for method, data in regressions["histories"].items():
        if not expected.isin(data["factors"].index).all():
            missing = expected[~expected.isin(data["factors"].index)]
            raise ValueError(
                f"Noncontiguous {method} factor history before {day.date()}: {list(missing[:3])}"
            )
        residuals = data["residuals"].reindex(expected)
        eligible = eligible.intersection(residuals.columns[residuals.notna().all()])
    if len(eligible) < 100:
        raise ValueError(f"Only {len(eligible)} causally eligible assets on {day.date()}")
    snap = ExposureSnapshot(
        effective_at=current.effective_at,
        available_at=current.available_at,
        values=current.values.loc[eligible],
        source=current.source,
        market_caps=current.market_caps.loc[eligible],
        industries=current.industries.loc[eligible],
        industry_factors=current.industry_factors,
    )
    weights = {
        "equal": pd.Series(1 / len(eligible), index=eligible),
        "capitalization": snap.market_caps / snap.market_caps.sum(),
    }
    for sector in sorted(snap.industries.unique()):
        mask = snap.industries.eq(sector).astype(float)
        weights["industry:" + sector] = mask / mask.sum()
    return snap, expected, weights


def risk_components(snap, factors, residuals, config):
    covariance, _ = estimate_covariance(
        factors.to_numpy(),
        half_life=config.factor_half_life,
        lags=config.newey_west_lags,
        shrinkage=config.covariance_shrinkage,
    )
    factor_scale, _ = regime_multiplier(
        factors.to_numpy(), config, half_life=config.factor_half_life
    )
    specific, diagnostic = estimate_specific_risk(
        {a: residuals[a].to_list() for a in snap.values.index},
        snap,
        config,
        len(factors),
    )
    return covariance * factor_scale, specific / config.annualization, diagnostic


def legacy_components(factors, residuals, eligible):
    sample = factors.cov().to_numpy()
    covariance = 0.8 * sample + 0.2 * np.diag(np.diag(sample))
    raw = residuals.reindex(columns=eligible).var(ddof=1)
    specific = 0.8 * raw.reindex(eligible) + 0.2 * raw.mean()
    return covariance, specific, {"original_algorithm_revision": "54fcf4d"}


def evaluate(protocol, panel, regressions, grid, configs):
    records = []
    for count, row in enumerate(grid, 1):
        i, day, end = row
        snap, history, weights = forecast_inputs(panel, regressions, row)
        opening = panel["prices"]["adjusted_close"].loc[day, snap.values.index]
        closing = panel["prices"]["adjusted_close"].loc[end, snap.values.index]
        actual = closing / opening - 1
        stale = panel["prices"]["adjusted_close"].loc[:end].ffill()
        marked = stale.loc[end, snap.values.index] / stale.loc[day, snap.values.index] - 1
        for name, config in configs.items():
            method = "legacy_ols" if name == "legacy_ols" else config.regression_weighting
            data = regressions["histories"][method]
            if name == "legacy_ols":
                F, D, diagnostic = legacy_components(
                    data["factors"].loc[history],
                    data["residuals"].loc[history],
                    snap.values.index,
                )
                design = snap.values.drop(columns="market")
            else:
                F, D, diagnostic = risk_components(
                    snap,
                    data["factors"].loc[history],
                    data["residuals"].loc[history],
                    config,
                )
                design = snap.values
            for portfolio, w in weights.items():
                exposure = design.T @ w
                variance = (
                    float(exposure.to_numpy() @ F @ exposure.to_numpy() + (w * w * D).sum())
                    * config.risk_horizon
                )
                if not np.isfinite(variance) or variance <= 0:
                    raise ValueError("Nonpositive forecast variance")
                held = w.ne(0)
                complete = bool(actual[held].notna().all())
                realized = float((w[held] * actual[held]).sum()) if complete else None
                marked_return = (
                    float((w[held] * marked[held]).sum()) if marked[held].notna().all() else None
                )
                family = "industry" if portfolio.startswith("industry:") else portfolio
                records.append(
                    {
                        "date": str(day.date()),
                        "end": str(end.date()),
                        "model": name,
                        "portfolio": portfolio,
                        "family": family,
                        "assets": int(held.sum()),
                        "universe_assets": len(w),
                        "variance": variance,
                        "return": realized,
                        "qlike": None
                        if realized is None
                        else float(np.log(variance) + realized**2 / variance),
                        "stale_return": marked_return,
                        "stale_qlike": None
                        if marked_return is None
                        else float(np.log(variance) + marked_return**2 / variance),
                        "missing_assets": list(actual[held][actual[held].isna()].index),
                    }
                )
        if count % 10 == 0:
            print(
                json.dumps(
                    {
                        "forecast_dates": count,
                        "of": len(grid),
                        "through": str(day.date()),
                        "candidates": len(configs),
                    }
                ),
                flush=True,
            )
    return pd.DataFrame(records)


def daily_scores(records, column="qlike"):
    # Each family has 1/3 total weight, regardless of its number of portfolios.
    family = records.groupby(["model", "date", "family"])[column].mean().unstack("family")
    complete = family.dropna(subset=["equal", "capitalization", "industry"])
    return complete.mean(axis=1).unstack("model").dropna()


def save_records(frame, path):
    frame.assign(missing_assets=frame.missing_assets.map(json.dumps)).to_csv(
        path, index=False, encoding="utf-8"
    )


def select():
    if (ROOT / "active-run.json").exists():
        raise ValueError("A frozen selection already exists; preserve it instead of reselecting")
    protocol, panel, regressions = load()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = ROOT / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    baseline, candidates = configurations(protocol)
    selection, _ = date_grid(panel)
    records = evaluate(
        protocol,
        panel,
        regressions,
        selection,
        {
            "baseline": baseline,
            "legacy_ols": replace(baseline, industry_constraint=False),
            **candidates,
        },
    )
    scores = daily_scores(records)
    if len(scores) < protocol["minimum_selection_dates"]:
        raise ValueError(f"Only {len(scores)} complete selection dates")
    means = scores.mean().to_dict()
    selected = min(candidates, key=lambda name: (means[name], name))
    frozen = {
        "status": "selection_frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": digest(ROOT / "protocol.json"),
        "clarifications_sha256": digest(ROOT / "protocol-clarifications.json"),
        "panel_sha256": digest(ROOT / "clean/panel.pkl"),
        "regressions_sha256": digest(ROOT / "clean/regressions.pkl"),
        "source_manifest_sha256": digest(ROOT / "clean/source-manifest.json"),
        "dates": len(scores),
        "scores": means,
        "selected_id": selected,
        "selected_config": asdict(candidates[selected]),
        "baseline_config": asdict(baseline),
        "holdout_not_used_in_selection": True,
    }
    frozen["numerical_code_sha256"] = numerical_code_hashes()
    save_records(records, run / "selection-forecasts.csv")
    (run / "selection.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "active-run.json").write_text(
        json.dumps(
            {
                "run": str(run.relative_to(ROOT)),
                "selection_sha256": digest(run / "selection.json"),
            }
        ),
        encoding="utf-8",
    )
    print(json.dumps(frozen, ensure_ascii=False), flush=True)


def bootstrap_delta(values, replications=2000, block=10, seed=20261013):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    estimates = []
    for _ in range(replications):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        indices = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        estimates.append(float(values[indices].mean()))
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def ablate():
    protocol, panel, regressions = load()
    active = json.loads((ROOT / "active-run.json").read_text(encoding="utf8"))
    run = ROOT / active["run"]
    if digest(run / "selection.json") != active["selection_sha256"]:
        raise ValueError("Selection changed")
    selection = json.loads((run / "selection.json").read_text(encoding="utf8"))
    cfg = FactorModelConfig(**selection["selected_config"])
    configs = {
        "selected": cfg,
        "without_cap_weighting": replace(cfg, regression_weighting="equal"),
        "without_time_decay": replace(cfg, factor_half_life=None, specific_half_life=None),
        "without_serial_adjustment": replace(cfg, newey_west_lags=0, specific_newey_west_lags=0),
        "without_regime_adjustment": replace(cfg, regime_window=0),
        "without_group_bayesian_specific": replace(
            cfg, specific_target="global", specific_shrinkage_method="fixed"
        ),
    }
    grid, _ = date_grid(panel)
    records = evaluate(protocol, panel, regressions, grid, configs)
    scores = daily_scores(records)
    save_records(records, run / "selection-ablations.csv")
    (run / "selection-ablations.json").write_text(
        json.dumps(
            {
                "purpose": "selection-period mechanism diagnostics only; never used to reselect or tune holdout",
                "mean_qlike": scores.mean().to_dict(),
                "date_count": len(scores),
            },
            indent=2,
        ),
        encoding="utf8",
    )


def holdout():
    protocol, panel, regressions = load()
    active = json.loads((ROOT / "active-run.json").read_text(encoding="utf8"))
    run = ROOT / active["run"]
    if digest(run / "selection.json") != active["selection_sha256"]:
        raise ValueError("Frozen selection changed")
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    for label, path in (
        ("protocol", ROOT / "protocol.json"),
        ("panel", ROOT / "clean/panel.pkl"),
        ("regressions", ROOT / "clean/regressions.pkl"),
    ):
        if digest(path) != selection[label + "_sha256"]:
            raise ValueError(f"{label} changed after selection")
    if digest(ROOT / "protocol-clarifications.json") != selection["clarifications_sha256"]:
        raise ValueError("Clarifications changed")
    if digest(ROOT / "clean/source-manifest.json") != selection["source_manifest_sha256"]:
        raise ValueError("Source evidence changed after selection")
    if numerical_code_hashes() != selection["numerical_code_sha256"]:
        raise ValueError("Numerical code changed after selection")
    if (run / "holdout.json").exists():
        raise ValueError("Holdout already scored; preserve the original run")
    _, grid = date_grid(panel)
    configs = {
        "baseline": FactorModelConfig(**selection["baseline_config"]),
        "enhanced": FactorModelConfig(**selection["selected_config"]),
    }
    configs["legacy_ols"] = replace(configs["baseline"], industry_constraint=False)
    records = evaluate(protocol, panel, regressions, grid, configs)
    scores = daily_scores(records)
    coverage_ok = len(scores) >= protocol["minimum_holdout_dates"]
    if len(scores) < protocol["bootstrap"]["moving_block_length"]:
        raise ValueError("Too few complete dates even for the frozen bootstrap")
    delta = scores.enhanced - scores.baseline
    interval = bootstrap_delta(delta.to_numpy())
    bias = {}
    for model in configs:
        sub = records[(records.model == model) & records["return"].notna()].copy()
        sub["squared_return"] = sub["return"] ** 2
        sums = sub.groupby("family")[["squared_return", "variance"]].sum()
        bias[model] = {
            family: float(np.sqrt(r.squared_return / r.variance)) for family, r in sums.iterrows()
        }
    # A 10% deterioration in the absolute log bias is material for this decision.
    bias_ok = all(
        abs(np.log(bias["enhanced"][family])) <= abs(np.log(bias["baseline"][family])) + 0.1
        for family in bias["baseline"]
    )
    stale = daily_scores(records, "stale_qlike")
    result = {
        "status": "completed_historical_holdout",
        "selection_sha256": active["selection_sha256"],
        "date_count": len(scores),
        "scheduled_date_count": len(grid),
        "minimum_holdout_dates": protocol["minimum_holdout_dates"],
        "coverage_gate_passed": coverage_ok,
        "mean_qlike": scores.mean().to_dict(),
        "enhanced_minus_baseline_qlike": float(delta.mean()),
        "paired_block_bootstrap_95_ci": interval,
        "volatility_bias": bias,
        "bias_guardrail_passed": bias_ok,
        "missing_endpoint_portfolios": records.groupby("model")["return"]
        .apply(lambda x: int(x.isna().sum()))
        .to_dict(),
        "stale_mark_sensitivity_delta": float((stale.enhanced - stale.baseline).mean()),
        "statistical_improvement": bool(interval[1] < 0),
        "performance_gate_passed": bool(interval[1] < 0 and bias_ok and coverage_ok),
        "production_parameters_changed": False,
        "scope": "200-name historical main-board cohort; public-document PIT reconstruction; neither live feed replay nor all-market validation",
    }
    legacy_delta = scores.enhanced - scores.legacy_ols
    result["enhanced_minus_original_ols_qlike"] = float(legacy_delta.mean())
    result["original_ols_comparison_95_ci"] = bootstrap_delta(legacy_delta.to_numpy())
    result["performance_gate_passed"] = bool(
        result["performance_gate_passed"] and result["original_ols_comparison_95_ci"][1] < 0
    )
    save_records(records, run / "holdout-forecasts.csv")
    scores.assign(delta=delta).to_csv(run / "paired-daily-scores.csv", encoding="utf-8")
    (run / "holdout.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"select": select, "ablate": ablate, "holdout": holdout}[sys.argv[1]]()
