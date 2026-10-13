"""Real-data causal and numerical acceptance checks before any holdout scoring."""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import build_panel
from model_bootstrap import (
    AssetReturnObservation,
    ExposureSnapshot,
    fit_barra_style_risk_model,
)
from public_sources import ROOT
from run_study import (
    configurations,
    date_grid,
    digest,
    forecast_inputs,
    legacy_components,
    load,
    risk_components,
)


def original_module():
    path = ROOT / "raw/code/original_factor_model.py"
    assert digest(path) == "f8c141f80d0afdd8eb9ea18f5466ed7ca7c168caf025bcf22ded32f58da9dbd7"
    spec = importlib.util.spec_from_file_location("original_ols_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_check(panel, regressions, row, config, legacy=False):
    snap, history, weights = forecast_inputs(panel, regressions, row)
    module = original_module() if legacy else None
    Snap = module.ExposureSnapshot if legacy else ExposureSnapshot
    Obs = module.AssetReturnObservation if legacy else AssetReturnObservation
    Fit = module.fit_barra_style_risk_model if legacy else fit_barra_style_risk_model
    snapshots = []
    observations = []
    for day in history:
        position = panel["dates"].get_loc(day)
        prior_day = panel["dates"][position - 1]
        prior = regressions["snapshots"][prior_day]
        returns = panel["prices"]["returns"].loc[day].dropna()
        assets = prior.values.index.intersection(returns.index)
        values = prior.values.loc[assets]
        args = {
            "effective_at": prior.effective_at,
            "available_at": prior.available_at,
            "values": values.drop(columns="market") if legacy else values,
            "source": prior.source,
        }
        if not legacy:
            args.update(
                market_caps=prior.market_caps.loc[assets],
                industries=prior.industries.loc[assets],
                industry_factors=prior.industry_factors,
            )
        snapshots.append(Snap(**args))
        end = day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
        observations.append(
            Obs(
                period_start=prior.effective_at,
                period_end=end,
                available_at=end.normalize() + pd.Timedelta(days=1),
                values=returns.loc[assets],
                source="archived public multiplicative-adjusted prices",
            )
        )
    args = {
        "effective_at": snap.effective_at,
        "available_at": snap.available_at,
        "values": snap.values.drop(columns="market") if legacy else snap.values,
        "source": snap.source,
    }
    if not legacy:
        args.update(
            market_caps=snap.market_caps,
            industries=snap.industries,
            industry_factors=snap.industry_factors,
        )
    snapshots.append(Snap(**args))
    if legacy:
        cfg = module.FactorModelConfig(
            model_kind="fundamental_style",
            min_periods=252,
            min_assets_per_period=100,
            min_asset_observations=252,
            covariance_shrinkage=0.2,
            specific_variance_shrinkage=0.2,
        )
        data = regressions["histories"]["legacy_ols"]
        F, D, _ = legacy_components(
            data["factors"].loc[history],
            data["residuals"].loc[history],
            snap.values.index,
        )
    else:
        cfg = config
        data = regressions["histories"][config.regression_weighting]
        F, D, _ = risk_components(
            snap, data["factors"].loc[history], data["residuals"].loc[history], config
        )
    fitted = Fit(
        exposure_snapshots=snapshots,
        return_observations=observations,
        as_of=snap.available_at,
        config=cfg,
    )
    design = snap.values.drop(columns="market") if legacy else snap.values
    cached = design.to_numpy() @ F @ design.to_numpy().T + np.diag(D)
    direct = fitted.asset_covariance.loc[design.index, design.index].to_numpy() / 252
    np.testing.assert_allclose(cached, direct, rtol=2e-9, atol=1e-12)
    error = float(np.abs(cached - direct).max())
    return {
        "model": "original_ols" if legacy else config.regression_weighting,
        "date": str(row[1].date()),
        "assets": len(snap.values),
        "max_daily_covariance_absolute_error": error,
        "passed": True,
    }


def financial_future_poison(panel):
    facts = pd.read_csv(
        ROOT / "clean/financial_facts.csv", dtype={"asset": str, "report_period": str}
    )
    cutoff = pd.Timestamp("2023-12-29", tz="Asia/Shanghai")
    dates = panel["dates"][(panel["dates"] >= "2023-12-01") & (panel["dates"] <= "2023-12-29")]
    base = build_panel.financial_daily(dates, panel["assets"])
    future = pd.to_datetime(facts.available_at, utc=True) > cutoff
    facts.loc[future, "value"] = 1e30
    original_root = build_panel.ROOT
    with tempfile.TemporaryDirectory(prefix="pit-poison-", dir=ROOT) as temp:
        root = Path(temp)
        (root / "clean").mkdir()
        facts.to_csv(root / "clean/financial_facts.csv", index=False)
        try:
            build_panel.ROOT = root
            changed = build_panel.financial_daily(dates, panel["assets"])
        finally:
            build_panel.ROOT = original_root
    for field in base:
        pd.testing.assert_frame_equal(base[field], changed[field])
    return {
        "future_original_facts_poisoned": int(future.sum()),
        "assets_checked": len(panel["assets"]),
        "past_dates_checked": len(dates),
        "past_financial_features_unchanged": True,
    }


def quality_summary(panel, regressions):
    facts = pd.read_csv(
        ROOT / "clean/financial_facts.csv", dtype={"asset": str, "report_period": str}
    )
    announcements = (
        pd.to_datetime(facts.available_at, utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    assert (announcements >= pd.to_datetime(facts.report_period)).all()
    assert np.isfinite(facts.value).all()
    assert not facts.duplicated(["asset", "report_period", "field", "available_at"]).any()
    counts = panel["prices"]["returns"].notna().sum()
    future_exit = panel["prices"]["close"].apply(lambda x: x.last_valid_index())
    details = []
    for day, snapshot in regressions["snapshots"].items():
        processing = snapshot.provenance.get("descriptor_processing", {})
        for field, row in processing.items():
            details.append(
                {
                    "date": day,
                    "field": field,
                    **{k: v for k, v in row.items() if isinstance(v, (int, float, str, bool))},
                }
            )
    pd.DataFrame(details).to_csv(ROOT / "clean/exposure-data-quality.csv", index=False)
    return {
        "cohort_assets": len(panel["assets"]),
        "original_facts": len(facts),
        "original_fact_assets": facts.asset.nunique(),
        "dates": len(panel["dates"]),
        "prices_begin": str(panel["dates"][0].date()),
        "prices_end": str(panel["dates"][-1].date()),
        "observed_return_days_min": int(counts.min()),
        "observed_return_days_median": float(counts.median()),
        "historical_names_with_last_quote_before_2025_end": {
            a: str(d.date()) for a, d in future_exit.items() if d < panel["dates"][-1]
        },
        "regression_dates": {k: len(v["factors"]) for k, v in regressions["histories"].items()},
        "facts_available_after_report_period": True,
    }


def main():
    protocol, panel, regressions = load()
    baseline, candidates = configurations(protocol)
    selection, _ = date_grid(panel)
    checks = []
    for name, config, legacy in [
        ("baseline", baseline, False),
        ("enhanced", next(iter(candidates.values())), False),
        ("original_ols", baseline, True),
    ]:
        result = canonical_check(panel, regressions, selection[0], config, legacy)
        result["check"] = name
        checks.append(result)
        print(result, flush=True)
    result = {
        "status": "passed",
        "canonical_model_agreement": checks,
        "future_filing_poison": financial_future_poison(panel),
        "data_quality": quality_summary(panel, regressions),
        "no_holdout_performance_read": True,
    }
    (ROOT / "clean/preselection-validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
