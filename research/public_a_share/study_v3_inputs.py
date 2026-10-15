"""Read-only reconstruction and quality evidence for the fixed 200-name study."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import build_panel
import confirm_exposure_v2 as confirmation
import exposure_v2_study as prior
from model_bootstrap import SOURCE
from study_v3_archive import digest, write_json


def freeze(old, exposure, out):
    """Freeze finite experiments before producing new results, never backdate."""
    prior_protocol = json.loads((exposure / "protocol.json").read_text(encoding="utf8"))
    inputs = {
        "cohort": old / "cohort.json",
        "old_panel": old / "clean/panel.pkl",
        "old_source_manifest": old / "clean/source-manifest.json",
        "v2_protocol": exposure / "protocol.json",
        "v2_selection": exposure / "selection.json",
        "confirmation_manifest": exposure / "confirmation/source-manifest.json",
        "financial_facts": exposure / "confirmation/merged-inputs/clean/financial_facts.csv",
        "industry_history": exposure / "confirmation/merged-inputs/clean/industry_history.csv",
    }
    selected = json.loads((exposure / "selection.json").read_text(encoding="utf8"))["selected"]
    if selected != "residual_volatility":
        raise ValueError("unexpected prior frozen candidate; review the study protocol")
    modules = [
        SOURCE / n
        for n in (
            "factor_model.py",
            "exposure_builder.py",
            "style_descriptors.py",
            "risk_estimators.py",
        )
    ]
    modules += [Path(__file__), Path(build_panel.__file__), Path(prior.__file__)]
    binding = {k: digest(v) for k, v in inputs.items()}
    specification = {
        "schema": "public-barra-200-study/v3",
        "cohort_size": 200,
        "cutoff": "2026-09-30",
        "design": "retrospective; 2024-2026 already inspected; no independent promotion evidence",
        "selection_year": 2023,
        "diagnostic_start": "2024-01-01",
        "horizon": 5,
        "lookback": 252,
        "min_assets": 100,
        "variants": ["legacy", "repaired", selected],
        "inputs": binding,
        "old_protocol": prior_protocol["old_protocol"],
        "bootstrap": {
            "replications": 5000,
            "block_dates": 10,
            "seed": 20261015,
            "correction": "Bonferroni simultaneous 95% intervals per experiment family",
        },
        "promotion": {
            "independent_complete_windows": 80,
            "maximum_log_bias_deterioration": 0.1,
            "maximum_coverage_drop": 0.05,
            "raw_coverage_floor": 0.8,
            "maximum_exposure_failure_fraction": 0.05,
            "automatic": False,
        },
        "portfolio": {
            "capital": 1000000,
            "budgets": [0.95, 0.8],
            "max_weight": 0.05,
            "turnover_l1": 0.2,
            "initial_turnover_l1": 2.0,
            "tracking_error_limits": [0.03, 0.06],
            "timeout_seconds": 30,
            "commission": 0.0003,
            "min_commission": 5.0,
            "slippage": 0.0005,
            "statutory_stamp_tax": True,
            "benchmark": "research-sample-cap-weighted",
            "signal": "lagged 252-21 momentum",
            "ic_lookback": 120,
            "ic_min_periods": 20,
        },
    }
    target = out / "protocol.json"
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf8"))
        if any(previous.get(k) != v for k, v in specification.items()):
            raise ValueError("frozen research inputs/specification changed; use a new run")
        return previous
    specification["created_at"] = datetime.now(timezone.utc).isoformat()
    specification["baseline_code"] = {p.name: digest(p) for p in modules}
    specification["repository_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE.parents[1], text=True
    ).strip()
    write_json(target, specification)
    return specification


def prepare(old, exposure, out, protocol):
    with (old / "clean/panel.pkl").open("rb") as stream:
        original = pickle.load(stream)
    old_market = confirmation.read_market(old / "raw/benchmark/csi300.json")
    new_market = confirmation.read_market(
        exposure / "confirmation-inputs/raw/benchmark/csi300.json"
    )
    market = pd.concat([old_market, new_market.loc["2026"]])
    if market.index.has_duplicates or str(market.index.max().date()) != protocol["cutoff"]:
        raise ValueError("market calendar does not match the frozen cutoff")
    prices, bridges = confirmation.merge_prices(original, exposure / "confirmation-inputs", market)
    merged = exposure / "confirmation/merged-inputs"
    previous_root = build_panel.ROOT
    build_panel.ROOT = merged
    try:
        dates, assets = market.index, original["assets"]
        financial = build_panel.financial_daily(dates, assets)
        shares = build_panel.capital_daily(dates, assets, financial["shares"])
        groups = build_panel.industry_daily(
            dates, assets, protocol["old_protocol"]["sector_mapping"]
        )
        market_returns = market.pct_change(fill_method=None)
        caps, descriptors = build_panel.descriptors_from_history(
            prices, financial, shares, market_returns
        )
    finally:
        build_panel.ROOT = previous_root
    past = original["dates"][original["dates"] >= "2021-11-11"]
    for k, frame in original["prices"].items():
        expected = frame.loc[past].copy()
        expected.index = expected.index.as_unit("ns")
        expected.columns = expected.columns.astype(object)
        pd.testing.assert_frame_equal(
            prices[k].loc[past],
            expected,
            check_dtype=False,
            check_freq=False,
            check_names=False,
            rtol=1e-12,
            atol=1e-12,
        )
    expected_caps = original["caps"].loc[past].copy()
    expected_caps.index = expected_caps.index.as_unit("ns")
    expected_caps.columns = expected_caps.columns.astype(object)
    pd.testing.assert_frame_equal(
        caps.loc[past],
        expected_caps,
        check_dtype=False,
        check_freq=False,
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )
    from quant_risk_monitor.style_descriptors import (
        additional_market_descriptors,
        annual_growth_descriptors,
        paired_market_descriptors,
    )

    facts = pd.read_csv(
        merged / "clean/financial_facts.csv", dtype={"asset": str, "report_period": str}
    )
    paired, counts = paired_market_descriptors(prices["returns"], market_returns)
    additional = additional_market_descriptors(prices["returns"], prices["turnover"])
    growth, growth_quality = annual_growth_descriptors(facts, dates, assets)
    panel = {
        "dates": dates,
        "assets": assets,
        "prices": prices,
        "caps": caps,
        "groups": groups,
        "descriptors": descriptors,
    }
    enhanced = {**panel, "descriptors": {**descriptors, **paired, **additional, **growth}}
    bundle = {
        "panel": panel,
        "v2": enhanced,
        "market_returns": market_returns,
        "growth_quality": growth_quality,
        "paired_counts": counts,
    }
    with (out / "panel.pkl").open("wb") as stream:
        pickle.dump(bundle, stream, protocol=5)
    write_json(
        out / "panel-binding.json",
        {
            "protocol": digest(out / "protocol.json"),
            "panel": digest(out / "panel.pkl"),
            "input_files": protocol["inputs"],
        },
    )
    write_json(out / "price-bridges.json", bridges)
    return bundle


def status_history(old, exposure, dates, assets):
    result = pd.DataFrame(np.nan, index=dates, columns=assets)
    observed = []
    for asset in assets:
        frames = []
        for root in (old, exposure / "confirmation-inputs"):
            path = root / "raw/bao-prices" / (asset + ".csv")
            frame = pd.read_csv(path, parse_dates=["date"])
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            if root != old:
                frame = frame[frame.date.dt.year == 2026]
            frames.append(frame)
        frame = pd.concat(frames).set_index("date")
        frame = frame[~frame.index.duplicated(keep="last")]
        result[asset] = frame.tradestatus.reindex(dates)
        if len(frame):
            observed.append(
                {
                    "asset": asset,
                    "first_source_row": str(frame.index.min().date()),
                    "last_source_row": str(frame.index.max().date()),
                }
            )
    return result, observed


def quality(bundle, old, exposure, out):
    panel, enhanced = bundle["panel"], bundle["v2"]
    dates, assets = panel["dates"], panel["assets"]
    status, observed = status_history(old, exposure, dates, assets)
    records = []
    coverage = []
    for name, frame in enhanced["descriptors"].items():
        for day, row in frame.loc["2021-11-11":].iterrows():
            coverage.append(
                {
                    "date": str(day.date()),
                    "descriptor": name,
                    "observed": int(np.isfinite(row).sum()),
                    "cohort": len(assets),
                }
            )
    base_names = [
        "size_log_mcap",
        "beta_252d",
        "momentum_252_21",
        "residual_vol_252d",
        "liquidity_log_turnover_20d",
    ]
    for asset in assets:
        returns = panel["prices"]["returns"][asset]
        count = returns.notna().rolling(252, min_periods=1).sum().shift(1).fillna(0)
        for day in dates[dates >= "2021-11-11"]:
            missing = [
                n for n in base_names if not np.isfinite(enhanced["descriptors"][n].at[day, asset])
            ]
            record = {
                "date": day,
                "asset": asset,
                "quote_observed": bool(pd.notna(panel["prices"]["close"].at[day, asset])),
                "observed_suspension": bool(status.at[day, asset] == 0),
                "trading_status_unknown": bool(pd.isna(status.at[day, asset])),
                "prior_return_observations": int(count.loc[day]),
                "short_return_history": bool(count.loc[day] < 126),
                "price_descriptor_missing": ",".join(missing),
                "industry": panel["groups"].at[day, asset],
                "capitalization_available": bool(np.isfinite(panel["caps"].at[day, asset])),
            }
            records.append(record)
    daily = pd.DataFrame(records).sort_values(["asset", "date"])
    daily["industry_changed"] = daily.groupby("asset").industry.transform(
        lambda s: s.ne(s.shift()) & s.shift().notna()
    )
    daily.to_parquet(out / "daily-quality.parquet", index=False)
    pd.DataFrame(coverage).to_parquet(out / "descriptor-coverage.parquet", index=False)
    status.to_parquet(out / "source-trading-status.parquet")
    write_json(out / "source-observation-spans.json", observed)
    write_json(
        out / "quality-summary.json",
        {
            "asset_days": len(daily),
            "cohort_assets": len(assets),
            "suspension_asset_days": int(daily.observed_suspension.sum()),
            "unknown_trading_status_asset_days": int(daily.trading_status_unknown.sum()),
            "short_history_asset_days": int(daily.short_return_history.sum()),
            "industry_changes": int(daily.industry_changed.sum()),
            "delisting_inference": "absence of a quote is not evidence of delisting; verify original events",
            "data_use": "quality observations do not become decision-time trading knowledge before availability",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("old", "exposure", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = freeze(args.old, args.exposure, args.out)
    bundle = prepare(args.old, args.exposure, args.out, protocol)
    quality(bundle, args.old, args.exposure, args.out)
    print(
        json.dumps({"stage": "inputs_complete", "assets": len(bundle["panel"]["assets"])}),
        flush=True,
    )
    prior.VARIANTS = {name: prior.VARIANTS[name] for name in protocol["variants"]}
    history_protocol = {**protocol, "maximum_exposure_failure_fraction": 0.05}
    prior.build_histories(bundle["panel"], bundle["v2"], args.out, history_protocol)
    write_json(
        args.out / "history-binding-v3.json",
        {
            "histories": digest(args.out / "histories.pkl"),
            "panel": digest(args.out / "panel.pkl"),
            "protocol": digest(args.out / "protocol.json"),
        },
    )
    print(json.dumps({"stage": "histories_complete"}), flush=True)


if __name__ == "__main__":
    main()
