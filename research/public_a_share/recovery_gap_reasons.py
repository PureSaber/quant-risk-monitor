"""Explain every remaining descriptor cell without manufacturing observations."""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from recovery_inventory import digest, write_json


def financial_reasons(known, day, field):
    reasons, needed = [], []
    fields = {"book_to_price": ("book_equity",), "book_leverage": ("book_equity", "liabilities")}
    if field in fields:
        for raw in fields[field]:
            values = known[raw]
            if not values:
                reasons.append("no_known_" + raw)
                continue
            period = max(values)
            if (day - pd.Timestamp(period)).days > 550:
                reasons.append("stale_" + raw)
            elif field == "book_leverage" and raw == "book_equity" and values[period] <= 0:
                reasons.append("nonpositive_equity_ratio_undefined")
        return reasons, needed
    values = known["profit_ytd"]
    if not values:
        return ["no_known_profit"], needed
    period = max(values)
    if (day - pd.Timestamp(period)).days > 550:
        return ["stale_latest_profit"], needed
    prior = str(int(period[:4]) - 1) + period[4:]
    annual = str(int(period[:4]) - 1) + "1231"
    required = (prior,) if field == "earnings_growth_yoy" else (() if period.endswith("1231") else (prior, annual))
    for target in required:
        if target not in values:
            reasons.append("missing_known_profit_component")
            needed.append(target)
    if field == "earnings_growth_yoy" and prior in values and abs(values[prior]) <= 1:
        reasons.append("near_zero_profit_denominator")
    return reasons, needed


def run(base, recovered, lifecycle, out):
    if (out / "remaining-reason-summary.csv").exists():
        raise ValueError("completed diagnosis is immutable")
    run_protocol = read_json(recovered.parent / "protocol.json")
    panel_path = base / "runs/main/panel.pkl"
    expected = next(r["sha256"] for r in run_protocol["inputs"] if Path(r["path"]).name == "panel.pkl")
    if digest(panel_path) != expected:
        raise ValueError("frozen panel changed")
    with panel_path.open("rb") as stream:
        bundle = pickle.load(stream)
    panel = bundle["v2"]
    descriptors = dict(panel["descriptors"])
    for field in descriptors:
        path = recovered / (field + ".parquet")
        if path.exists():
            descriptors[field] = pd.read_parquet(path)
    dates = panel["dates"][(panel["dates"] >= "2023-01-03") & (panel["dates"] <= "2026-09-30")]
    life = pd.read_parquet(lifecycle / "lifecycle-overlay.parquet")
    facts = pd.read_parquet(recovered / "financial-vintages.parquet")
    facts["available_at"] = pd.to_datetime(facts.available_at, utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    growth_quality = {f: pd.read_parquet(recovered / (f + "-quality.parquet")) for f in ("revenue_growth_3y", "profit_growth_3y")}
    fin = {"book_to_price", "earnings_yield_ttm", "earnings_growth_yoy", "book_leverage"}
    records, components = [], []
    for asset in panel["assets"]:
        events = list(facts[facts.asset.eq(asset)].sort_values(["available_at", "report_period", "field"]).itertuples())
        known, cursor = defaultdict(dict), 0
        for day in dates:
            while cursor < len(events) and events[cursor].available_at <= day:
                event = events[cursor]
                known[event.field][event.report_period] = event.value
                cursor += 1
            for field, frame in descriptors.items():
                if pd.notna(frame.at[day, asset]):
                    continue
                reasons, needed = [], []
                if field in growth_quality:
                    reasons = ["annual_" + str(growth_quality[field].at[day, asset])]
                elif field in fin:
                    reasons, needed = financial_reasons(known, day, field)
                    if field in {"book_to_price", "earnings_yield_ttm"} and not np.isfinite(panel["caps"].at[day, asset]):
                        reasons.append("missing_lagged_market_cap")
                elif field in {"size_log_mcap", "nonlinear_size"}:
                    reasons = ["missing_or_nonpositive_lagged_market_cap"]
                elif field in {"beta_252d", "residual_vol_252d"}:
                    count = bundle["paired_counts"].at[day, asset]
                    reasons = ["paired_return_observations_below_126" if count < 126 else "paired_regression_undefined"]
                elif field == "momentum_252_21":
                    reasons = ["missing_lagged_momentum_endpoint"]
                elif field == "cumulative_range_252d":
                    reasons = ["complete_252_return_sessions_required"]
                elif field.startswith("liquidity_"):
                    reasons = ["insufficient_observed_turnover_or_nonpositive_mean"]
                elif field == "daily_vol_252d":
                    reasons = ["observed_returns_below_227_in_252_sessions"]
                if not reasons:
                    raise ValueError((asset, day, field, "unexplained_missing_descriptor"))
                records.append({"asset": asset, "date": day, "field": field,
                    "reason": ";".join(sorted(set(reasons))), "trading_state": life.at[day, asset]})
                for period in needed:
                    components.append({"asset": asset, "date": day, "field": field, "needed_profit_period": period})
    frame = pd.DataFrame(records)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "remaining-cell-reasons.parquet", index=False)
    summary = frame.groupby(["field", "reason", "trading_state"], dropna=False).agg(
        company_days=("date", "size"), companies=("asset", "nunique"), first=("date", "min"), last=("date", "max")
    ).reset_index()
    summary.to_csv(out / "remaining-reason-summary.csv", index=False, encoding="utf-8-sig")
    component_frame = pd.DataFrame(components, columns=["asset", "date", "field", "needed_profit_period"])
    component_frame.groupby(["asset", "field", "needed_profit_period"]).agg(
        missing_days=("date", "size"), first=("date", "min"), last=("date", "max")
    ).reset_index().to_csv(out / "remaining-profit-components.csv", index=False, encoding="utf-8-sig")
    expected_cells = sum(int(f.reindex(dates).isna().sum().sum()) for f in descriptors.values())
    assert len(frame) == expected_cells
    write_json(out / "remaining-reason-receipt.json", {"explained_cells": len(frame),
        "fields": len(descriptors), "producer_sha256": digest(__file__),
        "financial_input_sha256": digest(recovered / "financial-vintages.parquet"),
        "scope": "Formula/input-state explanation; missing component is not proof its source report does not exist."})
    print(summary.to_string(index=False), flush=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "recovered", "lifecycle", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.recovered, args.lifecycle, args.out)
