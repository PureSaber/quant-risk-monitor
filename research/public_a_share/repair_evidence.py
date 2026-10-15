"""Classify frozen risk-union failures and historical corporate-action stops."""

import argparse
import json
import pickle
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pandas as pd

from repair_failure_audit import date_rows, decimal_column, frozen_accounts
from study_v3_archive import digest, write_json


def run(source, coverage, config_dir, out):
    from quant_data_kit import FixedPoint
    from quant_data_kit.share_entitlements import EntitlementUnresolved, ShareEntitlement

    from study_v3_portfolios import scenarios

    out.mkdir(parents=True, exist_ok=True)
    with (source / "portfolio-risk-models.pkl").open("rb") as stream:
        original = pickle.load(stream)
    with (coverage / "portfolio-risk-models.pkl").open("rb") as stream:
        expanded = pickle.load(stream)
    rules = json.loads((config_dir / "share-rules-strict.json").read_text())["rules"]
    gap_rows, asset_rows, actions = [], [], []
    for scenario, folder in frozen_accounts(source):
        positions = pd.read_parquet(folder / "positions.parquet")
        decisions = json.loads((folder / "allocation-decisions.json").read_text())
        _, model, _, _ = list(scenarios())[scenario]
        for row in decisions:
            if row.get("status") != "input_unavailable":
                continue
            day = pd.Timestamp(row["session"])
            held = date_rows(positions, day)
            held = set(held.loc[held.quantity_units.ne(0), "instrument_id"])
            union = held | set(original[day]["assets"]) | set(original[day]["benchmark"].index)
            old_missing = sorted(union - set(original[day]["models"][model]["exposures"].index))
            new = expanded[day]["models"][model]
            new_missing = sorted(union - set(new["exposures"].index))
            assert old_missing
            gap_rows.append(
                {
                    "scenario": scenario,
                    "date": row["session"],
                    "model": model,
                    "old_missing": ",".join(old_missing),
                    "remaining_missing": ",".join(new_missing),
                    "risk_union_covered": not new_missing,
                    "account_source": str(folder),
                    "not_a_new_allocation": True,
                }
            )
            for asset in old_missing:
                asset_rows.append(
                    {
                        "scenario": scenario,
                        "date": row["session"],
                        "model": model,
                        "asset": asset,
                        "now_covered": asset not in new_missing,
                        **new["coverage"].get(asset, {"status": "missing_coverage_record"}),
                    }
                )
        stop_path = folder / "full-horizon-stop.json"
        if not stop_path.exists():
            continue
        stop = json.loads(stop_path.read_text())
        asset = "600763" if str(stop["at"]).startswith("2024-07-05") else "603007"
        rule = next(item for item in rules.values() if item["entitlement_id"].startswith(asset))
        record = pd.Timestamp(rule["record_date"])
        held = date_rows(positions, record)
        held = held[held.instrument_id.eq(asset)].iloc[-1]
        quantity = decimal_column(held, "quantity")
        ratio = Decimal(rule["ordinary_additional_ratio"])
        terms = ShareEntitlement(
            entitlement_id=rule["entitlement_id"],
            account_id=str(held.account_id),
            record_date=record.date(),
            eligible_quantity=FixedPoint(int(quantity), 0),
            ordinary_additional_ratio=FixedPoint(int(ratio * 10**6), 6),
            known_at=pd.Timestamp(rule["known_at"]).tz_convert("UTC").to_pydatetime(),
            credit_at=pd.Timestamp(rule["credit_at"]).tz_convert("UTC").to_pydatetime(),
            sellable_date=pd.Timestamp(rule["sellable_date"]).date(),
            evidence_source=rule["evidence_source"],
            evidence_sha256=rule["evidence_sha256"],
            fractional_rule=rule["fractional_rule"],
        )
        try:
            exact = str(terms.resolved_quantity())
            strict_status = "exact_pro_rata_integer"
        except EntitlementUnresolved:
            exact, strict_status = None, "entitlement_unresolved"
        from dataclasses import replace

        floor = replace(terms, evidence_status="sensitivity_floor").resolved_quantity()
        ceil = (
            replace(terms, evidence_status="sensitivity_ceil").resolved_quantity()
            if terms.fractional_rule == "largest_fraction"
            else floor
        )
        actions.append(
            {
                "scenario": scenario,
                "asset": asset,
                "original_stop": stop["at"],
                "record_date": rule["record_date"],
                "eligible_quantity": int(quantity),
                "unrounded_additional_shares": str(quantity * ratio),
                "strict_status": strict_status,
                "strict_confirmed_quantity": exact,
                "sensitivity_floor": str(floor),
                "sensitivity_upper": str(ceil),
                "fractional_rule": terms.fractional_rule,
                "unresolved_fact": (
                    "global fractional-share registry allocation for this virtual account"
                    if asset == "600763"
                    else "approximate disclosed ratio, final registered quantity and delivery evidence"
                ),
                "later_confirmation_is_not_a_decision_feature": True,
                "source": terms.evidence_source,
                "source_sha256": terms.evidence_sha256,
                "account_source": str(folder),
                "account_sha256": digest(folder / "positions.parquet"),
            }
        )
    assert len(gap_rows) == 2893 and len(actions) == 19
    pd.DataFrame(gap_rows).to_csv(out / "frozen-risk-union-cases.csv", index=False)
    pd.DataFrame(asset_rows).to_json(
        out / "frozen-risk-union-assets.jsonl", orient="records", lines=True
    )
    pd.DataFrame(actions).to_csv(out / "corporate-action-cases.csv", index=False)
    write_json(
        out / "data-failure-evidence.json",
        {
            "frozen_risk_union_failures": len(gap_rows),
            "fully_covered_by_new_model": sum(row["risk_union_covered"] for row in gap_rows),
            "still_uncovered": sum(not row["risk_union_covered"] for row in gap_rows),
            "asset_coverage_status": dict(Counter(row["status"] for row in asset_rows)),
            "corporate_action_stops": len(actions),
            "corporate_action_assets": dict(Counter(row["asset"] for row in actions)),
            "strict_entitlement_status": dict(Counter(row["strict_status"] for row in actions)),
            "expanded_models_sha256": digest(coverage / "portfolio-risk-models.pkl"),
            "producer_sha256": digest(__file__),
            "candidate_or_benchmark_changed": False,
            "coverage_is_not_predictive_quality_acceptance": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "coverage", "config-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.coverage, args.config_dir, args.out)
