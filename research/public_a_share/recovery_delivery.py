"""Inspectable coverage, vintage changes and unresolved-case receipts."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from recovery_inventory import digest, write_json


def run(base, recovered, out):
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((recovered / "summary.json").read_text(encoding="utf-8"))
    if digest(base / "runs/main/panel.pkl") != summary["baseline_sha256"]:
        raise ValueError("original_panel_changed")
    with (base / "runs/main/panel.pkl").open("rb") as stream:
        panel = pickle.load(stream)["v2"]
    old = pd.read_parquet(base / "runs/main/financial-vintages.parquet")
    new = pd.read_parquet(recovered / "financial-vintages.parquet")
    keys = ["asset", "field", "report_period", "available_at"]
    for frame in (old, new):
        frame["available_at"] = pd.to_datetime(frame.available_at, utc=True)
    delta = old[keys + ["value"]].merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
    delta = delta[(delta._merge != "both") | ~np.isclose(delta.value_old, delta.value_new, equal_nan=True)]
    delta.to_csv(out / "fact-vintage-changes.csv", index=False, encoding="utf-8-sig")
    inventory = pd.read_csv(recovered.parent / "recovery-inventory.csv", dtype=str)
    annual = new[new.report_period.str.endswith("1231") & (new.available_at <= "2026-09-30T15:59:59Z")]
    matrix = []
    for asset in panel["assets"]:
        for year in range(2018, 2026):
            period = str(year) + "1231"
            reports = inventory[(inventory.asset == asset) & (inventory.report_period == period)]
            for field in ("revenue_ytd", "profit_ytd", "assets", "liabilities", "book_equity"):
                facts = annual[(annual.asset == asset) & (annual.report_period == period) & (annual.field == field)].sort_values("available_at")
                matrix.append({"asset": asset, "year": year, "field": field, "known_report_versions": len(reports),
                               "admitted_fact_versions": len(facts), "first_available": str(facts.available_at.min()) if len(facts) else None,
                               "last_available": str(facts.available_at.max()) if len(facts) else None,
                               "status": "available_fact" if len(facts) else "report_known_field_not_admitted" if len(reports) else "no_report_in_search_scope",
                               "latest_value": float(facts.value.iloc[-1]) if len(facts) else None})
    pd.DataFrame(matrix).to_csv(out / "company-year-field-matrix.csv", index=False, encoding="utf-8-sig")
    before_masks, after_masks, company, gaps, losses = [], [], [], [], []
    for field, old_values in panel["descriptors"].items():
        before = old_values.loc["2023-01-03":"2026-09-30"]
        path = recovered / (field + ".parquet")
        after = pd.read_parquet(path).reindex(before.index) if path.exists() else before
        before_masks.append(before.notna())
        after_masks.append(after.notna())
        quality = recovered / (field + "-quality.parquet")
        q = pd.read_parquet(quality) if quality.exists() else None
        for asset in before.columns:
            b, a = before[asset], after[asset]
            company.append({"asset": asset, "field": field, "dates": len(a), "before_observed": int(b.notna().sum()), "after_observed": int(a.notna().sum())})
            missing = a.index[a.isna()]
            if len(missing):
                reasons = q.loc[missing, asset].value_counts().to_dict() if q is not None else {}
                gaps.append({"asset": asset, "field": field, "missing_days": len(missing),
                             "first_missing": str(missing[0].date()), "last_missing": str(missing[-1].date()),
                             "descriptor_statuses": json.dumps(reasons),
                             "interpretation": "Financial annual history remains unavailable/expired; see field decisions" if q is not None else "Unchanged original input or rolling-history gap; no synthetic price, return or financial fact"})
            for day in b.index[b.notna() & a.isna()]:
                losses.append({"asset": asset, "field": field, "date": str(day.date()), "before": float(b.loc[day]),
                               "after": None, "reason": "recomputed_formula_no_longer_defined; inspect restored denominator"})
    pd.DataFrame(company).to_csv(out / "company-descriptor-coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gaps).to_csv(out / "remaining-descriptor-gaps.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(losses).to_csv(out / "lost-descriptor-cells.csv", index=False, encoding="utf-8-sig")
    all_before = np.logical_and.reduce([f.to_numpy() for f in before_masks])
    all_after = np.logical_and.reduce([f.to_numpy() for f in after_masks])
    receipt = {"all_fields_company_days_before": int(all_before.sum()), "all_fields_company_days_after": int(all_after.sum()),
               "company_days": int(all_before.size), "companies_with_any_gap_before": int((~all_before).any(axis=0).sum()),
               "companies_with_any_gap_after": int((~all_after).any(axis=0).sum()), "fact_changes": len(delta),
               "new_vintages": int((delta._merge == "right_only").sum()),
               "corrected_vintages": int((delta._merge == "both").sum()), "matrix_rows": len(matrix),
               "input_sha256": digest(recovered / "summary.json"), "original_retained": True,
               "default_model_changed": False}
    write_json(out / "coverage-receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("base", "recovered", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.recovered, args.out)
