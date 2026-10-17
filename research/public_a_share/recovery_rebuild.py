"""Rebuild only changed descriptor matrices; preserve immutable baseline panels."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from recovery_admission import FINANCIAL_FIELDS, merge_evidence, opinion_evidence
from recovery_inventory import digest, write_json

import model_bootstrap  # noqa: F401
from quant_risk_monitor.style_descriptors import annual_growth_descriptors


def recovered_rows(cache):
    import pymupdf

    rows, documents = [], []
    for path in sorted(cache.glob("*.json")):
        if path.name == "statuses.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if digest(record["pdf"]) != record["sha256"]:
            raise ValueError("recovered_document_digest_changed")
        with pymupdf.open(record["pdf"]) as document:
            text = "\n".join(p.get_text() for p in document)
        opinion = opinion_evidence(text, int(record["original_report_period"][:4]))
        mirror = record["role"] == "post_delisting_original_mirror"
        reason = (
            "mirror_requires_exchange_crosscheck"
            if mirror
            else "audit_opinion_requires_review"
            if opinion["status"] not in {"unqualified", "unqualified_with_emphasis"}
            else "eligible"
        )
        documents.append(
            {
                "asset": record["asset"],
                "id": record["announcement_id"],
                "title": record["title"],
                "opinion": opinion["status"],
                "excerpt": opinion["excerpt"],
                "admission_reason": reason,
                "pdf": record["pdf"],
                "sha256": record["sha256"],
                "url": record["url"],
            }
        )
        for fact in record["accepted"]:
            if fact["field"] not in FINANCIAL_FIELDS:
                continue
            rows.append(
                {
                    **fact,
                    "asset": record["asset"],
                    "available_at": record["available_at"],
                    "announcement_id": record["announcement_id"],
                    "title": record["title"],
                    "url": record["url"],
                    "source_pdf": record["pdf"],
                    "source_sha256": record["sha256"],
                    "parser_sha256": record["parser_sha256"],
                    "audit_status": opinion["status"],
                    "admission_reason": reason,
                }
            )
    return pd.DataFrame(rows), documents


def coverage(old, new, label, dates):
    old, new = old.reindex(dates), new.reindex(dates)
    before, after = old.notna(), new.notna()
    changed = before & after & ~np.isclose(old, new, rtol=1e-10, atol=1e-12, equal_nan=True)
    return {
        "field": label,
        "denominator": old.size,
        "companies": len(old.columns),
        "dates": len(dates),
        "before_observed": int(before.sum().sum()),
        "after_observed": int(after.sum().sum()),
        "before_coverage": float(before.to_numpy().mean()),
        "after_coverage": float(after.to_numpy().mean()),
        "restored_cells": int((~before & after).sum().sum()),
        "lost_cells": int((before & ~after).sum().sum()),
        "changed_nonmissing": int(changed.sum().sum()),
        "before_companies_with_gap": int((~before).any().sum()),
        "after_companies_with_gap": int((~after).any().sum()),
    }


def rebuild(base, run, cache, audit, target=None, corrections=None):
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    for item in protocol["inputs"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("frozen_input_changed")
    # Only deserialize the trusted local panel after verifying its frozen digest.
    with (base / "runs/main/panel.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    panel = payload["v2"]
    dates, assets = panel["dates"], panel["assets"]
    evaluation = pd.DatetimeIndex(dates)
    evaluation = evaluation[(evaluation >= protocol["start"]) & (evaluation <= protocol["end"])]
    facts = pd.read_parquet(base / "runs/main/financial-vintages.parquet")
    additions, documents = recovered_rows(cache)
    reviewed = json.loads(corrections.read_text(encoding="utf-8")) if corrections else []
    merged, decisions = merge_evidence(facts, additions, reviewed_corrections=reviewed)
    target = target or run / "rebuilt"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "summary.json").exists():
        raise ValueError("use_new_output_version_for_completed_rebuild")
    write_json(target / "document-admission.json", documents)
    decisions.to_json(
        target / "field-decisions.json",
        orient="records",
        force_ascii=False,
        indent=2,
        date_format="iso",
    )
    flat = decisions.drop(columns=[c for c in ["bbox", "column_headers"] if c in decisions])
    flat.to_csv(target / "field-decisions.csv", index=False, encoding="utf-8-sig")
    merged[["asset", "field", "report_period", "value", "available_at"]].to_parquet(
        target / "financial-vintages.parquet", index=False
    )
    growth, quality = annual_growth_descriptors(merged, dates, assets)
    # Preserve all nonfinancial inputs, caps, share histories, prices and calendar.
    new_descriptors = dict(panel["descriptors"])
    new_descriptors.update(growth)
    import build_panel

    old_root = build_panel.ROOT
    clean = target / "clean"
    clean.mkdir(exist_ok=True)
    merged[["asset", "field", "report_period", "value", "available_at"]].to_csv(
        clean / "financial_facts.csv", index=False
    )
    try:
        build_panel.ROOT = target
        financial = build_panel.financial_daily(dates, assets)
    finally:
        build_panel.ROOT = old_root
    new_descriptors.update(
        {
            "book_to_price": financial["book_equity"] / panel["caps"],
            "earnings_yield_ttm": financial["profit_ttm"] / panel["caps"],
            "earnings_growth_yoy": financial["growth"],
            "book_leverage": financial["liabilities"]
            / financial["book_equity"].where(financial["book_equity"] > 0),
        }
    )
    metrics, files = [], []
    for name, old in panel["descriptors"].items():
        new = new_descriptors[name]
        metrics.append(coverage(old, new, name, evaluation))
        if name in growth or name in {
            "book_to_price",
            "earnings_yield_ttm",
            "earnings_growth_yoy",
            "book_leverage",
        }:
            path = target / (name + ".parquet")
            new.to_parquet(path)
            files.append({"field": name, "path": str(path), "sha256": digest(path)})
    for name, frame in quality.items():
        frame.to_parquet(target / (name + "-quality.parquet"))
    pd.DataFrame(metrics).to_csv(target / "coverage.csv", index=False, encoding="utf-8-sig")
    cases = pd.read_csv(audit / "annual-gap-classification.csv", dtype={"asset": str})
    closed = []
    for row in cases.to_dict("records"):
        subset = decisions[
            (decisions.asset == row["asset"])
            & (decisions.field == row["field"])
            & (decisions.report_period == str(row["year"]) + "1231")
        ]
        admitted = subset[subset.decision == "new_vintage_admitted"]
        first = (
            pd.Timestamp(row["first_affected"])
            if "first_affected" in row
            else pd.Timestamp(row["first_date"])
        )
        known = admitted[
            pd.to_datetime(admitted.available_at, utc=True) <= first.tz_localize("Asia/Shanghai")
        ]
        status = (
            "restored_by_first_affected_date"
            if len(known)
            else "restored_later_only"
            if len(admitted)
            else "not_admitted"
        )
        closed.append(
            {
                **row,
                "recovery_status": status,
                "admitted_sources": "|".join(admitted.announcement_id.unique()),
                "remaining_reason": ""
                if len(admitted)
                else "|".join(sorted(set(subset.decision))) or "no_verified_matching_field",
            }
        )
    pd.DataFrame(closed).to_csv(
        target / "annual-case-results.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "schema": "recovered-input-overlay/1",
        "default_model_changed": False,
        "baseline_panel": str(base / "runs/main/panel.pkl"),
        "baseline_sha256": digest(base / "runs/main/panel.pkl"),
        "descriptor_files": files,
        "original_facts": len(facts),
        "merged_facts": len(merged),
        "decision_counts": decisions.decision.value_counts().to_dict(),
        "document_admission": pd.Series([r["admission_reason"] for r in documents])
        .value_counts()
        .to_dict(),
        "case_counts": pd.Series([r["recovery_status"] for r in closed]).value_counts().to_dict(),
        "coverage": metrics,
        "warning": "Recovered research overlay only; no model promotion or live use.",
    }
    write_json(target / "summary.json", summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"descriptor_files", "coverage"}},
            default=str,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["base", "run", "cache", "audit"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--corrections", type=Path)
    args = parser.parse_args()
    rebuild(args.base, args.run, args.cache, args.audit, args.target, args.corrections)
