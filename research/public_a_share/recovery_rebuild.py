"""Rebuild only changed descriptor matrices; preserve immutable baseline panels."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import model_bootstrap  # noqa: F401
from quant_risk_monitor.style_descriptors import annual_growth_descriptors
from recovery_admission import FINANCIAL_FIELDS, merge_evidence, opinion_evidence
from recovery_inventory import digest, write_json


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


def reviewed_source_rows(facts, corrections):
    """Re-read explicitly reviewed non-annual sources without changing discovery."""
    import re

    import pymupdf
    from pypdf import PdfReader

    rows = []
    for correction in corrections:
        if not correction.get("reverify_document"):
            continue
        path = correction["source_pdf"]
        if digest(path) != correction["source_sha256"]:
            raise ValueError("reviewed_document_changed")
        page = correction["page"]
        period = correction["report_period"]
        with pymupdf.open(path) as document:
            text = document[page - 1].get_text()
        reader = PdfReader(path)
        other = reader.pages[page - 1].extract_text()
        for content in (text, other):
            compact = re.sub(r"\s+", "", content)
            row = re.search(r"负债合计\s*(\d{1,3}(?:,\d{3})+)", content)
            if (correction["field"] != "liabilities" or "合并资产负债表" not in compact
                    or "金额单位为人民币百万元" not in compact or not row
                    or float(row[1].replace(",", "")) * 1e6 != correction["new_value"]):
                raise ValueError("reviewed_source_failed_independent_recheck")
        verified = {"field": "liabilities", "report_period": period,
                    "value": correction["new_value"], "scale": 1e6, "page": page,
                    "label": "负债合计", "scope": "reviewed_consolidated_continuation"}
        original = facts[(facts.asset == correction["asset"])
                         & (facts.field == correction["field"])
                         & (facts.report_period == period)
                         & (pd.to_datetime(facts.available_at, utc=True)
                            == pd.Timestamp(correction["available_at"]))]
        if len(original) != 1:
            raise ValueError("reviewed_original_not_unique")
        rows.append({**original.iloc[0].to_dict(), **verified,
                     "admission_reason": "eligible", "audit_status": "reviewed_legacy",
                     "source_pdf": path, "source_sha256": correction["source_sha256"],
                     "url": correction["url"], "parser_sha256": digest(__file__),
                     "verification": "explicit_review_and_two_reader_recheck"})
    return rows


def comparative_values(text, label, years):
    import re

    compact = re.sub(r"\s+", "", text)
    header = compact.split("营业收入", 1)[0]
    seen = list(dict.fromkeys(int(y) for y in re.findall(r"(20\d{2})年度", header)))
    if seen != years or "金额单位为人民币百万元" not in header or "合并利润表" not in header:
        raise ValueError("reviewed_comparative_header_changed")
    row = re.search(re.escape(label) + r"\s*((?:\d{1,3}(?:,\d{3})+\s*){3})", text)
    if not row:
        raise ValueError("reviewed_comparative_row_changed")
    return [float(v.replace(",", "")) * 1e6 for v in re.findall(r"\d{1,3}(?:,\d{3})+", row[1])]


def reviewed_comparatives(path):
    import pymupdf
    from pypdf import PdfReader

    rows = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        source = item["source_pdf"]
        if digest(source) != item["source_sha256"]:
            raise ValueError("reviewed_comparative_source_changed")
        with pymupdf.open(source) as document:
            text = document[item["page"] - 1].get_text()
        other = PdfReader(source).pages[item["page"] - 1].extract_text()
        for field, label in (("revenue_ytd", "营业收入"), ("profit_ytd", "归属于母公司股东的净利润")):
            for content in (text, other):
                if comparative_values(content, label, item["column_years"]) != item[field]:
                    raise ValueError("reviewed_comparative_value_changed")
            for year, value in zip(item["column_years"], item[field], strict=True):
                rows.append({**{k: item[k] for k in ("asset", "announcement_id", "source_pdf", "source_sha256", "available_at", "url", "page")},
                             "field": field, "report_period": str(year) + "1231", "value": value,
                             "label": label, "scale": 1e6, "scope": "reviewed_restated_comparative",
                             "admission_reason": "eligible", "audit_status": "unqualified",
                             "verification": "explicit_comparative_columns_two_pdf_readers",
                             "parser_sha256": digest(__file__)})
    return rows


def rebuild(base, run, cache, audit, target=None, corrections=None, comparatives=None):
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
    extra = reviewed_source_rows(facts, reviewed)
    if comparatives:
        extra.extend(reviewed_comparatives(comparatives))
    if extra:
        additions = pd.concat([additions, pd.DataFrame(extra)], ignore_index=True)
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
    parser.add_argument("--comparatives", type=Path)
    args = parser.parse_args()
    rebuild(args.base, args.run, args.cache, args.audit, args.target, args.corrections, args.comparatives)
