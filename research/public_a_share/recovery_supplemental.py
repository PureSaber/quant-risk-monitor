"""Admit reviewed interim evidence without weakening the annual audit gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pymupdf
from financial_table_recovery import compact
from pypdf import PdfReader
from recovery_admission import opinion_evidence
from recovery_inventory import digest, write_json
from report_discovery import original_period


def recheck_review(item):
    """Bind manual column selection to original bytes, header, and both readers."""
    if "value" in item:
        token = item["raw_cell"]
        if (float(token.replace(",", "")) * item["scale"] != item["value"]
                or not any(token in needles for needles in item["checks"].values())):
            raise ValueError("reviewed_value_not_bound_to_original_token")
    if digest(item["source_pdf"]) != item["source_sha256"]:
        raise ValueError("reviewed_supplement_source_changed")
    with pymupdf.open(item["source_pdf"]) as document:
        first = {page: compact(document[int(page) - 1].get_text()) for page in item["checks"]}
    reader = PdfReader(item["source_pdf"])
    for page, needles in item["checks"].items():
        for content in (first[page], compact(reader.pages[int(page) - 1].extract_text() or "")):
            if any(compact(needle) not in content for needle in needles):
                raise ValueError("reviewed_supplement_header_or_value_changed")
    if pd.Timestamp(item["available_at"]) < pd.Timestamp(item["source_available_at"]):
        raise ValueError("supplement_backdated")


def build(geometry, reviews, target):
    review = json.loads(reviews.read_text(encoding="utf-8"))
    for item in review["manual_facts"] + review["entity_scope"]:
        recheck_review(item)
    rows = json.loads(geometry.read_text(encoding="utf-8"))
    facts, documents, sources = [], [], {str(reviews): digest(reviews), str(geometry): digest(geometry)}
    exceptions = {x["announcement_id"]: x for x in review["cover_exceptions"]}
    for row in rows:
        if digest(row["pdf"]) != row["sha256"]:
            raise ValueError("interim_source_changed")
        sources[row["pdf"]] = row["sha256"]
        with pymupdf.open(row["pdf"]) as document:
            text = "\n".join(p.get_text() for p in document)
            cover = document[0].get_text()[:600]
        reason = "eligible"
        try:
            period = original_period(row["title"], cover, row["published"])
            if period != row["report_period"]:
                reason = "report_identity_requires_review"
        except ValueError:
            exception = exceptions.get(row["announcement_id"])
            if not exception or (exception["source_sha256"], exception["report_period"]) != (row["sha256"], row["report_period"]):
                reason = "report_identity_requires_review"
        annual = row["report_period"].endswith("1231")
        opinion = opinion_evidence(text, int(row["report_period"][:4]))
        if annual and opinion["status"] not in {"unqualified", "unqualified_with_emphasis"}:
            reason = "audit_opinion_requires_review"
        if row["announcement_id"] in review["disputed_sources"]:
            reason = "issuer_assurance_disputed"
        document_record = {"asset": row["asset"], "id": row["announcement_id"], "pdf": row["pdf"],
                           "sha256": row["sha256"], "url": row["url"], "title": row["title"],
                           "opinion": opinion["status"] if annual else "interim_no_annual_audit_assumption",
                           "admission_reason": reason}
        documents.append(document_record)
        for value in row["accepted"]:
            fact_reason, scope_proof = reason, None
            if value["scope"] == "issuer_only_review_required":
                eligible = [s for s in review["entity_scope"] if s["asset"] == row["asset"]
                            and pd.Timestamp(s["available_at"]) <= pd.Timestamp(row["available_at"])
                            and s["report_period"] <= row["report_period"]]
                if not eligible:
                    fact_reason = "entity_scope_not_known_at_publication"
                else:
                    scope_proof = max(eligible, key=lambda s: s["available_at"])
            facts.append({**value, "asset": row["asset"], "available_at": row["available_at"],
                          "announcement_id": row["announcement_id"], "source_pdf": row["pdf"],
                          "source_sha256": row["sha256"], "url": row["url"], "title": row["title"],
                          "parser_sha256": row["parser_sha256"], "admission_reason": fact_reason,
                          "audit_status": document_record["opinion"], "entity_scope_evidence": scope_proof})
    for item in review["manual_facts"]:
        sources[item["source_pdf"]] = item["source_sha256"]
        facts.append({**item, "admission_reason": "eligible", "parser_sha256": digest(__file__),
                      "verification": "reviewed_column_and_two_pdf_reader_recheck"})
    for item in review["entity_scope"]:
        sources[item["source_pdf"]] = item["source_sha256"]
    if target.exists():
        raise ValueError("use_new_supplement_version")
    write_json(target, {"schema": "supplemental-financial-evidence/1", "facts": facts,
                        "documents": documents, "sources": [{"path": k, "sha256": v} for k, v in sources.items()]})
    print(json.dumps({"facts": len(facts), "documents": len(documents),
                      "admitted": sum(f["admission_reason"] == "eligible" for f in facts)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("geometry", "reviews", "target"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.geometry, args.reviews, args.target)
