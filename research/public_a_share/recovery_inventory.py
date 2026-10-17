"""Content-addressed, read-only input inventory for a new recovery run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from report_discovery import is_full_report, title_period


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def inventory(base, audit, out):
    """Inventory every cohort issuer; select audited blockers plus peer controls."""
    if (out / "protocol.json").exists():
        raise ValueError("existing_run_requires_explicit_resume_not_refreeze")
    members = json.loads((base / "inputs/old/cohort.json").read_text(encoding="utf-8"))["members"]
    audited = json.loads((audit / "document-audit.json").read_text(encoding="utf-8"))
    checked = {x["announcement_id"]: x for x in audited if x["status"] == "read"}
    comparisons = pd.read_csv(audit / "comparative-table-audit.csv", dtype=str)
    selected_ids = set(checked) | set(comparisons.source_id)
    docs = {}
    roots = [base / "inputs/old", base / "inputs/exposure/confirmation-inputs"]
    for member in members:
        code = member["code"]
        for root in roots:
            index = json.loads(
                (root / f"raw/announcements/{code}-index.json").read_text(encoding="utf-8")
            )
            for ann in index.get("all_records", []):
                if not is_full_report(ann["announcementTitle"]):
                    continue
                aid = str(ann["announcementId"])
                period = title_period(ann["announcementTitle"])
                if not period and checked.get(aid, {}).get("cover_year"):
                    period = str(checked[aid]["cover_year"]) + "1231"
                if (
                    not period
                    or not period.endswith("1231")
                    or not "20180101" <= period <= "20251231"
                ):
                    continue
                published = pd.Timestamp(ann["announcementTime"], unit="ms", tz="UTC").tz_convert(
                    "Asia/Shanghai"
                )
                if published.date().isoformat() > "2026-09-30":
                    continue
                paths = [r / f"raw/filings/{code}/{aid}.pdf" for r in roots]
                paths += [base / f"inputs/exposure/raw/filings/{code}/{aid}.pdf"]
                if aid in checked:
                    paths.append(Path(checked[aid]["pdf"]))
                path = next((p for p in paths if p.is_file()), None)
                # Control sampling uses a stable issuer code rule, not observed performance.
                control = int(code) % 17 == 0 and period in {"20201231", "20231231", "20251231"}
                docs[aid] = {
                    "asset": code,
                    "name": member["name_2021"],
                    "announcement_id": aid,
                    "title": ann["announcementTitle"],
                    "report_period": period,
                    "published": published.date().isoformat(),
                    "available_at": (published.normalize() + pd.Timedelta(days=1)).isoformat(),
                    "url": "https://static.cninfo.com.cn/" + ann["adjunctUrl"].lstrip("/"),
                    "pdf": str(path) if path else "",
                    "selected": aid in selected_ids or control,
                    "role": "audited_gap"
                    if aid in selected_ids
                    else "observed_control"
                    if control
                    else "inventory_only",
                    "sha256": digest(path) if path and (aid in selected_ids or control) else "",
                }
    rows = list(docs.values())
    matrix = []
    for m in members:
        for year in range(2018, 2026):
            available = [
                r for r in rows if r["asset"] == m["code"] and r["report_period"] == f"{year}1231"
            ]
            matrix.append(
                {
                    "asset": m["code"],
                    "name": m["name_2021"],
                    "year": year,
                    "indexed_reports": len(available),
                    "archived_reports": sum(bool(r["pdf"]) for r in available),
                    "status": "indexed"
                    if available
                    else "no_original_annual_index_not_proof_of_absence",
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "report-inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(matrix).to_csv(out / "annual-matrix.csv", index=False, encoding="utf-8-sig")
    paths = [
        base / "runs/main" / n
        for n in [
            "panel.pkl",
            "financial-vintages.parquet",
            "protocol.json",
            "daily-quality.parquet",
        ]
    ]
    paths += [base / "inputs/old/cohort.json", audit / "annual-gap-classification.csv"]
    protocol = {
        "schema": "barra-data-recovery/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cohort_size": 200,
        "start": "2023-01-03",
        "end": "2026-09-30",
        "inputs": [{"path": str(p), "sha256": digest(p)} for p in paths],
        "default_model_changed": False,
        "financial_fill_allowed": False,
        "development_year": 2023,
        "diagnostic_years": [2024, 2025, 2026],
        "imputation": {
            "seed": 20261017,
            "fraction_grid": [0.05, 0.1, 0.2],
            "block_sessions": 63,
            "methods": ["industry_median", "industry_size_ridge"],
            "acceptance": "research_only; no automatic promotion or historical holdout claims",
        },
        "budget": {
            "data_root": str(base),
            "data_bytes_limit": 25 * 2**30,
            "minimum_system_free_bytes": 40 * 2**30,
        },
    }
    write_json(out / "protocol.json", protocol)
    print(
        json.dumps(
            {
                "annual_versions": len(rows),
                "selected": sum(r["selected"] for r in rows),
                "selected_missing_pdf": sum(r["selected"] and not r["pdf"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["base", "audit", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    inventory(a.base, a.audit, a.out)
