"""Archive original disclosure metadata and financial-report PDFs for PIT matching."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pymupdf
from report_discovery import is_full_report

from public_sources import ROOT, fetch


def announcements(code, org, *, start="2019-01-01", end="2025-12-31"):
    rows = []
    payload = {
        "pageNum": 1,
        "pageSize": 30,
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": f"{code},{org}" if org else "",
        "searchkey": "" if org else code,
        "secid": "",
        "category": "category_ndbg_szsh;category_bndbg_szsh;category_yjdbg_szsh;category_sjdbg_szsh;category_bcgz_szsh;category_tbclts_szsh;category_tszlq_szsh",
        "trade": "",
        "seDate": f"{start}~{end}",
        "sortName": "time",
        "sortType": "asc",
        "isHLtitle": "false",
    }
    expected_total = None
    seen = set()
    received = 0
    for page in range(1, 41):
        payload["pageNum"] = page
        result = json.loads(
            fetch(
                f"announcements/{code}-{page:02}.json",
                "https://www.cninfo.com.cn/new/hisAnnouncement/query",
                data=payload,
            )
        )
        total = result.get("totalAnnouncement")
        if not isinstance(total, int) or total < 0:
            raise ValueError("Missing/invalid announcement total; archive completeness unknown")
        if expected_total is not None and total != expected_total:
            raise ValueError("Announcement result changed during pagination; retry a fresh archive")
        expected_total = total
        batch = result.get("announcements") or []
        if not batch and received < total:
            raise ValueError("Empty announcement page before declared total")
        for row in batch:
            key = (row["secCode"], str(row["announcementId"]))
            if key in seen:
                raise ValueError("Repeated announcement across pages; incomplete pagination")
            seen.add(key)
            if row["secCode"] == code:
                rows.append(row)
        received += len(batch)
        if received >= total:
            break
    else:
        raise ValueError("Announcement pagination limit exceeded; cannot silently truncate")
    return rows


def extract_pdf(row):
    ann_id = str(row["announcementId"])
    rel = f"filings/{row['secCode']}/{ann_id}.pdf"
    url = "https://static.cninfo.com.cn/" + row["adjunctUrl"].lstrip("/")
    fetch(rel, url, binary=True)
    output = ROOT / "raw" / rel.replace(".pdf", ".logical.txt")
    marker = output.with_suffix(".extract.json")
    safe_marker = (
        marker.exists()
        and json.loads(marker.read_text(encoding="utf8")).get("isolated_process") is True
    )
    if not output.exists() or not safe_marker:
        reader = pymupdf.open(ROOT / "raw" / rel)
        # Full originals preserve annual consolidated statements and capital tables.
        text = "\n".join(
            f"\n[[PAGE {i + 1}]]\n" + page.get_text(sort=False) for i, page in enumerate(reader)
        )
        output.write_text(text, encoding="utf-8")
        marker.write_text(
            json.dumps(
                {
                    "parser": "pymupdf-1.28.2",
                    "pages": len(reader),
                    "full_document": True,
                    "isolated_process": True,
                    "reading_order": "original_cell_order",
                }
            ),
            encoding="utf-8",
        )
        reader.close()
    return {
        "id": ann_id,
        "title": row["announcementTitle"],
        "announcement_time_ms": row["announcementTime"],
        "pdf": rel,
        "text": str(output.relative_to(ROOT)),
        "url": url,
    }


def one(code, org, pdfs=False):
    rows = announcements(code, org)
    selected = [r for r in rows if is_full_report(r["announcementTitle"])]
    selected = [
        r
        for r in selected
        if "正文" not in r["announcementTitle"]
        or not any(
            s["announcementTime"] == r["announcementTime"]
            and "全文" in s["announcementTitle"]
            and s["announcementTitle"].split("报告")[0] == r["announcementTitle"].split("报告")[0]
            for s in selected
        )
    ]
    # Includes original and amended full reports. Never deduplicate by fiscal period.
    result = {
        "code": code,
        "announcements": len(rows),
        "report_count": len(selected),
        "reports": [],
        "errors": [],
    }
    if pdfs:
        for row in selected:
            try:
                result["reports"].append(extract_pdf(row))
            except Exception as exc:
                result["errors"].append({"id": row["announcementId"], "error": str(exc)[:240]})
    (ROOT / "raw/announcements" / (code + "-index.json")).write_text(
        json.dumps({**result, "all_records": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {k: v for k, v in result.items() if k != "reports"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    members = json.loads((ROOT / "cohort.json").read_text(encoding="utf-8"))["members"]
    mapping = json.loads(
        fetch(
            "announcements/stock-map.json",
            "https://www.cninfo.com.cn/new/data/szse_stock.json",
        )
    )
    orgs = {r["code"]: r["orgId"] for r in mapping["stockList"]}
    codes = [m["code"] for m in members]
    if len(sys.argv) > 2:
        codes = sys.argv[2:]
    pdfs = len(sys.argv) > 1 and sys.argv[1] == "pdfs"
    status_path = ROOT / ("filings-status.json" if pdfs else "announcements-status.json")
    previous = json.loads(status_path.read_text(encoding="utf8")) if status_path.exists() else []
    results = {r["code"]: r for r in previous}
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(one, code, orgs.get(code), pdfs): code for code in codes}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:
                result = {"code": futures[future], "error": str(exc)[:250]}
            results[result["code"]] = result
            status_path.write_text(
                json.dumps(list(results.values()), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps({"completed": i, **result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
