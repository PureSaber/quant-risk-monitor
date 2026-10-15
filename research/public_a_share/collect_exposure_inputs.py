"""Supplement original annual disclosures for exposure warmup; preserve v1 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def extract_annual(args):
    old, target, code = map(str, args)
    os.environ["PUBLIC_PIT_ROOT"] = target
    from filings import extract_pdf
    from original_financials import amount_row, disclosure_period, normalize

    index = json.loads(
        (Path(old) / "raw/announcements" / (code + "-index.json")).read_text(encoding="utf8")
    )
    chosen = [
        r
        for r in index["all_records"]
        if re.search(r"2018年.*年度报告", r["announcementTitle"])
        and not re.search("摘要|取消|英文", r["announcementTitle"])
    ]
    rows, audit = [], []
    for ann in chosen:
        try:
            report = extract_pdf(ann)
            path = Path(target) / report["text"]
            text = path.read_text(encoding="utf8")
            publication = (
                pd.Timestamp(ann["announcementTime"], unit="ms", tz="UTC")
                .tz_convert("Asia/Shanghai")
                .normalize()
            )
            period = disclosure_period(ann["announcementTitle"], text, publication)
            if period != "20181231":
                raise ValueError("original cover does not identify fiscal 2018")
            normalized = normalize(text)
            for field in ("profit_ytd", "revenue_ytd"):
                parsed = amount_row(normalized, field, period, normalized=True)
                summary = amount_row(normalized, field, period, normalized=True, summary_only=True)
                if (
                    parsed
                    and summary
                    and parsed["from_consolidated_statement"]
                    and abs(parsed["value"] - summary["value"])
                    > max(parsed["display_unit_scale"], summary["display_unit_scale"], 1) * 1.01
                ):
                    parsed = None
                if parsed:
                    rows.append(
                        {
                            "asset": code,
                            "field": field,
                            "report_period": period,
                            "available_at": (publication + pd.Timedelta(days=1)).isoformat(),
                            "announcement_id": str(ann["announcementId"]),
                            "document": report["pdf"],
                            "text_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "source": "independent_original_cninfo_disclosure",
                            **parsed,
                        }
                    )
                else:
                    audit.append(
                        {
                            "asset": code,
                            "id": ann["announcementId"],
                            "field": field,
                            "reason": "original_not_unambiguously_parsed",
                        }
                    )
        except Exception as exc:
            audit.append({"asset": code, "id": ann["announcementId"], "reason": str(exc)[:300]})
    return rows, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    members = json.loads((args.old_root / "cohort.json").read_text(encoding="utf8"))["members"]
    rows, audit = [], []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(extract_annual, (args.old_root, args.out, m["code"])) for m in members
        ]
        for i, future in enumerate(as_completed(futures), 1):
            found, missing = future.result()
            rows.extend(found)
            audit.extend(missing)
            if i % 10 == 0:
                print(
                    json.dumps({"issuers": i, "facts": len(rows), "issues": len(audit)}), flush=True
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["asset", "field", "available_at", "announcement_id"])
        frame = frame.drop_duplicates(
            ["asset", "field", "report_period", "available_at"], keep="last"
        )
    frame.to_csv(args.out / "annual-warmup-facts.csv", index=False, encoding="utf8")
    (args.out / "annual-warmup-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(
        json.dumps(
            {"facts": len(frame), "issuers": int(frame.asset.nunique()) if not frame.empty else 0}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
