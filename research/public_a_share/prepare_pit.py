"""Reconstruct dated public facts; amounts must be present in original disclosures."""

from __future__ import annotations

import json
import re
import sys

import pandas as pd
from pypdf import PdfReader

from public_sources import ROOT

FIELD_NAMES = {
    "book_equity": ["归属于母公司股东权益合计", "归属于母公司所有者权益合计"],
    "assets": ["资产总计"],
    "liabilities": ["负债合计"],
    "shares": ["实收资本(或股本)", "实收资本（或股本）", "股本"],
    "profit_ytd": ["归属于母公司所有者的净利润", "归属于母公司股东的净利润"],
    "revenue_ytd": ["营业总收入", "营业收入"],
}
ALIASES = {
    "book_equity": r"归属[于於]?(?:上市公司|母公司)(?:的)?(?:股东|所有者)(?:的)?(?:净资产|(?:所有者)?权益(?:[（(]或股东权益[)）])?(?:合计|总计)?)",
    "assets": r"(?:资产总计|总资产)",
    "liabilities": r"(?<![\u3400-\u9fff])(?:负债合计|总负债)",
    "shares": r"(?:股份总数|(?:公司)?总股本(?=\s*[（(]股[)）]))",
    "profit_ytd": r"归属[于於]?(?:上市公司|母公司)(?:的)?(?:股东|所有者)(?:的)?净利润",
    "revenue_ytd": r"(?:营业总收入|营业收入)",
}


def period(title):
    title = re.sub(r"\s+", "", title)
    year = re.search(r"(20\d{2})年", title)
    if not year:
        return None
    y = year[1]
    if "第一季度" in title or "一季度" in title:
        return y + "0331"
    if "第三季度" in title or "三季度" in title:
        return y + "0930"
    if "半年度" in title:
        return y + "0630"
    if "年度" in title:
        return y + "1231"
    return None


def industries():
    members = json.loads((ROOT / "cohort.json").read_text(encoding="utf-8"))["members"]
    records = [
        {
            "asset": r["code"],
            "industry": r["industry_code"],
            "sector": r["sector"],
            "available_at": "2021-11-11T00:00:00+08:00",
            "source": "CSRC original 2021Q3",
        }
        for r in members
    ]
    selected = {r["code"] for r in members}
    sources = json.loads((ROOT / "industry-sources.json").read_text(encoding="utf-8"))
    for source in sources:
        if "2021" in source["title"]:
            continue
        date = source["date_strings"][0]
        text = "\n".join(p.extract_text() or "" for p in PdfReader(ROOT / source["file"]).pages)
        (ROOT / source["file"]).with_suffix(".txt").write_text(text, encoding="utf-8")
        for line in text.splitlines():
            hit = re.match(r"(\d{6})\s+\S+\s+([A-S])\s+(.+)", line)
            if hit and hit[1] in selected:
                large = re.search(r"(?:^|\s)(\d{2})\s+", hit[3])
                if not large:
                    raise ValueError("Industry row missing broad class")
                records.append(
                    {
                        "asset": hit[1],
                        "industry": hit[2] + large[1],
                        "sector": hit[2],
                        "available_at": (
                            pd.Timestamp(date, tz="Asia/Shanghai") + pd.Timedelta(days=1)
                        ).isoformat(),
                        "source": source["page"],
                    }
                )
    out = ROOT / "clean"
    out.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(out / "industry_history.csv", index=False, encoding="utf-8")
    print(
        json.dumps(
            {
                "industry_records": len(records),
                "publications": len(set(r["available_at"] for r in records)),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"industries": industries}[sys.argv[1]]()
