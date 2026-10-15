"""Dated industry releases and market benchmark for the fixed confirmation period."""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd


def parse_sector_rows(text, members):
    """The model needs the explicit major sector, not a wrapped subindustry code."""
    found = {}
    for line in text.splitlines():
        match = re.match(r"^(\d{6})\s*\S+\s+([A-S])(?:\s|$)", line.strip())
        if match and match[1] in members:
            if match[1] in found and found[match[1]] != match[2]:
                raise ValueError("conflicting original sector rows")
            found[match[1]] = match[2]
    if len(found) < 100:
        raise ValueError("insufficient original sector coverage")
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["PUBLIC_PIT_ROOT"] = str(args.out)
    from pypdf import PdfReader

    from public_sources import fetch, industry_sources

    (args.out / "clean").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.old_root / "cohort.json", args.out / "cohort.json")
    industry_sources(year_pattern=r"202[56]", maximum_publication_year=2026)
    members = {
        r["code"]
        for r in json.loads((args.out / "cohort.json").read_text(encoding="utf8"))["members"]
    }
    rows = []
    for source in json.loads((args.out / "industry-sources.json").read_text(encoding="utf8")):
        available = pd.Timestamp(source["date_strings"][0], tz="Asia/Shanghai") + pd.Timedelta(
            days=1
        )
        if (
            not pd.Timestamp("2025-12-31", tz="Asia/Shanghai")
            < available
            <= pd.Timestamp("2026-09-30", tz="Asia/Shanghai")
        ):
            continue
        text = "\n".join(p.extract_text() or "" for p in PdfReader(args.out / source["file"]).pages)
        (args.out / source["file"]).with_suffix(".txt").write_text(text, encoding="utf8")
        for asset, sector in parse_sector_rows(text, members).items():
            rows.append(
                {
                    "asset": asset,
                    "industry": sector,
                    "sector": sector,
                    "available_at": available.isoformat(),
                    "source": source["page"],
                }
            )
    current = pd.DataFrame(rows, columns=["asset", "industry", "sector", "available_at", "source"])
    old = pd.read_csv(args.old_root / "clean/industry_history.csv", dtype={"asset": str})
    known = pd.to_datetime(current.available_at, utc=True)
    current = current[
        (known > pd.Timestamp("2025-12-31", tz="Asia/Shanghai"))
        & (known <= pd.Timestamp("2026-09-30", tz="Asia/Shanghai"))
    ]
    merged = pd.concat([old, current], ignore_index=True).drop_duplicates()
    merged.to_csv(args.out / "clean/industry_history.csv", index=False)
    fetch(
        "benchmark/csi300.json",
        "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        params={
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "klt": "101",
            "fqt": "0",
            "secid": "1.000300",
            "beg": "20251201",
            "end": "20260930",
        },
    )
    print(
        json.dumps(
            {
                "new_industry_rows": len(current),
                "new_publications": sorted(current.available_at.unique()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
