"""Collect issuer announcements only for evidenced execution-input discrepancies."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def collect(old, out, destination):
    from quant_data_kit.providers._network import configure_network

    import public_sources
    from study_v3_archive import write_json

    configure_network()
    public_sources.ROOT = destination
    issues = json.loads((out / "execution-input-issues.json").read_text(encoding="utf8"))
    assets = sorted({row["asset"] for row in issues})

    def one(asset):
        source = json.loads(
            (old / "raw/announcements" / (asset + "-01.json")).read_text(encoding="utf8")
        )
        orgs = {r["orgId"] for r in source["announcements"] if r["secCode"] == asset}
        if len(orgs) != 1:
            raise ValueError("ambiguous issuer identity")
        found = {}
        keywords = ("权益分派实施", "利润分配实施")
        if asset in {"600165", "603007"}:
            keywords += ("转增股本",)
        for keyword in keywords:
            for page in range(1, 11):
                payload = {
                    "pageNum": page,
                    "pageSize": 30,
                    "column": "szse",
                    "tabName": "fulltext",
                    "plate": "",
                    "stock": asset + "," + next(iter(orgs)),
                    "searchkey": keyword,
                    "secid": "",
                    "category": "",
                    "trade": "",
                    "seDate": "2023-01-01~2026-09-30",
                    "sortName": "time",
                    "sortType": "asc",
                    "isHLtitle": "false",
                }
                result = json.loads(
                    public_sources.fetch(
                        f"announcements/{asset}-{keyword}-{page}.json",
                        "https://www.cninfo.com.cn/new/hisAnnouncement/query",
                        data=payload,
                    )
                )
                for record in result.get("announcements") or []:
                    if record["secCode"] == asset:
                        found[str(record["announcementId"])] = record
                if page * 30 >= result.get("totalAnnouncement", 0):
                    break
            else:
                raise ValueError("announcement pagination limit reached")
        for identity, record in found.items():
            public_sources.fetch(
                f"filings/{asset}/{identity}.pdf",
                "https://static.cninfo.com.cn/" + record["adjunctUrl"],
                binary=True,
            )
        return {"asset": asset, "announcements": list(found.values())}

    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(one, a) for a in assets]):
            record = future.result()
            records.append(record)
            print(
                json.dumps(
                    {"asset": record["asset"], "announcements": len(record["announcements"])}
                ),
                flush=True,
            )
    write_json(destination / "supplemental-manifest.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("old", "out", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    collect(args.old, args.out, args.destination)
