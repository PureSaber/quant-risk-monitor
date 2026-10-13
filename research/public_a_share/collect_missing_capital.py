"""Dated public capital-history fallback for gaps in the CNINFO endpoint."""

import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd
from bs4 import BeautifulSoup

from public_sources import ROOT, fetch


def main():
    members = json.loads((ROOT / "cohort.json").read_text(encoding="utf8"))["members"]
    for member in members:
        code = member["code"]
        output = ROOT / "raw/shares" / (code + ".csv")
        if output.exists():
            continue
        url = (
            f"https://money.finance.sina.com.cn/corp/go.php/vCI_StockStructure/stockid/{code}.phtml"
        )
        content = fetch("shares/sina-" + code + ".html", url, binary=True)
        soup = BeautifulSoup(content.decode("gb18030", errors="strict"), "html.parser")
        records = []
        for table in soup.find_all("table"):
            if (
                table.find("table")
                or "变动日期" not in table.get_text()
                or "总股本" not in table.get_text()
            ):
                continue
            fields = {}
            for row in table.find_all("tr"):
                cells = [
                    c.get_text(" ", strip=True) for c in row.find_all(["td", "th"], recursive=False)
                ]
                if not cells:
                    continue
                for name in ["变动日期", "公告日期", "总股本"]:
                    if name in cells[0]:
                        fields[name] = cells[1:]
            if set(fields) != set(["变动日期", "公告日期", "总股本"]):
                continue
            for effective, ann, count in zip(
                fields["变动日期"], fields["公告日期"], fields["总股本"], strict=True
            ):
                if (
                    not re.fullmatch(r"\d{8}", ann)
                    or not re.fullmatch(r"\d{8}", effective)
                    or ann < "19900101"
                ):
                    continue
                value = re.fullmatch(r"([\d.,]+)\s*万股", count)
                if value is None:
                    raise ValueError("Unexpected share-count unit")
                records.append(
                    {
                        "证券代码": code,
                        "公告日期": pd.Timestamp(ann).date().isoformat(),
                        "变动日期": pd.Timestamp(effective).date().isoformat(),
                        "总股本": float(value[1].replace(",", "")),
                        "source": "Sina dated capital-history table",
                    }
                )
        if not records:
            raise ValueError(f"No dated capital history for {code}")
        pd.DataFrame(records).sort_values(["变动日期", "公告日期"]).drop_duplicates().to_csv(
            output, index=False
        )
        output.with_suffix(".receipt.json").write_text(
            json.dumps(
                {
                    "url": url,
                    "source": url,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "raw_document": "raw/shares/sina-" + code + ".html",
                    "rows": len(records),
                    "date_policy": "Require both effective and announcement dates; reject 19000101 placeholders; no current-only backfill",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf8",
        )
        print(code, len(records), flush=True)


if __name__ == "__main__":
    main()
