"""Public market-data archive with content hashes and an explicit host allowlist."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(os.environ.get("PUBLIC_PIT_ROOT", Path(__file__).resolve().parent)).resolve()
ALLOWED = {
    "www.cninfo.com.cn",
    "static.cninfo.com.cn",
    "quotes.sina.cn",
    "money.finance.sina.com.cn",
    "push2his.eastmoney.com",
    "www.capco.org.cn",
    "capco.org.cn",
    "sp.capco.org.cn",
    "capcofile.oss-cn-beijing.aliyuncs.com",
    "www.csrc.gov.cn",
}


def fetch(name, url, *, params=None, data=None, binary=False):
    if urlparse(url).hostname not in ALLOWED:
        raise ValueError("Source outside the public-only study allowlist")
    path = ROOT / "raw" / name
    receipt = path.with_suffix(path.suffix + ".receipt.json")
    if path.exists() and receipt.exists():
        meta = json.loads(receipt.read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"Changed cached input: {name}")
        return path.read_bytes() if binary else path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            response = requests.request(
                "POST" if data is not None else "GET",
                url,
                params=params,
                data=data,
                timeout=(10, 45),
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://www.cninfo.com.cn/",
                }
                if "cninfo" in url
                else {"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            content = (
                response.content
                if binary
                else response.content.decode("utf-8", errors="strict").encode("utf-8")
            )
            path.write_bytes(content)
            receipt.write_text(
                json.dumps(
                    {
                        "url": response.url,
                        "request_data": data,
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "bytes": len(content),
                        "source_excludes_huisheng": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return content if binary else content.decode("utf-8")
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def industry_sources():
    html = fetch("industry/index.html", "https://www.capco.org.cn/xhgg/hyfl/hyfljg/index.html")
    soup = BeautifulSoup(html, "html.parser")
    records = []
    for a in soup.find_all("a", href=True):
        title = a.get_text(strip=True)
        if "上市公司行业分类结果" not in title or not re.search(r"202[1345]", title):
            continue
        url = urljoin("https://www.capco.org.cn/xhgg/hyfl/hyfljg/index.html", a["href"])
        if "/2026" in url:
            continue
        if not url.endswith(".html"):
            continue
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        page = fetch(f"industry/{key}.html", url)
        details = BeautifulSoup(page, "html.parser")
        dates = re.findall(r"20\d{2}[-年/]\d{2}[-月/]\d{2}", details.get_text(" "))
        links = [
            (x.get_text(strip=True), urljoin(url, x["href"]))
            for x in details.find_all("a", href=True)
            if ".pdf" in x["href"].lower()
        ]
        if not links:
            continue
        # Prefer code-sorted PDFs, whose rows repeat their full industry classification.
        chosen = sorted(links, key=lambda x: ("代码" not in x[0], x[0]))[0]
        rel = f"industry/{key}.pdf"
        fetch(rel, chosen[1], binary=True)
        records.append(
            {
                "title": title,
                "page": url,
                "date_strings": dates,
                "pdf_url": chosen[1],
                "file": "raw/" + rel,
            }
        )
    (ROOT / "industry-sources.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"industry_documents": len(records)}, ensure_ascii=False), flush=True)


def cohort():
    if (ROOT / "cohort.json").exists():
        print("Preserving existing historical cohort; no resampling.", flush=True)
        return
    dst = ROOT / "raw/industry/csrc-2021q3.pdf"
    dst.parent.mkdir(parents=True, exist_ok=True)
    content = fetch(
        "industry/csrc-2021q3.pdf",
        "https://www.csrc.gov.cn/csrc/c100103/c1558619/1558619/files/1638277734844_11692.pdf",
        binary=True,
    )
    if (
        hashlib.sha256(content).hexdigest()
        != "0e8af6662353037135bb8e8a08a409dcebea9c47f71e7e25b48fe66a24f8c62b"
    ):
        raise ValueError("Original CSRC document hash differs")
    dst.write_bytes(content)
    all_text = "\n".join(p.extract_text() for p in PdfReader(dst).pages)
    (dst.with_suffix(".txt")).write_text(all_text, encoding="utf-8")
    # Capture the actual publication page independently in this study.
    fetch(
        "industry/csrc-2021q3.html",
        "https://www.csrc.gov.cn/csrc/c100103/c1558619/content.shtml",
    )
    rows = []
    current_major = "A"
    current_number = "01"
    for line in all_text.splitlines():
        major = re.search(r"[（(]([A-S])[)）]", line)
        if major:
            current_major = major[1]
        number = re.search(r"(?:^|\s)(\d{2})\s+[^\d]", line)
        if number:
            current_number = number[1]
        security = re.search(r"(?<!\d)((?:000|001|002|600|601|603|605)\d{3})(?!\d)\s*(\S+)", line)
        if security:
            rows.append(
                {
                    "code": security[1],
                    "name_2021": security[2],
                    "industry_code": current_major + current_number,
                    "sector": current_major,
                    "available_date": "2021-11-10",
                }
            )
    unique = {r["code"]: r for r in rows}
    rows = list(unique.values())
    if len(rows) < 2500:
        raise ValueError(f"Unexpected CSRC main-board universe: {len(rows)}")

    # A fixed, historical cohort selected without reading future returns or current listings.
    def key(r):
        return hashlib.sha256(("barra-public-v1:" + r["code"]).encode()).hexdigest()

    selected = {
        r["code"]: r
        for sector in sorted({r["sector"] for r in rows})
        for r in sorted([r for r in rows if r["sector"] == sector], key=key)[:8]
    }
    for row in sorted(rows, key=key):
        if len(selected) >= 200:
            break
        selected[row["code"]] = row
    result = {
        "selection_cutoff": "2021-11-10",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sampling": "8 per historical major sector then SHA256 rank to 200; Shanghai/Shenzhen main boards; never condition on future prices or survival",
        "population": len(rows),
        "members": sorted(selected.values(), key=lambda x: x["code"]),
    }
    (ROOT / "cohort.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"cohort": len(selected), "population": len(rows)}, ensure_ascii=False),
        flush=True,
    )


def security(code):
    results = {"code": code, "errors": []}
    for adjustment in ("0", "2"):
        try:
            params = {
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "ut": "7eea3edcaed734bea9cbfc24409ed989",
                "klt": "101",
                "fqt": adjustment,
                "secid": f"{1 if code.startswith('6') else 0}.{code}",
                "beg": "20190101",
                "end": "20251231",
            }
            payload = json.loads(
                fetch(
                    f"prices/{code}-{adjustment}.json",
                    "https://push2his.eastmoney.com/api/qt/stock/kline/get",
                    params=params,
                )
            )
            results["prices_" + adjustment] = len((payload.get("data") or {}).get("klines") or [])
        except Exception as exc:
            results["errors"].append({"stage": "price_" + adjustment, "error": str(exc)[:220]})
    for table in ("fzb", "lrb"):
        try:
            fetch(
                f"financial/{code}-{table}.json",
                "https://quotes.sina.cn/cn/api/openapi.php/CompanyFinanceService.getFinanceReport2022",
                params={
                    "paperCode": ("sh" if code.startswith("6") else "sz") + code,
                    "source": table,
                    "type": "0",
                    "page": "1",
                    "num": "1000",
                },
            )
        except Exception as exc:
            results["errors"].append({"stage": table, "error": str(exc)[:220]})
    return results


def download():
    members = json.loads((ROOT / "cohort.json").read_text(encoding="utf-8"))["members"]
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(security, row["code"]) for row in members]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            (ROOT / "download-status.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps({"completed": i, **result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    {"cohort": cohort, "industries": industry_sources, "download": download}[sys.argv[1]]()
