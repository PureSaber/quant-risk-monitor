"""Bounded incremental retrieval of issuer reports mirrored after an A-share exit.

Mirrors are labelled as such. They are not silently called exchange originals.
No present-day structured financial value is used as a historical fact.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from recovery_inventory import digest, write_json
from report_discovery import is_full_report, title_period

START = {
    "000150": 2023,
    "000613": 2022,
    "000666": 2023,
    "002621": 2024,
    "002699": 2024,
    "002731": 2025,
    "600190": 2025,
    "600766": 2024,
    "600896": 2022,
}
ALLOWED = {
    "money.finance.sina.com.cn",
    "vip.stock.finance.sina.com.cn",
    "file.finance.sina.com.cn",
    "www.neeq.com.cn",
}


def request(url, *, maximum=40 * 2**20):
    if urlparse(url).hostname not in ALLOWED:
        raise ValueError("unapproved_public_source")
    for attempt in range(3):
        try:
            with requests.get(url, timeout=(10, 35), stream=True) as response:
                response.raise_for_status()
                if urlparse(response.url).hostname not in ALLOWED:
                    raise ValueError("unapproved_redirect_source")
                chunks = []
                size = 0
                for block in response.iter_content(2**18):
                    size += len(block)
                    if size > maximum:
                        raise ValueError("document_download_budget_exceeded")
                    chunks.append(block)
                return b"".join(chunks)
        except requests.RequestException:
            if attempt == 2:
                raise


def soup(raw):
    return BeautifulSoup(raw, "html.parser", from_encoding="gb18030")


def collect(base, out):
    limit = 25 * 2**30
    used = sum(p.stat().st_size for p in base.rglob("*") if p.is_file())
    if shutil.disk_usage(Path.cwd().anchor).free < 40 * 2**30:
        raise ValueError("system_disk_reserve_exceeded")
    target = out / "supplemental-originals"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    statuses = []
    for code, start in START.items():
        index_url = f"https://money.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/{code}/page_type/ndbg.phtml"
        index_path = target / (code + "-annual-index.html")
        if not index_path.exists():
            raw = request(index_url, maximum=2**20)
            if used + len(raw) > limit:
                raise ValueError("data_root_budget_exceeded")
            index_path.write_bytes(raw)
            used += len(raw)
        links = []
        for a in soup(index_path.read_bytes()).find_all("a", href=True):
            title = a.get_text("", strip=True)
            period = title_period(title)
            if not is_full_report(title) or not period or not period.endswith("1231"):
                continue
            if not start <= int(period[:4]) <= 2025 or re.search(r"东海B|锦州港B", title):
                continue
            url = urljoin(index_url, a["href"])
            if "vCB_AllBulletinDetail" in url:
                links.append((title, period, url))
        unique = {url: (title, period, url) for title, period, url in links}
        for title, period, url in unique.values():
            aid = "sina-" + re.search(r"[?&]id=(\d+)", url)[1]
            try:
                html_path = target / (aid + ".html")
                if not html_path.exists():
                    raw = request(url, maximum=3 * 2**20)
                    if used + len(raw) > limit:
                        raise ValueError("data_root_budget_exceeded")
                    html_path.write_bytes(raw)
                    used += len(raw)
                page = soup(html_path.read_bytes())
                date = re.search(
                    r"公告日期\s*[:：]\s*(\d{4}-\d{2}-\d{2})", page.get_text(" ", strip=True)
                )
                if not date:
                    raise ValueError("missing_publication_date")
                published = date[1]
                if published > "2026-09-30":
                    continue
                pdfs = [
                    urljoin(url, a["href"])
                    for a in page.find_all("a", href=True)
                    if re.search(r"\.pdf(?:$|\?)", a["href"], re.I)
                ]
                if not pdfs:
                    raise ValueError("missing_original_pdf_link")
                pdf_url = pdfs[0].replace("http://", "https://", 1)
                pdf_path = target / (aid + ".pdf")
                if not pdf_path.exists():
                    raw = request(pdf_url, maximum=min(40 * 2**20, limit - used))
                    if not raw.startswith(b"%PDF"):
                        raise ValueError("not_original_pdf")
                    if used + len(raw) > limit:
                        raise ValueError("data_root_budget_exceeded")
                    pdf_path.write_bytes(raw)
                    used += len(raw)
                row = {
                    "asset": code,
                    "name": "",
                    "announcement_id": aid,
                    "title": title,
                    "report_period": period,
                    "published": published,
                    "available_at": (
                        pd.Timestamp(published, tz="Asia/Shanghai") + pd.Timedelta(days=1)
                    ).isoformat(),
                    "url": pdf_url,
                    "discovery_url": url,
                    "pdf": str(pdf_path),
                    "selected": True,
                    "role": (
                        "post_delisting_exchange_original"
                        if urlparse(pdf_url).hostname == "www.neeq.com.cn"
                        else "post_delisting_original_mirror"
                    ),
                    "sha256": digest(pdf_path),
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
                records.append(row)
                statuses.append(
                    {
                        "asset": code,
                        "id": aid,
                        "status": "archived",
                        "published": published,
                        "title": title,
                    }
                )
            except Exception as exc:
                statuses.append(
                    {"asset": code, "id": aid, "status": "failed", "error": str(exc), "url": url}
                )
        statuses.append(
            {
                "asset": code,
                "status": "index_searched",
                "annual_links": len(unique),
                "source": index_url,
                "index_sha256": digest(index_path),
                "absence_is_proven": False,
            }
        )
        print(
            json.dumps(
                {"asset": code, "downloaded": len(records), "root_GiB": round(used / 2**30, 3)}
            ),
            flush=True,
        )
        write_json(out / "supplemental-status.json", statuses)
    existing = pd.read_csv(out / "report-inventory.csv", dtype=str, keep_default_na=False)
    pd.concat([existing, pd.DataFrame(records)], ignore_index=True).to_csv(
        out / "recovery-inventory.csv", index=False, encoding="utf-8-sig"
    )
    write_json(out / "supplemental-receipts.json", records)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    collect(a.base, a.out)
