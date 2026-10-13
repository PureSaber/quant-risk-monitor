"""Retain announcement/effective dates for original public capital-change records."""

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import akshare as ak
import requests

from public_sources import ROOT

original_request = requests.sessions.Session.request


def bounded(self, *args, **kwargs):
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = (10, 35)
    return original_request(self, *args, **kwargs)


requests.sessions.Session.request = bounded


def collect(code):
    path = ROOT / "raw/shares" / (code + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return {"code": code, "cached": True}
    df = ak.stock_share_change_cninfo(symbol=code, start_date="20000101", end_date="20251231")
    df.to_csv(path, index=False, encoding="utf-8")
    meta = {
        "code": code,
        "rows": len(df),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": "https://webapi.cninfo.com.cn/api/stock/p_stock2215",
        "parameters": {"scode": code, "sdate": "2000-01-01", "edate": "2025-12-31"},
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "client": "akshare-1.19.1",
    }
    path.with_suffix(".receipt.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    members = json.loads((ROOT / "cohort.json").read_text(encoding="utf-8"))["members"]
    results = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(collect, r["code"]): r["code"] for r in members}
        for i, f in enumerate(as_completed(jobs), 1):
            try:
                row = f.result()
            except Exception as exc:
                row = {"code": jobs[f], "error": str(exc)[:250]}
            results.append(row)
            (ROOT / "shares-status.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if i % 10 == 0 or "error" in row:
                print(json.dumps({"completed": i, **row}, ensure_ascii=False), flush=True)
