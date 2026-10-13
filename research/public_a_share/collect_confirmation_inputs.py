"""Archive fresh-period inputs separately; never overwrite the v1 evidence.

Collection is outcome-blind. The scoring driver must bind its selected model
before reading these files. Missing/delisted members stay in the fixed cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def bao(root, code, start, end, flag, label, *, reuse_session=False):
    import baostock as bs

    path = root / "raw" / label / (code + ".csv")
    receipt = path.with_suffix(".receipt.json")
    if path.exists() and receipt.exists():
        if sha(path) != json.loads(receipt.read_text(encoding="utf8"))["sha256"]:
            raise ValueError("changed archived market input")
        return
    socket.setdefaulttimeout(25)
    if not reuse_session:
        response = bs.login()
        if response.error_code != "0":
            raise ValueError(response.error_msg)
    try:
        import baostock.common.context as context

        class BoundedSocket:
            _bounded = True

            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.started = time.monotonic()

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def send(self, data):
                self.started = time.monotonic()
                return self.wrapped.send(data)

            def recv(self, size):
                if time.monotonic() - self.started > 60:
                    raise TimeoutError("BaoStock response exceeded 60 seconds")
                result = self.wrapped.recv(size)
                if not result:
                    raise ConnectionError("BaoStock closed an incomplete response")
                return result

        if not getattr(context.default_socket, "_bounded", False):
            context.default_socket = BoundedSocket(context.default_socket)
        result = bs.query_history_k_data_plus(
            ("sh." if code.startswith("6") else "sz.") + code,
            "date,code,close,preclose,pctChg,tradestatus,adjustflag,volume,amount,turn",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag=flag,
        )
        rows = []
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        if result.error_code != "0":
            raise ValueError(result.error_msg)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=result.fields).to_csv(path, index=False)
        save_json(
            receipt,
            {
                "source": "BaoStock query_history_k_data_plus",
                "code": code,
                "start": start,
                "end": end,
                "adjustflag": flag,
                "rows": len(rows),
                "sha256": sha(path),
                "captured_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    finally:
        if not reuse_session:
            bs.logout()


def collect_one(args):
    old, target, code = args
    os.environ["PUBLIC_PIT_ROOT"] = str(target)
    import akshare as ak
    import requests

    import filings
    import original_financials
    import public_sources

    root = Path(target)
    for module in (public_sources, filings, original_financials):
        module.ROOT = root
    original_request = requests.sessions.Session.request

    def bounded(self, *args, **kwargs):
        kwargs.setdefault("timeout", (10, 35))
        if kwargs["timeout"] is None:
            kwargs["timeout"] = (10, 35)
        return original_request(self, *args, **kwargs)

    requests.sessions.Session.request = bounded
    errors = []
    try:
        public_sources.fetch(
            f"prices/{code}-0.json",
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            params={
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "ut": "7eea3edcaed734bea9cbfc24409ed989",
                "klt": "101",
                "fqt": "0",
                "secid": ("1." if code.startswith("6") else "0.") + code,
                "beg": "20251201",
                "end": "20260930",
            },
        )
    except Exception as exc:
        errors.append({"stage": "raw_prices", "reason": str(exc)[:250]})
    try:
        previous = json.loads(
            (Path(old) / "raw/announcements" / (code + "-index.json")).read_text(encoding="utf8")
        )
        org = next((r["orgId"] for r in reversed(previous["all_records"]) if r.get("orgId")), "")
        rows = filings.announcements(code, org, start="2026-01-01", end="2026-09-30")
        chosen = [
            r
            for r in rows
            if re.search(r"20\d{2}年.*(?:年度报告|半年度报告|季度报告)", r["announcementTitle"])
            and not re.search(
                r"取消|英文|摘要|提示|董事|监事|核查|审核|专项|独立|审计|更正.*公告|修订.*公告",
                r["announcementTitle"],
            )
        ]
        chosen = [
            r
            for r in chosen
            if "正文" not in r["announcementTitle"]
            or not any(
                s["announcementTime"] == r["announcementTime"]
                and "全文" in s["announcementTitle"]
                and s["announcementTitle"].split("报告")[0]
                == r["announcementTitle"].split("报告")[0]
                for s in chosen
            )
        ]
        reports, failures = [], []
        for row in chosen:
            try:
                reports.append(filings.extract_pdf(row))
            except Exception as exc:
                failures.append({"id": str(row["announcementId"]), "reason": str(exc)[:250]})
        save_json(
            root / "raw/announcements" / (code + "-index.json"),
            {
                "code": code,
                "report_count": len(chosen),
                "reports": reports,
                "errors": failures,
                "all_records": rows,
            },
        )
        if failures:
            raise ValueError(f"incomplete original archive: {len(failures)} reports")
        facts, missing = original_financials.extract_member(
            {"code": code}, first_period="20181231", last_period="20260930"
        )
        save_json(root / "extracted" / (code + ".json"), {"facts": facts, "missing": missing})
    except Exception as exc:
        errors.append({"stage": "filings", "reason": str(exc)[:250]})
    try:
        path = root / "raw/shares" / (code + ".csv")
        receipt = path.with_suffix(".receipt.json")
        if path.exists() and receipt.exists():
            if sha(path) != json.loads(receipt.read_text(encoding="utf8"))["sha256"]:
                raise ValueError("changed capital archive")
        else:
            data = ak.stock_share_change_cninfo(
                symbol=code, start_date="20250101", end_date="20260930"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(path, index=False, encoding="utf8")
            save_json(
                receipt,
                {
                    "source": "https://webapi.cninfo.com.cn/api/stock/p_stock2215",
                    "start": "2025-01-01",
                    "end": "2026-09-30",
                    "sha256": sha(path),
                    "rows": len(data),
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                },
            )
    except Exception as exc:
        errors.append({"stage": "capital", "reason": str(exc)[:250]})
    requests.sessions.Session.request = original_request
    return {"code": code, "errors": errors}


def main():
    sys.stdout.reconfigure(encoding="utf8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    members = json.loads((args.old_root / "cohort.json").read_text(encoding="utf8"))["members"]
    members = members[: args.limit]
    # One connection for sequential market requests avoids competing server
    # sessions and repeated login overhead. Document work has a separate pool.
    import baostock as bs

    socket.setdefaulttimeout(25)
    market_errors = {}
    login = bs.login()
    if login.error_code != "0":
        raise ValueError(login.error_msg)
    try:
        for i, member in enumerate(members, 1):
            code = member["code"]
            for attempt in range(3):
                try:
                    bao(
                        args.out,
                        code,
                        "2023-06-01",
                        "2023-06-30",
                        "3",
                        "turnover-check",
                        reuse_session=True,
                    )
                    bao(
                        args.out,
                        code,
                        "2025-12-01",
                        "2026-09-30",
                        "1",
                        "bao-prices",
                        reuse_session=True,
                    )
                    break
                except Exception as exc:
                    if attempt == 2:
                        market_errors[code] = str(exc)[:250]
                    else:
                        bs.logout()
                        login = bs.login()
                        if login.error_code != "0":
                            raise ValueError(login.error_msg) from exc
            if i % 10 == 0:
                print(json.dumps({"market_issuers": i, "errors": len(market_errors)}), flush=True)
    finally:
        bs.logout()
    results = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(collect_one, (args.old_root, args.out, m["code"]))
            for m in members[: args.limit]
        ]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row["code"] in market_errors:
                row["errors"].append({"stage": "bao", "reason": market_errors[row["code"]]})
            results.append(row)
            save_json(args.out / "collection-status.json", results)
            if i % 10 == 0 or row["errors"]:
                print(json.dumps({"completed": i, **row}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
