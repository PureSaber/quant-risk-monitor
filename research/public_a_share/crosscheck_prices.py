"""Independent public-source price comparison; never conditions the cohort."""

import hashlib
import json
import multiprocessing as mp
import socket
from datetime import datetime, timezone

import pandas as pd

from public_sources import ROOT


def collect(code):
    import baostock as bs

    socket.setdefaulttimeout(25)
    login = bs.login()
    if login.error_code != "0":
        raise ValueError(login.error_msg)
    folder = ROOT / "raw/price-crosscheck"
    folder.mkdir(exist_ok=True)
    for adjustment in ["3", "1"]:
        path = folder / f"{code}-{adjustment}.csv"
        if path.exists():
            continue
        result = bs.query_history_k_data_plus(
            ("sh." if code.startswith("6") else "sz.") + code,
            "date,code,close,pctChg,tradestatus",
            start_date="2021-01-01",
            end_date="2025-12-31",
            frequency="d",
            adjustflag=adjustment,
        )
        rows = []
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        if result.error_code != "0":
            raise ValueError(result.error_msg)
        pd.DataFrame(rows, columns=result.fields).to_csv(path, index=False)
        path.with_suffix(".receipt.json").write_text(
            json.dumps(
                {
                    "source": "BaoStock query_history_k_data_plus",
                    "code": code,
                    "adjustflag": adjustment,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "rows": len(rows),
                }
            ),
            encoding="utf8",
        )
    bs.logout()


def main():
    members = json.loads((ROOT / "cohort.json").read_text(encoding="utf8"))["members"]
    # Historical selection only; deliberately retain three later exits.
    codes = ["000150", "000613", "600190"] + [
        r["code"]
        for r in sorted(members, key=lambda r: hashlib.sha256(r["code"].encode()).hexdigest())[:5]
    ]
    for code in codes:
        worker = mp.Process(target=collect, args=(code,))
        worker.start()
        worker.join(75)
        if worker.is_alive():
            worker.terminate()
            worker.join()
        print(code, worker.exitcode, flush=True)
    records = []
    for code in codes:
        for ba, em in [("3", "0"), ("1", "2")]:
            path = ROOT / "raw/price-crosscheck" / f"{code}-{ba}.csv"
            if not path.exists():
                continue
            independent = pd.read_csv(path).set_index("date")["close"]
            payload = json.loads(
                (ROOT / "raw/prices" / f"{code}-{em}.json").read_text(encoding="utf8")
            )["data"]["klines"]
            primary = pd.Series({r.split(",")[0]: float(r.split(",")[2]) for r in payload})
            joined = pd.concat(
                [primary.rename("primary"), independent.rename("independent")], axis=1
            ).dropna()
            error = (joined.primary - joined.independent).abs()
            return_error = (
                joined.primary.pct_change(fill_method=None)
                - joined.independent.pct_change(fill_method=None)
            ).abs()
            records.append(
                {
                    "asset": code,
                    "adjustment": "raw" if ba == "3" else "back_adjusted",
                    "shared_dates": len(joined),
                    "max_price_absolute_difference": float(error.max()),
                    "median_return_absolute_difference": float(return_error.median()),
                    "p99_return_absolute_difference": float(return_error.quantile(0.99)),
                    "max_return_absolute_difference": float(return_error.max()),
                }
            )
    (ROOT / "clean").mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(ROOT / "clean/price-crosscheck.csv", index=False)
    print(pd.DataFrame(records).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
