"""Archive multiplicatively adjusted histories independently of EM additive HFQ."""

import hashlib
import json
import multiprocessing as mp
import socket
from datetime import datetime, timezone

import pandas as pd

from public_sources import ROOT


def one(code):
    import baostock as bs

    socket.setdefaulttimeout(25)
    folder = ROOT / "raw/bao-prices"
    folder.mkdir(exist_ok=True)
    path = folder / (code + ".csv")
    login = bs.login()
    if login.error_code != "0":
        raise ValueError(login.error_msg)
    result = bs.query_history_k_data_plus(
        ("sh." if code.startswith("6") else "sz.") + code,
        "date,code,close,preclose,pctChg,tradestatus,adjustflag",
        start_date="2019-01-01",
        end_date="2025-12-31",
        frequency="d",
        adjustflag="1",
    )
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0" or not rows:
        raise ValueError(f"{code}: {result.error_msg}; {len(rows)} rows")
    data = pd.DataFrame(rows, columns=result.fields)
    data.to_csv(path, index=False)
    path.with_suffix(".receipt.json").write_text(
        json.dumps(
            {
                "source": "BaoStock query_history_k_data_plus",
                "code": code,
                "start": "2019-01-01",
                "end": "2025-12-31",
                "adjustflag": "1",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": len(data),
            }
        ),
        encoding="utf8",
    )
    bs.logout()


def main():
    members = json.loads((ROOT / "cohort.json").read_text(encoding="utf8"))["members"]
    results = []
    for i, m in enumerate(members, 1):
        code = m["code"]
        path = ROOT / "raw/bao-prices" / (code + ".csv")
        receipt = path.with_suffix(".receipt.json")
        for attempt in range(3):
            if path.exists() and receipt.exists():
                meta = json.loads(receipt.read_text(encoding="utf8"))
                if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
                    raise ValueError("Changed BaoStock cache")
                break
            worker = mp.Process(target=one, args=(code,))
            worker.start()
            worker.join(65)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        valid = path.exists() and receipt.exists()
        results.append({"asset": code, "status": "ok" if valid else "failed"})
        (ROOT / "returns-status.json").write_text(json.dumps(results, indent=2), encoding="utf8")
        if i % 10 == 0 or not valid:
            print(json.dumps({"completed": i, "code": code, "valid": valid}), flush=True)
    if any(r["status"] != "ok" for r in results):
        raise ValueError("Incomplete multiplicative price archive")


if __name__ == "__main__":
    main()
