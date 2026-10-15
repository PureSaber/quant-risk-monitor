"""Incremental public corporate-action capture, preserving original responses."""

from __future__ import annotations

import argparse
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from study_v3_archive import digest, write_json


def collect(cohort, root):
    import akshare as ak
    import requests
    from quant_data_kit.providers._network import configure_network
    from quant_data_kit.providers.corporate_actions import normalize_actions

    configure_network()
    root.mkdir(parents=True, exist_ok=True)
    # All downloads in this job are bounded responses for 200 fixed companies.
    # Reserve 1 GiB before collecting; never delete evidence to make room.
    if shutil.disk_usage(root).free < 26 * 1024**3:
        raise ValueError("D drive reserve insufficient for incremental collection")
    state = threading.local()
    request = requests.sessions.Session.request

    def capture(session, method, url, **kwargs):
        kwargs.setdefault("timeout", (10, 30))
        response = request(session, method, url, **kwargs)
        response.raise_for_status()
        state.responses.append((url, response.content))
        return response

    def one(asset):
        folder = root / asset
        folder.mkdir(exist_ok=True)
        receipt_path = folder / "receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf8"))
            for name, sha in receipt["files"].items():
                if digest(folder / name) != sha:
                    raise ValueError("captured corporate-action file changed: " + asset)
            return receipt
        if any(folder.iterdir()):
            raise ValueError(
                "failed/incomplete capture is immutable; retry this asset in a new capture root: "
                + asset
            )
        captured = datetime.now(timezone.utc).isoformat()
        state.responses = []
        try:
            frame = ak.stock_dividend_cninfo(symbol=asset)
            actions = normalize_actions(frame, asset, captured)
            actions.to_parquet(folder / "normalized.parquet", index=False)
            frame.to_parquet(folder / "source-table.parquet", index=False)
            status, reason = "captured", None
        except Exception as exc:
            status, reason = "failed", type(exc).__name__ + ": " + str(exc)
        sources = []
        for i, (url, body) in enumerate(state.responses):
            path = folder / f"response-{i:02d}.json"
            path.write_bytes(body)
            sources.append({"url": url, "file": path.name, "sha256": digest(path)})
        files = {p.name: digest(p) for p in folder.iterdir() if p.is_file()}
        receipt = {
            "asset": asset,
            "status": status,
            "reason": reason,
            "captured_at": captured,
            "sources": sources,
            "files": files,
            "availability": "original announcement fields; capture time kept separately",
        }
        # Failed attempts remain separate, so retry cannot erase their evidence.
        if status == "captured":
            write_json(receipt_path, receipt)
        else:
            write_json(folder / ("failure-" + captured.replace(":", "-") + ".json"), receipt)
        return receipt

    requests.sessions.Session.request = capture
    receipts = []
    try:
        # AKShare's bundled MiniRacer initialization is not thread-safe on
        # Windows. Serialize this provider; a crash must not lose receipts.
        with ThreadPoolExecutor(max_workers=1) as pool:
            tasks = [pool.submit(one, member["code"]) for member in cohort["members"]]
            for task in as_completed(tasks):
                receipts.append(task.result())
                if len(receipts) % 20 == 0:
                    print(
                        json.dumps(
                            {
                                "actions_collected": len(receipts),
                                "failures": sum(r["status"] != "captured" for r in receipts),
                            }
                        ),
                        flush=True,
                    )
    finally:
        requests.sessions.Session.request = request
    write_json(root / "manifest.json", sorted(receipts, key=lambda r: r["asset"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    collect(json.loads(args.cohort.read_text(encoding="utf8")), args.root)
