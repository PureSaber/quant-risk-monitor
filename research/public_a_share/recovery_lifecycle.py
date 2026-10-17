"""Separate A-share lifecycle corrections from quote/return recovery."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pymupdf
import requests

from recovery_inventory import digest, write_json

CODES = ("000150", "000613", "000666", "002621", "002699", "600190", "600636", "600766", "600896")


def collect(base, out):
    out.mkdir(parents=True, exist_ok=True)
    used = sum(p.stat().st_size for p in base.rglob("*") if p.is_file())
    rows = []
    for code in CODES:
        choices = {}
        for name in ("old", "exposure/confirmation-inputs"):
            root = base / "inputs" / name
            data = json.loads(
                (root / f"raw/announcements/{code}-index.json").read_text(encoding="utf-8")
            )
            for row in data["all_records"]:
                if "摘牌" in row["announcementTitle"]:
                    choices[str(row["announcementId"])] = row
        for aid, row in choices.items():
            url = "https://static.cninfo.com.cn/" + row["adjunctUrl"].lstrip("/")
            path = out / (aid + ".pdf")
            if not path.exists():
                response = requests.get(url, timeout=(10, 30))
                response.raise_for_status()
                raw = response.content
                if not raw.startswith(b"%PDF") or len(raw) > 5 * 2**20:
                    raise ValueError("unexpected_lifecycle_document")
                if used + len(raw) > 25 * 2**30:
                    raise ValueError("data_root_budget_exceeded")
                path.write_bytes(raw)
                used += len(raw)
            with pymupdf.open(path) as document:
                text = "\n".join(p.get_text() for p in document)
            (out / (aid + ".txt")).write_text(text, encoding="utf-8")
            rows.append(
                {
                    "asset": code,
                    "announcement_id": aid,
                    "url": url,
                    "pdf": str(path),
                    "sha256": digest(path),
                    "published": pd.Timestamp(row["announcementTime"], unit="ms", tz="UTC")
                    .tz_convert("Asia/Shanghai")
                    .isoformat(),
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    write_json(out / "receipts.json", rows)


def state_overlay(statuses, events):
    """Past-state correction starts only once both fact and effective date are known."""
    result = pd.DataFrame("unknown", index=statuses.index, columns=statuses.columns)
    result[statuses.eq(0)] = "confirmed_suspension"
    result[statuses.eq(1)] = "observed_trading"
    for event in events:
        if not event.get("source_sha256") or not event.get("review_basis"):
            raise ValueError("unverified_lifecycle_event")
        known = pd.Timestamp(event["available_at"]).tz_convert("Asia/Shanghai").tz_localize(None)
        effective = pd.Timestamp(event["effective_date"])
        start = max(known, effective)
        # A future notice never turns earlier unknown periods into known input.
        result.loc[result.index >= start, event["asset"]] = "exited_a_share"
    return result


def apply_events(base, out, config):
    events = json.loads(config.read_text(encoding="utf-8"))
    receipts = {
        r["announcement_id"]: r
        for r in json.loads((out / "receipts.json").read_text(encoding="utf-8"))
    }
    for event in events:
        receipt = receipts[event["announcement_id"]]
        if event["asset"] != receipt["asset"] or digest(receipt["pdf"]) != event["source_sha256"]:
            raise ValueError("lifecycle_evidence_changed")
    source = base / "runs/main/source-trading-status.parquet"
    statuses = pd.read_parquet(source)
    overlay = state_overlay(statuses, events)
    overlay.to_parquet(out / "lifecycle-overlay.parquet")
    evaluation = overlay.loc["2023-01-03":"2026-09-30"]
    old = statuses.reindex(evaluation.index)
    rows = [
        {
            "asset": a,
            "before_unknown": int(old[a].isna().sum()),
            "after_unknown": int(evaluation[a].eq("unknown").sum()),
            "unknown_reclassified_as_exit": int(
                (old[a].isna() & evaluation[a].eq("exited_a_share")).sum()
            ),
        }
        for a in evaluation
    ]
    pd.DataFrame(rows).to_csv(out / "state-coverage.csv", index=False, encoding="utf-8-sig")
    write_json(
        out / "state-summary.json",
        {
            "source_sha256": digest(source),
            "events_sha256": digest(config),
            "before_unknown": int(old.isna().sum().sum()),
            "after_unknown": int(evaluation.eq("unknown").sum().sum()),
            "prices_filled": 0,
            "returns_filled": 0,
            "holdings_deleted": 0,
            "note": "Exited A-share venue is a state correction, not observed prices or permitted trading.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    args = parser.parse_args()
    if args.events:
        apply_events(args.base, args.out, args.events)
    else:
        collect(args.base, args.out)
