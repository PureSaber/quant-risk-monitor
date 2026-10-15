"""Original traded bars and evidenced distributions for the existing QExec ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from decimal import Decimal
from pathlib import Path

import pandas as pd

from study_v3_archive import digest, write_json

FIELDS = {
    "F006D": "实施方案公告日期",
    "F044V": "分红类型",
    "F011N": "转增比例",
    "F010N": "送股比例",
    "F012N": "派息比例",
    "F018D": "股权登记日",
    "F020D": "除权日",
    "F023D": "派息日",
    "F025D": "股份到账日",
    "F007V": "实施方案分红说明",
    "F001V": "报告时间",
}


def normalize_window(payload, asset, captured, start, end):
    from quant_data_kit.providers.corporate_actions import normalize_actions

    if payload.get("resultcode") != 200 or payload.get("count") != len(payload.get("records", [])):
        raise ValueError("incomplete corporate-action response")
    frame = pd.DataFrame(payload["records"]).rename(columns=FIELDS)
    if frame.empty:
        return normalize_actions(pd.DataFrame(columns=FIELDS.values()), asset, captured)
    frame["除权日"] = pd.to_datetime(frame["除权日"])
    announced = pd.to_datetime(frame["实施方案公告日期"])
    unknown = frame[frame["除权日"].isna() & announced.between(start, end)]
    if len(unknown):
        raise ValueError("unknown ex-date for an announcement in the execution window")
    frame = frame[frame["除权日"].between(start, end)].copy()
    rows = []
    for _, group in frame.groupby("除权日", sort=True):
        # Normalize individual source records first. Special + annual dividends
        # have the same entitlement share base; add cash and share increments.
        normalized = []
        for i in range(len(group)):
            row = normalize_actions(group.iloc[[i]], asset, captured)
            if not row.empty:
                normalized.append(row.iloc[0])
        if not normalized:
            continue
        first = normalized[0].to_dict()
        if len(normalized) > 1:
            kinds = group["分红类型"].tolist()
            if len(set(kinds)) != len(kinds) or not set(kinds) <= {
                "年度分红",
                "中期分红",
                "特别分红",
            }:
                raise ValueError("conflicting distribution records")
            for key in ("announced_date", "record_date", "pay_date"):
                if any(row[key] != first[key] for row in normalized):
                    raise ValueError("inconsistent entitlement dates")
            if sum(Decimal(row.share_ratio) != 1 for row in normalized) > 1:
                raise ValueError("multiple share distributions need independent review")
            first["cash_per_share"] = str(
                sum((Decimal(row.cash_per_share) for row in normalized), Decimal(0))
            )
            first["share_ratio"] = str(
                1 + sum((Decimal(row.share_ratio) - 1 for row in normalized), Decimal(0))
            )
            first["source_record"] = json.dumps(
                [json.loads(row.source_record) for row in normalized],
                ensure_ascii=False,
                sort_keys=True,
            )
            first["event_id"] = (
                "cninfo-combined:"
                + hashlib.sha256(first["source_record"].encode()).hexdigest()[:24]
            )
        rows.append(first)
    return pd.DataFrame(
        rows,
        columns=normalize_actions(pd.DataFrame(columns=FIELDS.values()), asset, captured).columns,
    )


def run(old, exposure, root, out):
    from a_share_multifactor.corporate_actions import action_events

    with (out / "panel.pkl").open("rb") as stream:
        panel = pickle.load(stream)["panel"]
    trading_status = pd.read_parquet(out / "source-trading-status.parquet")
    bars, catalog, actions, failures, evidence = [], [], [], [], []
    for asset in panel["assets"]:
        parts = []
        market_id = None
        for base in (old, exposure / "confirmation-inputs"):
            path = base / "raw/prices" / (asset + "-0.json")
            data = json.loads(path.read_text(encoding="utf8")).get("data")
            if not data:
                continue
            market_id = data["market"]
            frame = pd.DataFrame(
                [r.split(",") for r in data["klines"]],
                columns=[
                    "date",
                    "open",
                    "close",
                    "high",
                    "low",
                    "volume",
                    "amount",
                    "amplitude",
                    "pct_change",
                    "change",
                    "turnover",
                ],
            )
            frame["date"] = pd.to_datetime(frame.date)
            frame = frame[
                (frame.date.dt.year == 2026) if base != old else (frame.date.dt.year < 2026)
            ]
            parts.append(frame)
        frame = pd.concat(parts, ignore_index=True)
        frame = frame[frame.date.between("2023-01-01", "2026-09-30")].copy()
        frame = frame[trading_status[asset].reindex(frame.date).eq(1).to_numpy()].copy()
        for col in frame.columns.difference(["date"]):
            frame[col] = pd.to_numeric(frame[col], errors="raise")
        frame["volume"] *= 100  # Eastmoney historical kline volume is board lots.
        frame["symbol"], frame["adjustment"], frame["source"] = (
            asset,
            "none",
            "public.eastmoney.raw",
        )
        frame["composite_score"] = 0.0
        bars.append(frame)
        if market_id not in (0, 1):
            raise ValueError("unknown source-declared venue")
        catalog.append(
            {
                "symbol": asset,
                "asset_class": "equity",
                "product_type": "common_stock",
                "venue": "SSE" if market_id == 1 else "SZSE",
                "price_scale": 2,
                "price_tick": ".01",
                "quantity_step": 1,
                "lot_size": 100,
                "commission_rate": ".0003",
                "stamp_duty_rate": ".0005",
                "effective_from": "2022-12-01T00:00:00Z",
                "effective_to": "2026-10-01T00:00:00Z",
                "available_at": "2021-11-10T00:00:00Z",
                "scope": "research declared main-board rules; source-declared venue; not listing metadata",
            }
        )
        path = root / "corporate-actions" / asset / "response-00.json"
        receipt_paths = list(path.parent.glob("receipt.json")) or sorted(
            path.parent.glob("failure-*.json")
        )
        receipt = json.loads(receipt_paths[-1].read_text(encoding="utf8"))
        try:
            normalized = normalize_window(
                json.loads(path.read_text(encoding="utf8")),
                asset,
                receipt["captured_at"],
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2026-09-30"),
            )
            # Dates independently read from the issuer's original announcements.
            for ex_date, ann_id in (("2023-07-17", "1217254673"), ("2024-07-08", "1220505695")):
                if asset == "600188":
                    pdf = root / "action-evidence/raw/filings/600188" / (ann_id + ".pdf")
                    import pymupdf

                    with pymupdf.open(pdf) as document:
                        text = document[0].get_text()
                    day = pd.Timestamp(ex_date)
                    rendered_date = f"{day.year}/{day.month}/{day.day}"
                    if "新增无限售" not in text or text.count(rendered_date) < 3:
                        raise ValueError("issuer share-delivery evidence changed")
                    mask = normalized.ex_date.eq(pd.Timestamp(ex_date))
                    normalized.loc[mask, "shares_available_date"] = pd.Timestamp(ex_date)
                    evidence.append(
                        {
                            "asset": asset,
                            "ex_date": ex_date,
                            "field": "shares_available_date",
                            "value": ex_date,
                            "source": str(pdf),
                            "sha256": digest(pdf),
                            "page": 1,
                        }
                    )
            actions.append(normalized)
        except (ValueError, KeyError) as exc:
            failures.append({"asset": asset, "stage": "actions", "reason": str(exc)})
    raw = pd.concat(bars, ignore_index=True).sort_values(["date", "symbol"])
    normalized = pd.concat(actions, ignore_index=True)
    from study_v3_action_repairs import repair

    normalized, repairs, guards = repair(normalized, root)
    evidence.extend(repairs)
    write_json(out / "execution-input-guards.json", guards)
    raw.to_parquet(out / "execution-bars.parquet", index=False)
    normalized.to_parquet(out / "execution-actions.parquet", index=False)
    pd.DataFrame(catalog).to_csv(out / "execution-catalog.csv", index=False)
    events = []
    for asset, frame in raw.groupby("symbol"):
        adjusted = frame[["date", "symbol"]].copy()
        adjusted["close"] = panel["prices"]["adjusted_close"][asset].reindex(frame.date).to_numpy()
        # Empty action feeds are accepted only if the independently observed
        # adjustment factor supports no unrecorded cash/share events.
        try:
            events.extend(
                action_events(
                    normalized[normalized.symbol.eq(asset)],
                    frame,
                    adjusted,
                    frame.date.min(),
                    frame.date.max(),
                )
            )
        except ValueError as exc:
            failures.append(
                {"asset": asset, "stage": "adjustment_reconciliation", "reason": str(exc)}
            )
    with (out / "corporate-action-events.pkl").open("wb") as stream:
        pickle.dump(
            tuple(sorted(events, key=lambda e: (e.available_at, e.instrument_id))),
            stream,
            protocol=5,
        )
    write_json(out / "execution-input-issues.json", failures)
    write_json(out / "corporate-action-repair-evidence.json", evidence)
    print(
        json.dumps(
            {
                "raw_bars": len(raw),
                "actions": len(normalized),
                "events": len(events),
                "issues": failures,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("old", "exposure", "root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.old, args.exposure, args.root, args.out)
