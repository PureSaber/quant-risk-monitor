"""Explicit, source-bound corrections; never infer a missing payment date."""

from __future__ import annotations

import json
import re
from decimal import Decimal

import pandas as pd
import pymupdf

from study_v3_archive import digest


def repair(actions, root):
    result = actions.copy()
    evidence = []
    for i, row in result.iterrows():
        if Decimal(row.share_ratio) == 1 or pd.notna(row.shares_available_date):
            continue
        for path in sorted((root / "bao-dividends").glob(row.symbol + "-*.json")):
            payload = json.loads(path.read_text(encoding="utf8"))
            if payload["error_code"] != "0":
                continue
            for values in payload["rows"]:
                record = dict(zip(payload["fields"], values))
                if record["dividOperateDate"] != str(row.ex_date.date()):
                    continue
                cash = Decimal(record["dividCashPsBeforeTax"] or "0")
                shares = (
                    1
                    + Decimal(record["dividStocksPs"] or "0")
                    + Decimal(record["dividReserveToStockPs"] or "0")
                )
                date = record["dividStockMarketDate"]
                if (
                    cash != Decimal(row.cash_per_share)
                    or shares != Decimal(row.share_ratio)
                    or not date
                ):
                    continue
                result.loc[i, "shares_available_date"] = pd.Timestamp(date)
                evidence.append(
                    {
                        "asset": row.symbol,
                        "ex_date": str(row.ex_date.date()),
                        "field": "shares_available_date",
                        "value": date,
                        "source": str(path),
                        "sha256": digest(path),
                    }
                )

    # Read the actual exchange reference amounts from the reviewed issuer text,
    # distinct from the cash that the native account receives.
    references = [
        ("000631", "2023-05-17", "1216767461", "0.2475791"),
        ("000703", "2026-09-18", "1225559021", "0.8164176"),
        ("601666", "2025-06-09", "1223713319", "0.57"),
    ]
    for asset, day, identity, value in references:
        pdf = root / "action-evidence/raw/filings" / asset / (identity + ".pdf")
        with pymupdf.open(pdf) as document:
            text = re.sub(r"\s+", "", "".join(page.get_text() for page in document))
        if value + "元/股" not in text or "除权" not in text:
            raise ValueError("reviewed exchange-reference evidence changed")
        mask = result.symbol.eq(asset) & result.ex_date.eq(pd.Timestamp(day))
        if mask.sum() != 1:
            raise ValueError("exchange reference does not identify one action")
        result.loc[mask, "adjustment_cash_per_share"] = value
        result.loc[mask, "adjustment_reference_source"] = str(pdf)
        result.loc[mask, "adjustment_reference_sha256"] = digest(pdf)
        evidence.append(
            {
                "asset": asset,
                "ex_date": day,
                "field": "adjustment_cash_per_share",
                "value": value,
                "source": str(pdf),
                "sha256": digest(pdf),
            }
        )

    # These issuer distributions were absent from CNInfo's structured feed.
    declarations = [
        ("000666", "1217265503", "2023-07-14", "2023-07-17", "0.065"),
        ("600636", "1217042790", "2023-06-20", "2023-06-21", "0.10"),
        ("600636", "1220347479", "2024-06-19", "2024-06-20", "0.04"),
        ("600636", "1223859410", "2025-06-18", "2025-06-19", "0.04"),
        ("600636", "1224668269", "2025-09-24", "2025-09-25", "0.01"),
    ]
    manifest = json.loads(
        (root / "action-evidence/supplemental-manifest.json").read_text(encoding="utf8")
    )
    metadata = {
        str(row["announcementId"]): row for company in manifest for row in company["announcements"]
    }
    added = []
    for asset, identity, record_day, ex_day, cash in declarations:
        if (result.symbol.eq(asset) & result.ex_date.eq(pd.Timestamp(ex_day))).any():
            continue
        pdf = root / "action-evidence/raw/filings" / asset / (identity + ".pdf")
        receipt = json.loads(pdf.with_suffix(".pdf.receipt.json").read_text(encoding="utf8"))
        if digest(pdf) != receipt["sha256"]:
            raise ValueError("issuer evidence digest mismatch")
        with pymupdf.open(pdf) as document:
            text = re.sub(r"\s+", "", "".join(page.get_text() for page in document))
        day = pd.Timestamp(ex_day)
        source_date = (
            f"{day.year}/{day.month}/{day.day}"
            if asset == "600636"
            else f"{day.year}年{day.month}月{day.day}日"
        )
        if source_date not in text or (cash + "元" not in text and asset != "000666"):
            raise ValueError("reviewed issuer payment text changed")
        published = (
            pd.Timestamp(metadata[identity]["announcementTime"], unit="ms", tz="UTC")
            .tz_convert("Asia/Shanghai")
            .tz_localize(None)
            .normalize()
        )
        row = {
            "event_id": "issuer-pdf:" + identity,
            "symbol": asset,
            "announced_date": published,
            "record_date": pd.Timestamp(record_day),
            "ex_date": day,
            "pay_date": day,
            "shares_available_date": pd.NaT,
            "cash_per_share": cash,
            "share_ratio": "1",
            "source": receipt["url"],
            "captured_at": receipt["captured_at"],
            "source_record": json.dumps({"pdf_sha256": receipt["sha256"], "reviewed": True}),
        }
        added.append(row)
        evidence.append(
            {
                "asset": asset,
                "ex_date": ex_day,
                "field": "missing_cash_distribution",
                "value": cash,
                "source": str(pdf),
                "sha256": digest(pdf),
            }
        )
    result = pd.concat([result, pd.DataFrame(added)], ignore_index=True)
    guards = []
    reorganizations = [
        ("600165", "2025-12-11", "1224862404", "1.20", "不向原股东进行分配", None),
        ("603007", "2024-12-26", "1222129931", "7.11", "每10股转增约1.55346股", "1.155346"),
    ]
    for asset, day, identity, adjustment, phrase, entitlement in reorganizations:
        pdf = root / "action-evidence/raw/filings" / asset / (identity + ".pdf")
        with pymupdf.open(pdf) as document:
            text = re.sub(r"\s+", "", "".join(page.get_text() for page in document))
        if phrase not in text:
            raise ValueError("reviewed reorganization entitlement evidence changed")
        mask = result.symbol.eq(asset) & result.ex_date.eq(pd.Timestamp(day))
        if mask.sum() != 1:
            raise ValueError("reorganization does not identify one source action")
        result.loc[mask, "share_ratio"] = "1"
        result.loc[mask, "adjustment_cash_per_share"] = adjustment
        result.loc[mask, "adjustment_share_ratio"] = "1"
        result.loc[mask, "adjustment_reference_source"] = str(pdf)
        result.loc[mask, "adjustment_reference_sha256"] = digest(pdf)
        result.loc[mask, "reference_only"] = True
        result.loc[mask, "unresolved_holder_share_ratio"] = entitlement
        evidence.append(
            {
                "asset": asset,
                "ex_date": day,
                "field": "reorganization_entitlement",
                "source": str(pdf),
                "sha256": digest(pdf),
                "holder_share_ratio": entitlement or "1",
                "ledger_admission": "blocked_when_held"
                if entitlement
                else "no_distribution_to_existing_holders",
            }
        )
        if entitlement:
            guards.append(
                {
                    "asset": asset,
                    "available_at": day + "T09:00:00+08:00",
                    "source": str(pdf),
                    "sha256": digest(pdf),
                    "reason": "Reorganization gives ordinary holders ratio 1.155346, not structured-feed ratio 2.155346; actual delivery and fractional allocation require account entitlement records. No fabricated shares or cash.",
                }
            )
    return result, evidence, guards
