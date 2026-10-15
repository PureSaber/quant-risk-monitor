"""Auditable daily input quality and the ten previously incomplete windows."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from study_v3_archive import digest, write_json
from study_v3_risk import eligible_inputs


def endpoint_audit(old, exposure, out, panel):
    with (exposure / "confirmation/histories.pkl").open("rb") as stream:
        history = pickle.load(stream)
    forecasts = pd.read_csv(exposure / "confirmation/forecasts.csv")
    failures = forecasts.loc[forecasts["return"].isna(), ["date", "end"]].drop_duplicates()
    rows = []
    for case in failures.itertuples():
        day, end = pd.Timestamp(case.date), pd.Timestamp(case.end)
        assets, _ = eligible_inputs(panel, history, panel["dates"].get_loc(day), day)
        values = panel["prices"]["adjusted_close"]
        for asset in assets[(values.loc[end, assets] / values.loc[day, assets]).isna()]:
            path = exposure / "confirmation-inputs/raw/bao-prices" / (asset + ".csv")
            source = pd.read_csv(path, dtype={"date": str})
            for date in (case.date, case.end):
                if pd.notna(values.at[pd.Timestamp(date), asset]):
                    continue
                match = source.loc[source.date.eq(date)]
                suspended = len(match) == 1 and match.iloc[0].tradestatus == 0
                rows.append(
                    {
                        "forecast": case.date,
                        "end": case.end,
                        "asset": asset,
                        "missing_at": date,
                        "classification": "source_confirmed_suspension"
                        if suspended
                        else "unresolved_missing_quote",
                        "source_file": str(path),
                        "source_sha256": digest(path),
                        "source_row": json.loads(match.to_json(orient="records")),
                        "repair": "none; no traded endpoint price; return remains unobserved",
                    }
                )
    write_json(out / "prior-ten-window-audit.json", {"windows": len(failures), "endpoints": rows})


def financial_quality(old, exposure, out, panel):
    path = exposure / "confirmation/merged-inputs/clean/financial_facts.csv"
    facts = pd.read_csv(path, dtype={"asset": str, "report_period": str, "announcement_id": str})
    facts["available_at"] = pd.to_datetime(facts.available_at, utc=True)
    receipts = {}
    for root in (old, exposure, exposure / "confirmation-inputs"):
        for receipt in (root / "raw/filings").glob("*/*.pdf.receipt.json"):
            data = json.loads(receipt.read_text(encoding="utf8"))
            key = receipt.parent.name, receipt.name.split(".")[0]
            receipts[key] = data
    facts["captured_at"] = [
        receipts.get((r.asset, r.announcement_id), {}).get("captured_at")
        for r in facts.itertuples()
    ]
    facts["original_pdf_sha256"] = [
        receipts.get((r.asset, r.announcement_id), {}).get("sha256") for r in facts.itertuples()
    ]
    facts.to_parquet(out / "financial-vintages.parquet", index=False)
    rows = []
    dates = panel["dates"][panel["dates"] >= "2021-11-11"]
    for asset in panel["assets"]:
        events = list(
            facts[facts.asset.eq(asset)]
            .sort_values(["available_at", "report_period", "revision_rank"])
            .itertuples()
        )
        known, cursor = {}, 0
        for day in dates:
            cutoff = day.tz_localize("Asia/Shanghai")
            while cursor < len(events) and events[cursor].available_at <= cutoff:
                event = events[cursor]
                key = event.field, event.report_period
                known[key] = event
                cursor += 1
            periods = [p for f, p in known if f in {"book_equity", "profit_ytd"}]
            latest = max(periods) if periods else None
            age = (day - pd.Timestamp(latest)).days if latest else None
            rows.append(
                {
                    "date": day,
                    "asset": asset,
                    "latest_report_period": latest,
                    "latest_report_age_days": age,
                    "stale_financial": age is not None and age > 550,
                    "no_mature_financial": latest is None,
                    "known_revision_rows": sum(
                        getattr(e, "revision_rank", 0) > 0 for e in known.values()
                    ),
                }
            )
    pd.DataFrame(rows).to_parquet(out / "financial-daily-quality.parquet", index=False)
    write_json(
        out / "financial-quality-summary.json",
        {
            "facts": len(facts),
            "original_capture_missing": int(facts.captured_at.isna().sum()),
            "source_sha256": digest(path),
            "availability_rule": "original disclosure plus producer lag; actual capture recorded separately; retrospective reconstruction",
            "revision_rows": int((facts.revision_rank > 0).sum()),
        },
    )


def cache_quality(out):
    rows = []
    for path in sorted((out / "daily-cache").glob("*.json")):
        envelope = json.loads(path.read_text(encoding="utf8"))
        from quant_risk_monitor.exposure_cache import _digest

        if _digest(envelope["payload"]) != envelope["sha256"]:
            raise ValueError("corrupt exposure cache: " + path.name)
        snap = envelope["payload"]["snapshot"]
        config = envelope["payload"]["binding"]["configuration"]
        audit = snap["provenance"]["descriptor_processing"]
        for asset in snap["assets"]:
            imputed = [name for name, info in audit.items() if asset in info.get("imputed", {})]
            rows.append(
                {
                    "date": snap["effective_at"],
                    "asset": asset,
                    "source": snap["source"],
                    "definition": config["definition_version"],
                    "configuration_sha256": _digest(config),
                    "descriptors": len(audit),
                    "imputed_count": len(imputed),
                    "imputed_fraction": len(imputed) / len(audit),
                    "imputed_descriptors": ",".join(imputed),
                    "cache_key": path.stem,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "imputation-daily-quality.parquet", index=False)


def run(old, exposure, out):
    with (out / "panel.pkl").open("rb") as stream:
        panel = pickle.load(stream)["panel"]
    endpoint_audit(old, exposure, out, panel)
    financial_quality(old, exposure, out, panel)
    cache_quality(out)
    daily = pd.read_parquet(out / "daily-quality.parquet")
    basic_path = out.parents[1] / "inputs/stock-basic.json"
    if basic_path.exists():
        data = json.loads(basic_path.read_text(encoding="utf8"))
        basic = pd.DataFrame(data["rows"], columns=data["fields"])
        basic["asset"] = basic.code.str.split(".").str[-1]
        basic = basic[basic.asset.isin(panel["assets"])].copy()
        basic["captured_at"] = data["captured_at"]
        basic["historical_use"] = "quality audit only; not used to exclude future delistings"
        basic.to_parquet(out / "listing-exit-audit.parquet", index=False)
    summary = daily.groupby("asset").agg(
        days=("date", "size"),
        observed_quotes=("quote_observed", "sum"),
        suspension_days=("observed_suspension", "sum"),
        unknown_status=("trading_status_unknown", "sum"),
        short_history_days=("short_return_history", "sum"),
        classification_changes=("industry_changed", "sum"),
    )
    summary["raw_quote_coverage"] = summary.observed_quotes / summary.days
    summary.to_csv(out / "company-quality.csv")
    print(json.dumps({"quality_complete": True, "companies": len(summary)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("old", "exposure", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.old, args.exposure, args.out)
