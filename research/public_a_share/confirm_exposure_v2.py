"""Score frozen exposure definitions on newly collected 2026 public evidence."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

import build_panel
import exposure_v2_study as study
from model_bootstrap import FactorModelConfig

# isort: split

from quant_risk_monitor.style_descriptors import (
    additional_market_descriptors,
    annual_growth_descriptors,
    paired_market_descriptors,
)


def read_market(path):
    payload = json.loads(path.read_text(encoding="utf8"))["data"]
    return pd.Series(
        {pd.Timestamp(r.split(",")[0]): float(r.split(",")[2]) for r in payload["klines"]}
    ).sort_index()


def merge_prices(old_panel, root, market):
    additions, audit = [], []
    for asset in old_panel["assets"]:
        raw = json.loads(
            (root / "raw/prices" / (asset + "-0.json")).read_text(encoding="utf8")
        ).get("data")
        rows = [] if not raw else [r.split(",") for r in raw.get("klines", [])]
        frame = pd.DataFrame(
            rows,
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
        frame = frame.set_index("date").apply(pd.to_numeric, errors="raise")
        if frame.index.has_duplicates:
            raise ValueError("duplicate confirmation quote")
        path = root / "raw/bao-prices" / (asset + ".csv")
        receipt = json.loads(path.with_suffix(".receipt.json").read_text(encoding="utf8"))
        if study.digest(path) != receipt["sha256"]:
            raise ValueError("changed confirmation adjusted quotes")
        bao = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        adjusted = bao.close.where(bao.tradestatus.eq(1))
        if adjusted.index.has_duplicates or not frame.index.isin(adjusted.index).all():
            raise ValueError("adjusted quote calendar does not cover raw quotes")
        prior = old_panel["prices"]["adjusted_close"][asset]
        overlap = pd.concat([prior.rename("old"), adjusted.rename("new")], axis=1).dropna()
        future = frame[frame.index.year == 2026]
        scale, discrepancy = 1.0, None
        if len(future):
            if len(overlap) < 5:
                raise ValueError(f"insufficient adjustment bridge for {asset}")
            ratios = overlap.old / overlap.new
            scale = float(ratios.median())
            discrepancy = float((ratios / scale - 1).abs().max())
            if discrepancy > 1e-5:
                raise ValueError(
                    f"nonconstant adjusted history revision for {asset}: {discrepancy}"
                )
        frame["adjusted_close"] = adjusted.reindex(frame.index) * scale
        additions.append(frame.loc[frame.index.year == 2026].assign(asset=asset).reset_index())
        audit.append(
            {
                "asset": asset,
                "sessions_2026": len(future),
                "adjustment_scale": scale,
                "bridge_max_relative_difference": discrepancy,
            }
        )
    long = pd.concat(additions, ignore_index=True)
    dates = market.index
    prices = {}
    for name in ("close", "adjusted_close", "volume", "turnover"):
        new = long.pivot(index="date", columns="asset", values=name)
        prices[name] = pd.concat([old_panel["prices"][name], new]).reindex(
            index=dates, columns=old_panel["assets"]
        )
    prices["returns"] = prices["adjusted_close"].pct_change(fill_method=None)
    return prices, audit


def prepare(old, root, out, selection, protocol):
    status = json.loads((root / "collection-status.json").read_text(encoding="utf8"))
    members = json.loads((old / "cohort.json").read_text(encoding="utf8"))["members"]
    if {r["code"] for r in status} != {m["code"] for m in members}:
        raise ValueError("confirmation collection does not cover the fixed historical cohort")
    disallowed = [r for r in status if any(e["stage"] != "capital" for e in r["errors"])]
    if disallowed:
        raise ValueError(
            f"unresolved confirmation collection failures: {[r['code'] for r in disallowed]}"
        )
    for receipt in (root / "raw").rglob("*.receipt.json"):
        meta = json.loads(receipt.read_text(encoding="utf8"))
        base = receipt.with_name(receipt.name.removesuffix(".receipt.json"))
        path = base if base.exists() else base.with_suffix(".csv")
        if not path.exists() or study.digest(path) != meta["sha256"]:
            raise ValueError(f"confirmation source receipt mismatch: {receipt}")
    with (old / "clean/panel.pkl").open("rb") as stream:
        panel = pickle.load(stream)
    old_market = read_market(old / "raw/benchmark/csi300.json")
    new_market = read_market(root / "raw/benchmark/csi300.json")
    market = pd.concat([old_market, new_market[new_market.index.year == 2026]])
    if market.index.has_duplicates or market.index.max() != pd.Timestamp(
        protocol["confirmation_end"]
    ):
        raise ValueError("benchmark does not cover the frozen confirmation cutoff")
    prices, bridges = merge_prices(panel, root, market)
    for row in status:
        if row["errors"] and prices["close"].loc["2026", row["code"]].notna().any():
            raise ValueError(f"missing capital history for active issuer {row['code']}")
    extra = []
    for member in members:
        payload = json.loads(
            (root / "extracted" / (member["code"] + ".json")).read_text(encoding="utf8")
        )
        for row in payload["facts"]:
            path = root / row["document"]
            text = path.with_suffix(".logical.txt")
            if study.digest(text) != row["text_sha256"]:
                raise ValueError("extracted confirmation fact has changed original text")
        extra.extend(payload["facts"])
    facts = pd.concat(
        [
            pd.read_csv(
                old / "clean/financial_facts.csv", dtype={"asset": str, "report_period": str}
            ),
            pd.read_csv(
                out.parent / "annual-warmup-facts.csv", dtype={"asset": str, "report_period": str}
            ),
            pd.DataFrame(extra),
        ],
        ignore_index=True,
    )
    facts["revision_rank"] = facts.revision_rank.fillna(0)
    facts["announcement_id"] = facts.announcement_id.astype(str)
    facts = facts.sort_values(
        ["asset", "field", "report_period", "available_at", "revision_rank", "announcement_id"]
    )
    facts = facts.drop_duplicates(["asset", "field", "report_period", "available_at"], keep="last")
    facts.to_csv(root / "clean/financial_facts.csv", index=False)
    # Materialize original + new capital events in a separate build directory.
    merged_root = out / "merged-inputs"
    (merged_root / "clean").mkdir(parents=True, exist_ok=True)
    facts.to_csv(merged_root / "clean/financial_facts.csv", index=False)
    (merged_root / "clean/industry_history.csv").write_bytes(
        (root / "clean/industry_history.csv").read_bytes()
    )
    (merged_root / "raw/shares").mkdir(parents=True, exist_ok=True)
    for asset in panel["assets"]:
        pieces = [
            pd.read_csv(p, dtype={"证券代码": str})
            for p in (old / "raw/shares" / (asset + ".csv"), root / "raw/shares" / (asset + ".csv"))
            if p.exists()
        ]
        if pieces:
            pd.concat(pieces, ignore_index=True).drop_duplicates().to_csv(
                merged_root / "raw/shares" / (asset + ".csv"), index=False
            )
    previous_root = build_panel.ROOT
    build_panel.ROOT = merged_root
    try:
        dates, assets = prices["close"].index, panel["assets"]
        financial = build_panel.financial_daily(dates, assets)
        shares = build_panel.capital_daily(dates, assets, financial["shares"])
        groups = build_panel.industry_daily(
            dates, assets, protocol["old_protocol"]["sector_mapping"]
        )
        returns = market.pct_change(fill_method=None).reindex(dates)
        caps, descriptors = build_panel.descriptors_from_history(prices, financial, shares, returns)
        combined = {
            "dates": dates,
            "assets": assets,
            "prices": prices,
            "caps": caps,
            "groups": groups,
            "descriptors": descriptors,
        }
        corrected, _ = paired_market_descriptors(prices["returns"], returns)
        extra_market = additional_market_descriptors(prices["returns"], prices["turnover"])
        growth, _ = annual_growth_descriptors(facts, dates, assets)
        v2 = {**combined, "descriptors": {**descriptors, **corrected, **extra_market, **growth}}
    finally:
        build_panel.ROOT = previous_root
    manifest = {
        "old_panel": study.digest(old / "clean/panel.pkl"),
        "selection": selection,
        "files": {
            str(p.relative_to(root)): study.digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
    }
    study.write_json(out / "source-manifest.json", manifest)
    study.write_json(out / "price-bridges.json", bridges)
    return combined, v2, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.study_root / "confirmation-inputs", args.study_root / "confirmation"
    out.mkdir(parents=True, exist_ok=True)
    selection_path = args.study_root / "selection.json"
    selection_hash = study.digest(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf8"))
    original_protocol = json.loads((args.study_root / "protocol.json").read_text(encoding="utf8"))
    chosen = [n for n in study.VARIANTS if n in {"legacy", "repaired", selection["selected"]}]
    lock = {
        "selection_sha256": selection_hash,
        "selected": selection["selected"],
        "variants": chosen,
        "source_scope": "fixed cohort; newly collected 2026; retrospective reconstruction; no authenticated contemporaneous archive",
    }
    lock_path = out / "selection-lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf8")) != lock:
        raise ValueError("selection changed after confirmation was opened")
    study.write_json(lock_path, lock)
    panel, v2, _ = prepare(args.old_root, root, out, lock, original_protocol)
    protocol = {
        **original_protocol,
        "inputs": {"confirmation_source_manifest": study.digest(out / "source-manifest.json")},
    }
    study.VARIANTS = {n: study.VARIANTS[n] for n in chosen}
    snapshots, histories, admission = study.build_histories(panel, v2, out, protocol)
    common = {
        "model_kind": "fundamental_style",
        "risk_horizon": 5,
        "min_periods": 252,
        "min_assets_per_period": 100,
        "min_asset_observations": 252,
    }
    estimators = {
        "common_ols": FactorModelConfig(**common, **protocol["old_protocol"]["baseline"]),
        "frozen_enhanced": FactorModelConfig(
            **common,
            **protocol["old_protocol"]["enhanced_fixed"],
            covariance_shrinkage=0.1,
            factor_half_life=90,
            specific_half_life=60,
            specific_prior_dof=10,
        ),
    }
    first = panel["dates"].searchsorted(pd.Timestamp("2026-01-01"))
    grid = [
        (i, panel["dates"][i], panel["dates"][i + 5])
        for i in range(first, len(panel["dates"]) - 5, 5)
    ]
    rows, missing = study.evaluate(panel, snapshots, histories, grid, estimators)
    rows.to_csv(out / "forecasts.csv", index=False)
    summary = study.summarize(rows, protocol)
    legacy_comparison = (
        rows[rows.variant.isin(["legacy", selection["selected"]])].copy()
        if not rows.empty
        else rows
    )
    if not legacy_comparison.empty:
        legacy_comparison["variant"] = legacy_comparison.variant.map(
            {"legacy": "repaired", selection["selected"]: "selected_candidate"}
        )
    if study.digest(selection_path) != selection_hash:
        raise ValueError("selected model changed during confirmation")
    result = {
        "selected": selection["selected"],
        "selection_sha256": selection_hash,
        "confirmation_period": ["2026-01-01", protocol["confirmation_end"]],
        "metrics": summary,
        "selected_vs_legacy": study.summarize(legacy_comparison, protocol),
        "unavailable_dates": missing,
        "admission": admission,
        "minimum_confirmation_dates": protocol["minimum_confirmation_dates"],
        "status": "insufficient_independent_forecasts"
        if summary["common_ols"]["complete_dates"] < protocol["minimum_confirmation_dates"]
        else "requires_review",
        "production_changed": False,
        "source_authenticated": False,
    }
    study.write_json(out / "result.json", result)
    print(
        json.dumps(
            {"status": result["status"], "complete_dates": summary["common_ols"]["complete_dates"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
