"""Verify final source hashes and price adjustment behavior before calibration."""

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from public_sources import ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    members = json.loads((ROOT / "cohort.json").read_text(encoding="utf8"))["members"]
    entries = []
    price_checks = []
    documents = 0

    def include(path, receipt, role):
        actual = sha(path)
        if receipt.get("sha256") and actual != receipt["sha256"]:
            raise ValueError(f"Source hash changed: {path}")
        url = receipt.get("url", receipt.get("source", ""))
        if any(word in url.lower() for word in ["huisheng", "dataway", "hhhstz"]):
            raise ValueError("Excluded source")
        entries.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": actual,
                "bytes": path.stat().st_size,
                "source": url,
                "captured_at": receipt.get("captured_at"),
                "role": role,
            }
        )

    for member in members:
        code = member["code"]
        index_path = ROOT / "raw/announcements" / (code + "-index.json")
        index = json.loads(index_path.read_text(encoding="utf8"))
        if index.get("errors") or len(index.get("reports", [])) != index["report_count"]:
            raise ValueError(f"Incomplete original archive {code}")
        include(index_path, {}, "announcement_metadata")
        for report in index["reports"]:
            path = ROOT / "raw" / report["pdf"]
            receipt = json.loads(path.with_suffix(".pdf.receipt.json").read_text(encoding="utf8"))
            include(path, receipt, "original_financial_disclosure")
            documents += 1
            text_path = ROOT / report["text"]
            marker = json.loads(text_path.with_suffix(".extract.json").read_text(encoding="utf8"))
            if not marker.get("isolated_process"):
                raise ValueError("Unsafe/incomplete PDF extraction")
            include(text_path, {}, "original_disclosure_text")
        raw_path = ROOT / "raw/prices" / (code + "-0.json")
        include(
            raw_path,
            json.loads(raw_path.with_suffix(".json.receipt.json").read_text(encoding="utf8")),
            "raw_capitalization_price",
        )
        bao_path = ROOT / "raw/bao-prices" / (code + ".csv")
        include(
            bao_path,
            json.loads(bao_path.with_suffix(".receipt.json").read_text(encoding="utf8")),
            "multiplicative_return_price",
        )
        share_path = ROOT / "raw/shares" / (code + ".csv")
        if share_path.exists():
            include(
                share_path,
                json.loads(share_path.with_suffix(".receipt.json").read_text(encoding="utf8")),
                "dated_capital_events",
            )
            capital_receipt = json.loads(
                share_path.with_suffix(".receipt.json").read_text(encoding="utf8")
            )
            if capital_receipt.get("raw_document"):
                source_path = ROOT / capital_receipt["raw_document"]
                include(
                    source_path,
                    json.loads(
                        source_path.with_suffix(".html.receipt.json").read_text(encoding="utf8")
                    ),
                    "dated_capital_publication",
                )
        quotes = json.loads(raw_path.read_text(encoding="utf8"))["data"]["klines"]
        raw = pd.Series({row.split(",")[0]: float(row.split(",")[2]) for row in quotes})
        bao = pd.read_csv(bao_path).set_index("date")
        joined = pd.concat([raw.rename("raw"), bao.close.rename("adjusted")], axis=1).dropna()
        ratio = joined.adjusted / joined.raw
        relative_ratio_change = ratio.pct_change(fill_method=None).abs()
        # On ordinary days, adjustment must not distort the observed raw price return.
        ordinary = relative_ratio_change < 1e-5
        error = (
            joined.raw.pct_change(fill_method=None) - joined.adjusted.pct_change(fill_method=None)
        ).abs()
        if len(joined) < 100:
            raise ValueError(f"Insufficient independent prices: {code}")
        if not np.isfinite(joined.to_numpy()).all() or (joined <= 0).any().any():
            raise ValueError("Invalid observed price")
        price_checks.append(
            {
                "asset": code,
                "shared_price_days": len(joined),
                "ordinary_days": int(ordinary.sum()),
                "max_ordinary_day_return_difference": float(error[ordinary].max()),
                "adjustment_change_days": int((relative_ratio_change >= 1e-5).sum()),
                "missing_primary_dates_in_bao": int((~raw.index.isin(bao.index)).sum()),
            }
        )
    for path in sorted((ROOT / "raw/industry").glob("*")):
        if path.suffix not in {".html", ".pdf"}:
            continue
        receipt_path = path.with_suffix(path.suffix + ".receipt.json")
        receipt = (
            json.loads(receipt_path.read_text(encoding="utf8")) if receipt_path.exists() else {}
        )
        if path.name == "csrc-2021q3.pdf":
            receipt = {
                "source": "https://www.csrc.gov.cn/csrc/c100103/c1558619/1558619/files/1638277734844_11692.pdf",
                "sha256": "0e8af6662353037135bb8e8a08a409dcebea9c47f71e7e25b48fe66a24f8c62b",
            }
        include(path, receipt, "dated_industry_publication")
    benchmark_path = ROOT / "raw/benchmark/csi300.json"
    include(
        benchmark_path,
        json.loads(benchmark_path.with_suffix(".json.receipt.json").read_text(encoding="utf8")),
        "historical_benchmark_returns",
    )
    for relative in ["cohort.json", "clean/financial_facts.csv", "clean/industry_history.csv"]:
        include(ROOT / relative, {}, "derived_point_in_time_input")
    price_checks = pd.DataFrame(price_checks)
    price_checks.to_csv(ROOT / "clean/all-stock-price-validation.csv", index=False)
    if price_checks.missing_primary_dates_in_bao.sum():
        raise ValueError("Primary/independent price date mismatch; resolve before building panel")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "No Huisheng/Dataway; original disclosures and public market series only. Current Sina financial-statement snapshots and EM additive HFQ are audit references, never admitted as model inputs. Dated Sina capital events supplement the CNINFO capital interface.",
        "cohort_assets": len(members),
        "original_documents": documents,
        "verified_files": len(entries),
        "total_bytes": sum(r["bytes"] for r in entries),
        "price_assets_checked": len(price_checks),
        "source_entries": entries,
    }
    (ROOT / "clean/source-manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "source_entries"}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
