"""Compare original extraction to a secondary table without changing admission."""

import json

import pandas as pd

from prepare_pit import FIELD_NAMES
from public_sources import ROOT


def audit():
    facts = pd.read_csv(
        ROOT / "clean/financial_facts.csv", dtype={"asset": str, "report_period": str}
    )
    latest = facts.sort_values(["available_at", "announcement_id"]).drop_duplicates(
        ["asset", "report_period", "field"], keep="last"
    )
    comparisons = []
    for asset, group in latest.groupby("asset"):
        tables = {}
        for kind in ("fzb", "lrb"):
            data = json.loads(
                (ROOT / "raw/financial" / (asset + "-" + kind + ".json")).read_text(
                    encoding="utf-8"
                )
            )
            tables.update(
                (data.get("result", {}).get("data") or {}).get("report_list", {})
                if kind == "fzb"
                else {}
            )
            reports = (data.get("result", {}).get("data") or {}).get("report_list", {})
            fields = (
                ["book_equity", "assets", "liabilities"]
                if kind == "fzb"
                else ["profit_ytd", "revenue_ytd"]
            )
            for row in group[group.field.isin(fields)].itertuples():
                report = reports.get(row.report_period)
                if not report:
                    continue
                items = {r["item_title"]: r["item_value"] for r in report["data"]}
                ref = next(
                    (
                        items[n]
                        for n in FIELD_NAMES[row.field]
                        if items.get(n) not in (None, "", "--")
                    ),
                    None,
                )
                if ref is None:
                    continue
                reference = float(ref)
                relative = abs(row.value - reference) / max(abs(reference), 1)
                comparisons.append(
                    {
                        "asset": asset,
                        "period": row.report_period,
                        "field": row.field,
                        "original": row.value,
                        "secondary": reference,
                        "relative_difference": relative,
                        "announcement_id": row.announcement_id,
                        "label": row.matched_label,
                        "consolidated": row.from_consolidated_statement,
                    }
                )
    result = pd.DataFrame(comparisons)
    result.to_csv(ROOT / "clean/original-secondary-comparison.csv", index=False, encoding="utf-8")
    print(
        result.groupby("field")
        .relative_difference.agg(["size", "median", lambda x: (x > 0.01).mean()])
        .to_string()
    )
    print(
        result[result.relative_difference > 0.5]
        .sort_values("relative_difference", ascending=False)[
            ["asset", "period", "field", "original", "secondary", "label"]
        ]
        .head(12)
        .to_string(index=False, max_colwidth=150)
    )


if __name__ == "__main__":
    audit()
