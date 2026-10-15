"""Exposure stability and turnover-source audits, independent of outcome scores."""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from exposure_v2_study import write_json


def stability(root):
    with (root / "histories.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    results, comparisons = [], []
    for variant, snapshots in saved["snapshots"].items():
        previous = None
        for day, snapshot in sorted(snapshots.items()):
            columns = [
                c for c in snapshot.values if c != "market" and not c.startswith("industry:")
            ]
            if previous is not None:
                shared = snapshot.values.index.intersection(previous.values.index, sort=False)
                before, after = (
                    previous.values.loc[shared, columns],
                    snapshot.values.loc[shared, columns],
                )
                rank = before.rank().corrwith(after.rank())
                for c in columns:
                    results.append(
                        {
                            "variant": variant,
                            "year": day.year,
                            "factor": c,
                            "rank_correlation": rank[c],
                            "mean_absolute_change": (before[c] - after[c]).abs().mean(),
                            "shared_assets": len(shared),
                        }
                    )
            if variant != "legacy":
                original = saved["snapshots"]["legacy"][day]
                assets = snapshot.values.index.intersection(original.values.index, sort=False)
                correlation = snapshot.values.loc[assets, columns].corrwith(
                    original.values.loc[assets, columns]
                )
                for c in columns:
                    comparisons.append(
                        {
                            "variant": variant,
                            "year": day.year,
                            "factor": c,
                            "correlation_with_legacy": correlation[c],
                        }
                    )
            previous = snapshot
    for name, rows in (
        ("exposure-stability", results),
        ("exposure-definition-differences", comparisons),
    ):
        pd.DataFrame(rows).groupby(["variant", "year", "factor"]).mean().to_csv(
            root / (name + ".csv")
        )


def turnover(old, root):
    records = []
    for path in sorted((root / "confirmation-inputs/raw/turnover-check").glob("*.csv")):
        asset = path.stem
        bao = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        data = json.loads((old / "raw/prices" / (asset + "-0.json")).read_text(encoding="utf8"))[
            "data"
        ]["klines"]
        em = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(r.split(",")[0]),
                    "turn": float(r.split(",")[-1]),
                    "volume": float(r.split(",")[5]) * 100,
                }
                for r in data
            ]
        ).set_index("date")
        joined = (
            bao[["volume", "turn"]].join(em, how="inner", lsuffix="_bao", rsuffix="_em").dropna()
        )
        joined = joined[(joined.turn_bao > 0) & (joined.turn_em > 0)]
        capital_path = old / "raw/shares" / (asset + ".csv")
        capital = pd.read_csv(capital_path) if capital_path.exists() else pd.DataFrame()
        if "人民币普通股" not in capital:
            capital = pd.DataFrame()
        if not capital.empty:
            capital["effective"] = pd.to_datetime(capital["变动日期"], errors="coerce")
            capital["available"] = pd.to_datetime(
                capital["公告日期"], errors="coerce"
            ) + pd.Timedelta(days=1)
            capital["float_a"] = pd.to_numeric(capital["人民币普通股"], errors="coerce") * 10000
            capital = capital.dropna(subset=["effective", "available", "float_a"])
        for day, r in joined.iterrows():
            known = (
                capital[(capital.available <= day) & (capital.effective <= day)].sort_values(
                    ["effective", "available"]
                )
                if not capital.empty
                else capital
            )
            historical_float = None if known.empty else float(known.float_a.iloc[-1])
            implied = r.volume_bao / r.turn_bao * 100
            records.append(
                {
                    "asset": asset,
                    "date": str(day.date()),
                    "turnover_difference_percentage_points": abs(r.turn_em - r.turn_bao),
                    "volume_relative_difference": abs(r.volume_em / r.volume_bao - 1),
                    "implied_float_shares": implied,
                    "dated_cninfo_float_a": historical_float,
                    "float_denominator_relative_difference": None
                    if historical_float is None or historical_float <= 0
                    else abs(implied / historical_float - 1),
                }
            )
    data = pd.DataFrame(records)
    data.to_csv(root / "turnover-source-comparison.csv", index=False)
    if data.empty:
        raise ValueError("no shared historical turnover observations")
    finite = data.float_denominator_relative_difference.dropna()
    write_json(
        root / "turnover-audit.json",
        {
            "stock_dates": len(data),
            "issuers": data.asset.nunique(),
            "window": ["2023-06-01", "2023-06-30"],
            "fraction_turnover_within_rounding_0_011_percentage_points": float(
                (data.turnover_difference_percentage_points <= 0.011).mean()
            ),
            "p99_turnover_difference_percentage_points": float(
                data.turnover_difference_percentage_points.quantile(0.99)
            ),
            "dated_capital_checks": len(finite),
            "fraction_implied_float_within_one_percent_of_known_cninfo_float_a": float(
                (finite < 0.01).mean()
            )
            if len(finite)
            else None,
            "denominator": "circulating A shares; cross-source check with dated CNINFO records where present",
            "source_authenticated": False,
            "limitations": [
                "one historical month does not authenticate all daily float-share vintages",
                "later corrections to supplier histories remain possible",
                "unexplained discrepancies stay in the audit and are not overwritten",
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--turnover-only", action="store_true")
    args = parser.parse_args()
    if not args.turnover_only:
        stability(args.out)
    turnover(args.old_root, args.out)


if __name__ == "__main__":
    main()
