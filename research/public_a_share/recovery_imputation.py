"""Opt-in, time-local descriptor experiments. Never impute financial facts.

Known truth is withheld in persistent issuer blocks. Predictors see only that
day's observed peers. Missing prices, capitalizations and classifications cannot
be repaired by this module. No evaluation-year hyperparameter selection occurs.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from recovery_inventory import digest, write_json

FIELDS = ("revenue_growth_3y", "profit_growth_3y")
METHODS = ("industry_median", "industry_size_ridge")


def foundation_eligibility(panel):
    """Descriptor estimation never repairs unreliable market/core financial inputs."""
    valid = panel["caps"].gt(0) & panel["groups"].notna()
    for field in ("size_log_mcap", "book_to_price", "earnings_yield_ttm", "book_leverage"):
        valid &= np.isfinite(panel["descriptors"][field])
    return valid


def audit_eligibility(recovered, dates, assets):
    """Block a known adverse current annual report, even if quarterly cells exist.

    Unreviewed legacy sources retain their legacy status; this is not a claim
    that all old reports have been audited again by this recovery campaign.
    """
    inventory = pd.read_csv(recovered.parent / "recovery-inventory.csv", dtype=str)
    opinions = {
        r["id"]: r["opinion"]
        for r in json.loads((recovered / "document-admission.json").read_text(encoding="utf-8"))
    }
    inventory["available_at"] = pd.to_datetime(inventory.available_at, utc=True)
    blocked = {"qualified", "disclaimer_or_adverse", "unclassified"}
    allowed = pd.DataFrame(True, index=dates, columns=assets)
    for asset in assets:
        events = list(
            inventory[inventory.asset == asset]
            .sort_values(["available_at", "report_period", "announcement_id"])
            .itertuples()
        )
        cursor, latest_period, status = 0, "", None
        for day in dates:
            cutoff = pd.Timestamp(day).tz_localize("Asia/Shanghai")
            while cursor < len(events) and events[cursor].available_at <= cutoff:
                event = events[cursor]
                if event.report_period >= latest_period:
                    latest_period = event.report_period
                    status = opinions.get(event.announcement_id, "legacy_unreviewed")
                cursor += 1
            if status in blocked:
                allowed.at[day, asset] = False
    return allowed


def estimate(observed, groups, sizes, requested, method, *, minimum_peers=5, ridge=1.0):
    """Return predictions and reasons; the held-out target is not an argument."""
    if method not in METHODS:
        raise ValueError("unknown_imputation_method")
    observed = observed.replace([np.inf, -np.inf], np.nan)
    valid = observed.notna() & groups.notna() & sizes.notna() & np.isfinite(sizes)
    train = observed[valid]
    result = pd.Series(np.nan, index=requested, dtype=float)
    reasons = pd.Series("no_comparable_peers", index=requested, dtype=object)
    if len(train) < 20:
        return result, reasons
    categories = sorted(groups.loc[train.index].unique())
    design = pd.get_dummies(groups.loc[train.index]).reindex(columns=categories).astype(float)
    size_mean, size_scale = sizes.loc[train.index].mean(), sizes.loc[train.index].std()
    if not np.isfinite(size_scale) or size_scale <= 0:
        size_scale = 1.0
    design["__size"] = (sizes.loc[train.index] - size_mean) / size_scale
    center, scale = train.mean(), train.std()
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    x = design.to_numpy()
    coefficients = np.linalg.solve(
        x.T @ x + ridge * np.eye(x.shape[1]), x.T @ ((train - center) / scale)
    )
    for asset in requested:
        if pd.isna(groups.get(asset)) or not np.isfinite(sizes.get(asset, np.nan)):
            reasons.at[asset] = "missing_reliable_predictor"
            continue
        peers = train.index[groups.loc[train.index] == groups[asset]]
        if len(peers) < minimum_peers:
            continue
        # Do not extrapolate across an unsupported size range.
        if not sizes.loc[peers].min() <= sizes[asset] <= sizes.loc[peers].max():
            reasons.at[asset] = "outside_peer_size_support"
            continue
        if method == "industry_median":
            result.at[asset] = train.loc[peers].median()
        else:
            row = np.array(
                [float(groups[asset] == g) for g in categories]
                + [(sizes[asset] - size_mean) / size_scale]
            )
            result.at[asset] = center + scale * row @ coefficients
        reasons.at[asset] = "estimated_not_observed"
    return result, reasons


def block_mask(truth, groups, sizes, fraction, seed, block_sessions=63):
    """Mask issuer blocks, stratifying by contemporaneous industry and size.

    The sampled set stays fixed within each block; natural missing values never
    become validation truth. Year boundaries separate development/diagnostics.
    """
    rng = np.random.default_rng(seed)
    result = pd.DataFrame(False, index=truth.index, columns=truth.columns)
    for year in sorted(truth.index.year.unique()):
        days = truth.index[truth.index.year == year]
        for offset in range(0, len(days), block_sessions):
            block = days[offset : offset + block_sessions]
            day = block[0]
            eligible = truth.columns[
                truth.loc[day].notna() & groups.loc[day].notna() & sizes.loc[day].notna()
            ]
            if not len(eligible):
                continue
            strata = (
                groups.loc[day, eligible].astype(str)
                + ":"
                + pd.qcut(sizes.loc[day, eligible].rank(method="first"), 3, labels=False).astype(
                    str
                )
            )
            sampled = []
            for _, members in strata.groupby(strata):
                ordered = rng.permutation(members.index)
                sampled.extend(ordered[: max(1, int(round(len(ordered) * fraction)))])
            # Exact global fraction, preserving a balanced round-robin over strata.
            target = max(1, int(round(len(eligible) * fraction)))
            chosen = rng.choice(sampled, min(target, len(sampled)), replace=False)
            result.loc[block, chosen] = True
    return result & truth.notna()


def summarize(records):
    rows = []
    keys = ["field", "method", "fraction", "year", "mask_pattern"]
    for key, group in records.groupby(keys):
        good = group[group.prediction.notna()]
        error = (good.prediction - good.truth) / good.training_scale
        daily = []
        for _, part in good.groupby("date"):
            if len(part) >= 3 and part.prediction.nunique() > 1 and part.truth.nunique() > 1:
                daily.append(part.prediction.corr(part.truth, method="spearman"))
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "masked_count": len(group),
                "predicted_count": len(good),
                "prediction_coverage": len(good) / len(group),
                "normalized_mae": float(error.abs().mean()),
                "normalized_rmse": float(np.sqrt((error**2).mean())),
                "normalized_bias": float(error.mean()),
                "p95_absolute_error": float(error.abs().quantile(0.95)),
                "mean_date_rank_correlation": float(np.mean(daily)) if daily else None,
                "predicted_truth_variance_ratio": float(good.prediction.var() / good.truth.var())
                if good.truth.var() > 0
                else None,
                "extreme_truth_nmae": float(
                    error[good.truth.abs() > 2 * good.training_scale].abs().mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def run(base, recovered, out):
    binding = json.loads((recovered / "summary.json").read_text(encoding="utf-8"))
    if digest(base / "runs/main/panel.pkl") != binding["baseline_sha256"]:
        raise ValueError("baseline_digest_changed")
    with (base / "runs/main/panel.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    panel = payload["v2"]
    for file in binding["descriptor_files"]:
        if digest(file["path"]) != file["sha256"]:
            raise ValueError("recovered_descriptor_changed")
        panel["descriptors"][file["field"]] = pd.read_parquet(file["path"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.csv").exists():
        raise ValueError("completed_experiment_requires_new_version")
    protocol = {
        "seed": 20261017,
        "fractions": [0.05, 0.1, 0.2],
        "block_sessions": 63,
        "minimum_peers": 5,
        "ridge": 1.0,
        "development": 2023,
        "retrospective_diagnostics": [2024, 2025, 2026],
        "mask_patterns": ["stratified_63_sessions", "issuer_full_year"],
        "methods": METHODS,
        "acceptance": "research sensitivity only; no automatic model admission",
        "source_summary_sha256": digest(recovered / "summary.json"),
        "code_sha256": digest(__file__),
    }
    write_json(out / "protocol.json", protocol)
    dates = panel["dates"][(panel["dates"] >= "2023-01-03") & (panel["dates"] <= "2026-09-30")]
    foundation = foundation_eligibility(panel).loc[dates] & audit_eligibility(
        recovered, dates, panel["assets"]
    )
    groups = panel["groups"].loc[dates].where(foundation)
    sizes = panel["descriptors"]["size_log_mcap"].loc[dates]
    records, real = [], []
    for f, field in enumerate(FIELDS):
        raw_truth = pd.read_parquet(recovered / (field + ".parquet")).loc[dates]
        truth = raw_truth.where(foundation)
        for pattern, length in [("stratified_63_sessions", 63), ("issuer_full_year", 252)]:
            for fraction in protocol["fractions"]:
                mask = block_mask(truth, groups, sizes, fraction, protocol["seed"] + f, length)
                for day in dates[::5]:
                    requested = truth.columns[mask.loc[day]]
                    observed = truth.loc[day].mask(mask.loc[day])
                    scale = observed.std()
                    for method in METHODS:
                        prediction, reason = estimate(
                            observed, groups.loc[day], sizes.loc[day], requested, method
                        )
                        for asset in requested:
                            records.append(
                                {
                                    "date": str(day.date()),
                                    "year": day.year,
                                    "asset": asset,
                                    "field": field,
                                    "method": method,
                                    "fraction": fraction,
                                    "mask_pattern": pattern,
                                    "truth": truth.at[day, asset],
                                    "prediction": prediction[asset],
                                    "training_scale": scale,
                                    "reason": reason[asset],
                                    "industry": groups.at[day, asset],
                                }
                            )
        # Real gaps have no hidden truth: report coverage and provenance only.
        for day in dates[::5]:
            requested = raw_truth.columns[raw_truth.loc[day].isna()]
            for method in METHODS:
                prediction, reason = estimate(
                    truth.loc[day], groups.loc[day], sizes.loc[day], requested, method
                )
                for asset in requested:
                    real.append(
                        {
                            "date": str(day.date()),
                            "asset": asset,
                            "field": field,
                            "method": method,
                            "prediction": prediction[asset],
                            "status": reason[asset],
                        }
                    )
    records = pd.DataFrame(records)
    records.to_parquet(out / "masked-predictions.parquet", index=False)
    pd.DataFrame(real).to_parquet(out / "real-gap-sensitivity.parquet", index=False)
    summary = summarize(records)
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    write_json(
        out / "admission.json",
        {
            "default_model_changed": False,
            "promoted": False,
            "reason": "Masking accuracy alone cannot establish calibration for genuinely unobserved issuer histories.",
            "original_financial_facts_imputed": False,
            "candidate_values_published_as_observations": False,
            "records": len(records),
            "real_gap_candidate_records": len(real),
        },
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["base", "recovered", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.recovered, args.out)
