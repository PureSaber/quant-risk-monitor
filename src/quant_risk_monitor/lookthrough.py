"""Point-in-time fund look-through and cross-strategy overlap evidence."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from itertools import combinations
from pathlib import Path

import pandas as pd
from quant_data_kit.financial.common import number, utc
from quant_data_kit.financial.holdings import (
    COLUMNS,
    exposure_summary,
    holdings_asof,
    look_through,
    validate_holdings,
)

ZERO = Decimal(0)
ONE = Decimal(1)
POSITION_TYPES = {"security", "fund", "cash"}
COVERAGE_BUCKETS = ("covered", "unknown", "stale", "future", "cycle", "depth_limit")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _json_number(value: Decimal) -> int | float:
    return 0 if not value else float(value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _currency(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 3 or not result.isalpha() or result != result.upper():
        raise ValueError(f"{name} must be an uppercase three-letter currency")
    return result


def _exact_fields(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} requires exactly {sorted(expected)}")
    return value


def _frame_payload(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for row in frame.sort_values(["disclosure_id", "instrument_id"]).itertuples(index=False):
        rows.append(
            {
                "disclosure_id": row.disclosure_id,
                "fund_id": row.fund_id,
                "holding_date": row.holding_date.date().isoformat(),
                "available_at": row.available_at.isoformat(),
                "instrument_id": row.instrument_id,
                "asset_type": row.asset_type,
                "currency": row.currency,
                "weight": str(row.weight),
                "source": row.source,
                "evidence_id": row.evidence_id,
            }
        )
    return rows


def _future_disclosure_exists(
    disclosures: pd.DataFrame, fund_id: str, business_at: pd.Timestamp, knowledge_at: pd.Timestamp
) -> bool:
    rows = disclosures.loc[disclosures.fund_id == fund_id]
    if rows.empty:
        return False
    business_day = business_at.tz_localize(None).normalize()
    return bool(((rows.holding_date > business_day) | (rows.available_at > knowledge_at)).any())


def _coverage_bucket(
    leaf, disclosures: pd.DataFrame, business_at: pd.Timestamp, knowledge_at: pd.Timestamp
) -> str:
    if bool(leaf.known):
        return "covered"
    if leaf.reason == "stale":
        return "stale"
    if leaf.reason in {"cycle", "depth_limit"}:
        return str(leaf.reason)
    missing_fund = str(leaf.path[-1]) if leaf.path else ""
    if leaf.reason == "missing_disclosure" and _future_disclosure_exists(
        disclosures, missing_fund, business_at, knowledge_at
    ):
        return "future"
    return "unknown"


def _selected_disclosure_hashes(frame: pd.DataFrame, at: pd.Timestamp) -> dict[str, str]:
    if frame.empty:
        return {}
    selected = holdings_asof(frame, at)
    return {
        str(fund): _digest(_frame_payload(rows))
        for fund, rows in selected.groupby("fund_id", sort=True)
    }


def _disclosure_inventory_hashes(frame: pd.DataFrame) -> dict[str, str]:
    return {
        str(fund): _digest(_frame_payload(rows))
        for fund, rows in frame.groupby("fund_id", sort=True)
    }


def _validate_identity(
    identities: dict[str, tuple[str, str]], instrument: str, asset_type: str, currency: str
) -> None:
    if asset_type == "fund":
        return
    identity = (asset_type, currency)
    if instrument in identities and identities[instrument] != identity:
        raise ValueError(f"instrument {instrument!r} has conflicting asset type or currency")
    identities[instrument] = identity


def _normalize_study(spec: object) -> dict:
    expected = {
        "schema",
        "evidence_kind",
        "business_at",
        "knowledge_at",
        "base_currency",
        "position_weight_basis",
        "max_depth",
        "max_age_days",
        "strategies",
        "disclosures",
    }
    value = _exact_fields(spec, expected, "portfolio exposure study")
    if value["schema"] != "quant-risk.portfolio-exposure-study/v1":
        raise ValueError("unsupported portfolio exposure study schema")
    if value["evidence_kind"] not in {"synthetic", "retrospective", "historical_pit"}:
        raise ValueError("explicit evidence_kind required")
    if value["position_weight_basis"] != "fraction_of_strategy_nav_in_base_currency":
        raise ValueError("position weights must be fractions of strategy NAV in base currency")
    business_at = utc(value["business_at"], "business_at")
    knowledge_at = utc(value["knowledge_at"], "knowledge_at")
    if knowledge_at > business_at:
        raise ValueError("knowledge_at cannot be after business_at")
    base_currency = _currency(value["base_currency"], "base_currency")
    for name in ("max_depth", "max_age_days"):
        item = value[name]
        minimum = 1 if name == "max_depth" else 0
        if type(item) is not int or item < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(value["strategies"], list) or len(value["strategies"]) < 2:
        raise ValueError("at least two strategies are required")
    if not isinstance(value["disclosures"], list):
        raise TypeError("disclosures must be a list")

    if any(
        not isinstance(item, dict) or set(item) != set(COLUMNS) for item in value["disclosures"]
    ):
        raise ValueError("each disclosure requires the complete QDK holdings schema")
    disclosure_frame = pd.DataFrame(value["disclosures"], columns=COLUMNS)
    disclosure_frame = validate_holdings(disclosure_frame)

    identities: dict[str, tuple[str, str]] = {}
    for row in disclosure_frame.itertuples():
        _validate_identity(identities, row.instrument_id, row.asset_type, row.currency)

    strategies = []
    strategy_ids = set()
    position_identities: dict[str, tuple[str, str]] = {}
    strategy_fields = {"strategy_id", "effective_at", "available_at", "source", "positions"}
    position_fields = {"instrument_id", "asset_type", "currency", "weight", "source"}
    for raw_strategy in value["strategies"]:
        strategy = _exact_fields(raw_strategy, strategy_fields, "strategy")
        strategy_id = _text(strategy["strategy_id"], "strategy_id")
        if strategy_id in strategy_ids:
            raise ValueError("strategy_id values must be unique")
        strategy_ids.add(strategy_id)
        effective_at = utc(strategy["effective_at"], "strategy effective_at")
        available_at = utc(strategy["available_at"], "strategy available_at")
        if effective_at > available_at:
            raise ValueError("strategy snapshot cannot be available before it is effective")
        if effective_at > business_at or available_at > knowledge_at:
            raise ValueError("future strategy snapshot is unavailable at the requested cutoffs")
        source = _text(strategy["source"], "strategy source")
        if not isinstance(strategy["positions"], list) or not strategy["positions"]:
            raise ValueError("each strategy requires at least one explicit position")
        positions = []
        for raw_position in strategy["positions"]:
            position = _exact_fields(raw_position, position_fields, "position")
            instrument = _text(position["instrument_id"], "position instrument_id")
            asset_type = _text(position["asset_type"], "position asset_type")
            if asset_type not in POSITION_TYPES:
                raise ValueError("position asset_type must be security, fund, or cash")
            currency = _currency(position["currency"], "position currency")
            if asset_type == "cash" and instrument != f"CASH:{currency}":
                raise ValueError("cash identity must be CASH:<currency>")
            if asset_type != "cash" and instrument.startswith("CASH:"):
                raise ValueError("cash identity cannot label a security or fund")
            weight = number(position["weight"])
            position_source = _text(position["source"], "position source")
            position_identity = (asset_type, currency)
            if (
                instrument in position_identities
                and position_identities[instrument] != position_identity
            ):
                raise ValueError(
                    f"position instrument {instrument!r} has conflicting asset type or currency"
                )
            position_identities[instrument] = position_identity
            _validate_identity(identities, instrument, asset_type, currency)
            positions.append(
                {
                    "instrument_id": instrument,
                    "asset_type": asset_type,
                    "currency": currency,
                    "weight": weight,
                    "source": position_source,
                }
            )
        strategies.append(
            {
                "strategy_id": strategy_id,
                "effective_at": effective_at,
                "available_at": available_at,
                "source": source,
                "positions": positions,
                "source_sha256": _digest(raw_strategy),
            }
        )
    return {
        "business_at": business_at,
        "knowledge_at": knowledge_at,
        "base_currency": base_currency,
        "max_depth": value["max_depth"],
        "max_age_days": value["max_age_days"],
        "evidence_kind": value["evidence_kind"],
        "strategies": strategies,
        "disclosures": disclosure_frame,
    }


def _strategy_report(study: dict, strategy: dict, visible: pd.DataFrame) -> tuple[dict, dict]:
    aggregate_positions: dict[tuple[str, str, str], Decimal] = defaultdict(Decimal)
    position_hashes: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for position in strategy["positions"]:
        key = (position["instrument_id"], position["asset_type"], position["currency"])
        aggregate_positions[key] += position["weight"]
        position_hashes[key].add(
            _digest(
                {
                    "strategy_id": strategy["strategy_id"],
                    "effective_at": strategy["effective_at"].isoformat(),
                    "available_at": strategy["available_at"].isoformat(),
                    **{k: str(v) if k == "weight" else v for k, v in position.items()},
                }
            )
        )

    selected_hashes = _selected_disclosure_hashes(visible, study["business_at"])
    inventory_hashes = _disclosure_inventory_hashes(study["disclosures"])
    paths = []
    for key in sorted(aggregate_positions):
        instrument, asset_type, currency = key
        weight = aggregate_positions[key]
        if not weight:
            continue
        hashes = sorted(position_hashes[key])
        if asset_type != "fund":
            paths.append(
                {
                    "instrument_id": instrument,
                    "asset_type": asset_type,
                    "currency": currency,
                    "weight": weight,
                    "gross_weight": abs(weight),
                    "reason": "direct",
                    "coverage_bucket": "covered",
                    "path": (),
                    "source_sha256": hashes,
                }
            )
            continue
        unit = look_through(
            visible,
            {instrument: ONE},
            study["business_at"],
            max_depth=study["max_depth"],
            max_age_days=study["max_age_days"],
        )
        for leaf in unit.itertuples():
            leaf_weight = weight * leaf.weight
            bucket = _coverage_bucket(
                leaf,
                study["disclosures"],
                study["business_at"],
                study["knowledge_at"],
            )
            source_hashes = hashes + [
                selected_hashes[fund] for fund in leaf.path if fund in selected_hashes
            ]
            if bucket == "future" and leaf.path and leaf.path[-1] in inventory_hashes:
                source_hashes.append(inventory_hashes[leaf.path[-1]])
            paths.append(
                {
                    "instrument_id": str(leaf.instrument_id),
                    "asset_type": (
                        "cash" if str(leaf.instrument_id).startswith("CASH:") else "security"
                    ),
                    "currency": None if pd.isna(leaf.currency) else str(leaf.currency),
                    "weight": leaf_weight,
                    "gross_weight": abs(leaf_weight),
                    "reason": str(leaf.reason),
                    "coverage_bucket": bucket,
                    "path": tuple(str(item) for item in leaf.path),
                    "source_sha256": sorted(set(source_hashes)),
                }
            )

    declared_net = sum(aggregate_positions.values(), ZERO)
    declared_gross = sum((abs(value) for value in aggregate_positions.values()), ZERO)
    path_net = sum((leaf["weight"] for leaf in paths), ZERO)
    path_gross = sum((leaf["gross_weight"] for leaf in paths), ZERO)
    if path_net != declared_net or path_gross != declared_gross:
        raise ArithmeticError("portfolio look-through failed signed or gross weight conservation")

    coverage = {bucket: ZERO for bucket in COVERAGE_BUCKETS}
    for leaf in paths:
        coverage[leaf["coverage_bucket"]] += leaf["gross_weight"]
    if sum(coverage.values(), ZERO) != declared_gross:
        raise ArithmeticError("coverage buckets do not conserve declared gross exposure")

    exposures: dict[tuple[str, str, str | None], dict] = {}
    for leaf in paths:
        if leaf["coverage_bucket"] != "covered":
            continue
        key = (leaf["instrument_id"], leaf["asset_type"], leaf["currency"])
        if key not in exposures:
            exposures[key] = {"weight": ZERO, "source_sha256": set()}
        exposures[key]["weight"] += leaf["weight"]
        exposures[key]["source_sha256"].update(leaf["source_sha256"])

    exposure_rows = []
    for (instrument, asset_type, currency), item in sorted(exposures.items()):
        if not item["weight"]:
            continue
        exposure_rows.append(
            {
                "instrument_id": instrument,
                "asset_type": asset_type,
                "currency": currency,
                "weight": _json_number(item["weight"]),
                "source_sha256": sorted(item["source_sha256"]),
            }
        )
    aggregate_gross = sum((abs(item["weight"]) for item in exposures.values()), ZERO)
    unavailable = declared_gross - coverage["covered"]
    ratios = {
        bucket: (_json_number(value / declared_gross) if declared_gross else None)
        for bucket, value in coverage.items()
    }
    report = {
        "strategy_id": strategy["strategy_id"],
        "snapshot": {
            "effective_at": strategy["effective_at"].isoformat(),
            "available_at": strategy["available_at"].isoformat(),
            "source_sha256": strategy["source_sha256"],
        },
        "declared": {
            "net_weight": _json_number(declared_net),
            "gross_weight": _json_number(declared_gross),
            "position_rows": len(strategy["positions"]),
            "aggregated_positions": sum(bool(value) for value in aggregate_positions.values()),
        },
        "resolved": {
            "path_net_weight": _json_number(path_net),
            "path_gross_weight": _json_number(path_gross),
            "aggregated_known_net_weight": _json_number(
                sum((item["weight"] for item in exposures.values()), ZERO)
            ),
            "aggregated_known_gross_weight": _json_number(aggregate_gross),
            "internal_netting_weight": _json_number(coverage["covered"] - aggregate_gross),
        },
        "coverage_gross_weight": {
            "denominator": _json_number(declared_gross),
            **{bucket: _json_number(value) for bucket, value in coverage.items()},
            "unavailable": _json_number(unavailable),
        },
        "coverage_ratio": ratios,
        "comparison_eligible": bool(declared_gross and not unavailable),
        "exposures": exposure_rows,
        "unavailable_paths": [
            {
                "instrument_id": leaf["instrument_id"],
                "weight": _json_number(leaf["weight"]),
                "gross_weight": _json_number(leaf["gross_weight"]),
                "reason": leaf["reason"],
                "coverage_bucket": leaf["coverage_bucket"],
                "path": list(leaf["path"]),
                "source_sha256": leaf["source_sha256"],
            }
            for leaf in paths
            if leaf["coverage_bucket"] != "covered"
        ],
    }
    machine = {
        (row["instrument_id"], row["currency"]): Decimal(str(row["weight"]))
        for row in exposure_rows
        if row["asset_type"] == "security"
    }
    return report, machine


def _pair_report(left: dict, right: dict, left_map: dict, right_map: dict) -> dict:
    common = []
    shared = aligned = opposing = ZERO
    for key in sorted(set(left_map) & set(right_map)):
        left_weight, right_weight = left_map[key], right_map[key]
        if not left_weight or not right_weight:
            continue
        value = min(abs(left_weight), abs(right_weight))
        direction = "aligned" if left_weight * right_weight > 0 else "opposing"
        shared += value
        if direction == "aligned":
            aligned += value
        else:
            opposing += value
        common.append(
            {
                "instrument_id": key[0],
                "currency": key[1],
                "left_weight": _json_number(left_weight),
                "right_weight": _json_number(right_weight),
                "shared_abs_weight": _json_number(value),
                "direction": direction,
            }
        )
    left_gross = sum((abs(value) for value in left_map.values()), ZERO)
    right_gross = sum((abs(value) for value in right_map.values()), ZERO)
    denominator = min(left_gross, right_gross)
    complete = left["comparison_eligible"] and right["comparison_eligible"]
    reasons = []
    if not left["comparison_eligible"]:
        reasons.append(f"{left['strategy_id']}:incomplete_coverage")
    if not right["comparison_eligible"]:
        reasons.append(f"{right['strategy_id']}:incomplete_coverage")
    if not denominator:
        reasons.append("no_known_non_cash_security_gross_exposure")
    available = complete and bool(denominator)
    diagnostic = shared / denominator if denominator else None
    return {
        "left_strategy_id": left["strategy_id"],
        "right_strategy_id": right["strategy_id"],
        "status": "available" if available else "unavailable",
        "unavailable_reasons": reasons,
        "metric_scope": "position_overlap_not_return_correlation_or_risk_forecast",
        "known_security_gross": {
            "left": _json_number(left_gross),
            "right": _json_number(right_gross),
            "smaller_strategy_denominator": _json_number(denominator),
        },
        "shared_known_gross_weight": _json_number(shared),
        "aligned_known_gross_weight": _json_number(aligned),
        "opposing_known_gross_weight": _json_number(opposing),
        "overlap_ratio_of_smaller_gross": _json_number(diagnostic) if available else None,
        "diagnostic_known_only_ratio": _json_number(diagnostic) if diagnostic is not None else None,
        "common_exposures": common,
    }


def evaluate_portfolio_overlap(spec: object, *, risk_forecast: dict | None = None) -> dict:
    """Evaluate exact PIT overlap from caller-supplied NAV weights and disclosures."""
    study = _normalize_study(spec)
    visible = (
        study["disclosures"]
        .loc[study["disclosures"].available_at <= study["knowledge_at"]]
        .reset_index(drop=True)
    )
    strategy_reports, exposure_maps = [], []
    for strategy in study["strategies"]:
        report, machine = _strategy_report(study, strategy, visible)
        strategy_reports.append(report)
        exposure_maps.append(machine)
    pairwise = [
        _pair_report(strategy_reports[i], strategy_reports[j], exposure_maps[i], exposure_maps[j])
        for i, j in combinations(range(len(strategy_reports)), 2)
    ]
    disclosures = study["disclosures"]
    selected_disclosures = holdings_asof(visible, study["business_at"])
    result = {
        "schema": "quant-risk.portfolio-exposure-report/v1",
        "status": "complete",
        "evidence_kind": study["evidence_kind"],
        "business_at": study["business_at"].isoformat(),
        "knowledge_at": study["knowledge_at"].isoformat(),
        "base_currency": study["base_currency"],
        "valuation_contract": {
            "position_weight_basis": "fraction_of_strategy_nav_in_base_currency",
            "caller_responsibility": "NAV, market value, denominator and FX conversion",
            "certification": "this report does not verify market prices, FX rates or real holdings",
            "residual_cash": "never inferred from weights that do not net to one",
            "negative_weight": "short exposure; negative cash is borrowing",
            "leverage": "permitted and reported separately as signed net and absolute gross",
        },
        "provenance": {
            "input_sha256": _digest(spec),
            "disclosures_sha256": _digest(_frame_payload(disclosures)),
            "causally_visible_disclosures_sha256": _digest(_frame_payload(visible)),
            "selected_disclosures_sha256": _digest(_frame_payload(selected_disclosures)),
            "strategy_source_sha256": {
                item["strategy_id"]: item["source_sha256"] for item in study["strategies"]
            },
        },
        "strategies": strategy_reports,
        "pairwise": pairwise,
        "risk_forecast": risk_forecast,
        "limitations": [
            "overlap measures common position exposure, not return correlation or forecast risk",
            "known-only diagnostics are not complete comparisons when status is unavailable",
            "disclosures are lagged evidence and are not certified as real-time positions",
            "no missing constituent, denominator, market value or FX input is filled with zero",
        ],
    }
    _canonical(result)
    return result


def run_portfolio_overlap(input_path: str | Path, output: str | Path, *, risk_run=None) -> dict:
    """Validate a study, optionally verify forecast evidence, and write one new JSON report."""
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    if input_path == output:
        raise ValueError("portfolio overlap output cannot replace the input")
    if output.exists():
        raise FileExistsError("portfolio overlap output already exists")
    original = input_path.read_bytes()
    spec = json.loads(original)
    risk_forecast = None
    if risk_run is not None:
        from quant_risk_monitor.risk_validation import (
            readable_risk_forecast_summary,
            verify_risk_validation,
        )

        risk_forecast = readable_risk_forecast_summary(
            verify_risk_validation(risk_run),
            verification_status="verified_by_hash_and_full_recomputation",
        )
    report = evaluate_portfolio_overlap(spec, risk_forecast=risk_forecast)
    report["provenance"]["input_file_sha256"] = hashlib.sha256(original).hexdigest()
    if input_path.read_bytes() != original:
        raise ValueError("portfolio exposure input changed during evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def check_fund_concentration(
    weights, disclosures, at, *, max_security_weight="0.1", max_unknown_weight="0", **coverage
):
    """Preserve the original single-portfolio QDK concentration wrapper."""
    limit = number(max_security_weight, nonnegative=True)
    unknown_limit = number(max_unknown_weight, nonnegative=True)
    if limit > 1 or unknown_limit > 1:
        raise ValueError("weight limits cannot exceed one")
    summary = exposure_summary(look_through(disclosures, weights, at, **coverage))
    breaches = [
        {**x, "code": "LOOKTHROUGH_CONCENTRATION"}
        for x in summary["exposures"]
        if x["weight"] > limit and not x["instrument_id"].startswith("CASH:")
    ]
    if summary["unknown_weight"] > unknown_limit:
        breaches.append({"code": "LOOKTHROUGH_UNKNOWN", "weight": summary["unknown_weight"]})
    return {
        "allowed": not breaches,
        "breaches": breaches,
        "coverage": summary,
        "scope": "disclosed_holdings_not_realtime_positions",
    }
