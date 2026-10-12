"""Prespecified chronological surveillance; alerts are evidence, never trading authority."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from quant_risk_monitor.factor_model import _timestamp

DOMAINS = (
    "data_drift",
    "forecast_error",
    "signal_decay",
    "risk_miscalibration",
    "cost_miscalibration",
    "strategy_performance",
    "operational_failures",
)

DOMAIN_LABELS = {
    "data_drift": "single_feature_bounded_mean_shift",
    "forecast_error": "normalized_absolute_regression_error",
    "signal_decay": "increase_in_direction_classification_error",
    "risk_miscalibration": "absolute_return_tail_exceedance",
    "cost_miscalibration": "normalized_cost_underestimation_bps",
    "strategy_performance": "normalized_net_return_underperformance",
    "operational_failures": "reported_non_ok_operation_rate",
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _number(value, name):
    if isinstance(value, bool):
        raise TypeError(f"boolean {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"finite {name} required")
    return result


def validate_plan(plan):
    required = {
        "schema",
        "frozen_at",
        "reference_rows",
        "window_rows",
        "block_rows",
        "min_blocks",
        "max_windows",
        "family_alpha",
        "forecast_bound",
        "feature_bounds",
        "cost_error_bound_bps",
        "active_return_bound",
        "thresholds",
        "risk_exceedance_probability",
        "currency",
        "frequency",
        "independent_blocks_assumed",
    }
    if set(plan) != required or plan["schema"] != "quant.model-diagnostic-plan/v1":
        raise ValueError("closed quant.model-diagnostic-plan/v1 required")
    _timestamp(plan["frozen_at"], "frozen_at")
    for key in ("reference_rows", "window_rows", "block_rows", "min_blocks", "max_windows"):
        if type(plan[key]) is not int or plan[key] < 1:
            raise ValueError(f"positive integral {key} required")
    if plan["min_blocks"] < 2 or any(
        plan[k] % plan["block_rows"] for k in ("reference_rows", "window_rows")
    ):
        raise ValueError("windows must contain whole blocks and min_blocks >= 2")
    if not 0 < _number(plan["family_alpha"], "alpha") < 1:
        raise ValueError("alpha in (0,1) required")
    if not 0 < _number(plan["risk_exceedance_probability"], "risk probability") < 1:
        raise ValueError("risk exceedance probability in (0,1) required")
    for key in ("forecast_bound", "cost_error_bound_bps", "active_return_bound"):
        if _number(plan[key], key) <= 0:
            raise ValueError(f"positive {key} required")
    bounds = plan["feature_bounds"]
    if len(bounds) != 2 or _number(bounds[0], "feature min") >= _number(bounds[1], "feature max"):
        raise ValueError("ordered feature bounds required")
    if set(plan["thresholds"]) != set(DOMAINS):
        raise ValueError("one prespecified threshold per domain required")
    for key, value in plan["thresholds"].items():
        if not 0 <= _number(value, key) <= 1:
            raise ValueError("normalized thresholds in [0,1] required")
    if (
        not plan["currency"]
        or not plan["frequency"]
        or type(plan["independent_blocks_assumed"]) is not bool
    ):
        raise ValueError("currency, frequency and explicit block-independence assumption required")
    return plan


def _validate_rows(rows, plan, as_of):
    previous_end, ids = None, set()
    visible, withheld = [], []
    required = {
        "row_id",
        "forecast_at",
        "period_start",
        "period_end",
        "outcome_known_at",
        "regime_id",
        "source_sha256",
    }
    optional = {
        "prediction",
        "target",
        "feature",
        "risk_limit",
        "realized_return",
        "predicted_cost_bps",
        "actual_cost_bps",
        "net_return",
        "baseline_return",
        "operation",
    }
    for row in rows:
        if not required <= set(row) or set(row) - required - optional:
            raise ValueError("invalid diagnostic row fields")
        start, end, forecast, known = [
            _timestamp(row[k], k)
            for k in ("period_start", "period_end", "forecast_at", "outcome_known_at")
        ]
        if not forecast <= start < end <= known or (
            previous_end is not None and start < previous_end
        ):
            raise ValueError("causal, ordered, nonoverlapping periods required")
        if _timestamp(plan["frozen_at"], "frozen_at") > forecast:
            raise ValueError("diagnostic plan must be fixed before reference forecasts")
        if row["row_id"] in ids or not row["row_id"] or not row["regime_id"]:
            raise ValueError("unique row IDs and explicit regime identity required")
        sha = row["source_sha256"]
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("exact row source SHA256 required")
        ids.add(row["row_id"])
        previous_end = end
        # Outcomes beyond as_of do not enter any diagnostics, including missingness.
        if known > as_of:
            withheld.append(row["row_id"])
            continue
        if withheld:
            raise ValueError("outcome availability creates a gap in chronological surveillance")
        copy = dict(row)
        for key in optional - {"operation"}:
            if copy.get(key) is not None:
                copy[key] = _number(copy[key], key)
        if copy.get("operation") not in {None, "ok", "rejected", "failed", "missing"}:
            raise ValueError("unknown operation status")
        visible.append(copy)
    return visible, withheld


def _score(row, domain, plan):
    keys = {
        "data_drift": ("feature",),
        "forecast_error": ("prediction", "target"),
        "signal_decay": ("prediction", "target"),
        "risk_miscalibration": ("risk_limit", "realized_return"),
        "cost_miscalibration": ("predicted_cost_bps", "actual_cost_bps"),
        "strategy_performance": ("net_return", "baseline_return"),
        "operational_failures": ("operation",),
    }[domain]
    if any(row.get(k) is None for k in keys):
        return None
    values = [row[k] for k in keys]
    if domain == "data_drift":
        lo, hi = map(float, plan["feature_bounds"])
        if not lo <= values[0] <= hi:
            raise ValueError("feature outside prespecified bounds; revise as a new study")
        return (values[0] - lo) / (hi - lo)
    if domain in {"forecast_error", "signal_decay"}:
        bound = float(plan["forecast_bound"])
        if any(abs(v) > bound for v in values):
            raise ValueError("forecast/target outside prespecified bounds")
        if domain == "forecast_error":
            return abs(values[0] - values[1]) / (2 * bound)
        # Zero is a separate class; this descriptive classification loss is not PnL.
        return float((values[0] > 0) - (values[0] < 0) != (values[1] > 0) - (values[1] < 0))
    if domain == "risk_miscalibration":
        if values[0] <= 0:
            raise ValueError("positive one-period absolute-return risk limit required")
        return float(abs(values[1]) > values[0])
    if domain == "operational_failures":
        return float(values[0] != "ok")
    bound = float(
        plan["cost_error_bound_bps"]
        if domain == "cost_miscalibration"
        else plan["active_return_bound"]
    )
    if domain == "cost_miscalibration" and any(v < 0 for v in values):
        raise ValueError("nonnegative cost bps required")
    difference = values[1] - values[0]
    if abs(difference) > bound:
        raise ValueError("cost/active-return difference outside prespecified bounds")
    return difference / bound  # positive = underestimated cost / underperformance


def _estimate(rows, domain, plan, alpha):
    scores = [_score(r, domain, plan) for r in rows]
    block = plan["block_rows"]
    complete = len(scores) // block * block
    blocks = [
        sum(scores[i : i + block]) / block
        for i in range(0, complete, block)
        if all(v is not None for v in scores[i : i + block])
    ]
    n = len(blocks)
    low, high = (
        (-1.0, 1.0) if domain in {"cost_miscalibration", "strategy_performance"} else (0.0, 1.0)
    )
    result = {
        "rows": len(rows),
        "complete_blocks": n,
        "missing_rows": sum(v is None for v in scores),
        "unused_partial_rows": len(scores) - complete,
        "status": "insufficient_sample",
    }
    if result["missing_rows"]:
        result.update(status="partial_observation", mean=sum(blocks) / n if n else None)
        return result
    if n < plan["min_blocks"]:
        return result
    mean = sum(blocks) / n
    radius = (high - low) * math.sqrt(math.log(2 / alpha) / (2 * n))
    result.update(
        status="available",
        mean=mean,
        interval=[max(low, mean - radius), min(high, mean + radius)],
        interval_method="bounded_block_mean_hoeffding",
        per_interval_alpha=alpha,
    )
    return result


def diagnose_models(payload):
    if (
        set(payload) != {"schema", "plan", "rows", "evaluation_as_of", "evidence_kind"}
        or payload["schema"] != "quant.model-diagnostics-input/v1"
    ):
        raise ValueError("closed quant.model-diagnostics-input/v1 required")
    if payload["evidence_kind"] not in {
        "synthetic",
        "retrospective",
        "historical_pit",
        "independent_forward",
    }:
        raise ValueError("explicit supported evidence kind required")
    plan = validate_plan(payload["plan"])
    as_of = _timestamp(payload["evaluation_as_of"], "evaluation_as_of")
    rows, withheld = _validate_rows(payload["rows"], plan, as_of)
    reference = rows[: plan["reference_rows"]]
    recent = rows[plan["reference_rows"] :]
    if len(recent) > plan["window_rows"] * plan["max_windows"]:
        raise ValueError("prespecified monitoring horizon exhausted; no unlimited peeking")
    # Bonferroni for all domains, planned windows, and reference/current intervals.
    alpha = float(plan["family_alpha"]) / (len(DOMAINS) * plan["max_windows"] * 2)
    windows = []
    for i in range(0, len(recent), plan["window_rows"]):
        current = recent[i : i + plan["window_rows"]]
        regimes = {r["regime_id"] for r in reference + current}
        findings = {}
        for domain in DOMAINS:
            ref, now = (
                _estimate(reference, domain, plan, alpha),
                _estimate(current, domain, plan, alpha),
            )
            threshold = float(plan["thresholds"][domain])
            finding = {
                "reference": ref,
                "current": now,
                "diagnostic_label": DOMAIN_LABELS[domain],
                "threshold": threshold,
                "status": "insufficient_sample",
            }
            if len(regimes) != 1:
                finding.update(
                    status="structural_change",
                    reason="regime differs; reference calibration cannot certify current regime",
                )
            elif ref["missing_rows"] or now["missing_rows"]:
                finding.update(
                    status="partial_observation",
                    reason="missing outcomes can select favorable blocks; no full-window inference",
                )
            elif len(reference) < plan["reference_rows"] or len(current) < plan["window_rows"]:
                finding["reason"] = "unfinished prespecified window"
            elif ref["status"] == now["status"] == "available":
                if domain in {"data_drift", "signal_decay"}:
                    value = now["mean"] - ref["mean"]
                    interval = [
                        now["interval"][0] - ref["interval"][1],
                        now["interval"][1] - ref["interval"][0],
                    ]
                    breach = interval[0] > threshold or (
                        domain == "data_drift" and interval[1] < -threshold
                    )
                    finding["metric"] = "current_minus_reference"
                else:
                    offset = (
                        float(plan["risk_exceedance_probability"])
                        if domain == "risk_miscalibration"
                        else 0
                    )
                    value = now["mean"] - offset
                    interval = [v - offset for v in now["interval"]]
                    breach = interval[0] > threshold
                    finding["metric"] = "excess_exceedance_rate" if offset else "current_mean"
                finding.update(
                    value=value, interval=interval, status="alert" if breach else "no_clear_breach"
                )
                if not plan["independent_blocks_assumed"]:
                    finding.update(
                        status="descriptive_only",
                        reason="independence assumption not accepted; interval has no coverage claim",
                    )
            findings[domain] = finding
        windows.append(
            {
                "index": len(windows),
                "period_start": current[0]["period_start"],
                "period_end": current[-1]["period_end"],
                "findings": findings,
            }
        )
    return {
        "schema": "quant.model-diagnostics/v1",
        "status": "completed" if windows else "insufficient_sample",
        "evidence_kind": payload["evidence_kind"],
        "evaluation_as_of": as_of.isoformat(),
        "plan_sha256": _digest(plan),
        "input_sha256": _digest(payload),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "currency": plan["currency"],
        "units": {"returns": "decimal", "costs": "bps", "scores": "normalized_bounded"},
        "frequency": plan["frequency"],
        "windows": windows,
        "withheld_outcomes": withheld,
        "execution_authorized": False,
        "strategy_invalid": None,
        "uncertainty": {
            "family_alpha": plan["family_alpha"],
            "max_windows": plan["max_windows"],
            "independent_blocks_assumed": plan["independent_blocks_assumed"],
            "multiple_testing": "Bonferroni across all planned domain/window/reference-current intervals",
        },
        "limitations": [
            "Forecast/classification errors do not establish strategy failure or causal attribution.",
            "Hoeffding coverage requires independent bounded blocks; within-block dependence allowed.",
            "No automatic trade, retrain, account recovery, or revised thresholds.",
            "Regime IDs and source timing are producer assertions; unknown data drift is not automatically detected.",
            "One declared feature proxy and absolute-return tail risk; not full distribution or VaR certification.",
        ],
    }


def run_model_diagnostics(input_path, output):
    payload = json.loads(Path(input_path).read_text(encoding="utf-8-sig"))
    result = diagnose_models(payload)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    for name, value in (("input.json", payload), ("result.json", result)):
        with (root / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (root / "README.txt").write_text(
        "Chronological diagnostics. Each domain reports its own uncertainty and sample sufficiency.\n"
        "No finding authorizes trading or establishes strategy invalidity.\n"
        + "\n".join(
            f"window {w['index']}: "
            + ", ".join(f"{d}={f['status']}" for d, f in w["findings"].items())
            for w in result["windows"]
        ),
        encoding="utf-8",
    )
    return result
