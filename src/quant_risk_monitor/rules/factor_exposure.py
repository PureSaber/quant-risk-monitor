from __future__ import annotations

import math

import pandas as pd

from quant_risk_monitor.analytics import factor_exposure_drift
from quant_risk_monitor.models import Alert, Severity


def check_factor_exposure_drift(
    current: pd.Series,
    baseline: pd.Series,
    *,
    z_threshold: float = 2.0,
) -> list[Alert]:
    """Alert when factor exposure drifts too far from baseline (z-score)."""
    if not math.isfinite(z_threshold) or z_threshold <= 0:
        raise ValueError("z_threshold must be positive and finite")
    try:
        report = factor_exposure_drift(current, baseline)
    except ValueError as exc:
        return [
            Alert(
                rule_id="factor_exposure_drift_not_evaluable",
                severity=Severity.CRITICAL,
                message="factor exposure drift cannot be evaluated",
                details={"reason": str(exc)},
            )
        ]
    z = report["z_score"]
    worst = z.abs().idxmax()
    worst_z = float(z.loc[worst])
    if abs(worst_z) <= z_threshold:
        return []
    return [
        Alert(
            rule_id="factor_exposure_drift",
            severity=Severity.WARNING,
            message=f"factor {worst} exposure drift z={worst_z:.2f}",
            details={"factor": str(worst), "z_score": round(worst_z, 4)},
        )
    ]
