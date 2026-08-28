from __future__ import annotations

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
    if current.empty or baseline.empty:
        return []
    try:
        report = factor_exposure_drift(current, baseline)
    except ValueError:
        # Preserve the legacy alert API: missing usable overlap produces no alert.
        return []
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
