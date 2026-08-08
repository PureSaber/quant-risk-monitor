from __future__ import annotations

import pandas as pd

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
    aligned = pd.concat([current.rename("cur"), baseline.rename("base")], axis=1).dropna()
    if aligned.empty:
        return []
    diff = aligned["cur"] - aligned["base"]
    std = float(diff.std(ddof=0)) or 1.0
    z = diff / std
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
