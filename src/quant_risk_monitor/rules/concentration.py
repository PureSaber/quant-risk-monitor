from __future__ import annotations

import pandas as pd

from quant_risk_monitor.models import Alert, Severity


def check_single_name_weight(weights: pd.Series, max_weight: float) -> list[Alert]:
    if weights.empty:
        return []
    total = float(weights.sum())
    if total <= 0:
        return []
    normalized = weights / total
    worst_symbol = normalized.idxmax()
    worst_weight = float(normalized.max())
    if worst_weight > max_weight:
        return [
            Alert(
                rule_id="single_name_weight",
                severity=Severity.CRITICAL,
                message=f"{worst_symbol} weight {worst_weight:.2%} exceeds {max_weight:.2%}",
                details={"symbol": str(worst_symbol), "weight": round(worst_weight, 6)},
            )
        ]
    return []
