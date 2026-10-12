"""Independent deterministic null/change controls, not measured market performance."""

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def fixture(*, changed=True, count=256):
    domains = [
        "data_drift",
        "forecast_error",
        "signal_decay",
        "risk_miscalibration",
        "cost_miscalibration",
        "strategy_performance",
        "operational_failures",
    ]
    plan = {
        "schema": "quant.model-diagnostic-plan/v1",
        "frozen_at": "2020-01-01T00:00:00Z",
        "reference_rows": count,
        "window_rows": count,
        "block_rows": 1,
        "min_blocks": 20,
        "max_windows": 2,
        "family_alpha": 0.05,
        "forecast_bound": 0.1,
        "feature_bounds": [0, 1],
        "cost_error_bound_bps": 100,
        "active_return_bound": 0.2,
        "thresholds": {d: 0.1 for d in domains},
        "risk_exceedance_probability": 0.05,
        "currency": "USD",
        "frequency": "daily",
        "independent_blocks_assumed": True,
    }
    rows = []
    for i in range(count * 2):
        at = datetime(2020, 1, 2, tzinfo=timezone.utc) + timedelta(days=i)
        end = at + timedelta(days=1)
        change = changed and i >= count
        rows.append(
            {
                "row_id": str(i),
                "forecast_at": at.isoformat(),
                "period_start": at.isoformat(),
                "period_end": end.isoformat(),
                "outcome_known_at": end.isoformat(),
                "regime_id": "declared-stable",
                "source_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                "prediction": 0.1,
                "target": -0.1 if change else 0.1,
                "feature": 0.9 if change else 0.1,
                "risk_limit": 0.1,
                "realized_return": -0.2 if change else 0,
                "predicted_cost_bps": 0,
                "actual_cost_bps": 100 if change else 0,
                "net_return": -0.2 if change else 0,
                "baseline_return": 0,
                "operation": "failed" if change else "ok",
            }
        )
    return {
        "schema": "quant.model-diagnostics-input/v1",
        "plan": plan,
        "rows": rows,
        "evaluation_as_of": rows[-1]["outcome_known_at"],
        "evidence_kind": "synthetic",
    }


if __name__ == "__main__":
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=False)
    for name, changed in (("null", False), ("change", True)):
        (root / f"{name}.json").write_text(
            json.dumps(fixture(changed=changed), indent=2), encoding="utf-8"
        )
