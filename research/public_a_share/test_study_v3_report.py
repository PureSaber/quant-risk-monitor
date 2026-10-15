"""Independent account audit rejects corrupt published evidence."""

import json

import pandas as pd
import pytest

from study_v3_report import batch_results, verify_account


@pytest.mark.parametrize("fault", [None, "journal", "integer_residual", "fees", "weights", "cash"])
def test_account_audit_detects_corruption(tmp_path, fault):
    pd.DataFrame({
        "transaction_id": ["one", "one"], "currency": ["CNY", "CNY"],
        "amount_scale": [4, 4], "amount_units": [100, -99 if fault == "journal" else -100],
    }).to_parquet(tmp_path / "cash_ledger.parquet")
    pd.DataFrame({"quantity_scale": [0], "quantity_units": [100]}).to_parquet(tmp_path / "positions.parquet")
    pd.DataFrame({"cash_value_units": [500]}).to_parquet(tmp_path / "portfolio_snapshots.parquet")
    execution = {
        "constraint_residuals": {"budget": 1e-5 if fault == "integer_residual" else 0.0},
        "cash": "-1" if fault == "cash" else "500",
        "non_cash_assets": "0", "post_fee_nav": "1000",
        "costs": "1" if fault == "fees" else "0",
        "integer_weights": {"A": 0.6 if fault == "weights" else 0.5},
    }
    decisions = [{
        "session": "2023-01-03", "nav": 1000, "weights": {"A": 0.5},
        "diagnostics": {"max_constraint_residual": 0.0, "rebalance_seconds": 0.1, "execution": execution},
    }]
    (tmp_path / "allocation-decisions.json").write_text(json.dumps(decisions), encoding="utf8")
    if fault:
        with pytest.raises(ValueError):
            verify_account(tmp_path)
    else:
        result = verify_account(tmp_path)
        assert result["integer_fee_cash_reconciliation"]
        assert result["maximum_integer_residual"] == 0


@pytest.mark.parametrize("indices", [list(range(61)), list(range(61)) + [60], list(range(61)) + [62]])
def test_missing_or_duplicated_scenario_cannot_claim_completion(tmp_path, indices):
    logs = tmp_path / "portfolio-logs-test"
    logs.mkdir()
    (logs / "progress.json").write_text(json.dumps([{"scenario": i} for i in indices]), encoding="utf8")
    with pytest.raises(ValueError):
        batch_results(tmp_path, "test", allow_partial=False)
