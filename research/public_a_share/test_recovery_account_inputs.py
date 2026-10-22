"""Recovered overlays cannot alter the frozen calendar or market inputs."""

import numpy as np
import pandas as pd
import pytest
from recovery_account_inputs import FIELDS, apply_overlay


def test_overlay_keeps_market_inputs_and_rejects_axis_changes(tmp_path):
    original = pd.DataFrame([[1.0]], index=pd.to_datetime(["2023-01-03"]), columns=["000001"])
    prices = {"close": original.copy()}
    panel = {"descriptors": {field: original.copy() for field in FIELDS}, "prices": prices}
    bundle = {"panel": panel, "v2": panel}
    for field in FIELDS:
        (original * 2).to_parquet(tmp_path / (field + ".parquet"))
    result = apply_overlay(bundle, tmp_path)
    assert result["panel"]["prices"] is prices
    np.testing.assert_array_equal(panel["descriptors"][FIELDS[0]], [[1]])
    np.testing.assert_array_equal(result["panel"]["descriptors"][FIELDS[0]], [[2]])
    original.rename(columns={"000001": "000002"}).to_parquet(tmp_path / (FIELDS[0] + ".parquet"))
    with pytest.raises(ValueError, match="axis"):
        apply_overlay(bundle, tmp_path)
