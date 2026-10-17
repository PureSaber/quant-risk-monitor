import numpy as np
import pandas as pd
from recovery_imputation import block_mask, estimate
from recovery_lifecycle import state_overlay


def sample():
    assets = [f"a{i:02d}" for i in range(30)]
    size = pd.Series(np.arange(30), index=assets, dtype=float)
    values = size * 0.1 - 1
    groups = pd.Series("industry", index=assets)
    return values, groups, size


def test_predictions_do_not_receive_withheld_values():
    values, groups, sizes = sample()
    requested = ["a10", "a11"]
    observed = values.copy()
    observed.loc[requested] = np.nan
    for method in ("industry_median", "industry_size_ridge"):
        prediction, reason = estimate(observed, groups, sizes, requested, method)
        assert prediction.notna().all()
        assert set(reason) == {"estimated_not_observed"}
        assert observed.loc[requested].isna().all()
        poisoned = values.copy()
        poisoned.loc[requested] = 1e12
        poisoned.loc[requested] = np.nan
        same, _ = estimate(poisoned, groups, sizes, requested, method)
        pd.testing.assert_series_equal(same, prediction)


def test_unreliable_predictor_and_unsupported_peer_group_stay_missing():
    values, groups, sizes = sample()
    values.loc[["a10", "a11"]] = np.nan
    sizes.at["a10"] = np.nan
    groups.at["a11"] = "unseen"
    predictions, reasons = estimate(values, groups, sizes, ["a10", "a11"], "industry_size_ridge")
    assert predictions.isna().all()
    assert reasons.tolist() == ["missing_reliable_predictor", "no_comparable_peers"]


def test_block_mask_is_persistent_and_reproducible_without_future_truth():
    values, groups, sizes = sample()
    dates = pd.bdate_range("2023-01-02", periods=130)
    frames = [pd.DataFrame([s] * len(dates), index=dates) for s in [values, groups, sizes]]
    mask = block_mask(*frames, 0.2, 42)
    assert (mask.iloc[:63].to_numpy() == mask.iloc[0].to_numpy()).all()
    assert not mask.iloc[0].equals(mask.iloc[63])
    frames[0].iloc[100:] = 1e12
    repeated = block_mask(*frames, 0.2, 42)
    pd.testing.assert_frame_equal(mask, repeated)


def test_late_lifecycle_evidence_never_changes_earlier_known_state():
    days = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"])
    statuses = pd.DataFrame({"a": [1, np.nan, np.nan]}, index=days)
    events = [
        {
            "asset": "a",
            "effective_date": "2023-01-03",
            "available_at": "2023-01-04T00:00:00+08:00",
            "source_sha256": "hash",
            "review_basis": "issuer notice",
        }
    ]
    result = state_overlay(statuses, events)
    assert result.a.tolist() == ["observed_trading", "unknown", "exited_a_share"]
    assert statuses.a.isna().sum() == 2
