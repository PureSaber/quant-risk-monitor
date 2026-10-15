"""Independent arithmetic and evidence-integrity checks for the bounded study."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/public_a_share"))

from study_v3_archive import checked_copy, contained, digest, write_json
from study_v3_risk import factor_variance, forecast_grid, paired_summary


def test_archive_restore_detects_tampering_and_source_races(tmp_path):
    source, restored = tmp_path / "original", tmp_path / "restored"
    source.write_bytes(b"public original evidence")
    sha = digest(source)
    checked_copy(source, restored, sha)
    assert restored.read_bytes() == source.read_bytes()
    checked_copy(source, restored, sha)
    restored.write_bytes(b"changed")
    with pytest.raises(ValueError, match="destination changed"):
        checked_copy(source, restored, sha)
    source.write_bytes(b"source revision")
    with pytest.raises(ValueError, match="source changed"):
        checked_copy(source, tmp_path / "new", sha)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("relative", ["../outside", "."])
def test_archive_paths_cannot_escape_dedicated_root(tmp_path, relative):
    with pytest.raises(ValueError, match="escapes"):
        contained(tmp_path, relative)


def test_json_evidence_rejects_nonfinite_results(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "receipt.json", {"risk": float("nan")})
    assert not (tmp_path / "receipt.json").exists()


def test_factor_risk_and_gradient_equal_independent_dense_reference():
    rng = np.random.default_rng(200)
    x, a = rng.normal(size=(12, 4)), rng.normal(size=(4, 4))
    f, d = a @ a.T / 100, rng.uniform(0.01, 0.1, 12)
    dense = x @ f @ x.T + np.diag(d)
    benchmark = np.ones(12) / 12
    w = np.arange(1, 13, dtype=float)
    w *= 0.8 / w.sum()
    active = w - benchmark
    for weights in (w, active):
        total, common, specific = factor_variance(x, f, d, weights)
        assert total == pytest.approx(weights @ dense @ weights, rel=1e-13)
        assert total == pytest.approx(common + specific)
        gradient = 2 * (x @ f @ x.T @ weights + d * weights)
        numeric = []
        for i in range(12):
            step = np.eye(12)[i] * 1e-6
            numeric.append(
                (
                    factor_variance(x, f, d, weights + step)[0]
                    - factor_variance(x, f, d, weights - step)[0]
                )
                / 2e-6
            )
        np.testing.assert_allclose(gradient, numeric, atol=1e-10, rtol=1e-8)
    # Cash is real active exposure; normalizing away the uninvested 20% changes risk.
    assert factor_variance(x, f, d, active)[0] != pytest.approx(
        factor_variance(x, f, d, w / 0.8 - benchmark)[0]
    )


def test_date_block_inference_preserves_within_date_correlation_and_missingness():
    rows = []
    for i, day in enumerate(pd.bdate_range("2023-01-01", periods=35)):
        for model, delta in (("control", 0), ("candidate", -0.2)):
            for portfolio in ("one", "two"):
                rows.append(
                    {
                        "date": day,
                        "model": model,
                        "family": "equal",
                        "portfolio": portfolio,
                        "qlike": i / 100 + delta,
                    }
                )
    frame = pd.DataFrame(rows)
    result = paired_summary(frame, ["control", "candidate"], "control")
    duplicated = pd.concat([frame] * 3, ignore_index=True)
    check = paired_summary(duplicated, ["control", "candidate"], "control")
    np.testing.assert_allclose(
        result["comparisons"]["candidate"]["simultaneous_95_interval"], [-0.2, -0.2]
    )
    np.testing.assert_allclose(
        check["comparisons"]["candidate"]["simultaneous_95_interval"], [-0.2, -0.2]
    )
    frame.loc[0, "qlike"] = np.nan
    assert paired_summary(frame, ["control", "candidate"], "control")["complete_dates"] == 34


def test_five_session_windows_never_cross_development_boundary():
    dates = pd.bdate_range("2022-01-01", "2026-09-30")
    grid = forecast_grid(dates)
    assert all(end == dates[i + 5] for i, _, end in grid)
    assert all(day.year != 2023 or end.year == 2023 for _, day, end in grid)
    assert all(grid[i + 1][0] - grid[i][0] >= 5 for i in range(len(grid) - 1))
