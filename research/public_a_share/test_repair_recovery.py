import gzip
import hashlib

import pytest
from repair_recovery import restore_selection, verify_object


def test_recovery_verifies_decoded_bytes_and_refuses_overwriting_evidence(tmp_path):
    payload = b"fixed historical account\x00\xff"
    source = tmp_path / "object.gz"
    source.write_bytes(gzip.compress(payload, mtime=0))
    meta = {"encoding": "gzip", "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload)}
    target = tmp_path / "restored" / "account"
    verify_object(source, meta, target)
    assert target.read_bytes() == payload
    verify_object(source, meta, target)
    with pytest.raises(ValueError, match="payload changed"):
        verify_object(source, {**meta, "sha256": "0" * 64})
    target.write_bytes(b"another run")
    with pytest.raises(ValueError, match="not overwritten"):
        verify_object(source, meta, target)
    assert target.read_bytes() == b"another run"


def test_recovery_materializes_both_corporate_action_cases_in_every_branch():
    for branch in ("strict", "sensitivity_floor", "sensitivity_ceil"):
        for case in ("016", "046"):
            assert restore_selection(f"run/{branch}/portfolios/{case}-identity/positions.parquet")
    assert restore_selection("run/coverage/portfolio-risk-models.pkl")
    assert not restore_selection("run/strict/portfolios/030-identity/positions.parquet")
