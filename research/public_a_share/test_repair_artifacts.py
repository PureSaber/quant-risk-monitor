import gzip
import json

import pytest

from study_v3_portfolios import write_decisions
from study_v3_report import read


def test_compressed_decisions_are_lossless_deterministic_and_legacy_compatible(tmp_path):
    rows = [{"session": "2024-01-05", "status": "repair_failed", "source": "原始证据",
             "weights": {"000713": 0.05}, "trace": [{"residual": 1e-8}]}]
    plain = tmp_path / "allocation-decisions.json"
    plain.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf8")
    assert read(plain) == rows
    first = tmp_path / "compressed-a"
    second = tmp_path / "compressed-b"
    first.mkdir()
    second.mkdir()
    for folder in (first, second):
        write_decisions(folder, rows)
        assert read(folder / plain.name) == rows
    name = "allocation-decisions.json.gz"
    assert (first / name).read_bytes() == (second / name).read_bytes()
    assert gzip.decompress((first / name).read_bytes()).decode("utf8") == json.dumps(
        rows, ensure_ascii=False, indent=2, allow_nan=False
    )
    assert not (first / plain.name).exists()


def test_ambiguous_or_corrupt_compressed_decisions_are_rejected(tmp_path):
    write_decisions(tmp_path, [{"quantity": 135}])
    plain = tmp_path / "allocation-decisions.json"
    plain.write_text("[]")
    with pytest.raises(ValueError, match="ambiguous"):
        read(plain)
    plain.unlink()
    compressed = tmp_path / "allocation-decisions.json.gz"
    compressed.write_bytes(compressed.read_bytes()[:-5])
    with pytest.raises(EOFError):
        read(plain)
