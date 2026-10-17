import gzip
import json

import pytest
from recovery_inventory import digest
from recovery_restore import contained, run


@pytest.mark.parametrize("key", ["../outside", "/absolute", "C:/outside", "folder\\..\\escape"])
def test_restore_refuses_paths_outside_declared_root(tmp_path, key):
    with pytest.raises(ValueError):
        contained(tmp_path, key)


def test_restore_verifies_both_hashes_and_refuses_overwrite(tmp_path):
    archive = tmp_path / "archive"
    folder = archive / "version"
    folder.mkdir(parents=True)
    original = tmp_path / "source.bin"
    original.write_bytes(b"reviewed original observations")
    packed = folder / "object.gz"
    packed.write_bytes(gzip.compress(original.read_bytes(), mtime=0))
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps({"files": {"run/facts.bin": {
        "sha256": digest(original), "bytes": original.stat().st_size,
        "object": "version/object.gz", "encoding": "gzip", "encoded_sha256": digest(packed),
    }}}))
    target = tmp_path / "restored"
    result = run(manifest, digest(manifest), target, ["run/facts.bin"])
    assert result["unique_objects_verified"] == 1
    assert (target / "run/facts.bin").read_bytes() == original.read_bytes()
    (target / "run/facts.bin").write_bytes(b"newer data must remain")
    with pytest.raises(ValueError, match="not overwritten"):
        run(manifest, digest(manifest), target, ["run/facts.bin"])
    assert (target / "run/facts.bin").read_bytes() == b"newer data must remain"
    with pytest.raises(ValueError, match="manifest digest"):
        run(manifest, "wrong digest")
