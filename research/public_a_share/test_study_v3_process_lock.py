"""A duplicate process must wait; a crashed owner must not strand a scenario."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from study_v3_process_lock import process_lock


def child(path, code):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent)
    return subprocess.run([sys.executable, "-c", code, str(path)], env=env,
                          capture_output=True, text=True, timeout=15)


def test_duplicate_process_times_out_then_acquires_after_release(tmp_path):
    path = tmp_path / "account.lock"
    code = "from pathlib import Path; import sys; from study_v3_process_lock import process_lock; "
    code += "\nwith process_lock(Path(sys.argv[1]), timeout=.1, poll_seconds=.01): print('owned')"
    with process_lock(path):
        blocked = child(path, code)
        assert blocked.returncode != 0
        assert "TimeoutError" in blocked.stderr
        assert "owned" not in blocked.stdout
    acquired = child(path, code)
    assert acquired.returncode == 0
    assert acquired.stdout.strip() == "owned"


def test_crashed_owner_releases_lock_without_deleting_evidence(tmp_path):
    path = tmp_path / "account.lock"
    result = child(path, "from pathlib import Path; import sys,os; from study_v3_process_lock import process_lock; "
                   "\nwith process_lock(Path(sys.argv[1])): os._exit(7)")
    assert result.returncode == 7
    with process_lock(path, timeout=.1):
        assert path.exists()
    assert path.read_bytes() == b"0"


def test_lock_releases_after_exception_and_validates_timing(tmp_path):
    path = tmp_path / "account.lock"
    with pytest.raises(ValueError, match="boom"), process_lock(path):
        raise ValueError("boom")
    with process_lock(path, timeout=0):
        pass
    with pytest.raises(ValueError, match="timing"), process_lock(path, timeout=-1):
        pass
