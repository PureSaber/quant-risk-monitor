"""Cross-platform OS locks for duplicate scenario processes.

The file is a stable lock address, not a stale PID marker. The operating system
releases the lock when its owner exits, including an abnormal process exit.
Never unlink it while another process may be waiting on the same address.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from time import monotonic, sleep


@contextmanager
def process_lock(path: Path, *, timeout=14400, poll_seconds=.1):
    if timeout < 0 or poll_seconds <= 0:
        raise ValueError("invalid process lock timing")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        started = monotonic()
        while True:
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, PermissionError):
                if monotonic() - started >= timeout:
                    raise TimeoutError("scenario is already owned by another process: " + str(path)) from None
                sleep(poll_seconds)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
