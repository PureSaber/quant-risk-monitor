"""Preserve exact ordinary-install source bytes and producer scripts by SHA-256."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from study_v3_archive import checked_copy, digest, write_json


def run(out):
    rows, packages = {}, {}
    for name in ("quant-risk-monitor", "quant-portfolio", "a-share-multifactor", "quant-data-kit", "quant-execution", "quant-lab", "quant-factors"):
        distribution = metadata.distribution(name)
        packages[name] = distribution.version
        for item in distribution.files or ():
            path = Path(distribution.locate_file(item)).resolve()
            if path.is_file() and path.suffix in {".py", ".csv", ".json"}:
                sha = digest(path)
                checked_copy(path, out / "objects" / sha, sha)
                rows[name+"/"+str(item).replace("\\", "/")] = {"sha256": sha, "installed_path": str(path)}
    for path in Path(__file__).parent.glob("*.py"):
        sha = digest(path)
        checked_copy(path, out / "objects" / sha, sha)
        rows["producer/"+path.name] = {"sha256": sha, "source_path": str(path)}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    receipt = {"captured_at": timestamp, "packages": packages, "files": rows,
               "installation": "ordinary installs; exact source bytes retained"}
    write_json(out / (timestamp+".json"), receipt)
    print(timestamp, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
