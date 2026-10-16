"""Check economic parity across plain and compressed native-account artifacts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from study_v3_archive import digest, write_json
from study_v3_report import read


def run(plain, compressed, out):
    left, right = [read(folder / "allocation-decisions.json") for folder in (plain, compressed)]
    assert len(left) == len(right) > 0
    maximum = 0.0
    for a, b in zip(left, right, strict=True):
        assert a["session"] == b["session"] and a.get("status") == b.get("status")
        assert a.get("targets") == b.get("targets")
        if "weights" in a:
            wa, wb = pd.Series(a["weights"]), pd.Series(b["weights"])
            assert set(wa.index) == set(wb.index)
            difference = float(np.abs(wa - wb).max())
            maximum = max(maximum, difference)
            assert difference <= 1e-12
            for field in ("cash", "costs", "non_cash_assets", "post_fee_nav"):
                assert a["diagnostics"]["execution"][field] == b["diagnostics"]["execution"][field]
    ignored = {"account_id", "strategy_id", "run_id", "order_id", "fill_id", "cost_id",
               "event_id", "idempotency_key", "transaction_id", "reference_id", "venue_trade_id"}
    checked = {}
    for name in ("returns", "positions", "portfolio_snapshots", "fills", "costs", "cash_ledger"):
        a, b = [pd.read_parquet(folder / (name + ".parquet")) for folder in (plain, compressed)]
        columns = [column for column in a.columns if column not in ignored]
        a, b = a[columns], b[columns]
        if name == "cash_ledger":
            a, b = [frame.sort_values(columns, kind="stable").reset_index(drop=True) for frame in (a, b)]
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        checked[name] = len(a)
    specifications = [read(folder / "specification.json") for folder in (plain, compressed)]
    module_hashes = [
        {k: v for k, v in specification["inputs"].items() if k != "research_code"}
        for specification in specifications
    ]
    assert module_hashes[0] == module_hashes[1]
    result = {
        "status": "passed", "decisions": len(left), "maximum_weight_difference": maximum,
        "targets_equal": True, "exact_economic_rows": checked,
        "ignored_identity_columns": sorted(ignored), "numerical_module_hashes_equal": True,
        "plain_specification_sha256": digest(plain / "specification.json"),
        "compressed_specification_sha256": digest(compressed / "specification.json"),
        "plain_bytes": (plain / "allocation-decisions.json").stat().st_size,
        "compressed_bytes": (compressed / "allocation-decisions.json.gz").stat().st_size,
        "scope": "six historical sessions; validates transport parity, not full-study performance",
    }
    write_json(out, result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("plain", "compressed", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.plain, args.compressed, args.out)
