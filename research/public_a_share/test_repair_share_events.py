import json
from pathlib import Path

import pandas as pd
import pytest

from study_v3_portfolios import additional_share_events


@pytest.mark.parametrize("branch", [
    "strict", "sensitivity_floor", "sensitivity_ceil",
    "sensitivity_floor_extended", "sensitivity_ceil_extended",
])
def test_frozen_share_config_constructs_utc_native_events_without_lookahead(branch):
    path = Path(__file__).parent / "failure_repair_configs" / f"share-rules-{branch}.json"
    config = json.loads(path.read_text())
    assert additional_share_events(config, pd.Timestamp("2024-12-25")) == ()
    events = additional_share_events(config, pd.Timestamp("2024-12-26"))
    assert len(events) == 1
    assert events[0].available_at == pd.Timestamp("2024-12-26T01:00:00Z")
    assert events[0].event_id in config["rules"]
