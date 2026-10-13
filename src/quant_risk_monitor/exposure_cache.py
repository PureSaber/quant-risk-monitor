"""Immutable, content-bound daily market exposure snapshots shared by portfolios."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from quant_risk_monitor.factor_model import ExposureSnapshot, _timestamp


def _canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class DailyExposureStore:
    """One snapshot per cutoff/config/input digest, never per portfolio.

    The producer supplies digests of the actual source partitions and code used.
    A revised input creates a new immutable key rather than rewriting PIT history.
    This cache does not authenticate a producer or discover upstream revisions.
    """

    def __init__(self, directory):
        self.directory = Path(directory)

    def get_or_build(self, *, as_of, configuration, input_digests, build):
        if not input_digests or any(
            not isinstance(x, str) or len(x) != 64 or any(c not in "0123456789abcdef" for c in x)
            for x in input_digests.values()
        ):
            raise ValueError("input_digests must contain SHA-256 values")
        cutoff = _timestamp(as_of, "as_of")
        binding = json.loads(
            _canonical(
                {
                    "as_of": cutoff.isoformat(),
                    "configuration": configuration,
                    "input_digests": input_digests,
                }
            )
        )
        key = _digest(binding)
        path = self.directory / f"{key}.json"
        if path.exists():
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            if envelope["sha256"] != _digest(payload) or payload["binding"] != binding:
                raise ValueError("exposure cache content/binding mismatch")
            snapshot = self._decode(payload["snapshot"])
            self._check_time(snapshot, cutoff)
            return snapshot, True
        snapshot = build()
        if not isinstance(snapshot, ExposureSnapshot):
            raise TypeError("builder must return an ExposureSnapshot")
        self._check_time(snapshot, cutoff)
        payload = {"binding": binding, "snapshot": self._encode(snapshot)}
        envelope = {
            "schema": "quant-risk.daily-exposure-cache/v1",
            "payload": payload,
            "sha256": _digest(payload),
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix="exposure-", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(_canonical(envelope))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return snapshot, False

    @staticmethod
    def _check_time(snapshot, cutoff):
        if snapshot.effective_at != cutoff or snapshot.available_at > cutoff:
            raise ValueError("cached daily snapshot must match its cutoff and already be available")

    @staticmethod
    def _encode(snapshot):
        return {
            "effective_at": snapshot.effective_at.isoformat(),
            "available_at": snapshot.available_at.isoformat(),
            "assets": list(snapshot.values.index),
            "factors": list(snapshot.values.columns),
            "values": snapshot.values.to_numpy().tolist(),
            "source": snapshot.source,
            "market_caps": None if snapshot.market_caps is None else snapshot.market_caps.to_list(),
            "industries": None if snapshot.industries is None else snapshot.industries.to_list(),
            "industry_factors": list(snapshot.industry_factors),
            "provenance": snapshot.provenance,
        }

    @staticmethod
    def _decode(payload):
        assets = payload["assets"]
        return ExposureSnapshot(
            effective_at=payload["effective_at"],
            available_at=payload["available_at"],
            values=pd.DataFrame(payload["values"], index=assets, columns=payload["factors"]),
            source=payload["source"],
            market_caps=None
            if payload["market_caps"] is None
            else pd.Series(payload["market_caps"], index=assets),
            industries=None
            if payload["industries"] is None
            else pd.Series(payload["industries"], index=assets),
            industry_factors=tuple(payload["industry_factors"]),
            provenance=payload["provenance"],
        )
