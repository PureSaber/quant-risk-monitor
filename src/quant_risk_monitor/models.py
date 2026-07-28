from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    rule_id: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload


@dataclass
class CheckResult:
    alerts: list[Alert] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(a.severity == Severity.CRITICAL for a in self.alerts)

    def to_dict(self) -> dict[str, Any]:
        return {"alerts": [a.to_dict() for a in self.alerts], "count": len(self.alerts)}
