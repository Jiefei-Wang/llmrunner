from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ResultRecord:
    index: int
    messages: list[dict[str, Any]]
    status: str | None = None
    text: str | None = None
    parsed: Any = None
    error: str | None = None
    attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
