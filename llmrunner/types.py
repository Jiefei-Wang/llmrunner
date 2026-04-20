from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class Task(SimpleNamespace):
    def __init__(
        self,
        *,
        index: int | None = None,
        messages: list[dict[str, Any]],
        output: Any = None,
        status: str | None = None,
        error: str | None = None,
        attempts: int = 0,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            index=index,
            messages=messages,
            output=output,
            status=status,
            error=error,
            attempts=attempts,
            metadata={} if metadata is None else metadata,
            **extra,
        )


ResultRecord = Task
