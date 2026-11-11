from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class Task(SimpleNamespace):
    """LLM work item with local metadata and per-task backend request options."""

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
        backend_kwargs: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        if backend_kwargs is not None and not isinstance(backend_kwargs, dict):
            raise TypeError("backend_kwargs must be a dictionary or None")
        super().__init__(
            index=index,
            messages=messages,
            output=output,
            status=status,
            error=error,
            attempts=attempts,
            metadata={} if metadata is None else metadata,
            backend_kwargs={} if backend_kwargs is None else dict(backend_kwargs),
            **extra,
        )


def client_kwarg(**kwargs: Any) -> dict[str, Any]:
    """Build Task.backend_kwargs using keyword arguments, e.g. client_kwarg(max_tokens=512)."""
    return kwargs


ResultRecord = Task
