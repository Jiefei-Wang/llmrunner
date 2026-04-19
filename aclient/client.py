from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from .backends.base import Backend
from .types import ResultRecord


class _NullProgress:
    def update(self, _: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


class Client:
    def __init__(
        self,
        backend: Backend,
        max_concurrency: int = 128,
        max_retries: int = 0,
        retry_delay: float = 0.0,
        show_progress: bool = True,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")

        self.backend = backend
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.show_progress = show_progress
        self.last_results: list[ResultRecord] = []
        self._current_tasks: list[asyncio.Task[None]] = []
        self._interrupted = False
        self._states: dict[int, str] = {}

    def run(
        self,
        histories: Sequence[list[dict[str, Any]]],
        parse_fn: Callable[[str], Any] | None = None,
        **backend_kwargs: Any,
    ) -> list[ResultRecord]:
        return asyncio.run(self.arun(histories, parse_fn=parse_fn, **backend_kwargs))

    async def arun(
        self,
        histories: Sequence[list[dict[str, Any]]],
        parse_fn: Callable[[str], Any] | None = None,
        **backend_kwargs: Any,
    ) -> list[ResultRecord]:
        self._interrupted = False
        self._init_results(histories)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        progress = self._make_progress(total=len(histories))
        self._current_tasks = [
            asyncio.create_task(
                self._run_one(
                    index=index,
                    messages=messages,
                    parse_fn=parse_fn,
                    semaphore=semaphore,
                    backend_kwargs=backend_kwargs,
                )
            )
            for index, messages in enumerate(histories)
        ]

        try:
            for task in asyncio.as_completed(self._current_tasks):
                try:
                    await task
                finally:
                    progress.update(1)
        except (asyncio.CancelledError, KeyboardInterrupt):
            self._interrupted = True
            await self._cancel_unfinished_tasks()
            self._finalize_cancelled_slots()
            raise
        finally:
            progress.close()
            self._current_tasks = []
            self._states = {}

        return self.last_results

    def reset(self) -> None:
        self.last_results = []
        self._interrupted = False
        self._states = {}

    def get_successes(self) -> list[ResultRecord]:
        return [result for result in self.last_results if result.status == "success"]

    def get_failures(self) -> list[ResultRecord]:
        return [result for result in self.last_results if result.status in {"error", "cancelled"}]

    def _init_results(self, histories: Sequence[list[dict[str, Any]]]) -> None:
        self.last_results = [
            ResultRecord(index=index, messages=list(messages))
            for index, messages in enumerate(histories)
        ]
        self._states = {index: "pending" for index in range(len(histories))}

    async def _run_one(
        self,
        index: int,
        messages: list[dict[str, Any]],
        parse_fn: Callable[[str], Any] | None,
        semaphore: asyncio.Semaphore,
        backend_kwargs: dict[str, Any],
    ) -> None:
        record = self.last_results[index]
        max_attempts = self.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            record.attempts = attempt
            record.error = None
            self._states[index] = "running"

            try:
                async with semaphore:
                    text = await self.backend.generate(messages, **backend_kwargs)
            except asyncio.CancelledError:
                record.status = "cancelled"
                record.error = "Request cancelled"
                self._states[index] = "cancelled"
                raise
            except Exception as exc:
                if attempt < max_attempts:
                    record.metadata["last_error"] = str(exc)
                    record.metadata["retry_count"] = attempt
                    if self.retry_delay:
                        await asyncio.sleep(self.retry_delay)
                    continue

                record.status = "error"
                record.error = str(exc)
                record.metadata["error_type"] = type(exc).__name__
                record.metadata["retry_count"] = attempt - 1
                self._states[index] = "error"
                return

            record.text = text
            record.metadata["retry_count"] = attempt - 1

            if parse_fn is None:
                record.status = "success"
                self._states[index] = "success"
                return

            try:
                record.parsed = parse_fn(text)
            except Exception as exc:
                record.status = "error"
                record.error = str(exc)
                record.metadata["error_type"] = type(exc).__name__
                record.metadata["parse_failed"] = True
                self._states[index] = "error"
                return

            record.status = "success"
            self._states[index] = "success"
            return

    async def _cancel_unfinished_tasks(self) -> None:
        unfinished = [task for task in self._current_tasks if not task.done()]
        for task in unfinished:
            task.cancel()

        if unfinished:
            await asyncio.gather(*unfinished, return_exceptions=True)

    def _finalize_cancelled_slots(self) -> None:
        for record in self.last_results:
            if record.status is None:
                record.status = "cancelled"
                if record.error is None:
                    record.error = "Batch interrupted"

    def _make_progress(self, total: int) -> Any:
        if not self.show_progress or total == 0:
            return _NullProgress()

        try:
            from tqdm.auto import tqdm
        except ImportError:
            return _NullProgress()

        return tqdm(total=total, desc="LLM requests", unit="req")
