from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .types import Task

PipelineStage = Callable[[Task], Task | Any] | Callable[[Task], Awaitable[Task | Any]]


class _NullProgress:
    def update(self, _: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


class Runner:
    def __init__(
        self,
        client: Any,
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

        create = self._resolve_openai_create(client)
        self.client = client
        self._create = self._wrap_create(create)
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.show_progress = show_progress
        self.last_results: list[Task] = []
        self._current_tasks: list[asyncio.Task[None]] = []
        self._interrupted = False
        self._states: dict[int, str] = {}

    def run(
        self,
        inputs: Sequence[Task | list[dict[str, Any]]],
        pipeline: Sequence[PipelineStage] | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | Sequence[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        timeout: float | None = None,
        **request_kwargs: Any,
    ) -> list[Task]:
        return asyncio.run(
            self.arun(
                inputs,
                pipeline=pipeline,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                timeout=timeout,
                **request_kwargs,
            )
        )

    async def arun(
        self,
        inputs: Sequence[Task | list[dict[str, Any]]],
        pipeline: Sequence[PipelineStage] | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | Sequence[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        timeout: float | None = None,
        **request_kwargs: Any,
    ) -> list[Task]:
        request_kwargs = self._build_request_kwargs(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            timeout=timeout,
            extra_kwargs=request_kwargs,
        )
        self._validate_request_kwargs(request_kwargs)
        self._interrupted = False
        self._init_results(inputs)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        progress = self._make_progress(total=len(inputs))
        active_pipeline = list(pipeline or [])
        self._current_tasks = [
            asyncio.create_task(
                self._run_one(
                    task=self.last_results[index],
                    pipeline=active_pipeline,
                    semaphore=semaphore,
                    request_kwargs=request_kwargs,
                )
            )
            for index in range(len(inputs))
        ]

        try:
            for current in asyncio.as_completed(self._current_tasks):
                try:
                    await current
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

    def get_successes(self) -> list[Task]:
        return [result for result in self.last_results if result.status == "success"]

    def get_failures(self) -> list[Task]:
        return [result for result in self.last_results if result.status in {"error", "cancelled"}]

    def _init_results(self, inputs: Sequence[Task | list[dict[str, Any]]]) -> None:
        self.last_results = [self._normalize_task_input(index, item) for index, item in enumerate(inputs)]
        self._states = {index: "pending" for index in range(len(inputs))}

    def _normalize_task_input(self, index: int, item: Task | list[dict[str, Any]]) -> Task:
        if isinstance(item, Task):
            item.index = index
            return item

        return Task(index=index, messages=list(item))

    async def _run_one(
        self,
        task: Task,
        pipeline: Sequence[PipelineStage],
        semaphore: asyncio.Semaphore,
        request_kwargs: dict[str, Any],
    ) -> None:
        while True:
            self._states[task.index] = "running"
            text = await self._generate_with_retries(task, semaphore, request_kwargs)
            if task.status in {"error", "cancelled"}:
                return

            task.output = text
            task.metadata["last_text"] = text
            task.messages.append({"role": "assistant", "content": text})

            if not pipeline:
                task.status = "success"
                self._states[task.index] = "success"
                return

            try:
                final_output = await self._run_pipeline(task, pipeline)
            except Exception as exc:
                task.status = "error"
                task.error = str(exc)
                task.metadata["error_type"] = type(exc).__name__
                self._states[task.index] = "error"
                return

            if isinstance(final_output, Task):
                task = self._replace_task(task, final_output)
                continue

            task.output = final_output
            task.status = "success"
            self._states[task.index] = "success"
            return

    async def _generate_with_retries(
        self,
        task: Task,
        semaphore: asyncio.Semaphore,
        request_kwargs: dict[str, Any],
    ) -> str | None:
        max_attempts = self.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            task.attempts = attempt
            task.error = None

            try:
                async with semaphore:
                    task.metadata["total_backend_calls"] = task.metadata.get("total_backend_calls", 0) + 1
                    response = await self._create(messages=task.messages, **request_kwargs)
                    text = self._extract_text(response)
            except asyncio.CancelledError:
                task.status = "cancelled"
                task.error = "Request cancelled"
                self._states[task.index] = "cancelled"
                raise
            except Exception as exc:
                if attempt < max_attempts:
                    task.metadata["last_error"] = str(exc)
                    task.metadata["retry_count"] = attempt
                    if self.retry_delay:
                        await asyncio.sleep(self.retry_delay)
                    continue

                task.status = "error"
                task.error = str(exc)
                task.metadata["error_type"] = type(exc).__name__
                task.metadata["retry_count"] = attempt - 1
                self._states[task.index] = "error"
                return None

            task.metadata["retry_count"] = attempt - 1
            return text

        return None

    async def _run_pipeline(self, task: Task, pipeline: Sequence[PipelineStage]) -> Task | Any:
        current: Task | Any = task
        for stage in pipeline:
            if not isinstance(current, Task):
                return current

            result = stage(current)
            if inspect.isawaitable(result):
                result = await result
            current = result

        return current

    def _replace_task(self, current: Task, replacement: Task) -> Task:
        replacement.index = current.index
        self.last_results[current.index] = replacement
        return replacement

    async def _cancel_unfinished_tasks(self) -> None:
        unfinished = [task for task in self._current_tasks if not task.done()]
        for task in unfinished:
            task.cancel()

        if unfinished:
            await asyncio.gather(*unfinished, return_exceptions=True)

    def _finalize_cancelled_slots(self) -> None:
        for task in self.last_results:
            if task.status is None:
                task.status = "cancelled"
                if task.error is None:
                    task.error = "Batch interrupted"

    def _make_progress(self, total: int) -> Any:
        if not self.show_progress or total == 0:
            return _NullProgress()

        try:
            from tqdm.auto import tqdm
        except ImportError:
            return _NullProgress()

        return tqdm(total=total, desc="LLM requests", unit="req")

    def _resolve_openai_create(self, client: Any) -> Callable[..., Any]:
        create = getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)
        if callable(create):
            return create

        raise NotImplementedError(
            "Runner currently supports official OpenAI clients exposing `client.chat.completions.create(...)` only."
        )

    def _wrap_create(self, create: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
        async def call(**kwargs: Any) -> Any:
            result = create(**kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        return call

    def _validate_request_kwargs(self, request_kwargs: dict[str, Any]) -> None:
        if not request_kwargs.get("model"):
            raise ValueError("Runner requires `model` to be passed to `run()` or `arun()`.")

    def _build_request_kwargs(
        self,
        *,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: str | Sequence[str] | None,
        presence_penalty: float | None,
        frequency_penalty: float | None,
        timeout: float | None,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {"model": model}
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if stop is not None:
            request_kwargs["stop"] = stop
        if presence_penalty is not None:
            request_kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            request_kwargs["frequency_penalty"] = frequency_penalty
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        request_kwargs.update(extra_kwargs)
        return request_kwargs

    def _extract_text(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            raise ValueError("OpenAI response did not include any choices")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None and isinstance(first_choice, dict):
            message = first_choice.get("message")
        if message is None:
            raise ValueError("OpenAI response did not include a message")

        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        return self._normalize_content(content)

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                item_type = getattr(item, "type", None)
                item_text = getattr(item, "text", None)
                if item_type == "text" and isinstance(item_text, str):
                    parts.append(item_text)
                    continue

                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])

            text = "".join(parts).strip()
            if text:
                return text

        raise ValueError("OpenAI response content could not be normalized into plain text")
