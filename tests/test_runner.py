from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from llmrunner import Runner, Task


class _TextPart:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class AsyncCreate:
    def __init__(self, responses, delays=None):
        self.responses = responses
        self.delays = delays or {}
        self.calls = []
        self.counts = {}

    async def __call__(self, *, messages, **kwargs):
        index = messages[0]["index"]
        count = self.counts.get(index, 0)
        self.counts[index] = count + 1
        self.calls.append((index, count, kwargs["model"]))
        if index in self.delays:
            await asyncio.sleep(self.delays[index])

        response = self.responses[index]
        if isinstance(response, list):
            response = response[count]
        if isinstance(response, Exception):
            raise response
        return _Response(response)


class SyncCreate:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.counts = {}

    def __call__(self, *, messages, **kwargs):
        index = messages[0]["index"]
        count = self.counts.get(index, 0)
        self.counts[index] = count + 1
        self.calls.append((index, count, kwargs["model"]))

        response = self.responses[index]
        if isinstance(response, list):
            response = response[count]
        if isinstance(response, Exception):
            raise response
        return _Response(response)


class RetryAsyncCreate:
    def __init__(self):
        self.counts = {}

    async def __call__(self, *, messages, **kwargs):
        index = messages[0]["index"]
        turn = len(messages) - 1
        count = self.counts.get((index, turn), 0) + 1
        self.counts[(index, turn)] = count
        if count == 1:
            raise RuntimeError(f"temporary failure {index}-{turn}")
        return _Response(json.dumps({"index": index, "turn": turn, "attempt": count}))


class ControlledAsyncCreate:
    def __init__(self):
        self.started = asyncio.Event()
        self.finish_first = asyncio.Event()
        self.block_second = asyncio.Event()
        self.call_count = 0

    async def __call__(self, *, messages, **kwargs):
        self.call_count += 1
        index = messages[0]["index"]
        self.started.set()

        if index == 0:
            await self.finish_first.wait()
            return _Response("first-done")

        await self.block_second.wait()
        return _Response("second-done")


class UnsupportedClient:
    pass


def make_openai_client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def make_histories(count):
    return [[{"role": "user", "content": f"message-{index}", "index": index}] for index in range(count)]


def parse_json(task: Task):
    return json.loads(task.output)


def repair_json(task: Task):
    try:
        return json.loads(task.output)
    except json.JSONDecodeError as exc:
        task.history += [
            {"role": "assistant", "content": task.output, "index": task.index},
            {"role": "user", "content": f"Output is not valid JSON: {exc}", "index": task.index},
        ]
        return task


def replace_task_instance(task: Task):
    try:
        return json.loads(task.output)
    except json.JSONDecodeError as exc:
        return Task(
            index=task.index,
            history=task.history
            + [
                {"role": "assistant", "content": task.output, "index": task.index},
                {"role": "user", "content": f"Fix JSON: {exc}", "index": task.index},
            ],
            output=task.output,
            metadata=task.metadata,
            attempts=task.attempts,
        )


async def async_parse_json(task: Task):
    await asyncio.sleep(0)
    return json.loads(task.output)


def test_successful_batch_returns_ordered_results_for_async_client():
    client = make_openai_client(
        AsyncCreate(
            responses={0: "zero", 1: "one", 2: "two"},
            delays={0: 0.02, 1: 0.0, 2: 0.01},
        )
    )
    runner = Runner(client=client, max_concurrency=3, show_progress=False)

    results = runner.run(make_histories(3), model="gpt-4o-mini")

    assert [result.index for result in results] == [0, 1, 2]
    assert [result.output for result in results] == ["zero", "one", "two"]
    assert all(result.status == "success" for result in results)
    assert runner.last_results is results


def test_successful_batch_returns_ordered_results_for_sync_client():
    client = make_openai_client(SyncCreate(responses={0: "zero", 1: "one"}))
    runner = Runner(client=client, max_concurrency=2, show_progress=False)

    results = runner.run(make_histories(2), model="gpt-4o-mini")

    assert [result.output for result in results] == ["zero", "one"]
    assert all(result.status == "success" for result in results)


def test_pipeline_stage_returns_parsed_output():
    client = make_openai_client(AsyncCreate(responses={0: json.dumps({"name": "Ada"})}))
    runner = Runner(client=client, show_progress=False)

    results = runner.run(make_histories(1), pipeline=[parse_json], model="gpt-4o-mini")

    assert results[0].status == "success"
    assert results[0].output == {"name": "Ada"}
    assert results[0].metadata["last_text"] == json.dumps({"name": "Ada"})


def test_pipeline_stage_can_requeue_with_updated_history():
    client = make_openai_client(AsyncCreate(responses={0: ["not-json", json.dumps({"name": "Ada"})]}))
    runner = Runner(client=client, show_progress=False)

    results = runner.run(make_histories(1), pipeline=[repair_json], model="gpt-4o-mini")

    assert results[0].status == "success"
    assert results[0].output == {"name": "Ada"}
    assert results[0].metadata["total_backend_calls"] == 2
    assert results[0].history[-1]["content"].startswith("Output is not valid JSON")


def test_pipeline_can_replace_task_instance():
    client = make_openai_client(AsyncCreate(responses={0: ["not-json", json.dumps({"name": "Ada"})]}))
    runner = Runner(client=client, show_progress=False)

    results = runner.run(make_histories(1), pipeline=[replace_task_instance], model="gpt-4o-mini")

    assert results[0].status == "success"
    assert results[0].output == {"name": "Ada"}
    assert results[0].metadata["total_backend_calls"] == 2


def test_retry_is_applied_per_requeued_turn():
    runner = Runner(client=make_openai_client(RetryAsyncCreate()), max_retries=1, show_progress=False)

    def retrying_pipeline(task: Task):
        parsed = json.loads(task.output)
        if len(task.history) == 1:
            task.history += [
                {"role": "assistant", "content": task.output, "index": task.index},
                {"role": "user", "content": "Try again", "index": task.index},
            ]
            return task
        return parsed

    results = runner.run(make_histories(1), pipeline=[retrying_pipeline], model="gpt-4o-mini")

    assert results[0].status == "success"
    assert results[0].output == {"index": 0, "turn": 2, "attempt": 2}
    assert results[0].attempts == 2
    assert results[0].metadata["retry_count"] == 1
    assert results[0].metadata["total_backend_calls"] == 4


def test_async_pipeline_stage_is_awaited():
    client = make_openai_client(AsyncCreate(responses={0: json.dumps({"name": "Grace"})}))
    runner = Runner(client=client, show_progress=False)

    results = runner.run(make_histories(1), pipeline=[async_parse_json], model="gpt-4o-mini")

    assert results[0].status == "success"
    assert results[0].output == {"name": "Grace"}


def test_pipeline_exception_marks_only_that_slot_as_error():
    client = make_openai_client(
        AsyncCreate(responses={0: json.dumps({"name": "Ada"}), 1: "not-json"})
    )
    runner = Runner(client=client, max_concurrency=2, show_progress=False)

    results = runner.run(make_histories(2), pipeline=[parse_json], model="gpt-4o-mini")

    assert results[0].status == "success"
    assert results[1].status == "error"
    assert "Expecting value" in results[1].error
    assert results[1].metadata["last_text"] == "not-json"


def test_last_results_initialized_before_completion():
    runner = Runner(
        client=make_openai_client(AsyncCreate(responses={0: "ok"}, delays={0: 0.02})),
        show_progress=False,
    )

    async def check_initialization():
        task = asyncio.create_task(runner.arun(make_histories(1), model="gpt-4o-mini"))
        await asyncio.sleep(0)
        assert len(runner.last_results) == 1
        assert runner.last_results[0].status is None
        await task

    asyncio.run(check_initialization())


def test_cancellation_keeps_completed_results():
    controlled = ControlledAsyncCreate()
    runner = Runner(client=make_openai_client(controlled), max_concurrency=2, show_progress=False)

    async def cancel_mid_run():
        task = asyncio.create_task(runner.arun(make_histories(2), model="gpt-4o-mini"))
        await controlled.started.wait()
        controlled.finish_first.set()
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_mid_run())

    assert runner.last_results[0].status == "success"
    assert runner.last_results[0].output == "first-done"
    assert runner.last_results[1].status == "cancelled"


def test_unsupported_client_raises_clear_error():
    with pytest.raises(NotImplementedError, match="official OpenAI clients"):
        Runner(client=UnsupportedClient())


def test_missing_model_raises_clear_error():
    runner = Runner(client=make_openai_client(AsyncCreate(responses={0: "ok"})), show_progress=False)

    with pytest.raises(ValueError, match="requires `model`"):
        runner.run(make_histories(1))


def test_content_normalization_handles_text_parts():
    content = [_TextPart("Ada"), {"type": "text", "text": " Lovelace"}]
    runner = Runner(client=make_openai_client(AsyncCreate(responses={0: content})), show_progress=False)

    results = runner.run(make_histories(1), model="gpt-4o-mini")

    assert results[0].output == "Ada Lovelace"
