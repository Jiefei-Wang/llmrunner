from __future__ import annotations

import asyncio
import json

import pytest

from aclient import Backend, Client


class FakeBackend(Backend):
    def __init__(self, responses, delays=None):
        self.responses = responses
        self.delays = delays or {}
        self.calls = []

    async def generate(self, messages, **kwargs):
        index = messages[0]["index"]
        self.calls.append(index)
        if index in self.delays:
            await asyncio.sleep(self.delays[index])

        response = self.responses[index]
        if isinstance(response, Exception):
            raise response
        return response


class RetryBackend(Backend):
    def __init__(self):
        self.counts = {}

    async def generate(self, messages, **kwargs):
        index = messages[0]["index"]
        count = self.counts.get(index, 0) + 1
        self.counts[index] = count
        if count == 1:
            raise RuntimeError(f"temporary failure {index}")
        return json.dumps({"index": index, "attempt": count})


class ControlledBackend(Backend):
    def __init__(self):
        self.started = asyncio.Event()
        self.finish_first = asyncio.Event()
        self.block_second = asyncio.Event()
        self.call_count = 0

    async def generate(self, messages, **kwargs):
        self.call_count += 1
        index = messages[0]["index"]
        self.started.set()

        if index == 0:
            await self.finish_first.wait()
            return "first-done"

        await self.block_second.wait()
        return "second-done"


def make_histories(count):
    return [[{"role": "user", "content": f"message-{index}", "index": index}] for index in range(count)]


def test_successful_batch_returns_ordered_results():
    histories = make_histories(3)
    backend = FakeBackend(
        responses={0: "zero", 1: "one", 2: "two"},
        delays={0: 0.02, 1: 0.0, 2: 0.01},
    )
    client = Client(backend=backend, max_concurrency=3, show_progress=False)

    results = client.run(histories)

    assert [result.index for result in results] == [0, 1, 2]
    assert [result.text for result in results] == ["zero", "one", "two"]
    assert all(result.status == "success" for result in results)
    assert client.last_results is results


def test_one_backend_failure_does_not_fail_batch():
    histories = make_histories(3)
    backend = FakeBackend(responses={0: "ok-0", 1: RuntimeError("boom"), 2: "ok-2"})
    client = Client(backend=backend, max_concurrency=3, show_progress=False)

    results = client.run(histories)

    assert results[0].status == "success"
    assert results[1].status == "error"
    assert results[1].error == "boom"
    assert results[2].status == "success"


def test_parse_failure_preserves_raw_text_and_error():
    histories = make_histories(1)
    backend = FakeBackend(responses={0: "not-json"})
    client = Client(backend=backend, show_progress=False)

    results = client.run(histories, parse_fn=json.loads)

    assert results[0].status == "error"
    assert results[0].text == "not-json"
    assert results[0].parsed is None
    assert "Expecting value" in results[0].error


def test_retry_allows_eventual_success():
    histories = make_histories(2)
    backend = RetryBackend()
    client = Client(backend=backend, max_retries=1, show_progress=False)

    results = client.run(histories, parse_fn=json.loads)

    assert [result.status for result in results] == ["success", "success"]
    assert [result.attempts for result in results] == [2, 2]
    assert [result.metadata["retry_count"] for result in results] == [1, 1]
    assert results[0].parsed["attempt"] == 2


def test_last_results_initialized_before_completion():
    backend = FakeBackend(responses={0: "ok"}, delays={0: 0.02})
    client = Client(backend=backend, show_progress=False)

    async def check_initialization():
        task = asyncio.create_task(client.arun(make_histories(1)))
        await asyncio.sleep(0)
        assert len(client.last_results) == 1
        assert client.last_results[0].status is None
        await task

    asyncio.run(check_initialization())


def test_cancellation_keeps_completed_results():
    backend = ControlledBackend()
    client = Client(backend=backend, max_concurrency=2, show_progress=False)
    histories = make_histories(2)

    async def cancel_mid_run():
        task = asyncio.create_task(client.arun(histories))
        await backend.started.wait()
        backend.finish_first.set()
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_mid_run())

    assert client.last_results[0].status == "success"
    assert client.last_results[0].text == "first-done"
    assert client.last_results[1].status == "cancelled"
