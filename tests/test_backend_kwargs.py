from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llmrunner import Runner, Task, client_kwarg


class CaptureCreate:
    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first

    async def __call__(self, *, messages, **kwargs):
        self.calls.append({"messages": list(messages), "kwargs": dict(kwargs)})
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("temporary failure")
        return {"choices": [{"message": {"content": json.dumps({"id": messages[0]["content"]})}}]}


def make_runner(create, **kwargs):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return Runner(client=client, show_progress=False, **kwargs)


def make_task(note_id="note", **kwargs):
    return Task(messages=[{"role": "user", "content": note_id}], note_id=note_id, **kwargs)


def test_client_kwarg_and_independent_task_defaults():
    assert client_kwarg() == {}
    options = client_kwarg(max_tokens=100, response_format={"type": "json_object"})
    task = make_task(backend_kwargs=options)
    assert task.backend_kwargs == options
    assert task.backend_kwargs is not options
    task.backend_kwargs["max_tokens"] = 200
    assert options["max_tokens"] == 100
    first, second = make_task(), make_task(backend_kwargs=None)
    first.backend_kwargs["temperature"] = 0
    assert second.backend_kwargs == {}


@pytest.mark.parametrize("invalid", [[], "options", 42])
def test_task_rejects_non_dictionary_backend_kwargs(invalid):
    with pytest.raises(TypeError, match="backend_kwargs must be a dictionary"):
        make_task(backend_kwargs=invalid)


@pytest.mark.parametrize("stream", [False, True])
def test_task_options_override_defaults_without_leaking_metadata_or_sibling_options(stream):
    create = CaptureCreate()
    runner = make_runner(create)
    schemas = [{"type": "json_schema", "json_schema": {"name": f"Group{i}"}} for i in range(2)]
    tasks = [
        make_task("A", data_template="local", backend_kwargs=client_kwarg(
            response_format=schemas[0], max_tokens=128, temperature=0,
        )),
        make_task("B", backend_kwargs={"response_format": schemas[1]}),
        make_task("C"),
    ]
    defaults = {"type": "json_object"}
    results = runner.run(tasks, model="batch-model", temperature=0.5, max_tokens=512,
                         response_format=defaults, stream=stream, interactive=False)
    assert all(task.status == "success" for task in results)
    calls = {call["messages"][0]["content"]: call["kwargs"] for call in create.calls}
    assert calls["A"] == {"model": "batch-model", "temperature": 0, "max_tokens": 128,
                           "response_format": schemas[0], "stream": stream}
    assert calls["B"] == {"model": "batch-model", "temperature": 0.5, "max_tokens": 512,
                           "response_format": schemas[1], "stream": stream}
    assert calls["C"] == {"model": "batch-model", "temperature": 0.5, "max_tokens": 512,
                           "response_format": defaults, "stream": stream}
    assert defaults == {"type": "json_object"}
    assert tasks[1].backend_kwargs == {"response_format": schemas[1]}


def test_task_can_supply_or_override_model():
    create = CaptureCreate()
    runner = make_runner(create)
    tasks = [make_task("A", backend_kwargs=client_kwarg(model="model-a")),
             make_task("B", backend_kwargs=client_kwarg(model="model-b"))]
    runner.run(tasks, stream=False, interactive=False)
    assert [call["kwargs"]["model"] for call in create.calls] == ["model-a", "model-b"]


@pytest.mark.parametrize("reserved", ["messages", "stream"])
def test_reserved_task_options_fail_before_any_requests(reserved):
    create = CaptureCreate()
    runner = make_runner(create)
    with pytest.raises(ValueError, match="runner-controlled"):
        runner.run([make_task("valid"), make_task("invalid", backend_kwargs={reserved: None})],
                   model="model", interactive=False)
    assert create.calls == []


def test_mutated_invalid_options_fail_before_any_requests():
    create = CaptureCreate()
    task = make_task()
    task.backend_kwargs = []
    with pytest.raises(TypeError, match="task.backend_kwargs must be a dictionary"):
        make_runner(create).run([task], model="model", interactive=False)
    assert create.calls == []


def test_missing_effective_model_fails_before_any_requests():
    create = CaptureCreate()
    with pytest.raises(ValueError, match="requires `model`"):
        make_runner(create).run([make_task("A", backend_kwargs={"model": "model-a"}), make_task("B")],
                                interactive=False)
    assert create.calls == []


def test_backend_options_apply_to_transport_retries():
    create = CaptureCreate(fail_first=True)
    options = client_kwarg(max_tokens=100, response_format={"type": "json_object"})
    results = make_runner(create, max_retries=1).run(
        [make_task(backend_kwargs=options)], model="model", max_tokens=200,
        stream=False, interactive=False,
    )
    assert results[0].status == "success"
    assert len(create.calls) == 2
    assert create.calls[0]["kwargs"] == create.calls[1]["kwargs"]
    assert create.calls[1]["kwargs"]["max_tokens"] == 100


def test_pipeline_followup_uses_updated_task_options():
    create = CaptureCreate()

    def pipeline(task):
        if len(task.messages) == 2:
            task.backend_kwargs["max_tokens"] = 300
            task.messages.append({"role": "user", "content": "follow up"})
            return task
        return json.loads(task.output)

    results = make_runner(create).run(
        [make_task(backend_kwargs=client_kwarg(max_tokens=100))], pipeline=[pipeline],
        model="model", stream=False, interactive=False,
    )
    assert results[0].status == "success"
    assert [call["kwargs"]["max_tokens"] for call in create.calls] == [100, 300]


def test_replacement_task_can_preserve_backend_options():
    create = CaptureCreate()

    def pipeline(task):
        if len(task.messages) == 2:
            return Task(messages=task.messages + [{"role": "user", "content": "follow up"}],
                        backend_kwargs=task.backend_kwargs)
        return task.output

    results = make_runner(create).run(
        [make_task(backend_kwargs=client_kwarg(max_tokens=100))], pipeline=[pipeline],
        model="model", max_tokens=200, stream=False, interactive=False,
    )
    assert results[0].status == "success"
    assert [call["kwargs"]["max_tokens"] for call in create.calls] == [100, 100]


def test_backend_options_work_with_sync_client():
    calls = []

    def create(*, messages, **kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    results = make_runner(create).run([make_task(backend_kwargs=client_kwarg(max_tokens=64))],
                                     model="model", stream=False, interactive=False)
    assert results[0].output == "ok"
    assert calls[0]["max_tokens"] == 64
