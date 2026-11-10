from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from llmrunner import Runner, Task
from llmrunner.dashboard import DashboardModel, JupyterDisplay, NullDisplay


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


def make_chunk(content):
    return {"choices": [{"delta": {"content": content}}]}


class AsyncStream:
    def __init__(self, chunks, fail_after=None):
        self.chunks = list(chunks)
        self.fail_after = fail_after
        self.position = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        if self.fail_after is not None and self.position >= self.fail_after:
            raise RuntimeError("stream interrupted")
        if self.position >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.position]
        self.position += 1
        return chunk

    async def close(self):
        self.closed = True


class SyncStream:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.chunks)

    def close(self):
        self.closed = True


class AsyncStreamingCreate:
    def __init__(self, streams):
        self.streams = list(streams)
        self.calls = []

    async def __call__(self, *, messages, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs["stream"]:
            return self.streams.pop(0)
        return _Response("non-stream")


class SyncStreamingCreate:
    def __init__(self, stream):
        self.stream = stream
        self.calls = []

    def __call__(self, *, messages, **kwargs):
        self.calls.append(dict(kwargs))
        return self.stream


class CaptureDisplay(NullDisplay):
    def __init__(self, model):
        self.model = model


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
        self.calls.append({"index": index, "count": count, "kwargs": dict(kwargs)})
        if index in self.delays:
            await asyncio.sleep(self.delays[index])

        response = self.responses[index]
        if self._is_response_sequence(response):
            response = response[count]
        if isinstance(response, Exception):
            raise response
        return _Response(response)

    def _is_response_sequence(self, response):
        if not isinstance(response, list):
            return False

        for item in response:
            if isinstance(item, Exception):
                return True
            if isinstance(item, str):
                continue
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                return False
            if getattr(item, "type", None) == "text" and isinstance(getattr(item, "text", None), str):
                return False
            return True

        return True


class AsyncMethodCreate:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.counts = {}

    async def create(self, *, messages, **kwargs):
        index = messages[0]["index"]
        count = self.counts.get(index, 0)
        self.counts[index] = count + 1
        self.calls.append({"index": index, "count": count, "kwargs": dict(kwargs)})

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
        self.calls.append({"index": index, "count": count, "kwargs": dict(kwargs)})

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


def make_openai_client_from_methods(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def make_histories(count):
    return [[{"role": "user", "content": f"message-{index}", "index": index}] for index in range(count)]


def parse_json(task: Task):
    return json.loads(task.output)


def repair_json(task: Task):
    try:
        return json.loads(task.output)
    except json.JSONDecodeError as exc:
        task.messages.append(
            {"role": "user", "content": f"Output is not valid JSON: {exc}", "index": task.index}
        )
        return task


def replace_task_instance(task: Task):
    try:
        return json.loads(task.output)
    except json.JSONDecodeError as exc:
        return Task(
            index=task.index,
            messages=task.messages
            + [
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

    results = runner.run(make_histories(3), model="gpt-4o-mini", temperature=0.2)

    assert [result.index for result in results] == [0, 1, 2]
    assert [result.output for result in results] == ["zero", "one", "two"]
    assert all(result.status == "success" for result in results)
    assert runner.last_results is results


def test_successful_batch_returns_ordered_results_for_async_bound_method_client():
    completions = AsyncMethodCreate(responses={0: "zero", 1: "one"})
    runner = Runner(client=make_openai_client_from_methods(completions), max_concurrency=2, show_progress=False)

    results = runner.run(make_histories(2), model="gpt-4o-mini", temperature=0.2)

    assert [result.output for result in results] == ["zero", "one"]
    assert all(result.status == "success" for result in results)


def test_successful_batch_returns_ordered_results_for_sync_client():
    client = make_openai_client(SyncCreate(responses={0: "zero", 1: "one"}))
    runner = Runner(client=client, max_concurrency=2, show_progress=False)

    results = runner.run(make_histories(2), model="gpt-4o-mini", temperature=0.2)

    assert [result.output for result in results] == ["zero", "one"]
    assert all(result.status == "success" for result in results)


def test_runner_accepts_prebuilt_tasks_and_keeps_arbitrary_fields():
    client = make_openai_client(AsyncCreate(responses={0: "zero", 1: "one"}))
    runner = Runner(client=client, max_concurrency=2, show_progress=False)
    tasks = [
        Task(messages=make_histories(1)[0], request_id="req-0", payload={"topic": "ada"}),
        Task(messages=make_histories(2)[1], custom_note=["keep", "me"], metadata={"source": "user"}),
    ]

    results = runner.run(tasks, model="gpt-4o-mini", temperature=0.2)

    assert results[0] is tasks[0]
    assert results[1] is tasks[1]
    assert results[0].index == 0
    assert results[1].index == 1
    assert results[0].request_id == "req-0"
    assert results[0].payload == {"topic": "ada"}
    assert results[1].custom_note == ["keep", "me"]
    assert results[1].metadata == {"source": "user", "last_text": "one", "retry_count": 0, "total_backend_calls": 1}
    assert results[0].messages[-1] == {"role": "assistant", "content": "zero"}


def test_pipeline_stage_returns_parsed_output():
    client = make_openai_client(AsyncCreate(responses={0: json.dumps({"name": "Ada"})}))
    runner = Runner(client=client, show_progress=False)

    results = runner.run(make_histories(1), pipeline=[parse_json], model="gpt-4o-mini")

    assert results[0].status == "success"
    assert results[0].output == {"name": "Ada"}
    assert results[0].metadata["last_text"] == json.dumps({"name": "Ada"})


def test_common_request_settings_are_forwarded():
    create = AsyncCreate(responses={0: "ok"})
    runner = Runner(client=make_openai_client(create), show_progress=False)

    runner.run(
        make_histories(1),
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=64,
        top_p=0.9,
        stop=["END"],
        presence_penalty=0.1,
        frequency_penalty=0.2,
        timeout=30.0,
    )

    forwarded = create.calls[0]["kwargs"]
    assert forwarded["model"] == "gpt-4o-mini"
    assert forwarded["temperature"] == 0.2
    assert forwarded["max_tokens"] == 64
    assert forwarded["top_p"] == 0.9
    assert forwarded["stop"] == ["END"]
    assert forwarded["presence_penalty"] == 0.1
    assert forwarded["frequency_penalty"] == 0.2
    assert forwarded["timeout"] == 30.0


def test_pipeline_stage_can_requeue_with_updated_history():
    client = make_openai_client(AsyncCreate(responses={0: ["not-json", json.dumps({"name": "Ada"})]}))
    runner = Runner(client=client, show_progress=False)

    results = runner.run(make_histories(1), pipeline=[repair_json], model="gpt-4o-mini")

    assert results[0].status == "success"
    assert results[0].output == {"name": "Ada"}
    assert results[0].metadata["total_backend_calls"] == 2
    assert results[0].messages[-2]["content"].startswith("Output is not valid JSON")
    assert results[0].messages[-1] == {"role": "assistant", "content": json.dumps({"name": "Ada"})}


def test_pipeline_can_replace_task_instance():
    client = make_openai_client(AsyncCreate(responses={0: ["not-json", json.dumps({"name": "Ada"})]}))
    runner = Runner(client=client, show_progress=False)

    results = runner.run(make_histories(1), pipeline=[replace_task_instance], model="gpt-4o-mini")

    assert results[0].status == "success"
    assert results[0].output == {"name": "Ada"}
    assert results[0].metadata["total_backend_calls"] == 2


def test_latest_assistant_message_is_preserved_on_success():
    client = make_openai_client(AsyncCreate(responses={0: "final-answer"}))
    runner = Runner(client=client, show_progress=False)

    results = runner.run(make_histories(1), model="gpt-4o-mini")

    assert results[0].status == "success"
    assert results[0].messages[-1] == {"role": "assistant", "content": "final-answer"}


def test_retry_is_applied_per_requeued_turn():
    runner = Runner(client=make_openai_client(RetryAsyncCreate()), max_retries=1, show_progress=False)

    def retrying_pipeline(task: Task):
        parsed = json.loads(task.output)
        if len(task.messages) == 2:
            task.messages.append({"role": "user", "content": "Try again", "index": task.index})
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


def test_async_stream_is_aggregated_and_forwarded():
    stream = AsyncStream(
        [
            make_chunk("hello"),
            make_chunk(None),
            {"choices": []},
            make_chunk([{"type": "text", "text": "\nworld"}]),
        ]
    )
    create = AsyncStreamingCreate([stream])
    runner = Runner(client=make_openai_client(create), show_progress=False)

    results = runner.run(make_histories(1), model="gpt-4o-mini", interactive=False)

    assert results[0].output == "hello\nworld"
    assert results[0].messages[-1] == {"role": "assistant", "content": "hello\nworld"}
    assert create.calls[0]["stream"] is True
    assert stream.closed is True


def test_sync_stream_is_consumed_without_changing_result_contract():
    stream = SyncStream([make_chunk("sync"), make_chunk(" stream")])
    create = SyncStreamingCreate(stream)
    runner = Runner(client=make_openai_client(create), show_progress=False)

    results = runner.run(make_histories(1), model="gpt-4o-mini", interactive=False)

    assert results[0].output == "sync stream"
    assert stream.closed is True
    assert create.calls[0]["stream"] is True


def test_stream_false_uses_full_response_and_forwards_flag():
    create = AsyncStreamingCreate([])
    runner = Runner(client=make_openai_client(create), show_progress=False)

    results = runner.run(
        make_histories(1),
        model="gpt-4o-mini",
        stream=False,
        interactive=False,
    )

    assert results[0].output == "non-stream"
    assert create.calls[0]["stream"] is False


def test_partial_stream_is_discarded_before_retry(monkeypatch):
    first = AsyncStream([make_chunk("discard-me")], fail_after=1)
    second = AsyncStream([make_chunk("kept")])
    create = AsyncStreamingCreate([first, second])
    captured = {}

    def make_capture(model, **kwargs):
        captured["display"] = CaptureDisplay(model)
        return captured["display"]

    monkeypatch.setattr("llmrunner.runner.make_display", make_capture)
    runner = Runner(client=make_openai_client(create), max_retries=1, show_progress=False)

    results = runner.run(make_histories(1), model="gpt-4o-mini")

    assert results[0].output == "kept"
    assert results[0].messages[-1]["content"] == "kept"
    assert results[0].metadata["retry_count"] == 1
    assert captured["display"].model.snapshot(0).generated_chars == len("kept")


def test_dashboard_model_sorts_unfinished_first_and_formats_full_detail():
    tasks = [
        Task(index=0, messages=[{"role": "user", "content": "first\nline"}], status="success"),
        Task(index=1, messages=[{"role": "user", "content": "<unsafe>"}]),
    ]
    model = DashboardModel(tasks)
    model.mark_terminal(tasks[0])
    model.begin_attempt(1)
    model.append_delta(1, "draft")

    rows = model.sorted_snapshots("default", False)

    assert [row.index for row in rows] == [1, 0]
    assert rows[0].generated_chars == 5
    assert "<unsafe>" in rows[0].detail
    assert '"streaming": true' in rows[0].detail


def test_jupyter_dashboard_paginates_and_escapes_expanded_content():
    tasks = [Task(index=index, messages=[{"role": "user", "content": f"<{index}>\nnext"}]) for index in range(12)]
    model = DashboardModel(tasks)
    display = JupyterDisplay(model, show_progress=True)
    display._display = lambda _: None

    async def exercise():
        await display.start()
        assert len(display._table.children) == 11
        assert "1/2" in display._page_label.value
        button_ids = [id(row.expand) for row in display._rows]
        display._rows[0].expand.click()
        expanded = display._rows[0].detail
        assert "&lt;0&gt;" in expanded.value
        assert "&lt;0&gt;\nnext" in expanded.value
        assert "&lt;0&gt;\\nnext" not in expanded.value
        assert expanded.layout.max_height == "320px"
        assert expanded.layout.overflow == "auto"
        model.append_delta(0, "new text")
        display._refresh()
        assert [id(row.expand) for row in display._rows] == button_ids
        display._change_page(1)
        assert display._visible_indices[:2] == [10, 11]
        assert all(row.container.layout.display == "none" for row in display._rows[2:])
        await display.close()

    asyncio.run(exercise())


def test_expanded_messages_render_newlines_tabs_and_streamed_text_directly():
    content = "first\nsecond\tcolumn\r\nthird"
    task = Task(index=0, messages=[{"role": "user", "name": "example", "content": content}])
    model = DashboardModel([task])
    model.begin_attempt(0)
    model.append_delta(0, "streamed\nanswer")
    detail = model.snapshot(0).detail
    assert content in detail
    assert "streamed\nanswer" in detail
    assert "first\\nsecond" not in detail
    assert '"role": "user"' in detail
    assert '"name": "example"' in detail
    assert '"streaming": true' in detail
    assert task.messages == [{"role": "user", "name": "example", "content": content}]


def test_expanded_messages_preserve_literal_escapes_and_backslashes():
    content = r"C:\new\notes and regex \n"
    model = DashboardModel([Task(index=0, messages=[{"role": "user", "content": content}])])
    detail = model.snapshot(0).detail
    assert content in detail
    assert "C:\n" not in detail
    assert "regex \n" not in detail


def test_expanded_messages_keep_structured_content_and_errors_readable():
    message = {"role": "user", "content": [{"type": "text", "text": "example"}]}
    task = Task(index=0, messages=[message], error="first\nsecond")
    detail = DashboardModel([task]).snapshot(0).detail
    assert json.dumps(message, ensure_ascii=False, indent=2) in detail
    assert "ERROR\nfirst\nsecond" in detail


def test_expanded_detail_keeps_scroll_container_during_streaming():
    tasks = [Task(index=0, messages=[{"role": "user", "content": "long line\n" * 100}])]
    model = DashboardModel(tasks)
    display = JupyterDisplay(model, show_progress=False)
    model.begin_attempt(0)
    display._refresh(force=True)
    display._rows[0].expand.click()
    detail = display._rows[0].detail
    original_widget_id = detail.model_id
    original_layout_id = detail.layout.model_id
    original_layout = detail.layout.get_state()

    assert detail.layout.max_height == "320px"
    assert detail.layout.overflow == "auto"
    assert "overflow" not in detail.value
    assert "max-height" not in detail.value
    for index in range(10):
        model.append_delta(0, f"chunk {index}\n")
        display._refresh()
        assert display._rows[0].container.children[1] is detail
        assert detail.model_id == original_widget_id
        assert detail.layout.model_id == original_layout_id
        assert detail.layout.get_state() == original_layout
        assert f"chunk {index}" in detail.value
        assert "overflow" not in detail.value
        assert "max-height" not in detail.value

    display._rows[0].expand.click()
    assert detail.layout.display == "none"
    display._rows[0].expand.click()
    assert detail.layout.display == ""
    assert detail.model_id == original_widget_id
    assert "chunk 9" in detail.value


def test_jupyter_refresh_uses_native_comms_without_polling(monkeypatch):
    display = JupyterDisplay(DashboardModel([Task(index=0, messages=[])]), show_progress=False)

    async def dispatch_shell(msg, subshell_id=None, *, concurrent=False):
        pass

    kernel = SimpleNamespace(dispatch_shell=dispatch_shell, _main_asyncio_lock=asyncio.Lock())
    original_lock = kernel._main_asyncio_lock
    monkeypatch.setattr("IPython.get_ipython", lambda: SimpleNamespace(kernel=kernel))
    monkeypatch.setattr("llmrunner.dashboard.is_jupyter", lambda: True)

    def unexpected_polling():
        pytest.fail("Native async comm handling must not start legacy polling")

    monkeypatch.setitem(sys.modules, "jupyter_ui_poll", SimpleNamespace(ui_events=unexpected_polling))
    entered = []

    async def refresh(poll=None):
        entered.append(poll)
        assert kernel._main_asyncio_lock is original_lock

    monkeypatch.setattr(display, "_run_refresh_loop", refresh)
    asyncio.run(display._refresh_loop())
    assert entered == [None]


@pytest.mark.parametrize("has_dispatch_shell", [True, False])
def test_jupyter_refresh_retains_polling_for_older_kernels(monkeypatch, has_dispatch_shell):
    display = JupyterDisplay(DashboardModel([Task(index=0, messages=[])]), show_progress=False)

    async def dispatch_shell(msg, subshell_id=None):
        pass

    kernel = SimpleNamespace(dispatch_shell=dispatch_shell) if has_dispatch_shell else SimpleNamespace()
    monkeypatch.setattr("IPython.get_ipython", lambda: SimpleNamespace(kernel=kernel))
    monkeypatch.setattr("llmrunner.dashboard.is_jupyter", lambda: True)
    monkeypatch.setattr("llmrunner.dashboard.REFRESH_INTERVAL", 0)
    events = []

    async def poll(count):
        events.append(count)
        display._closed = True

    @asynccontextmanager
    async def ui_events():
        events.append("enter")
        try:
            yield poll
        finally:
            events.append("exit")

    monkeypatch.setitem(sys.modules, "jupyter_ui_poll", SimpleNamespace(ui_events=ui_events))
    asyncio.run(display._refresh_loop())
    assert events == ["enter", 10, "exit"]
    assert display._rows[0].index.value == "0"


def test_jupyter_native_refresh_cancels_cleanly(monkeypatch):
    display = JupyterDisplay(DashboardModel([Task(index=0, messages=[])]), show_progress=False)
    display._display = lambda _: None

    async def dispatch_shell(msg, subshell_id=None, *, concurrent=False):
        pass

    kernel = SimpleNamespace(dispatch_shell=dispatch_shell)
    monkeypatch.setattr("IPython.get_ipython", lambda: SimpleNamespace(kernel=kernel))
    monkeypatch.setattr("llmrunner.dashboard.is_jupyter", lambda: True)

    async def exercise():
        await display.start()
        await asyncio.sleep(0)
        assert not display._refresh_task.done()
        await display.close()
        assert display._refresh_task.cancelled()

    asyncio.run(exercise())


def test_terminal_dashboard_supports_paging_and_expansion():
    pytest.importorskip("textual")
    from textual.widgets import DataTable, Static

    from llmrunner.terminal_dashboard import TerminalDashboard

    tasks = [Task(index=index, messages=[{"role": "user", "content": f"message-{index}"}]) for index in range(12)]
    model = DashboardModel(tasks)
    app = TerminalDashboard(model, show_progress=True)

    async def exercise():
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#tasks", DataTable).row_count == 10
            await pilot.press("right")
            await pilot.pause()
            assert app.page == 1
            assert app.query_one("#tasks", DataTable).row_count == 2
            await pilot.press("enter")
            await pilot.pause()
            detail = app.query_one("#detail", Static).render()
            assert "message-10" in detail.plain

    asyncio.run(exercise())


def test_run_rejects_an_active_event_loop():
    runner = Runner(client=make_openai_client(AsyncCreate(responses={0: "ok"})), show_progress=False)

    async def exercise():
        with pytest.raises(RuntimeError, match="await runner.arun"):
            runner.run(make_histories(1), model="gpt-4o-mini", interactive=False)

    asyncio.run(exercise())
