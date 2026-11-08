from __future__ import annotations

import json
from pathlib import Path
from queue import Empty
import sys
import textwrap
import time

import pytest


@pytest.mark.parametrize("cancel", [False, True], ids=["complete", "cancel"])
def test_widgets_respond_during_active_jupyter_batch(tmp_path, cancel):
    pytest.importorskip("ipykernel", minversion="7.4")
    jupyter_client = pytest.importorskip("jupyter_client")
    pytest.importorskip("ipywidgets")
    # Keep the incompatible poller installed so the old dashboard regresses.
    pytest.importorskip("jupyter_ui_poll")

    root = Path(__file__).resolve().parents[1]
    manager = jupyter_client.KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}",
    ]
    client = None
    events = []
    output = ""
    pending = ""

    def read_message(timeout):
        nonlocal output, pending
        message = client.get_iopub_msg(timeout=timeout)
        content = message["content"]
        assert message["msg_type"] != "error", content
        if message["msg_type"] == "stream":
            output += content["text"]
            assert "Traceback (most recent call last)" not in output, output
            pending += content["text"]
            lines = pending.split("\n")
            pending = lines.pop()
            for line in lines:
                if line.startswith("LLMRUNNER_TEST "):
                    events.append(json.loads(line.removeprefix("LLMRUNNER_TEST ")))
        return message

    def wait_event(name, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for event in events:
                if event["event"] == name:
                    return event
            try:
                read_message(min(1, max(0.01, deadline - time.monotonic())))
            except Empty:
                pass
        pytest.fail(f"Timed out waiting for {name!r}; kernel output:\n{output}")

    def click(comm_id):
        # This is the browser's actual ipywidgets Button message, not .click().
        client.shell_channel.send(client.session.msg("comm_msg", {
            "comm_id": comm_id,
            "data": {"method": "custom", "content": {"event": "click"}},
        }))

    def finish_cell(message_id, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                message = read_message(min(1, max(0.01, deadline - time.monotonic())))
            except Empty:
                continue
            if (message["msg_type"] == "status"
                    and message["parent_header"].get("msg_id") == message_id
                    and message["content"]["execution_state"] == "idle"):
                break
        else:
            pytest.fail(f"Cell did not become idle; kernel output:\n{output}")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                reply = client.get_shell_msg(timeout=min(1, max(0.01, deadline - time.monotonic())))
            except Empty:
                continue
            if reply["parent_header"].get("msg_id") == message_id:
                assert reply["msg_type"] == "execute_reply", reply
                assert reply["content"]["status"] == "ok", reply
                return
        pytest.fail("Missing execute_reply")

    code = textwrap.dedent("""
        import asyncio
        import json
        from pathlib import Path
        import sys
        from types import SimpleNamespace

        sys.path.insert(0, str(Path.cwd()))
        import llmrunner
        import llmrunner.runner as runner_module
        from llmrunner import Runner
        from llmrunner.dashboard import JupyterDisplay

        assert Path(llmrunner.__file__).resolve().parent == Path.cwd() / 'llmrunner'
        kernel = get_ipython().kernel
        original_lock = kernel._main_asyncio_lock
        assert isinstance(original_lock, asyncio.Lock)
        gate = asyncio.Event()
        captured = {}
        callbacks = []
        original_make_display = runner_module.make_display

        def emit(event, **state):
            print('LLMRUNNER_TEST ' + json.dumps(dict(event=event, **state)), flush=True)

        def observe(event):
            display = captured['display']
            assert not run_task.done()
            assert runner._current_tasks
            assert all(result.status is None for result in runner.last_results)
            assert kernel._main_asyncio_lock is original_lock
            assert isinstance(kernel._main_asyncio_lock, asyncio.Lock)
            callbacks.append(event)
            emit(event, active=True, visible=display._visible_indices,
                 expanded=display._expanded, detail=display._rows[0].detail.value,
                 page=display._page, sort=display._sort_column)
            if event == 'sort':
                if CANCEL:
                    run_task.cancel()
                else:
                    gate.set()

        def capture_display(*args, **kwargs):
            display = original_make_display(*args, **kwargs)
            assert isinstance(display, JupyterDisplay)
            captured['display'] = display
            # Registered after the real callbacks, these only inspect their effects.
            display._next.on_click(lambda _: observe('page'))
            display._rows[0].expand.on_click(lambda _: observe('expand'))
            display._header_buttons['index'].on_click(lambda _: observe('sort'))
            return display

        async def create(*, messages, **kwargs):
            assert kwargs['stream'] is False
            await gate.wait()
            return {'choices': [{'message': {'content': 'result-' + str(messages[0]['index'])}}]}

        async def announce_widgets():
            # Give the dashboard's refresh loop time to enter its polling context.
            await asyncio.sleep(0.3)
            display = captured['display']
            assert kernel._main_asyncio_lock is original_lock, 'Kernel main lock was replaced'
            assert isinstance(kernel._main_asyncio_lock, asyncio.Lock)
            emit('ready', next=display._next.model_id,
                 expand=display._rows[0].expand.model_id,
                 sort=display._header_buttons['index'].model_id)

        runner_module.make_display = capture_display
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        runner = Runner(client, max_concurrency=12, show_progress=True)
        histories = [[{'role': 'user', 'content': '<message-' + str(i) + '>', 'index': i}]
                     for i in range(12)]
        run_task = asyncio.create_task(runner.arun(histories, model='fake',
                                                 stream=False, interactive=True))
        announcer = asyncio.create_task(announce_widgets())
        try:
            await asyncio.wait_for(announcer, timeout=5)
            try:
                results = await asyncio.wait_for(run_task, timeout=15)
            except asyncio.CancelledError:
                assert CANCEL
                results = runner.last_results
            else:
                assert not CANCEL
                assert [result.output for result in results] == ['result-' + str(i) for i in range(12)]
                assert all(result.messages[-1] == {'role': 'assistant', 'content': result.output}
                           for result in results)
            assert callbacks == ['page', 'expand', 'sort']
            assert [result.index for result in results] == list(range(12))
            assert all(result.status == ('cancelled' if CANCEL else 'success') for result in results)
            display = captured['display']
            assert display._closed and display._refresh_task.done()
            assert display._progress.value == 12
            assert kernel._main_asyncio_lock is original_lock
            assert isinstance(kernel._main_asyncio_lock, asyncio.Lock)
            assert not runner._current_tasks
            emit('finished')
        finally:
            announcer.cancel()
            run_task.cancel()
            await asyncio.gather(announcer, run_task, return_exceptions=True)
            runner_module.make_display = original_make_display
    """).replace("CANCEL", repr(cancel))

    with (tmp_path / "kernel.log").open("w+") as kernel_log:
        try:
            manager.start_kernel(cwd=str(root), stdout=kernel_log, stderr=kernel_log)
            client = manager.blocking_client()
            client.start_channels()
            client.wait_for_ready(timeout=30)
            message_id = client.execute(code, allow_stdin=False, stop_on_error=True)
            ready = wait_event("ready")
            click(ready["next"])
            page = wait_event("page")
            assert page["active"] and page["page"] == 1
            assert page["visible"] == [10, 11] + [None] * 8
            click(ready["expand"])
            expanded = wait_event("expand")
            assert expanded["active"] and expanded["expanded"] == 10
            assert "&lt;message-10&gt;" in expanded["detail"]
            click(ready["sort"])
            sorted_state = wait_event("sort")
            assert sorted_state["active"] and sorted_state["sort"] == "index"
            assert sorted_state["page"] == 0 and sorted_state["visible"] == list(range(10))
            wait_event("finished")
            finish_cell(message_id)
            # A subsequent execution also catches broken parent/lock cleanup.
            followup = client.execute(
                "assert kernel._main_asyncio_lock is original_lock\n"
                "assert isinstance(kernel._main_asyncio_lock, asyncio.Lock)\n"
                "emit('followup')", allow_stdin=False,
            )
            wait_event("followup")
            finish_cell(followup)
        finally:
            try:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            finally:
                if client is not None:
                    client.stop_channels()
        kernel_log.seek(0)
        log = kernel_log.read()
        assert "Traceback (most recent call last)" not in log, log
