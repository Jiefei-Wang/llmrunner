from __future__ import annotations

import asyncio
import html
import inspect
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any

from .types import Task

TERMINAL_STATES = {"success", "error", "cancelled"}
PAGE_SIZE = 10
REFRESH_INTERVAL = 0.1
UI_EVENTS_PER_REFRESH = 10
_fallback_warned = False


@dataclass(frozen=True)
class TaskSnapshot:
    index: int
    state: str
    generated_chars: int
    complete: bool
    detail: str


@dataclass
class _JupyterRow:
    container: Any
    index: Any
    chars: Any
    complete: Any
    expand: Any
    detail: Any


class DashboardModel:
    """Mutable batch state shared by the runner and presentation layers."""

    def __init__(self, tasks: list[Task]) -> None:
        self._tasks = tasks
        self._states = {task.index: "pending" for task in tasks}
        self._completed_chars = {task.index: 0 for task in tasks}
        self._drafts: dict[int, list[str]] = {task.index: [] for task in tasks}
        self._draft_chars = {task.index: 0 for task in tasks}
        self.started_at = time.monotonic()
        self.version = 0

    @property
    def total(self) -> int:
        return len(self._tasks)

    @property
    def completed(self) -> int:
        return sum(state in TERMINAL_STATES for state in self._states.values())

    @property
    def remaining(self) -> int:
        return self.total - self.completed

    @property
    def eta_seconds(self) -> float | None:
        if self.completed == 0 or self.remaining == 0:
            return 0.0 if self.remaining == 0 else None
        elapsed = max(0.0, time.monotonic() - self.started_at)
        return elapsed / self.completed * self.remaining

    def replace_task(self, index: int, task: Task) -> None:
        self._tasks[index] = task
        self._touch()

    def set_state(self, index: int, state: str) -> None:
        if self._states[index] != state:
            self._states[index] = state
            self._touch()

    def begin_attempt(self, index: int) -> None:
        self._drafts[index] = []
        self._draft_chars[index] = 0
        self.set_state(index, "running")
        self._touch()

    def append_delta(self, index: int, delta: str) -> None:
        if delta:
            self._drafts[index].append(delta)
            self._draft_chars[index] += len(delta)
            self._touch()

    def discard_attempt(self, index: int) -> None:
        if self._drafts[index]:
            self._drafts[index] = []
            self._draft_chars[index] = 0
            self._touch()

    def commit_response(self, index: int, text: str) -> None:
        self._completed_chars[index] += len(text)
        self._drafts[index] = []
        self._draft_chars[index] = 0
        self._touch()

    def mark_terminal(self, task: Task) -> None:
        self._drafts[task.index] = []
        self._draft_chars[task.index] = 0
        self._states[task.index] = task.status or "cancelled"
        self._touch()

    def snapshot(self, index: int) -> TaskSnapshot:
        task = self._tasks[index]
        state = self._states[index]
        draft = "".join(self._drafts[index])
        return TaskSnapshot(
            index=index,
            state=state,
            generated_chars=self._completed_chars[index] + self._draft_chars[index],
            complete=state in TERMINAL_STATES,
            detail=format_task_detail(task, draft),
        )

    def sorted_snapshots(self, column: str, reverse: bool) -> list[TaskSnapshot]:
        snapshots = [self.snapshot(index) for index in range(self.total)]
        if column == "default":
            return sorted(snapshots, key=lambda row: (row.complete, row.index))

        keys = {
            "index": lambda row: row.index,
            "chars": lambda row: row.generated_chars,
            "complete": lambda row: row.complete,
        }
        key = keys.get(column, keys["index"])
        return sorted(snapshots, key=lambda row: (key(row), row.index), reverse=reverse)

    def _touch(self) -> None:
        self.version += 1


def format_task_detail(task: Task, draft: str = "") -> str:
    blocks: list[str] = []
    messages = list(task.messages)
    if draft:
        messages.append({"role": "assistant", "content": draft, "streaming": True})
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            metadata = {key: value for key, value in message.items() if key != "content"}
            header = json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
            blocks.append(f"{header}\n{content}")
        else:
            blocks.append(json.dumps(message, ensure_ascii=False, indent=2, default=str))
    if task.error:
        blocks.append(f"ERROR\n{task.error}")
    return "\n\n".join(blocks)


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = max(0, math.ceil(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class DisplayController:
    async def start(self) -> None:
        return None

    def task_completed(self) -> None:
        return None

    async def close(self) -> None:
        return None


class NullDisplay(DisplayController):
    pass


class TqdmDisplay(DisplayController):
    def __init__(self, total: int) -> None:
        from tqdm.auto import tqdm

        self._progress = tqdm(total=total, desc="LLM requests", unit="req")

    def task_completed(self) -> None:
        self._progress.update(1)

    async def close(self) -> None:
        self._progress.close()


class JupyterDisplay(DisplayController):
    def __init__(self, model: DashboardModel, show_progress: bool) -> None:
        import ipywidgets as widgets
        from IPython.display import display

        self._widgets = widgets
        self._display = display
        self._model = model
        self._show_progress = show_progress
        self._sort_column = "default"
        self._sort_reverse = False
        self._page = 0
        self._expanded: int | None = None
        self._last_version = -1
        self._closed = False
        self._refresh_task: asyncio.Task[None] | None = None
        self._visible_indices: list[int | None] = [None] * PAGE_SIZE

        self._header_buttons = {
            "index": self._header_button("Task index", "index", "140px"),
            "chars": self._header_button("Characters", "chars", "140px"),
            "complete": self._header_button("Complete", "complete", "120px"),
        }
        header = widgets.HBox(
            [
                self._header_buttons["index"],
                self._header_buttons["chars"],
                self._header_buttons["complete"],
                widgets.HTML("<b>Expand</b>", layout=widgets.Layout(width="90px")),
            ]
        )
        self._rows = [self._make_row(slot) for slot in range(PAGE_SIZE)]
        self._table = widgets.VBox([header, *(row.container for row in self._rows)])
        self._page_label = widgets.HTML()
        self._previous = widgets.Button(description="Previous", layout=widgets.Layout(width="90px"))
        self._next = widgets.Button(description="Next", layout=widgets.Layout(width="90px"))
        self._previous.on_click(lambda _: self._change_page(-1))
        self._next.on_click(lambda _: self._change_page(1))
        pager = widgets.HBox([self._previous, self._page_label, self._next])

        children: list[Any] = [self._table, pager]
        self._progress = None
        self._progress_label = None
        if show_progress:
            self._progress = widgets.IntProgress(min=0, max=model.total, value=0)
            self._progress_label = widgets.HTML()
            children.extend([self._progress, self._progress_label])
        self.container = widgets.VBox(children)

    async def start(self) -> None:
        self._refresh(force=True)
        self._display(self.container)
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def close(self) -> None:
        self._closed = True
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            await asyncio.gather(self._refresh_task, return_exceptions=True)
        self._refresh(force=True)

    async def _refresh_loop(self) -> None:
        if not is_jupyter():
            await self._run_refresh_loop()
            return

        from IPython import get_ipython

        dispatch_shell = getattr(get_ipython().kernel, "dispatch_shell", None)
        # ipykernel 7.4+ dispatches widget comms during async cells natively.
        # Legacy UI polling replaces its lock with an incompatible nullcontext.
        if dispatch_shell is not None and "concurrent" in inspect.signature(dispatch_shell).parameters:
            await self._run_refresh_loop()
            return

        try:
            from jupyter_ui_poll import ui_events
        except ImportError:
            await self._run_refresh_loop()
            return

        try:
            async with ui_events() as poll:
                await self._run_refresh_loop(poll)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            warnings.warn(
                f"Jupyter UI event polling stopped; dashboard updates will continue: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            if not self._closed:
                await self._run_refresh_loop()

    async def _run_refresh_loop(self, poll: Any = None) -> None:
        while not self._closed:
            await asyncio.sleep(REFRESH_INTERVAL)
            if poll is not None:
                await poll(UI_EVENTS_PER_REFRESH)
            self._refresh()

    def _refresh(self, force: bool = False) -> None:
        if not force and self._last_version == self._model.version:
            self._refresh_progress()
            return
        self._last_version = self._model.version
        rows = self._model.sorted_snapshots(self._sort_column, self._sort_reverse)
        page_count = max(1, math.ceil(len(rows) / PAGE_SIZE))
        self._page = min(self._page, page_count - 1)
        visible = rows[self._page * PAGE_SIZE : (self._page + 1) * PAGE_SIZE]

        self._refresh_headers()
        for slot, widgets in enumerate(self._rows):
            if slot >= len(visible):
                self._visible_indices[slot] = None
                widgets.container.layout.display = "none"
                continue

            row = visible[slot]
            self._visible_indices[slot] = row.index
            widgets.container.layout.display = ""
            widgets.index.value = str(row.index)
            widgets.chars.value = str(row.generated_chars)
            widgets.complete.value = "Yes" if row.complete else "No"
            widgets.expand.description = "Collapse" if self._expanded == row.index else "Expand"
            if self._expanded == row.index:
                escaped = html.escape(row.detail)
                widgets.detail.value = (
                    f'<pre style="white-space:pre-wrap;margin:0">{escaped}</pre>'
                )
                widgets.detail.layout.display = ""
            else:
                widgets.detail.value = ""
                widgets.detail.layout.display = "none"

        self._page_label.value = f"<span style='padding:0 12px'>Page {self._page + 1}/{page_count}</span>"
        self._previous.disabled = self._page == 0
        self._next.disabled = self._page >= page_count - 1
        self._refresh_progress()

    def _header_button(self, label: str, column: str, width: str) -> Any:
        button = self._widgets.Button(description=label, layout=self._widgets.Layout(width=width))
        button.on_click(lambda _, selected=column: self._sort(selected))
        return button

    def _make_row(self, slot: int) -> _JupyterRow:
        index = self._widgets.Label(layout=self._widgets.Layout(width="140px"))
        chars = self._widgets.Label(layout=self._widgets.Layout(width="140px"))
        complete = self._widgets.Label(layout=self._widgets.Layout(width="120px"))
        expand = self._widgets.Button(description="Expand", layout=self._widgets.Layout(width="90px"))
        expand.on_click(lambda _, selected_slot=slot: self._toggle_slot(selected_slot))
        summary = self._widgets.HBox([index, chars, complete, expand])
        # Keep scrolling on the widget root, not the HTML replaced during streaming.
        detail = self._widgets.HTML(layout=self._widgets.Layout(
            width="100%", display="none", max_height="320px", overflow="auto",
            border="1px solid #ccc", padding="8px", margin="2px 0 8px 0",
        ))
        container = self._widgets.VBox([summary, detail], layout=self._widgets.Layout(width="100%"))
        return _JupyterRow(container, index, chars, complete, expand, detail)

    def _refresh_headers(self) -> None:
        labels = {"index": "Task index", "chars": "Characters", "complete": "Complete"}
        for column, button in self._header_buttons.items():
            suffix = ""
            if self._sort_column == column:
                suffix = " ↓" if self._sort_reverse else " ↑"
            button.description = labels[column] + suffix

    def _toggle_slot(self, slot: int) -> None:
        index = self._visible_indices[slot]
        if index is not None:
            self._toggle(index)

    def _sort(self, column: str) -> None:
        if self._sort_column == column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column
            self._sort_reverse = False
        self._page = 0
        self._refresh(force=True)

    def _change_page(self, delta: int) -> None:
        self._page = max(0, self._page + delta)
        self._refresh(force=True)

    def _toggle(self, index: int) -> None:
        self._expanded = None if self._expanded == index else index
        self._refresh(force=True)

    def _refresh_progress(self) -> None:
        if self._progress is None or self._progress_label is None:
            return
        self._progress.value = self._model.completed
        self._progress_label.value = (
            f"Completed {self._model.completed} / Remaining {self._model.remaining} / "
            f"ETA {format_eta(self._model.eta_seconds)}"
        )


class TerminalDisplay(DisplayController):
    def __init__(self, model: DashboardModel, show_progress: bool) -> None:
        from .terminal_dashboard import TerminalDashboard

        self._app = TerminalDashboard(model, show_progress)
        self._task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._app.run_async(inline=True, inline_no_clear=True))
        self._task.add_done_callback(self._report_failure)
        await asyncio.sleep(0)

    async def close(self) -> None:
        if self._task is None or self._task.done():
            if self._task is not None:
                await asyncio.gather(self._task, return_exceptions=True)
            return
        try:
            self._app.final_refresh_and_exit()
        except Exception:
            self._app.exit()
        await asyncio.gather(self._task, return_exceptions=True)

    @staticmethod
    def _report_failure(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            _warn_fallback(f"Textual dashboard stopped unexpectedly: {error}")


def is_jupyter() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except Exception:
        return False
    return shell is not None and getattr(shell, "kernel", None) is not None


def is_terminal() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def make_display(model: DashboardModel, *, interactive: bool, show_progress: bool) -> DisplayController:
    if model.total == 0:
        return NullDisplay()
    if not interactive:
        return _progress_fallback(model.total, show_progress)

    try:
        if is_jupyter():
            return JupyterDisplay(model, show_progress)
        if is_terminal():
            return TerminalDisplay(model, show_progress)
    except Exception as exc:
        _warn_fallback(f"Interactive dashboard unavailable; using fallback display: {exc}")
        return _progress_fallback(model.total, show_progress)

    _warn_fallback(
        "Interactive dashboard requires a Jupyter kernel or an interactive terminal; using fallback display."
    )
    return _progress_fallback(model.total, show_progress)


def _progress_fallback(total: int, show_progress: bool) -> DisplayController:
    if not show_progress:
        return NullDisplay()
    try:
        return TqdmDisplay(total)
    except ImportError:
        return NullDisplay()


def _warn_fallback(message: str) -> None:
    global _fallback_warned
    if _fallback_warned:
        return
    _fallback_warned = True
    warnings.warn(message, RuntimeWarning, stacklevel=2)
