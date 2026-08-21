from __future__ import annotations

import math

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Footer, ProgressBar, Static

from .dashboard import DashboardModel, PAGE_SIZE, REFRESH_INTERVAL, format_eta


class TerminalDashboard(App[None]):
    CSS = """
    TerminalDashboard {
        height: auto;
        max-height: 36;
    }
    #tasks {
        height: 14;
    }
    #pager {
        height: 3;
        align-horizontal: center;
    }
    #page-label {
        width: 18;
        height: 3;
        content-align: center middle;
    }
    #detail-container {
        height: 10;
        border: round $accent;
        padding: 0 1;
    }
    #progress-container {
        height: 4;
    }
    #progress-label {
        height: 1;
    }
    """

    BINDINGS = [
        Binding("q", "hide_dashboard", "Hide dashboard", priority=True),
        Binding("left", "previous_page", "Previous page", priority=True),
        Binding("right", "next_page", "Next page", priority=True),
    ]

    def __init__(self, model: DashboardModel, show_progress: bool) -> None:
        super().__init__()
        self.model = model
        self.show_progress = show_progress
        self.sort_column = "default"
        self.sort_reverse = False
        self.page = 0
        self.expanded: int | None = None
        self.last_version = -1

    def compose(self) -> ComposeResult:
        yield DataTable(id="tasks", cursor_type="row", zebra_stripes=True)
        with Horizontal(id="pager"):
            yield Button("Previous", id="previous")
            yield Static(id="page-label")
            yield Button("Next", id="next")
        with VerticalScroll(id="detail-container"):
            yield Static("Select a row to view the complete messages", id="detail")
        with Vertical(id="progress-container"):
            yield ProgressBar(total=self.model.total, show_eta=False, id="progress")
            yield Static(id="progress-label")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#tasks", DataTable)
        table.add_column("Task index", key="index")
        table.add_column("Characters", key="chars")
        table.add_column("Complete", key="complete")
        table.add_column("Expand", key="expand")
        if not self.show_progress:
            self.query_one("#progress-container").display = False
        self.set_interval(REFRESH_INTERVAL, self.refresh_model)
        self.refresh_model(force=True)

    def refresh_model(self, force: bool = False) -> None:
        if not self.is_mounted:
            return
        if not force and self.last_version == self.model.version:
            self._refresh_progress()
            return
        self.last_version = self.model.version
        rows = self.model.sorted_snapshots(self.sort_column, self.sort_reverse)
        page_count = max(1, math.ceil(len(rows) / PAGE_SIZE))
        self.page = min(self.page, page_count - 1)
        visible = rows[self.page * PAGE_SIZE : (self.page + 1) * PAGE_SIZE]

        table = self.query_one("#tasks", DataTable)
        table.clear(columns=False)
        for row in visible:
            table.add_row(
                str(row.index),
                str(row.generated_chars),
                "Yes" if row.complete else "No",
                "Collapse" if self.expanded == row.index else "Expand",
                key=str(row.index),
            )

        self.query_one("#page-label", Static).update(f"Page {self.page + 1}/{page_count}")
        self.query_one("#previous", Button).disabled = self.page == 0
        self.query_one("#next", Button).disabled = self.page >= page_count - 1
        self._refresh_detail(rows)
        self._refresh_progress()

    def final_refresh_and_exit(self) -> None:
        if self.is_mounted:
            self.refresh_model(force=True)
        self.exit()

    @on(DataTable.HeaderSelected)
    def sort_table(self, event: DataTable.HeaderSelected) -> None:
        column = str(event.column_key.value)
        if column == "expand":
            return
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False
        self.page = 0
        self.refresh_model(force=True)

    @on(DataTable.RowSelected)
    def toggle_row(self, event: DataTable.RowSelected) -> None:
        index = int(str(event.row_key.value))
        self.expanded = None if self.expanded == index else index
        self.refresh_model(force=True)

    @on(Button.Pressed, "#previous")
    def previous_button(self) -> None:
        self.action_previous_page()

    @on(Button.Pressed, "#next")
    def next_button(self) -> None:
        self.action_next_page()

    def action_previous_page(self) -> None:
        self.page = max(0, self.page - 1)
        self.refresh_model(force=True)

    def action_next_page(self) -> None:
        page_count = max(1, math.ceil(self.model.total / PAGE_SIZE))
        self.page = min(page_count - 1, self.page + 1)
        self.refresh_model(force=True)

    def action_hide_dashboard(self) -> None:
        self.exit()

    def _refresh_detail(self, rows) -> None:
        detail = self.query_one("#detail", Static)
        if self.expanded is None:
            detail.update("Select a row to view the complete messages")
            return
        selected = next((row for row in rows if row.index == self.expanded), None)
        if selected is None:
            detail.update("Task not found")
            return
        detail.update(Text(selected.detail))

    def _refresh_progress(self) -> None:
        if not self.show_progress:
            return
        self.query_one("#progress", ProgressBar).update(
            total=self.model.total,
            progress=self.model.completed,
        )
        self.query_one("#progress-label", Static).update(
            f"Completed {self.model.completed} / Remaining {self.model.remaining} / "
            f"ETA {format_eta(self.model.eta_seconds)}"
        )
