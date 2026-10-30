"""The browser app: `mtrain gui` (http://localhost:8090)."""

from __future__ import annotations

from nicegui import ui

from ..util import trust_system_certificates
from ..workspace import Workspace
from . import common
from .datasets_page import dataset_page, datasets_page
from .other_pages import (
    compare_page,
    glossary_page,
    home_page,
    playground_page,
    settings_page,
    sweeps_page,
)
from .run_page import run_page, runs_page
from .wizard_page import wizard_page


def register_pages() -> None:
    ui.page("/")(home_page)
    ui.page("/datasets")(datasets_page)
    ui.page("/dataset/{name}")(dataset_page)

    @ui.page("/new")
    def _new(
        dataset: str | None = None, from_run: str | None = None, template: str | None = None
    ) -> None:
        wizard_page(dataset, from_run, template)

    ui.page("/runs")(runs_page)
    ui.page("/run/{run_id}")(run_page)

    @ui.page("/compare")
    def _compare(runs: str = "") -> None:
        compare_page(runs)

    @ui.page("/playground")
    def _playground(run: str | None = None) -> None:
        playground_page(run)

    ui.page("/sweeps")(sweeps_page)
    ui.page("/settings")(settings_page)
    ui.page("/glossary")(glossary_page)


def run_app(ws: Workspace, port: int = 8090, show: bool = True) -> None:
    trust_system_certificates()
    common.STATE["ws"] = ws
    register_pages()
    ui.run(
        title="model-trainer",
        port=port,
        show=show,
        reload=False,
        favicon="🧠",
        storage_secret=f"model-trainer:{ws.root}",
    )
