"""Shared GUI pieces: the page frame (header, navigation), help icons, parameter inputs, charts."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from nicegui import app, ui

from ..params import Param
from ..util import gpu_status
from ..workspace import Project, Workspace

STATE = {"ws": None}
LEVEL_COLOUR = {"ok": "positive", "info": "info", "warn": "warning", "bad": "negative"}
LEVEL_ICON = {"ok": "check_circle", "info": "info", "warn": "warning", "bad": "error"}
STATE_COLOUR = {
    "created": "grey",
    "queued": "blue-grey",
    "preparing": "info",
    "running": "primary",
    "pausing": "orange",
    "paused": "orange",
    "stopping": "orange",
    "evaluating": "info",
    "stopped": "grey-8",
    "done": "positive",
    "failed": "negative",
    "interrupted": "deep-orange",
}
NAV = [
    ("/", "home", "Home"),
    ("/datasets", "photo_library", "Datasets"),
    ("/new", "add_circle", "New run"),
    ("/runs", "insights", "Runs"),
    ("/compare", "compare", "Compare"),
    ("/playground", "science", "Playground"),
    ("/sweeps", "tune", "Sweeps"),
    ("/settings", "settings", "Settings"),
    ("/glossary", "menu_book", "Glossary"),
]


def ws() -> Workspace:
    w = STATE["ws"]
    assert w is not None, "workspace not set"
    return w


def project() -> Project:
    name = None
    with contextlib.suppress(Exception):
        name = app.storage.user.get("project")
    return ws().project(name)


def set_project(name: str) -> None:
    app.storage.user["project"] = name
    s = ws().settings()
    s["project"] = name
    ws().save_settings(s)


def simple_mode() -> bool:
    try:
        return bool(app.storage.user.get("simple", True))
    except Exception:  # noqa: BLE001 - storage unavailable in tests
        return True


@contextlib.contextmanager
def frame(title: str) -> Iterator[None]:
    """Page frame: header with project switcher, Simple/Advanced switch, GPU chip; side navigation."""
    ui.add_css(
        ".mt-card { width: 100%; } .q-field__label { white-space: normal; } "
        ".mt-mono { font-family: ui-monospace, Consolas, monospace; font-size: 12px; white-space: pre-wrap; }"
    )
    with ui.header().classes("items-center justify-between bg-slate-800 py-1"):
        with ui.row().classes("items-center gap-2"):
            ui.button(icon="menu", on_click=lambda: drawer.toggle()).props("flat round color=white")
            ui.label("model-trainer").classes("text-lg font-bold")
            ui.label(f"/ {title}").classes("text-md opacity-80")
        with ui.row().classes("items-center gap-3"):
            projects = sorted(set(ws().projects()) | {project().name})
            ui.select(
                projects,
                value=project().name,
                on_change=lambda e: (set_project(e.value), ui.navigate.reload()),
                new_value_mode="add-unique",
                label="Project",
            ).props("dense dark options-dark outlined").classes("w-44").tooltip(
                "Projects group datasets and runs. Type a new name to create one."
            )
            sw = ui.switch("Advanced", value=not simple_mode()).props("color=amber dark")
            sw.on_value_change(
                lambda e: (app.storage.user.update(simple=not e.value), ui.navigate.reload())
            )
            sw.tooltip(
                "Simple mode shows the essential settings with recommended values; Advanced shows everything."
            )
            gpu_chip()
    with (
        ui.left_drawer(value=True)
        .classes("bg-slate-100 w-48")
        .props("width=190 breakpoint=900") as drawer
    ):
        for path, icon, label in NAV:
            ui.button(label, icon=icon, on_click=lambda p=path: ui.navigate.to(p)).props(
                "flat align=left no-caps"
            ).classes("w-full")
    with ui.column().classes("w-full p-4 gap-4"):
        yield


def gpu_chip() -> None:
    gpus = gpu_status()
    if not gpus:
        ui.chip("CPU only", icon="memory").props("dense color=grey-7 text-color=white").tooltip(
            "No NVIDIA GPU found: training will be slow."
        )
        return
    g = gpus[0]
    used, total = g["memory_used_mb"] or 0, g["memory_total_mb"] or 1
    chip = ui.chip(
        f"{g['name'].replace('NVIDIA GeForce ', '')}  {used / 1024:.1f}/{total / 1024:.0f} GB  {g['temperature_c']:.0f}°C",
        icon="memory",
    ).props("dense color=green-8 text-color=white")
    chip.tooltip(f"GPU load {g['load_pct']:.0f} %, driver {g['driver']}")


def help_icon(text: str, title: str | None = None) -> None:
    """A small (?) that explains something on hover/click."""
    with ui.icon("help_outline").classes("text-gray-500 cursor-help text-base"):
        with ui.tooltip().classes("text-sm max-w-md whitespace-pre-line"):
            if title:
                ui.label(title).classes("font-bold")
            ui.label(text)


def param_input(
    p: Param, value: Any, on_change: Callable[[str, Any], None], default: Any = None
) -> ui.element:
    """An input for one parameter, highlighted when it differs from the default."""
    default = p.default if default is None else default

    def changed(v: Any) -> None:
        on_change(p.key, v)
        box.classes(remove="bg-amber-100")
        if v != default:
            box.classes(add="bg-amber-100")

    def num(cast: type, raw: Any) -> Any:
        return default if raw is None else cast(raw)

    with ui.row().classes("items-center gap-1 w-full no-wrap") as row:
        box = ui.element("div").classes(
            "rounded px-2" + (" bg-amber-100" if value != default else "")
        )
        with box:
            if p.kind == "bool":
                ui.switch(p.label, value=bool(value), on_change=lambda e: changed(bool(e.value)))
            elif p.kind == "choice":
                opts = {
                    c: ("off" if c is False else "none" if c is None else str(c)) for c in p.choices
                }
                ui.select(
                    opts,
                    value=value if value in opts else p.default,
                    label=p.label,
                    on_change=lambda e: changed(e.value),
                ).classes("w-56")
            elif p.kind == "int":
                ui.number(
                    p.label,
                    value=value,
                    min=p.min,
                    max=p.max,
                    step=p.step or 1,
                    format="%d",
                    on_change=lambda e: changed(num(int, e.value)),
                ).classes("w-56")
            elif p.kind == "float":
                ui.number(
                    p.label,
                    value=value,
                    min=p.min,
                    max=p.max,
                    step=p.step or 0.01,
                    on_change=lambda e: changed(num(float, e.value)),
                ).classes("w-56")
            else:
                ui.input(
                    p.label, value=str(value or ""), on_change=lambda e: changed(e.value)
                ).classes("w-56")
        help_icon(p.help() + f"\n(default: {default})", p.label)
    return row


def line_chart(
    series: Mapping[str, Sequence[tuple[float, float]]],
    title: str = "",
    y_name: str = "",
    height: str = "260px",
    log_y: bool = False,
    x_name: str = "epoch",
) -> ui.echart:
    opts = {
        "title": {"text": title, "textStyle": {"fontSize": 13}},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 22, "type": "scroll"},
        "grid": {"top": 60, "left": 50, "right": 20, "bottom": 30},
        "xAxis": {"type": "value", "name": x_name, "minInterval": 1 if x_name == "epoch" else 0},
        "yAxis": {"type": "log" if log_y else "value", "name": y_name, "scale": True},
        "series": [
            {"name": k, "type": "line", "showSymbol": False, "data": [list(p) for p in v]}
            for k, v in series.items()
        ],
    }
    return ui.echart(opts).style(f"height:{height}").classes("w-full")


def set_series(chart: ui.echart, series: Mapping[str, Sequence[tuple[float, float]]]) -> None:
    chart.options["series"] = [
        {"name": k, "type": "line", "showSymbol": False, "data": [list(p) for p in v]}
        for k, v in series.items()
    ]
    chart.update()


def bar_chart(
    labels: Sequence[str],
    values: Mapping[str, Sequence[float]],
    title: str = "",
    height: str = "260px",
    horizontal: bool = False,
) -> ui.echart:
    cat = {
        "type": "category",
        "data": list(labels),
        "axisLabel": {"interval": 0, "rotate": 0 if horizontal else (30 if len(labels) > 6 else 0)},
    }
    val = {"type": "value"}
    opts = {
        "title": {"text": title, "textStyle": {"fontSize": 13}},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 22},
        "grid": {"top": 55, "left": 110 if horizontal else 50, "right": 20, "bottom": 60},
        "xAxis": val if horizontal else cat,
        "yAxis": cat if horizontal else val,
        "series": [
            {"name": k, "type": "bar", "stack": "total", "data": list(v)} for k, v in values.items()
        ],
    }
    return ui.echart(opts).style(f"height:{height}").classes("w-full")


def heatmap(
    matrix: Sequence[Sequence[float]],
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str = "",
    height: str = "360px",
    on_click: Callable[[int, int], None] | None = None,
) -> ui.echart:
    data = [[x, y, round(float(v), 3)] for y, row in enumerate(matrix) for x, v in enumerate(row)]
    vmax = max((d[2] for d in data), default=1) or 1
    opts = {
        "title": {"text": title, "textStyle": {"fontSize": 13}},
        "tooltip": {"position": "top"},
        "grid": {"top": 40, "left": 100, "right": 30, "bottom": 90},
        "xAxis": {
            "type": "category",
            "data": list(x_labels),
            "axisLabel": {"rotate": 40},
            "name": "true",
        },
        "yAxis": {"type": "category", "data": list(y_labels), "name": "predicted", "inverse": True},
        "visualMap": {
            "min": 0,
            "max": vmax,
            "show": False,
            "inRange": {"color": ["#f8fafc", "#2563eb"]},
        },
        "series": [
            {
                "type": "heatmap",
                "data": data,
                "label": {"show": len(x_labels) <= 15, "fontSize": 10},
            }
        ],
    }
    chart = ui.echart(opts).style(f"height:{height}").classes("w-full")
    if on_click:
        chart.on_point_click(lambda e: on_click(int(e.data[0]), int(e.data[1])))
    return chart


def state_badge(state: str) -> None:
    ui.badge(state, color=STATE_COLOUR.get(state, "grey"))


def empty(text: str, button: tuple[str, str] | None = None) -> None:
    with ui.card().classes("w-full items-center p-8"):
        ui.label(text).classes("text-gray-600")
        if button:
            ui.button(button[0], on_click=lambda: ui.navigate.to(button[1]))
