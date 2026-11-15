"""
Link Budget Calculator - NiceGUI desktop entry point.

Run with:  python desktop_app.py

Phase 2: three-pane layout (inputs | map | results) with a step tracker in the
Left Pane.
Phase 3: the Left Pane now holds the ITU import section and the Step 1 / Step 2 forms
(ui/left_pane.py).
Phase 5: the Middle Pane renders the interactive Folium coordination map
(ui/middle_pane.py).
Phase 6: the Right Pane shows the results - polar contour plot, border I/N table and
the mitigation sliders (ui/right_pane.py).
Part 3/4: a frequency selector in the top bar; the Middle (map) and Right (results) panes
show one map / one plot + I/N table per selected frequency, laid out by
ui.middle_pane.build_split_layout (1 frequency: single column, 2: two columns, 3+: grid).
Phase 7: Save / Open Project (top bar), session auto-load on startup and a 60 s
background autosave (autosave.json in the user data dir).
"""
import asyncio
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys
import logging
from logging.handlers import RotatingFileHandler

import multiprocessing
multiprocessing.freeze_support()

from core.app_paths import get_resource_path, get_user_data_dir

# ---------------------------------------------------------------------------
# Logging (set up first, so everything below - including import-time errors and,
# in a frozen exe, anything printed - ends up in <user data dir>/logs/app.log)
# ---------------------------------------------------------------------------
LOG_PATH = get_user_data_dir("logs") / "app.log"


class _StreamToLogger:
    """File-like object that forwards writes to a logger (replaces sys.stdout/stderr)."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._logger.log(self._level, line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buf.strip():
            self._logger.log(self._level, self._buf.rstrip())
        self._buf = ""

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise OSError("log stream has no file descriptor")


def _setup_logging() -> None:
    root = logging.getLogger()
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return  # already configured in this process
    handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    if getattr(sys, "frozen", False):
        # A windowed exe has no console (sys.stdout/stderr may even be None): send
        # print() output and tracebacks to the log file instead.
        sys.stdout = _StreamToLogger(logging.getLogger("stdout"), logging.INFO)
        sys.stderr = _StreamToLogger(logging.getLogger("stderr"), logging.ERROR)


_setup_logging()
log = logging.getLogger(__name__)

from nicegui import app, ui

from ui.left_pane import build_itu_import, build_step_form
from ui.middle_pane import STATIC_MAPS_DIR, STATIC_MAPS_URL, build_middle_pane
from ui.navigation import build_step_tracker, step_title
from ui.right_pane import build_right_pane
from core import map_visualization as mv
from ui.state import BUNDLED_DEM_SENTINEL, AppState, freq_key

# Folium maps are written as standalone HTML files into <user data dir>/maps and shown
# in an iframe, so that directory must exist and be served at /maps. Both happen
# at import time, before ui.run() (the native child process re-imports this
# module as __mp_main__ and needs them too).
STATIC_MAPS_DIR.mkdir(parents=True, exist_ok=True)
app.add_static_files(STATIC_MAPS_URL, str(STATIC_MAPS_DIR))  # STATIC_MAPS_URL == "/maps"

# Offline web assets (Leaflet, Bootstrap, jQuery, Font Awesome, ...). The saved map HTML has its
# CDN links rewritten to /web_assets/... (core.map_visualization.localize_map_file), and this
# serves the folder. It is read-only and bundled, so it is located with get_resource_path.
WEB_ASSETS_DIR = Path(get_resource_path(mv.WEB_ASSETS_RELPATH))      # ui/static/web_assets
if WEB_ASSETS_DIR.is_dir():
    app.add_static_files("/web_assets", str(WEB_ASSETS_DIR))
else:
    log.warning("Web assets folder missing (%s): maps will fall back to CDN links. "
                "Run tools/fetch_web_assets.py once while online.", WEB_ASSETS_DIR)

# Prebuilt DEM hillshade (tools/prepare_dem_overlay.py): served as a file so the map HTML only
# carries a URL, not a multi-MB base64 image.
_HILLSHADE_PNG = Path(get_resource_path(mv.HILLSHADE_PNG_RELPATH))
if _HILLSHADE_PNG.is_file():
    _HILLSHADE_URL = "/dem_overlay/dem_hillshade.png"
    app.add_static_file(local_file=str(_HILLSHADE_PNG), url_path=_HILLSHADE_URL)
    mv.configure_static_hillshade_url(_HILLSHADE_URL)

# One AppState for the whole process. Native mode is single-user, so a single
# shared instance is enough. Because it lives in the process (not the page), a
# page reload rebuilds the UI from it and keeps the user's place in the wizard.
state = AppState()

# ---------------------------------------------------------------------------
# Save / Open / autosave
# ---------------------------------------------------------------------------
AUTOSAVE_PATH = get_user_data_dir() / "autosave.json"
AUTOSAVE_INTERVAL_S = 60
DEFAULT_PROJECT_NAME = "project_state.json"      # fallback target when no native dialog exists
JSON_FILE_TYPES = ("JSON files (*.json)", "All files (*.*)")

# Notifications waiting for the next page build. Opening a project reloads the page
# (which would wipe a notification shown right before it), and the startup auto-load
# happens before any page exists, so both queue their messages here.
_pending_notices: List[Tuple[str, str]] = []


def _restore_warnings() -> List[Tuple[str, str]]:
    """Warnings about things a project file cannot carry. The rest of the state is kept."""
    notes: List[Tuple[str, str]] = []

    # The project stores the ITU file's (name, size) and the parsed stations, never the
    # .mdb path or its DataFrames - so the raw tables are always absent after a load.
    itu = state.itu_import
    if itu.file_key and itu.mdb_tables is None:
        name = itu.file_key[0] if isinstance(itu.file_key, (tuple, list)) and itu.file_key else "?"
        notes.append((
            "warning",
            f"ITU file \u201C{name}\u201D is not loaded. Its imported stations were restored; "
            "re-select the .mdb file only if you need the raw tables again.",
        ))

    # The one real file path in the state: a custom DEM GeoTIFF (Step 8 hillshade). The bundled
    # DEM is stored as the BUNDLED_DEM_SENTINEL and resolved at runtime, so it is never "missing".
    dem = state.border.dem_path
    if dem and dem != BUNDLED_DEM_SENTINEL and not Path(dem).is_file():
        notes.append(("warning", f"DEM file not found on this computer: {dem}. The hillshade overlay is disabled."))
    return notes


def autosave_now() -> None:
    """Silently write the current state to autosave.json. Never raises."""
    try:
        state.save_to_json(AUTOSAVE_PATH)
    except Exception as exc:  # noqa: BLE001 - a background save must not disturb the UI
        print(f"[autosave] failed: {exc}")


def autoload_on_startup() -> None:
    """Pick up the previous session from autosave.json, if there is one."""
    if not AUTOSAVE_PATH.is_file():
        return
    try:
        state.load_from_json(AUTOSAVE_PATH)   # validates first: a bad file changes nothing
    except Exception as exc:  # noqa: BLE001
        # Move the unreadable file aside so the first autosave doesn't overwrite it.
        bad = AUTOSAVE_PATH.with_name("autosave.corrupt.json")
        try:
            os.replace(AUTOSAVE_PATH, bad)
        except OSError:
            pass
        print(f"[autosave] could not restore {AUTOSAVE_PATH.name}: {exc} (moved to {bad.name})")
        _pending_notices.append(("warning", "The last autosave could not be read, starting fresh."))
        return
    _pending_notices.append(("info", "Restored your previous session."))
    _pending_notices.extend(_restore_warnings())


def _native_window():
    """The pywebview window in native mode, else None."""
    try:
        return app.native.main_window
    except Exception:  # noqa: BLE001
        return None


def _dialog_type(name: str):
    """webview.FileDialog.OPEN / .SAVE on pywebview >= 5, webview.OPEN_DIALOG / SAVE_DIALOG before."""
    import webview
    enum = getattr(webview, "FileDialog", None)
    if enum is not None:
        return getattr(enum, name)
    return getattr(webview, f"{name}_DIALOG")


def _first_path(result) -> Optional[str]:
    """create_file_dialog returns None (cancelled), a str, or a tuple/list of paths."""
    if isinstance(result, (list, tuple)):
        result = result[0] if result else None
    return str(result) if result else None


async def _read_upload(e) -> bytes:
    """Bytes of an uploaded file, for both the NiceGUI 3.x (e.file) and 1.x/2.x (e.content) APIs."""
    file = getattr(e, "file", None)
    if file is not None:
        return await file.read()
    return e.content.read()


def _finish_load() -> None:
    """After a successful load: queue the messages and rebuild every pane from `state`."""
    _pending_notices.append(("positive", "Project loaded successfully!"))
    _pending_notices.extend(_restore_warnings())
    ui.navigate.reload()   # index() -> build_layout(state) repopulates all three panes


async def save_project() -> None:
    path: Optional[str] = None
    window = _native_window()
    if window is not None:
        try:
            result = await window.create_file_dialog(
                _dialog_type("SAVE"), save_filename="project.json", file_types=JSON_FILE_TYPES
            )
        except Exception:  # noqa: BLE001 - dialog unavailable: use the default file below
            result, window = None, None
        else:
            path = _first_path(result)
            if path is None:
                return  # the user cancelled
    if path is None:
        path = str(Path.cwd() / DEFAULT_PROJECT_NAME)
    if not path.lower().endswith(".json"):
        path += ".json"

    try:
        state.save_to_json(path)
    except Exception as exc:  # noqa: BLE001
        ui.notify(f"Could not save the project: {exc}", type="negative")
        return
    ui.notify("Project saved successfully!", type="positive", caption=path)


def _open_upload_dialog() -> None:
    """Browser fallback when there is no native window: pick the file with ui.upload."""
    with ui.dialog() as dialog, ui.card():
        ui.label("Select a project (.json) file")

        async def handle(e) -> None:
            try:
                data = json.loads((await _read_upload(e)).decode("utf-8-sig"))
                state.load_from_json(data)
            except Exception as exc:  # noqa: BLE001
                ui.notify(f"Could not open the project: {exc}. Your current work was not changed.",
                          type="negative")
                return
            dialog.close()
            _finish_load()

        ui.upload(on_upload=handle, auto_upload=True, max_files=1).props("accept=.json")
        ui.button("Cancel", on_click=dialog.close).props("flat")
    dialog.open()


async def open_project() -> None:
    window = _native_window()
    if window is None:
        _open_upload_dialog()
        return
    try:
        result = await window.create_file_dialog(
            _dialog_type("OPEN"), allow_multiple=False, file_types=JSON_FILE_TYPES
        )
    except Exception:  # noqa: BLE001
        _open_upload_dialog()
        return
    path = _first_path(result)
    if path is None:
        return  # the user cancelled
    try:
        state.load_from_json(path)
    except Exception as exc:  # noqa: BLE001
        ui.notify(f"Could not open the project: {exc}. Your current work was not changed.",
                  type="negative")
        return
    _finish_load()

# ---------------------------------------------------------------------------
# Frequency selector (top bar)
# ---------------------------------------------------------------------------
_DIRECTION_LABEL = {"E": "Tx", "R": "Rx"}
FREQUENCY_DEBOUNCE_S = 0.35     # ticking several boxes quickly rebuilds the map only once


def _fmt_ghz(freq_ghz: float) -> str:
    return f"{freq_ghz:.6f}".rstrip("0").rstrip(".")


def frequency_options(state: AppState) -> Dict[float, str]:
    """{freq_key(GHz): "14.78 GHz (Tx)"} for every distinct frequency of the imported station,
    ascending. A frequency that appears as both Tx and Rx shows "(Tx/Rx)"; one with no
    resolved direction shows "(?)"."""
    directions: Dict[float, List[str]] = {}
    for g in state.itu_import.all_frequencies or []:
        try:
            key = freq_key(g["freq_center_ghz"])
        except (KeyError, TypeError, ValueError):
            continue
        label = _DIRECTION_LABEL.get(g.get("direction"), "?")
        seen = directions.setdefault(key, [])
        if label not in seen:
            seen.append(label)
    return {k: f"{_fmt_ghz(k)} GHz ({'/'.join(v)})" for k, v in sorted(directions.items())}


async def _call_maybe_async(cb: Callable[[], Any]) -> None:
    result = cb()
    if inspect.isawaitable(result):
        await result


# Pane widths as specified: 28% / 44% / 28%.
# Below ~1000 px (a narrowed window) the panes stack vertically and the page scrolls,
# instead of squeezing three columns into an unusable width.
LAYOUT_CSS = """
.nicegui-content { padding: 0 !important; gap: 0 !important; }
/* Top bar (Save / Open Project). The three panes get the rest of the window height. */
.lbc-topbar {
    display: flex !important; flex-wrap: nowrap !important; align-items: center !important;
    gap: 0.5rem !important; width: 100%; height: 48px; box-sizing: border-box;
    padding: 0 0.75rem; border-bottom: 1px solid rgba(128,128,128,0.35);
}
.lbc-topbar .lbc-freq-select { width: 20rem; max-width: 40vw; }
.lbc-root {
    display: flex !important; flex-wrap: nowrap !important; gap: 0 !important;
    align-items: stretch !important;
    width: 100%; height: calc(100vh - 48px);
}
.lbc-pane {
    display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important;
    align-items: stretch !important; gap: 0.5rem !important;
    height: 100%; min-width: 0; box-sizing: border-box; padding: 0.75rem;
}
.lbc-left   { flex: 0 0 28%; border-right: 1px solid rgba(128,128,128,0.35); }
.lbc-middle { flex: 0 0 44%; }
.lbc-right  { flex: 0 0 28%; border-left: 1px solid rgba(128,128,128,0.35); }
/* The scroll area takes whatever height is left in its pane. min-height:0 is what
   lets a flex child shrink below its content so it scrolls instead of overflowing. */
.lbc-scroll { flex: 1 1 0; min-height: 0; width: 100%; }
/* ---- Split view (ui.middle_pane.build_split_layout) ------------------------------------
   1 frequency: .lbc-split > cell.  2: .lbc-split > row > 2 cells.  3+: .lbc-split >
   .lbc-split-grid > rows of 2 cells.  Cells are equal width (flex 1 1 0) and may shrink. */
.lbc-split, .lbc-split-grid {
    display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important;
    align-items: stretch !important; gap: 0.5rem !important; width: 100%; min-width: 0;
}
.lbc-split-row {
    display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important;
    align-items: stretch !important; gap: 0.5rem !important; width: 100%; min-width: 0;
}
.lbc-split-cell {
    display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important;
    align-items: stretch !important; gap: 0.35rem !important; min-width: 0; box-sizing: border-box;
    overflow-x: auto;                       /* a narrow cell scrolls its table instead of clipping it */
}
.lbc-split > .lbc-split-cell { flex: 0 0 auto; width: 100%; }
.lbc-split-row > .lbc-split-cell, .lbc-split-spacer { flex: 1 1 0; min-width: 0; }
.lbc-split-title { padding: 0 0.15rem 0.15rem; border-bottom: 1px solid rgba(128,128,128,0.35); }
/* fill mode (Middle Pane): the split takes the rest of the pane height, the maps need it. */
.lbc-split-fill { flex: 1 1 0; min-height: 0; overflow-y: auto; }
.lbc-split-fill > .lbc-split-cell, .lbc-split-fill > .lbc-split-row { flex: 1 1 0; min-height: 0; }
.lbc-split-fill > .lbc-split-grid { flex: 1 0 auto; }
.lbc-split-fill .lbc-split-grid > .lbc-split-row { flex: 1 1 0; min-height: 240px; }
.lbc-split-fill .lbc-split-cell { overflow: hidden; }
/* Map host: fills the rest of its cell; the iframe fills the host. */
.lbc-map-host {
    flex: 1 1 0; min-height: 0; width: 100%; position: relative;
    border: 1px solid rgba(128,128,128,0.35); border-radius: 8px; overflow: hidden;
}
.lbc-map-frame { position: absolute; inset: 0; width: 100%; height: 100%; border: none; }
@media (max-width: 1000px) {
    .lbc-root { flex-direction: column !important; height: auto; }
    .lbc-pane { flex: none !important; width: 100%; height: 85vh; border: none; }
}
"""


def build_layout(state: AppState) -> None:
    """Build the three panes. Called once per page load."""
    ui.add_css(LAYOUT_CSS)

    # Everything in the Left Pane below the tracker depends on the current step,
    # so it is refreshable and re-rendered by the tracker's on_change callback.
    @ui.refreshable
    def step_inputs() -> None:
        n = state.current_step
        ui.label(f"Step {n} — {step_title(n)}").classes("text-subtitle1 text-weight-medium")
        build_step_form(state, n)

    # ---- Frequency selection: wiring ---------------------------------------------
    # The panes are built after the top bar, so they register themselves here once they exist.
    #   "right": Right Pane's refresh_all() - re-creates the split of plots + I/N tables
    #   "map":   Middle Pane's async rebuild() - re-creates the map cells and redraws every map
    # Both build their split view with ui.middle_pane.build_split_layout from
    # state.selected_frequencies.
    panes: Dict[str, Optional[Callable[[], Any]]] = {"right": None, "map": None}
    change_token = {"n": 0}

    async def refresh_for_frequencies(rebuild_map: bool = True) -> None:
        """Re-render the Middle and Right panes for state.selected_frequencies."""
        for name in ("right", "map"):
            cb = panes.get(name)
            if cb is None or (name == "map" and not rebuild_map):
                continue
            try:
                await _call_maybe_async(cb)
            except Exception as exc:  # noqa: BLE001 - one pane failing must not block the other
                ui.notify(f"Could not refresh the {name} pane: {exc}", type="negative")

    @ui.refreshable
    def frequency_selector() -> None:
        """Multi-select of the imported station's frequencies, bound to state.selected_frequencies."""
        options = frequency_options(state)
        state.sanitize_selected_frequencies()       # drop anything no longer in the station
        if not options:
            ui.label("Frequencies: import an ITU station to choose").classes("text-caption text-grey-7")
            return

        total = len(options)

        def _summary() -> str:
            return f"{len(state.selected_frequencies)} of {total} frequencies"

        async def _on_change(e) -> None:
            chosen = set(e.value or [])
            new = [f for f in options if f in chosen]          # keep ascending order
            if new == state.selected_frequencies:
                return
            state.selected_frequencies = new                   # the bound state field
            state.bump_map_version()
            select.props(f'display-value="{_summary()}"')
            # Debounce: only the last change of a quick series refreshes the panes.
            change_token["n"] += 1
            mine = change_token["n"]
            await asyncio.sleep(FREQUENCY_DEBOUNCE_S)
            if mine == change_token["n"]:
                await refresh_for_frequencies()

        ui.label("Frequencies").classes("text-caption text-grey-7")
        select = ui.select(options, value=list(state.selected_frequencies), multiple=True,
                           on_change=_on_change).props(
            f'dense outlined options-dense display-value="{_summary()}"'
        ).classes("lbc-freq-select")
        ui.button("All", on_click=lambda: select.set_value(list(options))) \
            .props("flat dense no-caps size=sm")
        ui.button("None", on_click=lambda: select.set_value([])) \
            .props("flat dense no-caps size=sm")

    def after_itu_import() -> None:
        """The import replaced all_frequencies (and reset the selection): rebuild the selector,
        the step form, the results and the (empty) map cells. No map is built until the user
        asks for it."""
        step_inputs.refresh()
        frequency_selector.refresh()
        if panes["right"] is not None:
            panes["right"]()
        layout_only = getattr(panes["map"], "refresh_layout", None)
        if layout_only is not None:
            layout_only()

    # ---- Top bar: project save / open + frequency selector -------------------------
    with ui.row().classes("lbc-topbar"):
        ui.button("\U0001F4BE Save Project", on_click=save_project).props("flat dense no-caps")
        ui.button("\U0001F4C2 Open Project", on_click=open_project).props("flat dense no-caps")
        ui.separator().props("vertical")
        frequency_selector()

    with ui.row().classes("lbc-root"):
        # ---- Left Pane (28%): step tracker + inputs -------------------------
        with ui.column().classes("lbc-pane lbc-left"):
            ui.label("Inputs").classes("text-h6")
            build_step_tracker(state, on_change=lambda _n: step_inputs.refresh())
            ui.separator()
            with ui.scroll_area().classes("lbc-scroll"):
                # The ITU import can change Gt/Pt/frequencies, so after "Proceed" the
                # current step's form is re-rendered to show the imported values.
                build_itu_import(state, on_imported=after_itu_import)
                ui.separator().classes("q-my-sm")
                step_inputs()

        # ---- Middle Pane (44%): map ----------------------------------------
        # No scroll area here: the map fills the pane and pans/zooms by itself.
        with ui.column().classes("lbc-pane lbc-middle"):
            ui.label("Map").classes("text-h6")
            # build_middle_pane lays out one map per selected frequency (build_split_layout) and
            # returns the async "rebuild maps" function; the Right Pane calls it after border
            # analysis / mitigation so the maps follow the results.
            rebuild_map = build_middle_pane(state)
            panes["map"] = rebuild_map

        # ---- Right Pane (28%): results -------------------------------------
        with ui.column().classes("lbc-pane lbc-right"):
            ui.label("Results").classes("text-h6")      # one plot + I/N table per selected frequency
            with ui.scroll_area().classes("lbc-scroll"):
                panes["right"] = build_right_pane(state, on_map_refresh=rebuild_map)

    # Silent background autosave, and the notices queued by a load / startup restore
    # (a short delay so the browser connection is up before they are sent).
    ui.timer(AUTOSAVE_INTERVAL_S, autosave_now)

    def _flush_notices() -> None:
        while _pending_notices:
            kind, message = _pending_notices.pop(0)
            ui.notify(message, type=kind)

    ui.timer(0.5, _flush_notices, once=True)


@ui.page("/")
def index() -> None:
    build_layout(state)


# The __mp_main__ half of this guard is required: native mode starts the
# pywebview window in a child process that re-imports this module, and without
# it ui.run() would fail (or spawn windows repeatedly) on Windows/macOS.
if __name__ == "__main__":
    # Main process only: the __mp_main__ child owns just the window and has its own,
    # empty AppState - it must never load or (on shutdown) overwrite the autosave.
    autoload_on_startup()
    app.on_shutdown(autosave_now)   # one last save when the window is closed

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Link Budget Calculator",
        native=True,
        window_size=(1600, 900),
        reload=False,  # auto-reload and native mode don't mix
    )