"""
Middle Pane (Phase 5, split view in Part 4): interactive Folium coordination map(s).

``build_middle_pane(state)`` draws a toolbar (borders / DEM hillshade / DEM opacity /
Rebuild Map) and, below it, ONE Folium map per selected frequency, laid out by
``build_split_layout`` (1 frequency: a single column; 2: two equal columns side by side;
3 or more: a grid of rows of two). Each map is a standalone HTML file shown in its own iframe.

``build_split_layout(selected_freqs, state)`` is shared with the Right Pane
(ui/right_pane.py): it creates the layout and hands back one empty cell per frequency for
the caller to fill.

How a map is built
------------------
1. On "Rebuild Map", once on page load if there is data, and automatically whenever
   ``AppState.map_rebuild_version`` changes (a 0.5 s poll, see ``build_middle_pane``), a
   plain-data snapshot of everything ONE frequency's map needs is taken
   from ``AppState`` on the UI thread (``_snapshot``): that frequency's contour and its own
   border-analysis rows (``FrequencyResults.border_rows``).
2. ``_build_map_file`` - which touches no NiceGUI or AppState objects - runs in a worker
   thread via ``run.io_bound``. It follows the same recipe as the old
   core/streamlit_map_component.py::render_coordination_map:
       create_base_map  (includes the earth-station marker)
       -> add_country_borders          (optional)
       -> add_coordination_contour     (if a Step 5 contour exists)
       -> add_interference_zones + legend (if border analysis rows exist)
       -> add_dem_overlay              (optional, only if the DEM file exists)
   and writes ``map_<freq_tag>_<8 hex>.html`` into ``STATIC_MAPS_DIR``. EVERY build gets a
   NEW file name, so a file an iframe is showing is never overwritten or replaced. The maps
   are built one after the other (the border / DEM data are cached on disk and are not safe to
   fetch concurrently).
3. Each iframe's ``src`` is ``/maps/<file>``; the new name means the browser never serves a
   cached older map. While a cell is rebuilding a "Rebuilding map..." overlay covers it; it
   goes away when the iframe has loaded the new file, and only then are the older files of
   that frequency deleted (also at the start of the next build, if the load event never came).
   Every iframe also has a unique ``name`` and ``data-map-id`` (``lbc-map-<freq>``).

``desktop_app.py`` must serve ``STATIC_MAPS_DIR`` at ``/maps``
(``app.add_static_files``) - it imports the constants from here.

The callable returned by ``build_middle_pane`` is the async "rebuild every map" function.
It also has ``.refresh_layout()``: re-create the (empty) map cells when the set of shown
frequencies changed, without building any map.
"""
from __future__ import annotations

import inspect
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from nicegui import run, ui

from core.app_paths import get_resource_path, get_user_data_dir
from ui.state import UNSPECIFIED_FREQ_KEY, AppState, FrequencyResults

log = logging.getLogger(__name__)

# Generated map HTML is written to the per-user data dir (writable even when the app is
# frozen / installed read-only); desktop_app serves it at STATIC_MAPS_URL.
def _dem_overlay_available(dem_path: Optional[str]) -> bool:
    """True if a DEM layer can be drawn: the GeoTIFF exists, or (bundled DEM only) the prebuilt
    PNG hillshade is shipped. map_visualization is imported lazily because folium may be missing;
    without it the plain file check is used."""
    try:
        try:
            from core import map_visualization as _mv
        except ImportError:
            import map_visualization as _mv  # type: ignore[no-redef]
        return bool(_mv.dem_overlay_available(dem_path))
    except Exception:  # noqa: BLE001 - never let a toolbar check break page building
        return bool(dem_path) and os.path.isfile(dem_path)


STATIC_MAPS_DIR = get_user_data_dir("maps")
STATIC_MAPS_URL = "/maps"
MAP_FILENAME = "coordination_map.html"        # legacy name (kept for imports; no longer written)
MAP_FILE_PREFIX = "map_"
AUTO_REBUILD_POLL_S = 0.5      # the debounced auto-rebuild poll (a change must hold for one tick)
LOAD_TIMEOUT_S = 12.0          # hide the overlay even if the iframe never reports "load"
STALE_MAP_MAX_AGE_S = 24 * 3600  # leftovers of earlier sessions are purged at page build
DEFAULT_ZOOM = 7
FREQ_TOL_GHZ = 1e-3        # same tolerance step5_pane uses to match a group to the saved frequency

_DIRECTION_LABEL = {"E": "Tx", "R": "Rx"}

# Why an azimuth got no PASS/FAIL sector on the map.
SKIP_NO_BORDER = "no border found"
SKIP_NO_OVERLAP = "no overlap (normal)"
SKIP_CALC_ERROR = "calc error"
SKIP_NO_LIMIT = "no limit"
SKIP_REASONS = (SKIP_NO_BORDER, SKIP_NO_OVERLAP, SKIP_CALC_ERROR, SKIP_NO_LIMIT)
_SKIP_BENIGN = (SKIP_NO_BORDER, SKIP_NO_OVERLAP)      # expected, never degrade the badge

# Persistent status line under each map: severity -> (icon, Quasar colour).
_BADGE_STYLE = {
    "ok": ("\U0001F7E2", "positive"),      # green circle
    "warn": ("\U0001F7E1", "warning"),     # yellow circle
    "error": ("\U0001F534", "negative"),   # red circle
    "off": ("\u26AA", "grey"),             # white circle: layer not requested / nothing to draw
}
_BADGE_LAYERS = (("contour", "Contour"), ("sectors", "Sectors"), ("borders", "Borders"), ("dem", "DEM"))


# ---------------------------------------------------------------------------
# Frequencies: which ones are shown, and each one's results
# ---------------------------------------------------------------------------
def _fmt_ghz(freq_ghz: float) -> str:
    return f"{float(freq_ghz):.6f}".rstrip("0").rstrip(".")


def _freq_tag(freq: Optional[float]) -> str:
    """File / DOM-safe unique tag of a frequency (None = the single, unlabelled result set)."""
    if freq is None:
        return "default"
    return f"{float(freq):.6f}".replace(".", "_").replace("-", "m")


def _new_map_filename(freq_tag: str) -> str:
    """A fresh, unique file name for ONE build: ``map_<freq_tag>_<8 hex>.html``."""
    return f"{MAP_FILE_PREFIX}{freq_tag}_{uuid.uuid4().hex[:8]}.html"


def _map_file_re(freq_tag: str) -> "re.Pattern[str]":
    return re.compile(rf"^{re.escape(MAP_FILE_PREFIX + freq_tag)}_[0-9a-f]{{8}}\.html$")


def _cleanup_old_maps(freq_tag: str, keep: Optional[str]) -> int:
    """Delete every map file of ``freq_tag`` except ``keep``. Never raises (a file that is
    locked or already gone is simply skipped). Returns how many files were removed."""
    pattern = _map_file_re(freq_tag)
    removed = 0
    try:
        names = os.listdir(STATIC_MAPS_DIR)
    except OSError:
        return 0
    for name in names:
        if name == keep or not pattern.match(name):
            continue
        try:
            (STATIC_MAPS_DIR / name).unlink()
            removed += 1
        except OSError:
            log.debug("could not delete old map %s", name, exc_info=True)
    return removed


def _purge_stale_maps(max_age_s: float = STALE_MAP_MAX_AGE_S) -> None:
    """Delete generated map files (and legacy ones) left over by earlier sessions."""
    cutoff = time.time() - max_age_s
    try:
        entries = list(STATIC_MAPS_DIR.iterdir())
    except OSError:
        return
    for path in entries:
        name = path.name
        generated = (name.startswith(MAP_FILE_PREFIX) or name.startswith("coordination_map")) \
            and (name.endswith(".html") or name.endswith(".tmp"))
        if not generated:
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def effective_frequencies(state: AppState) -> List[Optional[float]]:
    """The frequencies the Middle / Right panes show: the top-bar selection; with no
    selection (no ITU import, manual workflow) the active result set; with nothing at all
    ``[None]`` - one placeholder cell, so the panes never go blank."""
    chosen = [float(f) for f in (state.selected_frequencies or [])]
    if chosen:
        return chosen
    active = state.results.active_frequency
    return [active] if active is not None else [None]


def result_for_frequency(state: AppState, freq: Optional[float]) -> Optional[FrequencyResults]:
    """Results of one frequency. The selector offers the ITU group's centre frequency, but the
    saved ("base") frequency is stored under the saved value, which can differ by up to
    FREQ_TOL_GHZ - so fall back to the nearest key within that tolerance. None -> active."""
    res = state.results
    if freq is None:
        return res.active()
    fr = res.get(freq)
    if fr is not None:
        return fr
    best: Optional[Tuple[float, FrequencyResults]] = None
    for key, cand in res.results_by_frequency.items():
        d = abs(float(key) - float(freq))
        if d <= FREQ_TOL_GHZ and (best is None or d < best[0]):
            best = (d, cand)
    return best[1] if best else None


def border_rows_for(state: AppState, fr: Optional[FrequencyResults]) -> Optional[List[Dict[str, Any]]]:
    """Border-analysis rows of one frequency. Falls back to ``border.analysis_rows`` for the
    active frequency, so projects saved before per-frequency rows existed still show them."""
    if fr is None:
        return None
    if fr.border_rows is not None:
        return fr.border_rows
    return state.border.analysis_rows if fr is state.results.active() else None


def frequency_label(state: AppState, freq: Optional[float]) -> str:
    """"14.78 GHz (Tx)" - the same wording as the top-bar selector."""
    if freq is None or float(freq) == UNSPECIFIED_FREQ_KEY:
        return "Current results"
    fr = result_for_frequency(state, freq)
    direction = fr.direction if fr is not None else None
    if direction is None:
        for g in state.itu_import.all_frequencies or []:
            try:
                if abs(float(g["freq_center_ghz"]) - float(freq)) <= FREQ_TOL_GHZ:
                    direction = g.get("direction")
                    break
            except (KeyError, TypeError, ValueError):
                continue
    return f"{_fmt_ghz(freq)} GHz ({_DIRECTION_LABEL.get(direction, '?')})"


# ---------------------------------------------------------------------------
# Split layout (shared by the Middle and Right panes)
# ---------------------------------------------------------------------------
def build_split_layout(selected_freqs: List[Optional[float]], state: AppState,
                       fill: bool = False) -> List[Tuple[Optional[float], ui.column]]:
    """Create the split container for ``selected_freqs`` in the current NiceGUI context and
    return ``[(frequency, cell), ...]`` in order. Every cell is an empty ``ui.column`` that
    already carries a title with the frequency label; the caller fills it with ``with cell:``.

    * 1 frequency  -> a single full-width column (the original single-pane look);
    * 2 frequencies -> one ``ui.row`` with two equal columns;
    * 3 or more     -> a grid: a column of ``ui.row`` s, two cells each (a last odd cell
                       takes half a row).

    ``fill=True`` makes the layout take the whole height left in its pane (the map needs a
    definite height); the default lets the cells grow with their content (the results pane
    scrolls). The ``.lbc-split*`` classes are defined in desktop_app.LAYOUT_CSS.
    """
    freqs = list(selected_freqs) or [None]
    cells: List[Tuple[Optional[float], ui.column]] = []

    def make_cell(freq: Optional[float]) -> None:
        cell = ui.column().classes("lbc-split-cell")
        with cell:
            ui.label(frequency_label(state, freq)).classes("lbc-split-title text-subtitle2")
        cells.append((freq, cell))

    container = ui.column().classes("lbc-split" + (" lbc-split-fill" if fill else ""))
    with container:
        if len(freqs) == 1:
            make_cell(freqs[0])
        elif len(freqs) == 2:
            with ui.row().classes("lbc-split-row"):
                for f in freqs:
                    make_cell(f)
        else:
            with ui.column().classes("lbc-split-grid"):
                for i in range(0, len(freqs), 2):
                    with ui.row().classes("lbc-split-row"):
                        for f in freqs[i:i + 2]:
                            make_cell(f)
                        if len(freqs[i:i + 2]) == 1:       # keep the odd last cell half width
                            ui.element("div").classes("lbc-split-spacer")
    return cells


# ---------------------------------------------------------------------------
# Snapshot of state -> plain data for the worker thread
# ---------------------------------------------------------------------------
@dataclass
class _MapRequest:
    es_lat: float
    es_lon: float
    contour: Dict[float, float] = field(default_factory=dict)   # {azimuth: km}
    zones: List[Dict[str, Any]] = field(default_factory=list)   # add_interference_zones() input
    skipped_zones: int = 0
    skipped_by_reason: Dict[str, List[Any]] = field(default_factory=dict)   # reason -> [azimuth, ...]
    show_borders: bool = True
    show_dem: bool = False
    dem_path: Optional[str] = None
    dem_opacity: float = 0.5
    dem_requested: bool = False           # the DEM checkbox was ticked
    zoom: int = DEFAULT_ZOOM
    has_border_rows: bool = False
    freq_tag: str = "default"             # names this frequency's files inside STATIC_MAPS_DIR


def _earth_station_coords(state: AppState) -> Optional[Tuple[float, float]]:
    """Earth-station location: the accepted Step 5 parameters first, then the
    Step 1 calculator coordinates as a fallback."""
    p = state.step5_inputs.ppd_params or {}
    lat, lon = p.get("latitude"), p.get("longitude")
    if lat is None or lon is None:
        lat, lon = state.step1.es_lat, state.step1.es_lon
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _build_zone_results(
    rows: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Any]]]:
    """Step 8 border-analysis rows -> the dicts the sector drawing expects.

    Same rules as the Streamlit version: a row needs a border distance, a
    computed I/N result and a ``limit_db`` to get a PASS/FAIL sector; other rows
    are skipped. The sector colour reflects the ORIGINAL I/N; the optional Step 7
    diffraction recheck is only attached as an annotation.

    Returns ``(zones, skipped)`` where ``skipped`` maps a skip reason (one of
    SKIP_REASONS) to the azimuths skipped for it:

    * "no border found"      - no border crossing along that azimuth
    * "no overlap (normal)"  - a border exists but the contour does not reach it
    * "no limit"             - overlap, but the row carries no ``limit_db``
    * "calc error"           - overlap, but the I/N calculation failed / gave
                               no usable (finite) numbers
    """
    zones: List[Dict[str, Any]] = []
    skipped: Dict[str, List[Any]] = {r: [] for r in SKIP_REASONS}
    for row in rows or []:
        az = row.get("azimuth_deg")
        in_res = row.get("in_result")
        limit_db = row.get("limit_db")
        border_km = row.get("border_km")

        if border_km is None:
            skipped[SKIP_NO_BORDER].append(az)
            continue
        overlap = row.get("overlap")
        if overlap is None:                       # rows saved without the flag
            overlap = in_res is not None
        if not overlap:
            skipped[SKIP_NO_OVERLAP].append(az)
            continue
        if limit_db is None:
            skipped[SKIP_NO_LIMIT].append(az)
            continue
        if not in_res or "IN_ratio_dB" not in in_res:
            skipped[SKIP_CALC_ERROR].append(az)
            continue

        # in_ratio = in_res["IN_ratio_dB"]
        # required_km = row.get("required_km", border_km)
        # if not all(_is_finite_number(v) for v in (az, border_km, required_km, in_ratio, limit_db)):
        #     log.warning("Sector at azimuth %s skipped: NaN/inf/non-numeric input "
        #                 "(border=%s required=%s I/N=%s limit=%s)",
        #                 az, border_km, required_km, in_ratio, limit_db)
        #     skipped[SKIP_CALC_ERROR].append(az)
        #     continue

        # zone = {
        #     "azimuth": az,
        #     "border_distance_km": border_km,
        #     "required_distance_km": required_km,
        #     "in_ratio_db": in_ratio,
        #     "limit_db": limit_db,
        #     "status": "FAIL" if in_ratio > limit_db else "PASS",
        # }
        # diff = row.get("diffraction_result")
        # if diff and diff.get("diffraction_available"):
        #     zone["diffraction_loss_db"] = diff["diffraction_loss_dB"]
        #     zone["diffraction_in_ratio_db"] = diff["adjusted_IN_ratio_dB"]
        #     zone["diffraction_status"] = diff["adjusted_status"]
        # zones.append(zone)



        in_ratio = in_res["IN_ratio_dB"]
        required_km = row.get("required_km", border_km)
        if not all(_is_finite_number(v) for v in (az, border_km, required_km, in_ratio, limit_db)):
            log.warning("Sector at azimuth %s skipped: NaN/inf/non-numeric input "
                        "(border=%s required=%s I/N=%s limit=%s)",
                        az, border_km, required_km, in_ratio, limit_db)
            skipped[SKIP_CALC_ERROR].append(az)
            continue
            
        # 1. Determine the base status from the raw I/N check
        base_status = "FAIL" if in_ratio > limit_db else "PASS"
        
        zone = {
            "azimuth": az,
            "border_distance_km": border_km,
            "required_distance_km": required_km,
            "in_ratio_db": in_ratio,
            "limit_db": limit_db,
            "status": base_status,
        }
        
        # 2. Check if diffraction was applied
        diff = row.get("diffraction_result")
        if diff and diff.get("diffraction_available"):
            zone["diffraction_loss_db"] = diff["diffraction_loss_dB"]
            zone["diffraction_in_ratio_db"] = diff["adjusted_IN_ratio_dB"]
            zone["diffraction_status"] = diff["adjusted_status"]
            
            # 3. THE FIX: If diffraction turned a FAIL into a PASS, override the status!
            # This ensures map_visualization.py draws the sector in GREEN.
            if base_status == "FAIL" and diff["adjusted_status"] == "PASS":
                zone["status"] = "PASS"
                
        zones.append(zone)
    return zones, skipped


def _skip_summary(skipped: Dict[str, Any]) -> str:
    """'no border found 3, no overlap (normal) 20, calc error 1' (non-zero entries only).
    Values may be lists of azimuths or plain counts."""
    parts = []
    for reason in SKIP_REASONS:
        v = skipped.get(reason, 0)
        n = len(v) if isinstance(v, (list, tuple)) else int(v or 0)
        if n:
            parts.append(f"{reason} {n}")
    return ", ".join(parts)


def _snapshot(
    state: AppState, freq: Optional[float], *, show_borders: bool, show_dem: bool, dem_opacity: float
) -> Optional[_MapRequest]:
    """Copy what ONE frequency's map needs out of AppState (UI thread). None = no location yet."""
    coords = _earth_station_coords(state)
    if coords is None:
        return None
    es_lat, es_lon = coords

    # active_contour() is the AP7 combined max(mode 1, mode 2) contour when mode 2
    # was computed, otherwise exactly that frequency's mode 1 contour.
    fr = result_for_frequency(state, freq)
    profile = fr.active_contour() if fr is not None else None
    contour = {az: dist for az, dist in profile if dist is not None} if profile else {}

    rows = border_rows_for(state, fr)
    zones, skipped_by_reason = _build_zone_results(rows)

    dem_path = state.border.effective_dem_path          # sentinel -> real path
    dem_ok = _dem_overlay_available(dem_path)           # source GeoTIFF OR prebuilt PNG hillshade

    return _MapRequest(
        es_lat=es_lat,
        es_lon=es_lon,
        contour=dict(sorted(contour.items())),
        zones=zones,
        skipped_zones=sum(len(v) for v in skipped_by_reason.values()),
        skipped_by_reason=skipped_by_reason,
        show_borders=show_borders,
        show_dem=show_dem and dem_ok,
        dem_path=dem_path if dem_ok else None,
        dem_opacity=dem_opacity,
        dem_requested=bool(show_dem),
        has_border_rows=bool(rows),
        freq_tag=_freq_tag(freq),
    )


# ---------------------------------------------------------------------------
# Worker (runs in a thread - no NiceGUI / AppState access in here)
# ---------------------------------------------------------------------------
def _build_map_file(req: _MapRequest) -> Dict[str, Any]:
    """Build the Folium map and write it to a NEW, uniquely named file in STATIC_MAPS_DIR.

    Returns ``{"ok": bool, "error": str | None, "warnings": [str, ...], "filename": str | None,
    "layers_status": {...}}`` (``filename`` is set when ``ok``).

    ``layers_status`` has one entry per layer - "contour", "sectors", "borders", "dem" - each
    ``{"ok": bool, "reason": str, "count": int, "severity": "ok"|"warn"|"error"|"off", "detail": str}``
    (the "sectors" entry also has ``total`` and ``skipped``: {reason: count}). Optional layers
    that fail only add a warning and a status; a failure of the base map or of the file write
    is a hard error.
    """
    warnings: List[str] = []
    layers_status: Dict[str, Dict[str, Any]] = {}
    try:
        try:
            from core import map_visualization as mv
        except ImportError:
            import map_visualization as mv  # type: ignore[no-redef]
    except ImportError as e:
        return {
            "ok": False,
            "error": f"Map visualization unavailable - missing dependency: {e}. "
                     "Install with: pip install folium pyproj numpy rasterio matplotlib",
            "warnings": warnings,
            "filename": None,
            "layers_status": layers_status,
        }

    try:
        m = mv.create_base_map(req.es_lat, req.es_lon, zoom=req.zoom)
    except Exception as e:
        log.exception("create_base_map failed")
        return {"ok": False, "error": f"Could not build the base map: {e}", "warnings": warnings,
                "filename": None, "layers_status": layers_status}

    # ---- borders --------------------------------------------------------
    if req.show_borders:
        try:
            # Look for the bundled borders in data_sources (resolved with get_resource_path, so
            # it works from any working directory and inside a PyInstaller build).
            local_borders_path = None
            # Bundled GeoJSON first, then the older shapefile / coastline fallbacks.
            for name in ["ne_10m_admin_0_countries.geojson", "ne_10m_admin_0_countries.shp",
                         "ne_10m_coastline.shp"]:
                candidate = get_resource_path(os.path.join("data_sources", name))
                if os.path.exists(candidate):
                    local_borders_path = candidate
                    break

            if local_borders_path:
                st = mv.add_country_borders(m, geojson_path=local_borders_path)
            else:
                # Fallback to download only if the local file is missing
                st = mv.add_country_borders(m)
        except Exception as e:
            log.exception("country borders layer failed")
            st = mv.layer_status(False, "Error", severity="error", detail=str(e))
        layers_status["borders"] = st
        if not st["ok"]:
            warnings.append(f"Country borders unavailable: {st.get('detail') or st['reason']}")
    else:
        layers_status["borders"] = mv.layer_status(False, "Off", severity="off")

    # ---- coordination contour (pane above the DEM) ---------------------------
    if req.contour:
        try:
            m = mv.add_coordination_contour(
                m, req.es_lat, req.es_lon, req.contour, color="red", line_style="dashed"
            )
            layers_status["contour"] = mv.layer_status(True, "OK", count=len(req.contour))
        except Exception as e:
            log.exception("coordination contour failed")
            layers_status["contour"] = mv.layer_status(False, "Error", severity="error", detail=str(e))
            warnings.append(f"Coordination contour could not be drawn: {e}")
    else:
        layers_status["contour"] = mv.layer_status(False, "none - run Step 5", severity="off")

    # ---- PASS/FAIL sectors (pane above the DEM) - one try/except PER SECTOR --------
    skipped: Dict[str, List[Any]] = {r: list(req.skipped_by_reason.get(r, [])) for r in SKIP_REASONS}
    n_rows_total = len(req.zones) + sum(len(v) for v in skipped.values())
    if req.has_border_rows:
        drawn = 0
        try:
            from pyproj import Geod
            geod = Geod(ellps="WGS84")
            group = mv.create_interference_group(m)
            half_width = mv.interference_half_width(req.zones)
            for i, zone in enumerate(req.zones):
                try:
                    mv.draw_interference_sector(group, zone, req.es_lat, req.es_lon, half_width,
                                                geod=geod, index=i)
                    drawn += 1
                except Exception as e:      # one bad azimuth must not break the whole map
                    az = zone.get("azimuth")
                    log.warning("Sector at azimuth %s skipped (%s): %s", az, SKIP_CALC_ERROR, e)
                    skipped[SKIP_CALC_ERROR].append(az)
            if drawn:
                m.get_root().html.add_child(mv.folium.Element(mv.create_interference_legend()))
        except Exception as e:              # setup failure (pyproj missing, ...): no sectors at all
            log.exception("interference sectors could not be set up")
            for zone in req.zones[drawn:]:
                skipped[SKIP_CALC_ERROR].append(zone.get("azimuth"))
            warnings.append(f"Interference zones could not be drawn: {e}")

        n_error = len(skipped[SKIP_CALC_ERROR])
        n_nolimit = len(skipped[SKIP_NO_LIMIT])
        if n_error or n_nolimit:
            severity = "error" if drawn == 0 else "warn"
        else:
            severity = "ok"
        summary = _skip_summary(skipped)
        log.info("Sectors: drawn %d/%d%s", drawn, n_rows_total, f"; skipped: {summary}" if summary else "")
        for reason in SKIP_REASONS:
            if skipped[reason]:
                log.debug("Sectors skipped (%s): azimuths %s", reason, skipped[reason])
        layers_status["sectors"] = mv.layer_status(
            severity == "ok", f"{drawn}/{n_rows_total}", count=drawn, severity=severity,
            detail=("Skipped: " + summary) if summary else "",
            total=n_rows_total, skipped={r: len(v) for r, v in skipped.items()},
        )
    else:
        layers_status["sectors"] = mv.layer_status(
            False, "no analysis", severity="off", total=0, skipped={r: 0 for r in SKIP_REASONS})

    # ---- DEM hillshade (pane below the default overlay pane) ----------------------
    if req.show_dem and req.dem_path:
        try:
            st = mv.add_dem_overlay(m, req.dem_path, req.es_lat, req.es_lon, opacity=req.dem_opacity)
        except Exception as e:
            log.exception("DEM overlay failed")
            st = mv.layer_status(False, "Error", severity="error", detail=str(e))
        layers_status["dem"] = st
        if not st["ok"]:
            warnings.append(f"DEM overlay could not be added: {st.get('detail') or st['reason']}")
    elif req.dem_requested:
        layers_status["dem"] = mv.layer_status(False, "File not found", severity="error",
                                               detail="The DEM file does not exist.")
    else:
        layers_status["dem"] = mv.layer_status(False, "Off", severity="off")

    # ---- exactly one LayerControl, added after every layer ------------------------
    try:
        mv.ensure_layer_control(m)
    except Exception as e:
        log.warning("LayerControl could not be added: %s", e)

    # A brand-new file name for every build: nothing is replaced or overwritten, so a map an
    # iframe is showing right now is never touched. The iframe only learns the name after this
    # function returns, i.e. when the file is complete.
    filename = _new_map_filename(req.freq_tag)
    final_path = STATIC_MAPS_DIR / filename
    try:
        STATIC_MAPS_DIR.mkdir(parents=True, exist_ok=True)
        m.save(str(final_path))
        mv.localize_map_file(final_path)    # CDN links -> /web_assets (no-op if not bundled)
    except Exception as e:
        log.exception("saving the map failed")
        try:
            final_path.unlink()             # no half-written file left behind
        except OSError:
            pass
        return {"ok": False, "error": f"Could not save the map: {e}", "warnings": warnings,
                "filename": None, "layers_status": layers_status}

    return {"ok": True, "error": None, "warnings": warnings, "filename": filename,
            "layers_status": layers_status}


# ---------------------------------------------------------------------------
# Status badges (pure functions - no NiceGUI needed)
# ---------------------------------------------------------------------------
def badge_specs(layers_status: Optional[Dict[str, Dict[str, Any]]]) -> List[Tuple[str, str, str]]:
    """``layers_status`` -> ``[(severity, text, tooltip), ...]`` in display order, e.g.
    ``("ok", "Sectors (34/36)", "Skipped: ...")``, ``("warn", "Borders (Offline)", "...")``."""
    specs: List[Tuple[str, str, str]] = []
    for key, label in _BADGE_LAYERS:
        st = (layers_status or {}).get(key)
        if not st:
            continue
        severity = st.get("severity") or ("ok" if st.get("ok") else "error")
        reason = str(st.get("reason") or "")
        if key == "sectors" and severity != "off":
            text = f"{label} ({st.get('count', 0)}/{st.get('total', 0)})"
        elif severity == "ok" and reason in ("", "OK"):
            text = label
        else:
            text = f"{label} ({reason})" if reason else label
        tip = str(st.get("detail") or reason or label)
        specs.append((severity, text, tip))
    return specs


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@dataclass
class _MapCell:
    """The widgets (and load state) of one frequency's map cell."""
    freq: Optional[float]
    frame: Any          # the <iframe> element
    placeholder: Any    # centred text shown while there is no map
    status: Any         # per-map caption (what is / is not on this map)
    overlay: Any = None             # "Rebuilding map..." cover, shown while this cell is stale
    current_file: Optional[str] = None   # file the iframe shows / is loading (never deleted)
    awaiting_load: bool = False     # src was set, the iframe's "load" event has not come yet
    badges: Any = None              # persistent status line (row of badges) under the map


def build_middle_pane(state: AppState) -> Callable[[], Awaitable[None]]:
    """Build the Middle Pane contents (toolbar + one map iframe per selected frequency).

    Call this inside the Middle Pane's column. Returns the async ``rebuild``
    coroutine function so other parts of the UI can trigger a map refresh; it redraws the
    maps of the CURRENT frequency selection (re-creating the cells first if the selection
    changed). ``rebuild.refresh_layout()`` only re-creates the cells.

    The pane also watches ``state.map_rebuild_version`` (a 0.5 s ``ui.timer``): when it differs
    from the version the maps were last built for - and has stopped changing for one tick - the
    maps are rebuilt automatically, quietly (no info toasts), with an overlay on every cell.
    """
    building = False
    again = False                       # a rebuild was requested while one was running
    manual_pending = False              # ...and at least one such request was a button press
    layout: Dict[str, Any] = {"freqs": None}
    cells: Dict[str, _MapCell] = {}
    # built:   the map_rebuild_version the last build's snapshot was taken at
    # pending: the version the poll saw on its previous tick (debounce)
    versions = {"built": state.map_rebuild_version, "pending": state.map_rebuild_version}

    _purge_stale_maps()                 # leftovers of earlier sessions (older than a day)

    dem_path = state.border.effective_dem_path
    dem_available = _dem_overlay_available(dem_path)

    # ---- toolbar ---------------------------------------------------------
    with ui.row().classes("w-full items-center no-wrap q-gutter-sm"):
        borders_cb = ui.checkbox("Show country borders", value=True)
        dem_cb = ui.checkbox("Show DEM hillshade", value=False)
        if not dem_available:
            dem_cb.disable()
            dem_cb.tooltip(f"DEM not found at {dem_path!r}")
        with ui.column().classes("gap-0").style("min-width: 130px"):
            ui.label("DEM opacity").classes("text-caption text-grey-7")
            opacity_slider = ui.slider(min=0.0, max=1.0, step=0.05, value=0.5)
            opacity_slider.bind_enabled_from(dem_cb, "value")
        ui.space()
        rebuild_btn = ui.button("Rebuild Map", icon="refresh")

    status = ui.label("").classes("text-caption text-grey-7")

    # ---- loading overlay / iframe load handling ---------------------------
    def _set_overlay(cell: Optional[_MapCell], visible: bool) -> None:
        if cell is not None and cell.overlay is not None:
            cell.overlay.set_visibility(visible)

    def _finish_loading(tag: str, filename: Optional[str]) -> None:
        """The iframe has the new file (or we gave up waiting): drop the overlay and delete the
        older files of this frequency. A no-op if a newer build has taken over meanwhile."""
        cell = cells.get(tag)
        if cell is None or not cell.awaiting_load or cell.current_file != filename:
            return
        cell.awaiting_load = False
        _set_overlay(cell, False)
        _cleanup_old_maps(tag, keep=cell.current_file)

    # ---- map area: one cell per frequency --------------------------------
    def _idle_text() -> str:
        if _earth_station_coords(state) is None:
            return "No earth station location yet - complete Step 1 / Step 5, then press Rebuild Map."
        return "Press \u201cRebuild Map\u201d to draw the map."

    @ui.refreshable
    def map_grid() -> None:
        freqs = effective_frequencies(state)
        layout["freqs"] = list(freqs)
        cells.clear()
        for freq, cell in build_split_layout(freqs, state, fill=True):
            tag = _freq_tag(freq)
            with cell:
                with ui.element("div").classes("lbc-map-host"):
                    placeholder = ui.label(_idle_text()).classes("text-grey-7 absolute-center text-center")
                    # A plain <iframe> element (rather than ui.html) so this works the same
                    # in NiceGUI 2.x-4.x. Unique name / data attribute per frequency (NiceGUI owns
                    # the element's own id attribute, so it is not overridden).
                    frame = ui.element("iframe").classes("lbc-map-frame").props(
                        f'name="lbc-map-{tag}" data-map-id="lbc-map-{tag}"')
                    frame.set_visibility(False)
                    frame.on("load", lambda _e, t=tag: _on_frame_load(t))
                    # Covers the cell while its map is out of date, so a stale map is never
                    # mistaken for the current one. Neutral grey: readable in light and dark mode.
                    overlay = ui.element("div").style(
                        "position:absolute; inset:0; z-index:5; display:flex; flex-direction:column;"
                        "align-items:center; justify-content:center; gap:10px;"
                        "background:rgba(128,128,128,0.45); backdrop-filter:blur(1px);")
                    with overlay:
                        ui.spinner(size="xl")
                        ui.label("Rebuilding map\u2026").classes("text-subtitle1 text-weight-medium")
                    overlay.set_visibility(False)
                # Persistent layer-status line: one small badge per layer.
                badges = ui.row().classes("items-center no-wrap q-gutter-xs").style(
                    "flex-wrap:wrap; gap:4px; min-height:22px; padding-top:2px")
                cell_status = ui.label("").classes("text-caption text-grey-7")
            cells[tag] = _MapCell(freq, frame, placeholder, cell_status, overlay, badges=badges)

    def _render_badges(cell: Optional[_MapCell], layers_status: Optional[Dict[str, Any]]) -> None:
        """Redraw one cell's status line from a build's ``layers_status`` (empty = clear it)."""
        if cell is None or cell.badges is None:
            return
        cell.badges.clear()
        with cell.badges:
            for severity, text, tip in badge_specs(layers_status):
                icon, color = _BADGE_STYLE.get(severity, _BADGE_STYLE["off"])
                ui.badge(f"{icon} {text}").props(f"outline color={color}").classes(
                    "text-caption").tooltip(tip)

    def _on_frame_load(tag: str) -> None:
        cell = cells.get(tag)
        if cell is not None and cell.awaiting_load:
            _finish_loading(tag, cell.current_file)

    map_grid()

    def refresh_layout() -> None:
        """Re-create the cells if the frequencies to show changed (no map is built). In
        NiceGUI >= 3 the refresh then runs in the background, a moment later."""
        if layout["freqs"] != list(effective_frequencies(state)):
            map_grid.refresh()

    async def _ensure_layout() -> None:
        """Same, but returns only once the cells exist - the maps are drawn into them."""
        if layout["freqs"] != list(effective_frequencies(state)):
            pending = map_grid.refresh()        # an awaitable in NiceGUI >= 3, None before
            if inspect.isawaitable(pending):
                await pending

    # ---- rebuild ---------------------------------------------------------
    async def _rebuild_once(auto: bool) -> None:
        await _ensure_layout()
        freqs = list(layout["freqs"] or [])
        # No await between this line and the snapshots: everything the maps are drawn from
        # is read at exactly this version. A later bump makes the poll rebuild again.
        versions["built"] = state.map_rebuild_version
        snapshots: List[Tuple[Optional[float], Optional[_MapRequest]]] = [
            (f, _snapshot(state, f, show_borders=bool(borders_cb.value), show_dem=bool(dem_cb.value),
                          dem_opacity=float(opacity_slider.value)))
            for f in freqs
        ]
        if all(req is None for _f, req in snapshots):
            status.set_text("")
            for _f, _r in snapshots:
                cell = cells.get(_freq_tag(_f))
                if cell is not None:
                    cell.placeholder.set_text("No earth station location yet - complete Step 1 / Step 5 first.")
                    cell.placeholder.set_visibility(True)
                    cell.frame.set_visibility(False)
                    cell.status.set_text("")
                    _render_badges(cell, None)
                    _set_overlay(cell, False)
            if not auto:
                ui.notify("No earth station location yet - run Step 5 first.", type="warning")
            return

        # Delete what earlier builds left behind (keeps the file the iframe is showing), and
        # cover every cell that is about to be redrawn. Nothing else deletes if "load" never came.
        for freq, req in snapshots:
            cell = cells.get(_freq_tag(freq))
            if cell is None or req is None:
                continue
            _cleanup_old_maps(req.freq_tag, keep=cell.current_file)
            _set_overlay(cell, True)

        warnings_seen: List[str] = []
        errors: List[str] = []
        any_contour = False
        try:
            for freq, req in snapshots:
                tag = _freq_tag(freq)
                cell = cells.get(tag)
                if cell is None or req is None:
                    continue
                result = await run.io_bound(_build_map_file, req)   # sequential: see module docstring
                cell = cells.get(tag)           # the cells may have been re-created while we waited
                if cell is None:
                    continue
                label = frequency_label(state, freq)

                if not result["ok"]:
                    cell.status.set_text(result["error"])
                    _render_badges(cell, None)      # clear stale badges, then say why
                    if cell.badges is not None:
                        with cell.badges:
                            ui.badge(f"{_BADGE_STYLE['error'][0]} Map (build failed)").props(
                                f"outline color={_BADGE_STYLE['error'][1]}").classes(
                                "text-caption").tooltip(str(result["error"]))
                    errors.append(f"{label}: {result['error']}")
                    _set_overlay(cell, False)   # the old map (if any) stays; the caption says why
                    continue

                filename = result["filename"]
                cell.current_file = filename
                cell.awaiting_load = True
                cell.frame.props(f'src="{STATIC_MAPS_URL}/{filename}"')
                cell.frame.update()
                cell.frame.set_visibility(True)
                cell.placeholder.set_visibility(False)
                # The overlay stays until the iframe reports "load"; this timer is the safety net.
                ui.timer(LOAD_TIMEOUT_S, lambda t=tag, f=filename: _finish_loading(t, f), once=True)

                # Caption: what is (and isn't) on this frequency's map.
                layers = result.get("layers_status") or {}
                _render_badges(cell, layers)
                notes: List[str] = []
                if req.contour:
                    any_contour = True
                else:
                    notes.append("No contour for this frequency - run Step 5.")
                if not req.has_border_rows:
                    notes.append("Run border analysis to see PASS/FAIL sectors.")
                elif _skip_summary((layers.get("sectors") or {}).get("skipped") or {}):
                    notes.append("Sectors not drawn: "
                                 + _skip_summary(layers["sectors"]["skipped"]) + ".")
                elif not req.zones:
                    notes.append("Border analysis has no azimuth with both a border crossing and a limit_db.")
                if dem_cb.value and not req.show_dem:
                    notes.append("DEM hillshade skipped (file not found).")
                elif dem_cb.value and not (layers.get("dem") or {}).get("ok", True):
                    notes.append(f"DEM hillshade skipped ({layers['dem'].get('reason', 'error')}).")
                cell.status.set_text(" ".join(notes))
                for w in result["warnings"]:
                    if w not in warnings_seen:      # the same warning would repeat for every map
                        warnings_seen.append(w)
        finally:
            # An exception / cancellation must not leave a cell covered for ever. Cells whose
            # new file is still loading keep their overlay (they hide it themselves).
            for cell in cells.values():
                if not cell.awaiting_load:
                    _set_overlay(cell, False)

        for e in errors:
            ui.notify(e, type="negative", multi_line=True)
        if not any_contour and not errors and not auto:
            ui.notify("Run Step 5 to see the contour.", type="info")
        status.set_text(" ".join(warnings_seen))
        if not auto:                                # an auto rebuild must not re-toast every time
            for w in warnings_seen:
                ui.notify(w, type="warning", multi_line=True)

    async def _run(auto: bool) -> None:
        nonlocal building, again, manual_pending
        if building:
            again = True        # redo once when the running build is done, with the newest state
            manual_pending = manual_pending or not auto
            return
        building = True
        manual_pending = not auto
        if not auto:
            rebuild_btn.props("loading")
        try:
            while True:
                again = False
                await _rebuild_once(auto=auto and not manual_pending)
                if not again:
                    break
        finally:
            building = False
            manual_pending = False
            rebuild_btn.props(remove="loading")

    async def rebuild() -> None:
        await _run(auto=False)

    rebuild.refresh_layout = refresh_layout          # type: ignore[attr-defined]
    rebuild_btn.on_click(rebuild)

    # ---- debounced auto-rebuild -----------------------------------------
    async def _auto_tick() -> None:
        """Rebuild when AppState.map_rebuild_version moved since the last build. A change must
        be seen on two consecutive ticks (i.e. it has stopped changing) before it is acted on,
        so a burst of edits - typing a DEM path, ticking several frequencies - costs one build.
        Builds started by other code (Rebuild Map, frequency selector, border analysis) record
        the version they used, so they do not trigger a second build here."""
        version = state.map_rebuild_version
        if version == versions["built"]:
            versions["pending"] = version
            return
        if version != versions["pending"]:
            versions["pending"] = version       # still changing: wait for a quiet tick
            return
        if building:
            return                               # picked up on a later tick if it is still stale
        await _run(auto=True)

    ui.timer(AUTO_REBUILD_POLL_S, _auto_tick, once=False)

    # Initial load: draw the maps once if there is already something to show
    # (e.g. after a page reload, since AppState outlives the page).
    if _earth_station_coords(state) is not None:
        ui.timer(0.2, rebuild, once=True)

    return rebuild


# Hooking up auto-refresh: build_middle_pane returns the rebuild coroutine function, so in
# desktop_app.py keep it, e.g.
#     rebuild_map = build_middle_pane(state)
# and call ``await rebuild_map()`` (or ``ui.timer(0, rebuild_map, once=True)``).