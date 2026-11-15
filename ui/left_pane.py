"""
Left Pane content (Phase 3/4B): ITU .mdb import, Step 1 (Gt) and Step 2 (Pt).

Steps 3 (Gr / Pr(p)) and 4 (Lb(p)) live in ``ui/step3_pane.py`` and Step 5 (coordination
distance) in ``ui/step5_pane.py``; ``build_step_form`` below dispatches to them.

Public entry points
-------------------
``build_itu_import(state, on_imported)``
    The collapsible "Import from ITU filing database (.mdb)" section. Render it once,
    above the step form. ``on_imported`` is called after a successful "Proceed" so the
    caller can refresh the step form (the imported values show up in Steps 1/2).

``build_step_form(state, step)``
    Renders the input form for one step. Call it from the refreshable that
    ``desktop_app.py`` already re-renders when the tracker changes step.

Everything the old Streamlit app kept in ``st.session_state`` / ``core.draft_state`` now
lives on ``AppState``:
    * committed results  -> ``state.wizard_results`` (gt, gt_source, gt_geometry, pt, ...)
    * Step 1 coordinates -> ``state.step1`` (es_lat, es_lon, sat_lon, gain_calc scratch)
    * ITU import data    -> ``state.itu_import``
    * remembered field values (the old ``ds.*`` drafts) -> ``state.get_draft / put_draft``,
      using the SAME draft keys as the Streamlit app ("es_lat", "gw_Gt_gmax", "s2_manual", ...)
      so later phases can read them the same way.

Calculation and import logic is NOT reimplemented here: it calls ``core.calculations`` and
``core.itu_import``.
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from nicegui import run, ui

from core import calculations, itu_import
from core.app_paths import get_resource_path
from ui.state import AppState

# pyproj is optional: only the "auto-calculate alpha / eps_h" button needs it.
try:
    from pyproj import Geod

    _GEOD = Geod(ellps="WGS84")
except ImportError:  # pragma: no cover - depends on the environment
    _GEOD = None


# ===========================================================================
# Small helpers
# ===========================================================================
def _f(value: Any, default: float = 0.0) -> float:
    """float(value), or ``default`` for None / blank / NaN (a cleared ui.number gives None)."""
    try:
        if value is None:
            return default
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if v == v else default


def _fmt(value: Optional[float], unit: str = "") -> str:
    return "—" if value is None else f"{value:.4f} {unit}".rstrip()


def _draft(state: AppState, key: str, default: Any) -> Any:
    """Remembered value for ``key``, or ``default`` when nothing is stored (or it is None)."""
    v = state.get_draft(key)
    return default if v is None else v


def _drafted_number(state: AppState, label: str, key: str, default: float = 0.0,
                    fmt: str = "%.4f", **kwargs: Any) -> ui.number:
    """A ui.number whose value is remembered in ``state.drafts[key]`` as the user edits it."""
    el = ui.number(label, value=_draft(state, key, default), format=fmt, **kwargs)
    el.props("dense outlined")
    el.on_value_change(lambda e, k=key: state.put_draft(k, e.value))
    return el


def _readonly_field(label: str, value: str = "—") -> ui.input:
    return ui.input(label, value=value).props("readonly outlined dense").classes("w-full")


def _section_title(text: str) -> None:
    ui.label(text).classes("text-subtitle2 text-weight-medium q-mt-sm")


def _status(positive: bool, text: str) -> None:
    ui.label(text).classes("text-positive" if positive else "text-warning")


async def _read_upload(e: Any) -> Tuple[str, bytes]:
    """(filename, bytes) from an upload event: NiceGUI >= 3 passes ``e.file``, older ``e.content``."""
    f = getattr(e, "file", None)
    if f is not None:
        return f.name, await f.read()
    return e.name, e.content.read()


# ===========================================================================
# Horizon helpers for the optional "auto-calculate alpha and eps_h" button
# (ported unchanged from app.py; needs pyproj and the precomputed horizon grid DB)
# ===========================================================================
def _find_horizon_db(db_override: Optional[str]) -> Tuple[Optional[str], List[str]]:
    search_paths = [db_override] if (db_override and db_override.strip()) else [
        get_resource_path("data_sources/horizon_grid.db"),
        "horizon_grid.db",
    ]
    return next((p for p in search_paths if p and os.path.exists(p)), None), search_paths


def _interp_eps_h(profile: Dict[int, Dict[str, float]], azimuth_increment: float, target_az: float):
    """Linearly interpolate the horizon elevation at ``target_az`` from a profile sampled every
    ``azimuth_increment`` degrees. Returns (eps_h, az_lo, az_hi, eps_lo, eps_hi)."""
    target_az = target_az % 360.0
    az_lo = int(azimuth_increment * math.floor(target_az / azimuth_increment)) % 360
    az_hi = (az_lo + azimuth_increment) % 360
    if az_lo not in profile or az_hi not in profile:
        raise ValueError(
            f"Profile does not contain the azimuth bins needed to interpolate "
            f"{target_az:.2f}° (looked for {az_lo}° and {az_hi}°)."
        )
    eps_lo = profile[az_lo]["elevation_angle"]
    eps_hi = profile[az_hi]["elevation_angle"]
    frac = ((target_az - az_lo) % 360) / azimuth_increment if azimuth_increment else 0.0
    return eps_lo + frac * (eps_hi - eps_lo), az_lo, az_hi, eps_lo, eps_hi


def _compute_azimuth_and_eps_h(own_lat: float, own_lon: float, other_lat: float,
                               other_lon: float, db_override: Optional[str]) -> Dict[str, Any]:
    """Great-circle azimuth own -> other (alpha) and the interpolated horizon elevation (eps_h)
    at that azimuth from the horizon grid DB. Raises with a readable message on any problem."""
    if _GEOD is None:
        raise ImportError(
            "pyproj is not installed, so the azimuth between the two stations cannot be "
            "computed automatically. Install pyproj, or enter α and εh manually instead."
        )
    fwd_az, _back, _dist = _GEOD.inv(own_lon, own_lat, other_lon, other_lat)
    alpha_deg = fwd_az % 360.0

    db_path, searched = _find_horizon_db(db_override)
    if db_path is None:
        raise RuntimeError(
            "horizon_grid.db not found (checked: " + ", ".join(searched) + "). "
            f"Azimuth α was computed ({alpha_deg:.4f}°) but εh could not be looked up "
            "automatically — build/locate the grid DB, or enter εh manually."
        )

    from core.query_horizon import query_location  # lazy: only needed here

    record = query_location(db_path, own_lat, own_lon)
    eps_h_deg, az_lo, az_hi, eps_lo, eps_hi = _interp_eps_h(
        record["profile"], record["azimuth_increment"], alpha_deg)
    return {
        "alpha_deg": alpha_deg, "eps_h_deg": eps_h_deg,
        "grid_lat": record["lat"], "grid_lon": record["lon"], "offset_km": record["offset_km"],
        "azimuth_increment": record["azimuth_increment"],
        "az_lo": az_lo, "az_hi": az_hi, "eps_lo": eps_lo, "eps_hi": eps_hi,
    }


# ===========================================================================
# D/lambda defaults (shared by the ITU import and the Gt calculator)
# ===========================================================================
def _autofill_antenna_defaults(state: AppState, ns: str, freq_hint: Optional[float] = None):
    """If D/lambda (and D with lambda) are missing for calculator ``ns`` ("gw_Gt" / "gw_Gr"),
    derive them with ``calculations.resolve_antenna_defaults`` (ITU-R S.465/S.580, eta = 0.55
    from Gmax; 3.7 m fallback) and save them as drafts, so Eq. 97 never runs with D/lambda = 0.
    Returns the resolver result if it filled something in, else None."""
    gmax = _f(state.get_draft(f"{ns}_gmax"))
    diam = _f(state.get_draft(f"{ns}_D"))
    lam = _f(state.get_draft(f"{ns}_lambda"))
    dl = _f(state.get_draft(f"{ns}_dl"))
    if dl > 0 or (diam > 0 and lam > 0):
        return None
    res = calculations.resolve_antenna_defaults(gmax, diam, lam, None, freq_hint)
    state.put_draft(f"{ns}_dl", res["d_over_lambda"])
    if diam <= 0 and res["diameter_m"]:
        state.put_draft(f"{ns}_D", res["diameter_m"])
    if lam <= 0 and res["wavelength_m"]:
        state.put_draft(f"{ns}_lambda", res["wavelength_m"])
    if gmax <= 0 and res["gmax_estimated"]:
        state.put_draft(f"{ns}_gmax", res["gmax_dbi"])
    state.step1.gain_set(ns, "dl_source", res)
    return res


# ===========================================================================
# ITU import: applying a station to the AppState (no UI in here, so it is testable)
# ===========================================================================
Note = Tuple[str, str]  # (notify type, message)


def import_station(state: AppState, station: Dict[str, Any],
                   tables: Dict[str, Any]) -> Tuple[bool, str, List[Note]]:
    """Populate the AppState from one ``extract_earth_stations()`` entry.

    Returns ``(ok, message, notes)``. ``ok`` False means nothing was changed and ``message`` is
    the reason (shown as an error). Otherwise ``message`` summarises what was imported and
    ``notes`` are extra (type, text) hints to show as notifications.
    """
    if station["lat"] is None or station["lon"] is None:
        return False, "Could not convert this station's coordinates — see the warning above.", []

    # Joins by satellite NAME (not ntc_id) - see itu_import.classify_satellite().
    sat_class = itu_import.classify_satellite(
        station.get("sat_name"), tables.get("geo"), tables.get("non_geo"))
    if sat_class == "non_geo":
        return False, "This tool only supports Geostationary (GSO) satellites.", []

    bits: List[str] = []
    notes: List[Note] = []
    wr = state.wizard_results

    # A different earth station invalidates any Gt derived earlier, the calculators'
    # intermediate results, and everything computed from Lb(p). The import never writes a
    # final Gt/Gr (e_ant.gain is Gmax, not G(phi)), so the user derives Gt for this station.
    wr.gt = None
    wr.gt_source = None
    wr.gt_geometry = None
    state.step1.reset_all_gain_results()
    state.reset_derived()

    # --- coordinates -> Step 1 context + shared drafts (Gt calculator and Step 5) ---
    state.step1.es_lat = station["lat"]
    state.step1.es_lon = station["lon"]
    state.put_draft("es_lat", station["lat"])
    state.put_draft("es_lon", station["lon"])
    bits.append(
        f"coordinates {station['lat']:.4f}°{'N' if station['lat'] >= 0 else 'S'}, "
        f"{station['lon']:.4f}°{'E' if station['lon'] >= 0 else 'W'} (→ Gt calculator and Step 5)")

    # --- e_stn.long_nom -> satellite longitude (no join needed) ---
    if station.get("sat_long_nom") is not None:
        sat_lon = station["sat_long_nom"]
        state.put_draft("sat_lon", sat_lon)          # Gt calculator (shared draft)
        state.put_draft("gw_Gr_sat_lon", sat_lon)    # Gr calculator
        bits.append(f"satellite orbital longitude {sat_lon:.4f}° (→ Gt/Gr calculators)")

    # --- inclination: FYI only, never auto-filled (Plans/AP30B field, not Eq. 91-96's is) ---
    orbit = itu_import.extract_satellite_orbit_data(station.get("sat_name"), tables.get("geo"))
    if orbit and orbit.get("inclin_exc_deg") is not None:
        notes.append(("info",
                      f"FYI: the 'geo' table lists an inclination excursion of "
                      f"{orbit['inclin_exc_deg']:.4f}° for this satellite (Plans/AP30B field — "
                      "not necessarily Eq. 91-96's max orbital inclination; not auto-filled)."))

    # --- e_ant -> Gmax + diameter. Drop the previous station's antenna drafts first. ---
    for ns in ("gw_Gt", "gw_Gr"):
        for sfx in ("gmax", "D", "dl", "lambda"):
            state.put_draft(f"{ns}_{sfx}", 0.0)
    state.put_draft("vd_freq_hint2", 0.0)
    state.itu_import.victim_freq_ghz = None
    state.itu_import.freqs = None
    state.itu_import.all_frequencies = []

    e_ant = tables.get("e_ant")
    ant_tx = itu_import.extract_antenna_data(station["ntc_id"], e_ant, emi_rcp="E")
    ant_rx = itu_import.extract_antenna_data(station["ntc_id"], e_ant, emi_rcp="R")

    # e_ant.gain is the main-beam axis gain Gmax: it only seeds the calculators' Gmax field.
    if ant_tx and ant_tx["gain_dbi"] is not None:
        state.put_draft("gw_Gt_gmax", ant_tx["gain_dbi"])
        bits.append(f"main-beam gain Gmax = {ant_tx['gain_dbi']:.4f} dBi, transmit "
                    "(→ Gt calculator; Gt itself is calculated from it)")
    else:
        notes.append(("info", "No usable transmit-direction gain in 'e_ant' for this station — "
                              "Gmax was not auto-filled in the Gt calculator."))
    if ant_rx and ant_rx["gain_dbi"] is not None:
        state.put_draft("gw_Gr_gmax", ant_rx["gain_dbi"])
        bits.append(f"main-beam gain Gmax = {ant_rx['gain_dbi']:.4f} dBi, receive "
                    "(→ Gr calculator; Gr itself is calculated from it)")
    else:
        notes.append(("info", "No usable receive-direction gain in 'e_ant' for this station — "
                              "Gmax was not auto-filled in the Gr calculator."))

    def _pos_diam(a: Optional[Dict[str, Any]]) -> Optional[float]:
        d = (a or {}).get("ant_diam_m")
        return d if (d is not None and d > 0) else None

    d_tx = _pos_diam(ant_tx) if _pos_diam(ant_tx) is not None else _pos_diam(ant_rx)
    d_rx = _pos_diam(ant_rx) if _pos_diam(ant_rx) is not None else _pos_diam(ant_tx)
    if d_tx is not None:
        state.put_draft("gw_Gt_D", d_tx)
    if d_rx is not None:
        state.put_draft("gw_Gr_D", d_rx)
    if d_tx is not None or d_rx is not None:
        if d_tx is not None and d_rx is not None and abs(d_tx - d_rx) > 1e-9:
            bits.append(f"antenna diameter D = {d_tx:.4f} m (Gt) / {d_rx:.4f} m (Gr)")
        else:
            bits.append(f"antenna diameter {(d_tx if d_tx is not None else d_rx):.4f} m")

    # --- grp -> Tx frequency + max power (Step 2), Rx frequency (Step 3 victim) ---
    all_freqs = itu_import.extract_all_frequencies(station["ntc_id"], tables.get("grp"), e_ant)
    by_dir = itu_import.classify_frequencies_by_direction(all_freqs)
    state.itu_import.freqs = {"label": station["label"], **by_dir}
    # Keep EVERY frequency group (Tx, Rx and unassigned) for multi-frequency analysis.
    state.itu_import.all_frequencies = [dict(f) for f in all_freqs]
    # New station -> new frequency list: select every frequency (the top-bar selector
    # lets the user narrow it down).
    state.selected_frequencies = []
    state.sanitize_selected_frequencies(default_all=True)
    if all_freqs:
        bits.append(f"{len(all_freqs)} frequency group(s) stored for multi-frequency analysis "
                    f"({len(by_dir['transmit'])} Tx, {len(by_dir['receive'])} Rx, "
                    f"{len(by_dir['unknown'])} unassigned)")
    tx = by_dir["transmit"][0] if by_dir["transmit"] else None
    rx = by_dir["receive"][0] if by_dir["receive"] else None
    dir_txt = {"emi_rcp": "emi_rcp", "e_ant": "direction from e_ant",
               "heuristic": "direction suggested from frequency - please verify"}

    if tx is not None:
        grp_info = {"freq_ghz": tx["freq_center_ghz"], "pwr_max_dbw": tx["pwr_max_dbw"],
                    "match_count": len(by_dir["transmit"]),
                    "dir_note": f" (Tx/E, {dir_txt.get(tx['direction_source'], '')})"}
    else:  # no transmit group identifiable - legacy behaviour (first grp row)
        grp_info = itu_import.extract_group_data(station["ntc_id"], tables.get("grp"))

    if grp_info:
        state.put_draft("freq_ghz", grp_info["freq_ghz"])  # Step 5's Frequency field
        freq_bit = f"frequency {grp_info['freq_ghz']:.4f} GHz" + grp_info.get("dir_note", "")
        if grp_info["match_count"] > 1:
            freq_bit += f" [{grp_info['match_count']} matching 'grp' rows — used the first]"
        bits.append(freq_bit)
        if grp_info["pwr_max_dbw"] is not None:
            pwr = grp_info["pwr_max_dbw"]
            wr.pt = pwr
            wr.pt_source = "(imported from ITU SNS database, grp.pwr_max)"
            state.put_draft("s2_manual", pwr)
            bits.append(f"max transmit power {pwr:.4f} dBW (→ Pt)")
        else:
            notes.append(("info", "No usable 'pwr_max' in 'grp' for this station — Pt was not auto-filled."))
    else:
        notes.append(("info", "No matching frequency/power data in 'grp' for this station."))

    if rx is not None:
        state.itu_import.victim_freq_ghz = rx["freq_center_ghz"]
        state.put_draft("vd_freq_hint2", rx["freq_center_ghz"])
        bits.append(f"receive frequency {rx['freq_center_ghz']:.4f} GHz "
                    f"(Rx/R, {dir_txt.get(rx['direction_source'], '')}) (→ Step 3 victim)")
    else:
        notes.append(("info", "No receive (emi_rcp = R) frequency identified — "
                              "enter the victim frequency in Step 3."))
    if by_dir["unknown"]:
        notes.append(("info", f"{len(by_dir['unknown'])} frequency group(s) have no emi_rcp and fall "
                              "outside the Tx/Rx heuristic bands - left unassigned."))

    # --- D/lambda defaults (S.465/S.580, eta = 0.55; 3.7 m fallback) ---
    f_tx = grp_info["freq_ghz"] if grp_info else None
    f_rx = rx["freq_center_ghz"] if rx is not None else None
    for ns, lbl, fh in (("gw_Gt", "Gt", f_tx), ("gw_Gr", "Gr", f_rx)):
        res = _autofill_antenna_defaults(state, ns, fh)
        if res:
            bits.append(f"{lbl} calculator D/λ = {res['d_over_lambda']:.2f} ({res['note']})")

    if sat_class == "unknown":
        notes.append(("warning",
                      "This station's satellite (by name, from e_stn.sat_name) was not found in the "
                      "'geo' or 'non_geo' table — GSO status could not be verified. The import went "
                      "ahead, but please confirm manually."))

    return True, f"Imported {station['label']}: " + "; ".join(bits) + ".", notes


# ===========================================================================
# ITU import section
# ===========================================================================
def build_itu_import(state: AppState, on_imported: Optional[Callable[[], None]] = None) -> None:
    """Render the ITU import section in the current NiceGUI context."""
    itu = state.itu_import

    @ui.refreshable
    def station_details() -> None:
        stations = itu.stations or []
        if not stations:
            return
        idx = min(max(int(_draft(state, "itu_station_idx", 0)), 0), len(stations) - 1)
        st = stations[idx]
        if st["lat_error"] or st["lon_error"]:
            ui.label("Coordinate conversion issue for this station — "
                     f"lat: {st['lat_error'] or 'OK'}; lon: {st['lon_error'] or 'OK'}.") \
                .classes("text-warning")
        if st.get("sat_name"):
            ui.label(f"Associated satellite (e_stn.sat_name): {st['sat_name']}") \
                .classes("text-caption text-grey-7")
        else:
            ui.label("No associated satellite name on this station's e_stn row — the GSO/non-GSO "
                     "check can't confirm status for it (shown as “unknown”, not blocked).") \
                .classes("text-caption text-grey-7")

    @ui.refreshable
    def station_picker() -> None:
        if itu.error:
            ui.label(itu.error).classes("text-negative").style("white-space: pre-wrap")
            return
        if itu.stations is None:
            return  # nothing uploaded yet
        if not itu.stations:
            ui.label("No earth stations could be extracted from this file.").classes("text-grey-7")
            return

        tables = itu.mdb_tables or {}
        found = [t for t in ("geo", "non_geo", "grp", "e_ant") if tables.get(t) is not None]
        missing = [t for t in ("geo", "non_geo", "grp", "e_ant") if tables.get(t) is None]
        if found:
            ui.label(f"Optional tables found: {', '.join(found)}.").classes("text-caption text-grey-7")
        if missing:
            ui.label("Not present in this file (fine — related fields just won't auto-fill): "
                     f"{', '.join(missing)}.").classes("text-caption text-grey-7")
        if itu.last_import_msg:
            ui.label(itu.last_import_msg).classes("text-positive")

        n = len(itu.stations)
        idx = min(max(int(_draft(state, "itu_station_idx", 0)), 0), n - 1)

        def _on_select(e) -> None:
            if e.value is None:
                return
            state.put_draft("itu_station_idx", int(e.value))
            station_details.refresh()

        ui.select({i: s["label"] for i, s in enumerate(itu.stations)}, value=idx,
                  label="Select an earth station", with_input=True,
                  on_change=_on_select).props("dense outlined").classes("w-full")
        station_details()
        ui.button("Proceed", icon="arrow_forward", on_click=_proceed) \
            .props("unelevated no-caps color=primary")

    def _proceed() -> None:
        stations = itu.stations or []
        if not stations:
            return
        idx = min(max(int(_draft(state, "itu_station_idx", 0)), 0), len(stations) - 1)
        ok, message, notes = import_station(state, stations[idx], itu.mdb_tables or {})
        if not ok:
            ui.notify(message, type="negative", multi_line=True)
            return
        itu.last_import_msg = message
        ui.notify(f"Imported {stations[idx]['label']}.", type="positive")
        for kind, text in notes:
            ui.notify(text, type=kind, multi_line=True, timeout=8000)
        station_picker.refresh()
        if on_imported is not None:
            on_imported()

    async def _on_upload(e) -> None:
        name, data = await _read_upload(e)
        upload.reset()  # lets the same file be chosen again later
        key = (name, len(data))
        if itu.file_key == key and itu.stations:
            ui.notify(f"{name} is already loaded.", type="info")
            return

        itu.file_key = key
        itu.mdb_tables = itu.stations = itu.error = itu.last_import_msg = None
        busy = ui.notification(f"Reading {name}…", spinner=True, timeout=None)
        try:
            # load_mdb can take a while on a large filing database: keep the UI responsive.
            tables = await run.io_bound(itu_import.load_mdb, data, name)
            stations = itu_import.extract_earth_stations(tables["e_stn"])
            itu.mdb_tables, itu.stations = tables, stations
            state.put_draft("itu_station_idx", 0)
        except itu_import.ItuImportError as ex:
            itu.error = str(ex)
        except Exception as ex:  # noqa: BLE001 - surface anything else as a message, not a crash
            itu.error = f"Unexpected error reading file: {ex}"
        finally:
            busy.dismiss()
        if itu.error:
            ui.notify("Could not read the database file.", type="negative")
        else:
            ui.notify(f"Found {len(itu.stations or [])} earth station(s) in {name}.", type="positive")
        station_picker.refresh()

    with ui.expansion("Import from ITU filing database (.mdb)", icon="upload_file",
                      value=bool(itu.stations or itu.error)).classes("w-full"):
        backends = itu_import.list_available_backends()
        if backends:
            ui.label(f"Ready to read .mdb files via: {', '.join(backends)}.") \
                .classes("text-caption text-grey-7")
        else:
            ui.label("No .mdb-reading backend is installed. Uploading will fail with an explanation; "
                     "to make it work install one of: `pip install access-parser` (recommended, "
                     "cross-platform), mdbtools (Linux/macOS), or pyodbc + the Microsoft Access "
                     "Database Engine (Windows).").classes("text-warning")
        ui.label("Reads the e_stn, geo, non_geo, grp and e_ant tables of an exported ITU SNS database. "
                 "GSO earth stations only; a screening convenience, not an authoritative ITU filing "
                 "parser.").classes("text-caption text-grey-7")

        # The uploader lives OUTSIDE the refreshable picker, so refreshing the picker from its
        # upload handler never deletes the element that is running the handler.
        upload = ui.upload(
            label="ITU SNS database file (.mdb)", auto_upload=True, max_files=1,
            on_upload=_on_upload,
            on_rejected=lambda _e: ui.notify("Only .mdb files are accepted.", type="negative"),
        ).props('accept=".mdb" flat bordered').classes("w-full")
        station_picker()


# ===========================================================================
# Step 1 — Gt
# ===========================================================================
GAIN_MODES = [
    "Direct geometry (Eq. 84-89)",
    "Case 1 – no orbital inclination (Eq. 90)",
    "Case 2 – slightly inclined orbit (Eq. 91-96)",
]
MODE_DIRECT, MODE_CASE1, MODE_CASE2 = GAIN_MODES


def _compute_phi(mode: str, zeta: float, i_val: float, delta: float, K: float,
                 alpha: float, eps_h: float, is_max: float, step: float) -> Tuple[float, str]:
    """phi (deg) and a note naming the equations used, per the selected inclination method."""
    if mode == MODE_DIRECT:
        out, note = calculations.compute_all(zeta, i_val, delta, K, alpha, eps_h), "Eq. 84-89"
    elif mode == MODE_CASE1:
        out = calculations.phi_case1(alpha, eps_h, zeta, delta, K)
        note = out.get("case", "Eq. 90")
    else:
        out = calculations.phi_case2(alpha, eps_h, zeta, delta, is_max, K, step)
        note = out.get("case", "Eq. 91-96")
    if "phi" not in out:
        raise ValueError("No φ returned — check inputs.")
    return out["phi"], note


# Per-namespace settings of the shared Eq. 84-97 gain calculator. Gt keeps the original shared
# drafts ("es_lat" / "es_lon" / "sat_lon"), which Step 5 also reads. The Gr calculator is for
# the VICTIM station, so it has its own coordinate drafts and must never overwrite the
# interferer's coordinates in ``state.step1`` (those feed Step 5).
_GAIN_CALC_CFG: Dict[str, Dict[str, Any]] = {
    "gw_Gt": {
        "sym": "Gt",
        "title": "Determine Gt — interfering earth station, toward the victim's horizon",
        "coord_note": "Latitude, longitude and satellite longitude are saved as defaults and "
                      "reused in Step 5.",
        "lat_key": "es_lat", "lon_key": "es_lon", "sat_key": "sat_lon",
        "commit_ctx": True,
    },
    "gw_Gr": {
        "sym": "Gr",
        "title": "Determine Gr — victim (receiving) earth station, toward the horizon",
        "coord_note": "Enter the VICTIM station's latitude, longitude and satellite longitude "
                      "(kept separate from the interfering station's, which Step 5 uses).",
        "lat_key": "gw_Gr_es_lat", "lon_key": "gw_Gr_es_lon", "sat_key": "gw_Gr_sat_lon",
        "commit_ctx": False,
    },
}


def _gain_calc_freq_hint(state: AppState, ns: str) -> Optional[float]:
    """Operating frequency (GHz) used to estimate D/lambda: Step 5's frequency for Gt, the
    victim (receive) frequency for Gr."""
    if ns == "gw_Gr":
        return (_f(state.itu_import.victim_freq_ghz) or _f(state.get_draft("vd_freq_hint2"))
                or _f(state.victim_lookup.freq) or _f(state.step3.selected_freq)) or None
    return _f(state.get_draft("freq_ghz")) or None


def open_gain_calculator(state: AppState, ns: str,
                         on_accept: Callable[[float, Dict[str, Any]], None]) -> None:
    """Open the gain calculator dialog (Eq. 84-97) for namespace ``ns`` ("gw_Gt" or "gw_Gr").
    ``on_accept(gain, geometry)`` runs after the dialog has closed, when the user clicks
    "Use this value as Gt/Gr"."""
    cfg = _GAIN_CALC_CFG[ns]
    sym = cfg["sym"]
    s1 = state.step1
    freq_hint = _gain_calc_freq_hint(state, ns)

    # Fill missing D/lambda BEFORE the fields are drawn so they show the defaults.
    _autofill_antenna_defaults(state, ns, freq_hint)

    with ui.dialog() as dialog, ui.card().style(
            "width: 780px; max-width: 95vw; max-height: 92vh; overflow-y: auto;"):
        ui.label(cfg["title"]).classes("text-h6")
        ui.label("AP7 Annex 3, Eq. 84-97").classes("text-caption text-grey-7")

        # ---- §1-2 station & satellite geometry ------------------------------------------
        _section_title("§1-2 — Station & satellite geometry")
        ui.label("The longitude difference δ is calculated for you. " + cfg["coord_note"]) \
            .classes("text-caption text-grey-7")
        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            zeta_in = _drafted_number(state, "Earth station latitude ζ (deg, +N/−S)", cfg["lat_key"]).classes("col")
            lon_in = _drafted_number(state, "Earth station longitude (deg, +E/−W)", cfg["lon_key"]).classes("col")
            sat_in = _drafted_number(state, "Satellite orbital longitude (deg, +E/−W)", cfg["sat_key"]) \
                .classes("col").tooltip("e.g. 46.0 for Azerspace-1. Satellite latitude is taken as 0°.")
        delta_lbl = ui.label().classes("text-primary")

        def _update_delta(_e=None) -> None:
            delta_lbl.set_text(
                f"Calculated longitude difference (δ) = {_f(sat_in.value) - _f(lon_in.value):.4f}°")

        lon_in.on_value_change(_update_delta)
        sat_in.on_value_change(_update_delta)
        _update_delta()

        k_in = _drafted_number(state, "Orbit radius / Earth radius K", f"{ns}_K", calculations.K_DEFAULT)

        mode_radio = ui.radio(GAIN_MODES, value=_draft(state, f"{ns}_mode", MODE_DIRECT),
                              on_change=lambda e: state.put_draft(f"{ns}_mode", e.value))
        mode_radio.props("dense")
        with ui.row().classes("w-full") as row_direct:
            i_in = _drafted_number(state, "Sub-satellite latitude i (deg, +N/−S)", f"{ns}_i").classes("col")
        row_direct.bind_visibility_from(mode_radio, "value", backward=lambda m: m == MODE_DIRECT)
        with ui.row().classes("w-full no-wrap q-gutter-sm") as row_case2:
            ismax_in = _drafted_number(state, "Max orbital inclination iₛ (deg)", f"{ns}_ismax").classes("col")
            step_in = _drafted_number(state, "Grid search step (deg)", f"{ns}_step", 1.0).classes("col")
        row_case2.bind_visibility_from(mode_radio, "value", backward=lambda m: m == MODE_CASE2)

        # ---- alpha / eps_h --------------------------------------------------------------
        _section_title("α (azimuth) and εh (horizon elevation)")
        ui.label("Auto-calculated from the coordinates above (needs pyproj and the horizon grid DB), "
                 "or enter them manually.").classes("text-caption text-grey-7")
        db_in = ui.input("Grid DB path override (optional — blank = auto-search)",
                         value=_draft(state, f"{ns}_db_override", "")).props("dense outlined").classes("w-full")
        db_in.on_value_change(lambda e: state.put_draft(f"{ns}_db_override", e.value))

        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            alpha_in = _drafted_number(state, "Azimuth α (deg)", f"{ns}_alpha").classes("col")
            eps_in = _drafted_number(state, "Horizon elevation εh (deg)", f"{ns}_epsh").classes("col")
        diag_lbl = ui.label().classes("text-caption text-grey-7")

        def _show_diag() -> None:
            d = s1.gain_get(ns, "auto_diag")
            diag_lbl.set_text(
                "" if not d else
                f"Nearest grid point: {d['grid_lat']:.5f}, {d['grid_lon']:.5f}. εh interpolated from "
                f"εh({d['az_lo']}°)={d['eps_lo']:.4f}° and εh({d['az_hi']}°)={d['eps_hi']:.4f}°.")

        def auto_az() -> None:
            try:
                r = _compute_azimuth_and_eps_h(_f(zeta_in.value), _f(lon_in.value), 0.0,
                                               _f(sat_in.value), db_in.value)
            except Exception as ex:  # noqa: BLE001
                s1.gain_set(ns, "auto_diag", None)
                _show_diag()
                ui.notify(f"Error: {ex}", type="negative", multi_line=True)
                return
            alpha_in.value, eps_in.value = r["alpha_deg"], r["eps_h_deg"]  # also saved as drafts
            s1.gain_set(ns, "auto_diag", r)
            _show_diag()
            ui.notify(f"α = {r['alpha_deg']:.4f}°, εh = {r['eps_h_deg']:.4f}° "
                      f"(nearest grid point {r['offset_km']:.2f} km away).", type="positive")

        ui.button("Auto-calculate α and εh from coordinates", icon="my_location", on_click=auto_az) \
            .props("unelevated no-caps color=primary")
        _show_diag()

        phi_field = _readonly_field("φ =", _fmt(s1.gain_get(ns, "phi"), "°"))

        def calc_phi() -> None:
            try:
                phi_val, note = _compute_phi(
                    mode_radio.value, _f(zeta_in.value), _f(i_in.value),
                    _f(sat_in.value) - _f(lon_in.value), _f(k_in.value, calculations.K_DEFAULT),
                    _f(alpha_in.value), _f(eps_in.value), _f(ismax_in.value), _f(step_in.value, 1.0))
            except Exception as ex:  # noqa: BLE001
                s1.gain_set(ns, "phi", None)
                phi_field.set_value("—")
                ui.notify(f"Error: {ex}", type="negative", multi_line=True)
                return
            s1.gain_set(ns, "phi", phi_val)
            phi_field.set_value(_fmt(phi_val, "°"))
            ui.notify(f"φ = {phi_val:.4f}° calculated ({note}). Now compute G(φ) below.", type="positive")

        ui.button("Calculate φ", on_click=calc_phi).props("outline no-caps")

        ui.separator()

        # ---- §3 antenna gain G(phi) -----------------------------------------------------
        _section_title("§3 — Antenna gain towards the horizon G(φ) (Eq. 97)")
        ui.label("Enter D/λ directly, or λ and D, or click “Estimate D/λ from Gmax” (needs D/λ ≥ 35).") \
            .classes("text-caption text-grey-7")
        src = s1.gain_get(ns, "dl_source")
        if src:
            ui.label(f"D/λ = {src['d_over_lambda']:.2f} filled in automatically: {src['note']}. "
                     "The fields below are editable.").classes("text-info")
            for w in src.get("warnings", []):
                ui.label(w).classes("text-warning")

        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            # e_ant.gain is Gmax (main-beam axis). It is only the INPUT to Eq. 97; the value used
            # as Gt in the link budget is the horizon gain G(phi) computed below.
            gmax_in = _drafted_number(state, "Main-beam gain Gmax (dBi)", f"{ns}_gmax").classes("col") \
                .tooltip("Pre-filled from the ITU e_ant gain after an import. This is Gmax, not Gt: "
                         "Gt is the horizon gain G(φ) calculated from it.")
            lam_in = _drafted_number(state, "Wavelength λ (m)", f"{ns}_lambda", fmt="%.6f").classes("col")
            d_in = _drafted_number(state, "Antenna diameter D (m)", f"{ns}_D").classes("col")
            dl_in = _drafted_number(state, "D/λ (overrides λ/D)", f"{ns}_dl").classes("col")

        dl_est_lbl = ui.label().classes("text-caption text-grey-7")
        est0 = s1.gain_get(ns, "dl_estimated")
        if est0 is not None:
            dl_est_lbl.set_text(f"Last estimated D/λ = {est0:.4f}")

        def estimate_dl() -> None:
            try:
                dl = calculations.estimate_d_over_lambda(_f(gmax_in.value), freq_hint)
            except Exception as ex:  # noqa: BLE001
                ui.notify(f"Error: {ex}", type="negative", multi_line=True)
                return
            s1.gain_set(ns, "dl_estimated", dl)
            dl_in.value = dl  # drop it straight into the D/λ field (also saved as a draft)
            dl_est_lbl.set_text(f"Last estimated D/λ = {dl:.4f}")
            if _f(gmax_in.value) > 0:
                ui.notify(f"D/λ estimated from Gmax (S.465/S.580, η = {calculations.ETA_DEFAULT:.2f}): "
                          f"{dl:.4f}", type="positive")
            else:
                ui.notify(f"Gmax not known — D/λ = {dl:.4f} from the standard "
                          f"{calculations.D_FALLBACK_M:.1f} m diameter fallback.", type="info")

        gain_field = _readonly_field("G(φ) =", _fmt(s1.gain_get(ns, "gain"), "dBi"))
        branch_lbl = ui.label(s1.gain_get(ns, "gain_branch") or "").classes("text-caption text-grey-7")

        def calc_gain() -> None:
            phi_val = s1.gain_get(ns, "phi")
            if phi_val is None:
                ui.notify("Calculate φ above first.", type="negative")
                return
            try:
                gmax, lam, diam, dl_man = (_f(gmax_in.value), _f(lam_in.value),
                                           _f(d_in.value), _f(dl_in.value))
                est = s1.gain_get(ns, "dl_estimated")
                if dl_man > 0:
                    dl = dl_man
                elif lam > 0 and diam > 0:
                    dl = diam / lam
                elif est is not None:
                    dl = est
                else:  # nothing entered: ITU defaults instead of an error
                    r = calculations.resolve_antenna_defaults(gmax, diam, lam, None, freq_hint)
                    dl = r["d_over_lambda"]
                    s1.gain_set(ns, "dl_source", r)
                gmax_eff = gmax if gmax > 0 else calculations.gmax_from_d_over_lambda(dl)
                res = calculations.gain_at_phi(phi_val, dl, gmax_eff)
            except Exception as ex:  # noqa: BLE001
                s1.gain_set(ns, "gain", None)
                s1.gain_set(ns, "dl_used", None)
                gain_field.set_value("—")
                branch_lbl.set_text("")
                ui.notify(f"Error: {ex}", type="negative", multi_line=True)
                return
            s1.gain_set(ns, "gmax_used", gmax_eff)  # so "Use this value" saves the Gmax actually used
            s1.gain_set(ns, "gain", res["gain"])
            s1.gain_set(ns, "gain_branch", res["branch"])
            s1.gain_set(ns, "dl_used", dl)
            gain_field.set_value(_fmt(res["gain"], "dBi"))
            branch_lbl.set_text(f"Branch: {res['branch']}")
            ui.notify(f"G(φ) = {res['gain']:.4f} dBi calculated.", type="positive")

        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            ui.button("Estimate D/λ from Gmax", on_click=estimate_dl).props("outline no-caps")
            ui.button("Calculate G(φ)", on_click=calc_gain).props("outline no-caps")

        ui.separator()

        def use_value() -> None:
            gain = s1.gain_get(ns, "gain")
            if gain is None:
                ui.notify("Calculate G(φ) first.", type="negative")
                return
            lam, diam = _f(lam_in.value), _f(d_in.value)
            geometry = {
                "mode": mode_radio.value,
                "zeta": _f(zeta_in.value),
                "delta0": _f(sat_in.value) - _f(lon_in.value),
                "K": _f(k_in.value, calculations.K_DEFAULT),
                "i": _f(i_in.value) if mode_radio.value == MODE_DIRECT else None,
                "is_max": _f(ismax_in.value) if mode_radio.value == MODE_CASE2 else None,
                "step": _f(step_in.value, 1.0) if mode_radio.value == MODE_CASE2 else None,
                "gmax": s1.gain_get(ns, "gmax_used") or _f(gmax_in.value),
                "d_over_lambda": s1.gain_get(ns, "dl_used"),
                # Only known when the user gave both directly; Step 8's diameter slider uses them.
                "wavelength": lam if lam > 0 else None,
                "diameter": diam if diam > 0 else None,
            }
            # Gt only: commit the coordinates the downstream steps read (the editable defaults
            # already live in the shared drafts). The Gr calculator describes the victim
            # station, so it must not overwrite the interferer's coordinates.
            if cfg["commit_ctx"]:
                s1.es_lat, s1.es_lon, s1.sat_lon = _f(zeta_in.value), _f(lon_in.value), _f(sat_in.value)
            dialog.close()
            on_accept(gain, geometry)  # after close: it refreshes the form that owns this dialog

        with ui.row().classes("w-full justify-end q-gutter-sm"):
            ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
            ui.button(f"Use this value as {sym}", on_click=use_value).props("unelevated no-caps color=primary")

    dialog.open()


def open_gt_calculator(state: AppState,
                       on_accept: Callable[[float, Dict[str, Any]], None]) -> None:
    """Open the Gt calculator dialog (Eq. 84-97). ``on_accept(gain, geometry)`` runs after the
    dialog has closed, when the user clicks "Use this value as Gt"."""
    open_gain_calculator(state, "gw_Gt", on_accept)


@ui.refreshable
def step1_form(state: AppState) -> None:
    wr = state.wizard_results
    ui.label("AP7 Annex 3, Eq. 84-97").classes("text-caption text-grey-7")

    def _set_method(e) -> None:
        state.put_draft("s1_method", e.value)
        step1_form.refresh()

    method = state.get_draft("s1_method", "Manual")
    ui.radio(["Manual", "Calculate"], value=method, on_change=_set_method).props("inline")

    if method == "Manual":
        gt_in = ui.number("Gt (dBi)", value=_draft(state, "s1_manual", wr.gt if wr.gt is not None else 0.0),
                          format="%.4f").props("dense outlined").classes("w-full")
        gt_in.on_value_change(lambda e: state.put_draft("s1_manual", e.value))

        def use_manual() -> None:
            if gt_in.value is None:
                ui.notify("Enter a value for Gt first.", type="negative")
                return
            wr.gt = float(gt_in.value)
            wr.gt_source = "(entered manually)"
            # No geometry behind a manual Gt, so Step 5 can't recompute it per azimuth: drop any
            # geometry left over from an earlier calculation.
            wr.gt_geometry = None
            ui.notify(f"Gt = {wr.gt:.4f} dBi saved.", type="positive")
            step1_form.refresh()

        ui.button("Use this value as Gt", on_click=use_manual).props("unelevated no-caps color=primary")
    else:
        def accept(gain: float, geometry: Dict[str, Any]) -> None:
            wr.gt = gain
            wr.gt_source = "(calculated via Eq. 84-97, Step 1)"
            wr.gt_geometry = geometry
            ui.notify(f"Gt = {gain:.4f} dBi saved (calculated via Eq. 84-97).", type="positive")
            step1_form.refresh()

        ui.button("Open Gt calculator…", icon="calculate",
                  on_click=lambda: open_gt_calculator(state, accept)).props("unelevated no-caps color=primary")

    if wr.gt is not None:
        _status(True, f"Gt = {wr.gt:.4f} dBi {wr.gt_source or ''}".rstrip())
    else:
        _status(False, "Not yet determined.")
        gmax = _f(state.get_draft("gw_Gt_gmax"))
        if gmax > 0:
            ui.label(f"Main-beam gain Gmax = {gmax:.4f} dBi is saved as the default in the Gt calculator "
                     "(ITU import or your last entry). Gt is the horizon gain G(φ) and still has to be "
                     "calculated from it: choose “Calculate”. Coordinates, satellite longitude and "
                     "diameter are pre-filled there.").classes("text-caption text-grey-7")


# ===========================================================================
# Step 2 — Pt
# ===========================================================================
PT_OPTIONS = {"watts": "Watts → dBW", "eirp": "From EIRP", "dbw": "Known value (dBW)"}


@ui.refreshable
def step2_form(state: AppState) -> None:
    wr = state.wizard_results
    option = state.get_draft("ptd_option", "watts")

    def _set_option(e) -> None:
        state.put_draft("ptd_option", e.value)
        wr.ptd_result = None  # a result from another method must not be accepted by mistake
        step2_form.refresh()

    ui.radio(PT_OPTIONS, value=option, on_change=_set_option).props("inline")

    result_field: Optional[ui.input] = None
    if option == "watts":
        watts_in = _drafted_number(state, "Maximum aggregate power (W)", "ptd_watts", fmt="%.6f", min=0.0) \
            .classes("w-full").tooltip("Pt = 10·log₁₀(Watts)")
    elif option == "eirp":
        eirp_in = _drafted_number(state, "Maximum EIRP (dBW)", "ptd_eirp").classes("w-full")
        # EIRP is a main-beam-axis quantity, so the gain that belongs here is Gmax (from the Gt
        # calculator / ITU import), not the horizon gain Gt. Falls back to Gt if no Gmax is known.
        prefill = _f(state.get_draft("gw_Gt_gmax")) or (wr.gt if wr.gt is not None else 0.0)
        gain_in = _drafted_number(state, "Antenna gain used for this EIRP (dBi)", "ptd_gt", prefill) \
            .classes("w-full").tooltip("EIRP is a main-beam-axis quantity, so this is normally Gmax "
                                       "(pre-filled from the Gt calculator / ITU import); editable.")
    else:
        dbw_in = _drafted_number(
            state, "Pt (dBW)", "s2_manual", wr.pt if wr.pt is not None else 0.0).classes("w-full")

    if option in ("watts", "eirp"):
        result_field = _readonly_field("Pt =", _fmt(wr.ptd_result, "dBW"))

        def calc_pt() -> None:
            if option == "watts":
                watts = _f(watts_in.value)
                if watts <= 0:
                    ui.notify("Power must be > 0 W.", type="negative")
                    return
                pt_calc, how = 10.0 * math.log10(watts), f"10·log₁₀({watts:g}) dBW"
            else:
                eirp, gain = _f(eirp_in.value), _f(gain_in.value)
                pt_calc, how = eirp - gain, f"EIRP − G = {eirp:g} − {gain:g} dBW"
            wr.ptd_result = pt_calc
            result_field.set_value(_fmt(pt_calc, "dBW"))
            ui.notify(f"Pt = {pt_calc:.4f} dBW ({how}).", type="positive")

        ui.button("Calculate Pt", on_click=calc_pt).props("outline no-caps")

    def use_value() -> None:
        if option == "dbw":
            if dbw_in.value is None:
                ui.notify("Enter a value for Pt first.", type="negative")
                return
            wr.pt, wr.pt_source = float(dbw_in.value), "(entered manually)"
        else:
            if wr.ptd_result is None:
                ui.notify("Click “Calculate Pt” first.", type="negative")
                return
            wr.pt, wr.pt_source = wr.ptd_result, "(calculated, Step 2)"
        ui.notify(f"Pt = {wr.pt:.4f} dBW saved.", type="positive")
        step2_form.refresh()

    ui.button("Use this value as Pt", on_click=use_value).props("unelevated no-caps color=primary")

    if wr.pt is not None:
        _status(True, f"Pt = {wr.pt:.4f} dBW {wr.pt_source or ''}".rstrip())
    else:
        _status(False, "Not yet determined.")


# ===========================================================================
# Dispatcher used by desktop_app.py
# ===========================================================================
def build_step_form(state: AppState, step: int) -> None:
    """Render the input form for ``step`` in the current NiceGUI context."""
    # Imported here, not at module level: ui.step3_pane and ui.step5_pane import helpers from
    # this module, so a top-level import in both directions would be circular (and break
    # depending on which module happens to be imported first).
    from ui.step3_pane import step3_form, step4_form
    from ui.step5_pane import step5_form

    if step == 1:
        step1_form(state)
    elif step == 2:
        step2_form(state)
    elif step == 3:
        step3_form(state)
    elif step == 4:
        step4_form(state)
    elif step == 5:
        step5_form(state)
    else:
        ui.label("Input form placeholder.").classes("text-grey-7")  # Steps 6-9: later phases