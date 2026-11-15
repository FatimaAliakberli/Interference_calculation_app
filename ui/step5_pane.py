"""
Left Pane content (Phase 4B): Step 5 - coordination distance.

Public entry point
------------------
``step5_form(state)``
    Refreshable form for Step 5. It holds:
      * a button that opens the large "Propagation Parameters" dialog (the old Streamlit
        ``@st.dialog``): earth-station location, horizon-grid DB lookup, common parameters,
        Mode 1 radio-climatic zone + path composition, Mode 2 rain zone + elevation / Gx;
      * a "Calculate coordination distance" button that runs the iterative propagation
        mode (1) per-azimuth loop and propagation mode (2) in a worker thread, ONCE FOR EVERY
        frequency group of the imported ITU station (``state.itu_import.all_frequencies``;
        just the saved frequency when nothing was imported);
      * the results: polar plot (``ui.pyplot``), CSV download and a per-azimuth diagnostics table.

State used
----------
    * ``state.step5_inputs``  - horizon_profile, altitude_val, fetch_key, auto_zone(_name),
                                n0_val / be_val / dm_val / dx1_val / dc_val, autofill_sig,
                                ppd_params (the accepted parameter set) and ready.
    * ``state.step1``         - es_lat / es_lon (also the shared drafts "es_lat" / "es_lon").
    * ``state.step3``         - selected_freq (frequency pre-fill).
    * ``state.wizard_results``- pt, gt, gt_geometry, gr, pr_p (the Lb(p) terms).
    * ``state.itu_import``    - all_frequencies (every Tx / Rx frequency group, Part 1).
    * ``state.results``       - results_by_frequency {freq: FrequencyResults} and
                                active_frequency (the one the single-frequency panels show).
    * ``state.drafts``        - remembered field values, keys prefixed "s5_" (plus the shared
                                "es_lat", "es_lon", "freq_ghz", "p_percent").

Path composition (dtm / dlm / dc / land fraction) is either entered by hand or, by default,
measured along every azimuth with ``core.path_composition`` (great-circle rays over the Natural
Earth land + coastline data); see ``_gis_path_composition``.

Nothing in ``core/`` is modified; it is only imported. The slow calls (horizon DB lookup, R0.01
map lookup, the Mode 1 / Mode 2 iterations) all go through ``nicegui.run.io_bound`` so the
window never freezes.
"""
from __future__ import annotations

import csv
import io
import math
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple

from nicegui import run, ui

from core import calculations, propagation_calculator as pc, tables_data
from ui.left_pane import (
    _compute_phi,
    _draft,
    _drafted_number,
    _f,
    _find_horizon_db,
    _fmt,
    _readonly_field,
    _section_title,
    _status,
)
from ui.state import AppState, FrequencyResults, freq_key, get_resource_path
from ui.step3_pane import _row_values  # Gr / Pr(p) / Te / B of one ITU table row (same as Step 3)

try:  # matplotlib is only needed for the polar plot
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - depends on the environment
    plt = None

# Optional module: automatic radio-climatic zone (A1/A2/B/C) detection from lat/lon.
try:
    from core import radio_climatic_zone as _rcz
except ImportError:  # pragma: no cover - depends on the project
    _rcz = None

# Optional module: GIS path composition (dtm / dlm / dc / land fraction along each azimuth).
try:
    from core import path_composition as _pcomp
except ImportError:  # pragma: no cover - depends on the project / environment
    _pcomp = None

# Used when no horizon profile has been fetched and the user did not scan azimuths themselves.
DEFAULT_AZIMUTH_STEP_DEG = 5

_COASTLINE_NAME = "ne_10m_coastline.shp"
_coastline_cache: Dict[str, Any] = {}


def _find_coastline_shp() -> Tuple[Optional[str], List[str]]:
    """Locate ne_10m_coastline.shp independent of the working directory.

    Looks in the usual folders under the project root (or the PyInstaller bundle), then
    searches data_sources/ recursively, then falls back to paths relative to the cwd.
    Returns (path or None, every place that was checked)."""
    import os
    checked: List[str] = []
    candidates = [
        get_resource_path(os.path.join("data_sources", "coastline", _COASTLINE_NAME)),
        get_resource_path(os.path.join("data_sources", "costline", _COASTLINE_NAME)),
        get_resource_path(os.path.join("data_sources", "ne_10m_coastline", _COASTLINE_NAME)),
        get_resource_path(os.path.join("data_sources", _COASTLINE_NAME)),
        os.path.join("data_sources", "coastline", _COASTLINE_NAME),   # cwd-relative fallbacks
        os.path.join("data_sources", _COASTLINE_NAME),
        _COASTLINE_NAME,
    ]
    for p in candidates:
        checked.append(p)
        if os.path.isfile(p):
            return p, checked
    root = get_resource_path("data_sources")
    checked.append(os.path.join(root, "**", _COASTLINE_NAME))
    if os.path.isdir(root):
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.lower() == _COASTLINE_NAME:
                    return os.path.join(dirpath, f), checked
    return None, checked


def _load_coastline_gdf():
    """GeoDataFrame of the coastline, read once and cached (the 10m file is large)."""
    if "gdf" in _coastline_cache:
        return _coastline_cache["gdf"]
    import geopandas as gpd          # ImportError -> reported as such, not as "file not found"
    path, checked = _find_coastline_shp()
    if path is None:
        raise FileNotFoundError(
            f"{_COASTLINE_NAME} not found. Looked in: " + "; ".join(checked))
    try:
        gdf = gpd.read_file(path)    # needs the .shx/.dbf (and ideally .prj) next to the .shp
    except Exception as primary:     # noqa: BLE001 - typically a broken pyogrio/fiona + GDAL install
        try:
            gdf = _read_shapefile_pure_python(path)
        except Exception as fallback:  # noqa: BLE001
            raise RuntimeError(
                f"geopandas could not read the shapefile ({primary}). The pure-Python fallback "
                f"failed too ({fallback}). Fix the GDAL backend (reinstall pyogrio) or run: "
                f"pip install pyshp") from fallback
    _coastline_cache["gdf"] = gdf
    return gdf


def _read_shapefile_pure_python(path: str):
    """Read a shapefile with pyshp + shapely, bypassing GDAL (pyogrio / fiona) completely.
    Natural Earth data is WGS84, so EPSG:4326 is assigned."""
    import geopandas as gpd
    import shapefile                       # pip install pyshp
    from shapely.geometry import shape

    rows = []
    with shapefile.Reader(path) as reader:
        for sr in reader.iterShapeRecords():
            if sr.shape.shapeType == shapefile.NULL:
                continue
            row = dict(sr.record.as_dict())
            row["geometry"] = shape(sr.shape.__geo_interface__)
            rows.append(row)
    if not rows:
        raise ValueError("the shapefile contains no geometries")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# ===========================================================================
# Pure helpers (no UI - safe to call from a worker thread)
# ===========================================================================
# def _detect_radio_climatic_zone(lat: float, lon: float) -> Dict[str, Any]:
#     """Run ``core.radio_climatic_zone`` and normalise whatever it returns to
#     ``{"zone": str, "name": str | None, "dc": float | None}``.

#     The module's API is not part of the files this port was written from, so a few likely
#     function names are tried and str / dict / tuple results are accepted. Raises RuntimeError
#     when the module is missing or nothing usable comes back (the dialog then falls back to the
#     manual zone selector).
#     """
#     if _rcz is None:
#         raise RuntimeError("core.radio_climatic_zone is not available - choose the zone manually.")
#     fn = None
#     for name in ("detect_zone", "get_radio_climatic_zone", "get_zone", "lookup_zone",
#                  "classify_zone", "detect_radio_climatic_zone"):
#         fn = getattr(_rcz, name, None)
#         if callable(fn):
#             break
#     if not callable(fn):
#         raise RuntimeError("core.radio_climatic_zone has no recognised lookup function - "
#                            "choose the zone manually.")
#     out = fn(lat, lon)
#     zone: Optional[str] = None
#     name: Optional[str] = None
#     dc: Optional[float] = None
#     if isinstance(out, str):
#         zone = out
#     elif isinstance(out, dict):
#         zone = out.get("zone") or out.get("zone_code") or out.get("code")
#         name = out.get("zone_name") or out.get("name")
#         dc = out.get("dc") if out.get("dc") is not None else out.get("dc_km")
#     elif isinstance(out, (tuple, list)) and out:
#         zone = out[0]
#         name = out[1] if len(out) > 1 and isinstance(out[1], str) else None
#     zone = str(zone).upper() if zone else None
#     if zone not in pc.ZONES:
#         raise RuntimeError(f"Zone detection returned {out!r}, which is not one of {pc.ZONES}.")
#     return {"zone": zone, "name": name, "dc": dc}




def _detect_radio_climatic_zone(lat: float, lon: float, altitude: float) -> Dict[str, Any]:
    """Run `core.radio_climatic_zone` and normalise whatever it returns to
    `{"zone": str, "name": str | None, "dc": float | None}`."""
    if _rcz is None:
        raise RuntimeError("core.radio_climatic_zone is not available - choose the zone manually.")
    fn = None
    for name in ("detect_zone", "get_radio_climatic_zone", "get_zone", "lookup_zone",
                 "classify_zone", "detect_radio_climatic_zone", "determine_radio_climatic_zone"):
        fn = getattr(_rcz, name, None)
        if callable(fn):
            break
    if not callable(fn):
        raise RuntimeError("core.radio_climatic_zone has no recognised lookup function.")

    import inspect
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    out = None
    if len(params) >= 4 and 'coastline_gdf' in params:
        # It's the old GIS-heavy function: determine_radio_climatic_zone(lon, lat, alt, gdf)
        try:
            gdf = _load_coastline_gdf()
        except Exception as e:
            raise RuntimeError(f"Failed to load coastline shapefile for zone detection: {e}")
        # NOTE: the function signature is (station_lon, station_lat, station_alt, coastline_gdf).
        # Errors raised by the zone function itself are NOT labelled as a coastline-loading problem.
        out = fn(lon, lat, altitude, gdf)
    else:
        # New signature: (lat, lon)
        out = fn(lat, lon)

    zone: Optional[str] = None
    name: Optional[str] = None
    dc: Optional[float] = None
    if isinstance(out, str):
        zone = out
    elif isinstance(out, dict):
        zone = out.get("zone") or out.get("zone_code") or out.get("code")
        name = out.get("zone_name") or out.get("name")
        dc = out.get("dc") if out.get("dc") is not None else out.get("dc_km")
    elif isinstance(out, (tuple, list)) and out:
        zone = out[0]
        name = out[1] if len(out) > 1 and isinstance(out[1], str) else None
        dc = out[2] if len(out) > 2 else None

    zone = str(zone).upper() if zone else None
    if zone not in pc.ZONES:
        raise RuntimeError(f"Zone detection returned {out!r}, which is not one of {pc.ZONES}.")
    return {"zone": zone, "name": name, "dc": dc}



def _query_horizon(db_path: str, lat: float, lon: float) -> Dict[str, Any]:
    """Nearest precomputed horizon profile (slow: loads the whole grid index)."""
    from core.query_horizon import query_location  # lazy: pulls in scipy / pyproj

    return query_location(db_path, lat, lon)


def _detect_rain_zone(lat: float, lon: float, txt_path: Optional[str]) -> Dict[str, Any]:
    """R0.01 from the ITU-R P.837-8 map and the nearest rain zone (slow: parses a 1441x2881 grid
    the first time; ``rain_zone`` caches it afterwards)."""
    from core import rain_zone as rz  # lazy

    r001 = rz.get_r001_from_txt(lat, lon, txt_path or None)
    return {"r001": r001, "zone": rz.get_rain_zone_from_r001(r001)}


def _profile_dh(entry: Dict[str, Any]) -> Optional[float]:
    """Horizon distance (km) of one profile entry. The key name is not fixed in the files this
    port was written from, so the usual spellings are tried; None -> the calculator clamps to 0.5 km."""
    for key in ("distance_km", "horizon_distance_km", "distance", "dh", "dh_km"):
        v = entry.get(key)
        if v is not None:
            return _f(v, 0.5)
    return None


def _mode1_runner(freq_ghz: float):
    """The Annex 1 band function for this frequency."""
    if freq_ghz < 0.79:
        return pc.run_mode1_100_790
    if freq_ghz < 60.0:
        return pc.run_mode1_790_60000
    return pc.run_mode1_60_105


def _gt_at_azimuth(geom: Dict[str, Any], az: float, eps_h: float) -> float:
    """Gt(alpha): recompute the horizon gain G(phi) for azimuth ``az`` from the stored Step 1 recipe."""
    phi, _note = _compute_phi(
        geom["mode"], _f(geom.get("zeta")), _f(geom.get("i")), _f(geom.get("delta0")),
        _f(geom.get("K"), calculations.K_DEFAULT), az, eps_h,
        _f(geom.get("is_max")), _f(geom.get("step"), 1.0))
    res = calculations.gain_at_phi(phi, _f(geom["d_over_lambda"]), _f(geom["gmax"]))
    return float(res["gain"])


# ---------------------------------------------------------------------------
# GIS path composition
# ---------------------------------------------------------------------------
DEFAULT_GIS_RAY_KM = 1200.0          # ray length when no dmax1 is known
GIS_RAY_RANGE_KM = (100.0, 3000.0)
_PATH_CACHE_SIZE = 4
_path_cache: Dict[Tuple[Any, ...], Dict[float, Dict[str, float]]] = {}
_path_cache_lock = threading.Lock()


def _gis_ray_length_km(params: Dict[str, Any]) -> float:
    """Length of the ray along which the path composition is measured: dmax1 (the largest
    distance mode 1 can return), clamped to a sane range. dtm is later capped at the trial
    distance by the Annex 1 code (dt = min(dtm, di)), so measuring out to dmax1 is consistent."""
    v = _pos_num(params.get("dmax1"))
    if v is None:
        return DEFAULT_GIS_RAY_KM
    lo, hi = GIS_RAY_RANGE_KM
    return min(max(v, lo), hi)


def _gis_path_composition(params: Dict[str, Any], azimuths: List[int]
                          ) -> Tuple[Optional[Dict[float, Dict[str, float]]], Optional[str]]:
    """{azimuth: {dtm, dlm, dc, land_fraction}} for every azimuth, or (None, reason) when the GIS
    calculation is unavailable (libraries / data files missing, bad inputs). Results are cached,
    so a multi-frequency run measures each station / ray length only once."""
    if _pcomp is None:
        return None, "core.path_composition could not be imported"
    lat, lon = _f(params.get("latitude")), _f(params.get("longitude"))
    length = _gis_ray_length_km(params)
    thr = _pos_num(params.get("inland_threshold_km"))
    thr = 50.0 if thr is None else thr
    key = (round(lat, 5), round(lon, 5), round(length, 1), round(thr, 2), tuple(azimuths))
    with _path_cache_lock:
        hit = _path_cache.get(key)
    if hit is not None:
        return hit, None
    try:
        res = _pcomp.calculate_path_params_by_azimuth(lat, lon, azimuths, length,
                                                      inland_threshold_km=thr)
    except Exception as ex:  # noqa: BLE001 - PathCompositionError, ValueError, GDAL problems...
        print("[step5] GIS path composition FAILED:\n" + traceback.format_exc(), flush=True)
        return None, (f"GIS FAILED: {ex}. Check terminal logs for details. "
                      "Ensure shapely>=2.0, pyproj, and numpy are installed, and that "
                      "ne_10m_admin_0_countries.shp (plus .shx/.dbf) exists in data_sources/.")
    with _path_cache_lock:
        while len(_path_cache) >= _PATH_CACHE_SIZE:
            _path_cache.pop(next(iter(_path_cache)))
        _path_cache[key] = res
    return res, None


def compute_mode1_profile(params: Dict[str, Any], pt: float, gt: float, gr: float, pr_p: float,
                          gt_geometry: Optional[Dict[str, Any]],
                          frequency_ghz: Optional[float] = None) -> Dict[str, Any]:
    """Propagation mode (1) required distance for every azimuth of the horizon profile.

    Lb(p) = Pt + Gt + Gr - Pr(p), with Gt recomputed per azimuth when the Gt came from the
    Eq. 84-97 calculator (``gt_geometry`` is not None), otherwise the single Step 1 value.
    ``frequency_ghz`` selects the frequency (and so the Annex 1 band function); omitted, it is
    ``params["frequency_GHz"]``. For a multi-frequency run pass that frequency's own ``pt``,
    ``gr``, ``pr_p`` and ``gt_geometry`` (see build_frequency_plan).
    Path composition: when ``params["use_gis_path"]`` is true, dtm, dlm, dc and the land fraction
    are measured by ``core.path_composition`` along EVERY azimuth (ray of length dmax1 from the
    station) and passed to the Annex 1 runner for that azimuth. If the GIS data / libraries are
    unavailable, or one azimuth has no result, the manual values in ``params`` are used instead
    (and reported in ``out["path_composition"]`` / the diagnostics note).
    Returns {"profile": [(az, km)], "diag": [row dicts], "errors": [(az, message)],
             "path_composition": {"mode", "ray_km", "error"}}.
    """
    f = float(frequency_ghz if frequency_ghz is not None else params["frequency_GHz"])
    runner = _mode1_runner(f)
    horizon = (params.get("horizon_profile") or {}).get("profile") or {}
    if horizon:
        azimuths = sorted(int(a) for a in horizon)
    else:
        azimuths = list(range(0, 360, DEFAULT_AZIMUTH_STEP_DEG))

    n0 = _f(params.get("N0"))
    profile: List[Tuple[float, float]] = []
    diag: List[Dict[str, Any]] = []
    errors: List[Tuple[float, str]] = []

    gis: Optional[Dict[float, Dict[str, float]]] = None
    pc_info: Dict[str, Any] = {"mode": "manual", "ray_km": None, "error": None}
    if params.get("use_gis_path"):
        gis, gis_err = _gis_path_composition(params, azimuths)
        if gis is not None:
            pc_info.update(mode="gis", ray_km=_gis_ray_length_km(params))
        else:
            pc_info.update(mode="manual (GIS failed)", error=gis_err)

    for az in azimuths:
        entry = horizon.get(az) or {}
        eps_h = _f(entry.get("elevation_angle"), _f(params.get("eps_h")))
        dh = _profile_dh(entry) if entry else None
        if dh is None:
            dh = _f(params.get("dh"), 0.5)

        gt_az, gt_note = gt, ""
        if gt_geometry:
            try:
                gt_az = _gt_at_azimuth(gt_geometry, az, eps_h)
            except Exception as ex:  # noqa: BLE001 - keep the azimuth, fall back to the Step 1 Gt
                gt_note = f"Gt(α) failed ({ex}); used Step 1 Gt"
        lb_p = pt + gt_az + gr - pr_p

        # Path composition of this azimuth: GIS values, else the manual ones.
        dtm, dlm, dc, land_frac, pc_src = (params["dtm"], params["dlm"], params["dc"],
                                           params["land_fraction"], "manual")
        comp = gis.get(float(az)) if gis is not None else None
        if comp is not None:
            dtm, dlm, dc, land_frac, pc_src = (comp["dtm"], comp["dlm"], comp["dc"],
                                               comp["land_fraction"], "GIS")
        elif gis is not None:
            gt_note = (gt_note + "; " if gt_note else "") + "no GIS path composition; manual values used"

        try:
            r = runner(lb_p, f, params["p_percent"], params.get("zone"), eps_h, dh,
                       params["latitude"], n0 if n0 > 0 else None, dtm, dlm,
                       dc, params["is_transmitter"], land_frac,
                       params["step"])
        except Exception as ex:  # noqa: BLE001 - one bad azimuth must not kill the profile
            errors.append((az, str(ex)))
            continue
        km = float(r["required_distance_km"])
        profile.append((float(az), km))
        diag.append({
            "azimuth": az, "eps_h": eps_h, "dh": dh, "gt": gt_az, "lb_p": lb_p,
            "dtm": dtm, "dlm": dlm, "dc": dc, "land_fraction": land_frac, "path_source": pc_src,
            "km": km, "ah": r.get("Ah"), "iterations": r.get("iterations"),
            "terminated_by": (r.get("terminated_by") or "").split(" (")[0],
            "note": gt_note,
        })
    return {"profile": profile, "diag": diag, "errors": errors, "path_composition": pc_info}


def run_coordination(params: Dict[str, Any], pt: float, gt: float, gr: float, pr_p: float,
                     gt_geometry: Optional[Dict[str, Any]],
                     frequency_ghz: Optional[float] = None) -> Dict[str, Any]:
    """Mode 1 profile + Mode 2 + the AP7 S1.6 combined contour (worker-thread entry point;
    touches no UI).

    ``frequency_ghz`` runs both modes at that frequency instead of ``params["frequency_GHz"]``
    (``params`` itself is not modified). The other arguments must already be the values for
    that frequency: ``pt``, ``gr`` and ``pr_p`` (Lb(p) terms; Table 7/8/9 values for the
    frequency), ``gt_geometry`` (D/lambda and Gmax at that frequency) and the per-frequency
    entries of ``params`` (is_transmitter, gx, ...). ``build_frequency_plan`` prepares all of
    these. ``out["frequency_ghz"]`` is the frequency actually used. Called without
    ``frequency_ghz`` it behaves exactly as before (Step 8 mitigation relies on that).

    ``out["final_contour"]`` is ``[(azimuth, max(mode1_km, dr_km)), ...]``: mode (2) does not
    depend on azimuth, so its single distance ``dr_km`` is compared with every mode (1) value.
    It is None when mode (2) was not computed or there is no mode (1) profile. A mode (2)
    failure never discards the mode (1) results; it is recorded in ``mode2_error``.
    """
    if frequency_ghz is not None and float(frequency_ghz) != float(params["frequency_GHz"]):
        params = {**params, "frequency_GHz": float(frequency_ghz)}
    out = compute_mode1_profile(params, pt, gt, gr, pr_p, gt_geometry)
    out["frequency_ghz"] = float(params["frequency_GHz"])
    out["mode2"], out["mode2_error"], out["final_contour"] = None, None, None
    if params.get("rain_zone"):
        try:
            if float(params["p_percent"]) <= 0.001:
                # Same guard as the Streamlit app: Eq. 63-67 are defined for p > 0.001 %.
                raise ValueError(f"p = {params['p_percent']:g}% is outside the Annex 2 range "
                                 "(p must be > 0.001%).")
            mode2_result = pc.run_mode2(
                pt, pr_p, params["frequency_GHz"], params["p_percent"], params["rain_zone"],
                params["latitude"], params["longitude"], params["elevation_angle"],
                params["gx"], params["step"])
            dr_km = float(mode2_result["dr_km"])
            if out["profile"]:
                out["final_contour"] = [(az, max(d1, dr_km)) for az, d1 in out["profile"]]
            out["mode2"] = mode2_result
        except Exception as ex:  # noqa: BLE001
            out["mode2_error"] = str(ex)
    else:
        out["mode2_error"] = "No rain climatic zone selected - propagation mode (2) was skipped."
    return out


# ===========================================================================
# Multi-frequency planning (pure functions - no UI, safe in a worker thread)
# ===========================================================================
FREQ_MATCH_TOL_GHZ = 1e-3          # a group within 1 MHz of the saved frequency IS the saved one
FREQ_RANGE_GHZ = (0.1, 105.0)      # the range the Annex 1 band functions cover

# Which ITU table to try first for a frequency group, by direction (7 = transmitting ES,
# 8 = receiving ES, 9 = bidirectional). Receive / unknown keep Step 3's order (8, 7, 9).
_TABLE_ORDER: Dict[Optional[str], Tuple[str, ...]] = {
    "E": ("7", "9", "8"),
    "R": ("8", "7", "9"),
    None: ("8", "7", "9"),
}
_TABLE_LABEL = {"7": "Table 7", "8": "Table 8", "9": "Table 9"}


def _pos_num(value: Any) -> Optional[float]:
    """float(value) when it is a finite number > 0, else None (blank / 0 = missing)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if (v == v and 0.0 < v < float("inf")) else None


def _group_direction(group: Optional[Dict[str, Any]]) -> Optional[str]:
    d = str((group or {}).get("direction") or (group or {}).get("emi_rcp") or "").strip().upper()
    return d if d in ("E", "R") else None


def _dir_label(direction: Optional[str]) -> str:
    return {"E": "Tx", "R": "Rx"}.get(direction, "?")


def lookup_victim_for_frequency(freq_ghz: float, direction: Optional[str]) -> Optional[Dict[str, Any]]:
    """Gr, Pr(p), Te and B for one frequency from the ITU tables (RR Appendix 7 Tables 7/8/9).

    Same selection as Step 3's "Use frequencies from imported ITU MDB file": the first table (in
    ``_TABLE_ORDER`` for the direction) with a row at this frequency that has BOTH Gr and Pr(p);
    among such rows a GSO row is preferred (this tool only supports GSO). Returns
    {"gr", "pr", "te", "b", "table", "band", "orbit", "n_rows"} or None when nothing usable.
    """
    for alias in _TABLE_ORDER.get(direction, _TABLE_ORDER[None]):
        try:
            rows = tables_data.get_victim_parameters(alias, freq_ghz)
        except Exception:  # noqa: BLE001 - workbook missing / unreadable: caller falls back
            continue
        usable = [r for r in rows
                  if _row_values(alias, r)["gr"] is not None and _row_values(alias, r)["pr"] is not None]
        if not usable:
            continue
        best = min(usable, key=lambda r: 0 if str(r.get("Orbit")).strip().lower() == "gso" else 1)
        v = _row_values(alias, best)
        return {"gr": v["gr"], "pr": v["pr"], "te": v["te"], "b": v["b"],
                "table": _TABLE_LABEL.get(alias, alias), "band": best.get("Frequency Band"),
                "orbit": best.get("Orbit"), "n_rows": len(usable)}
    return None


def geometry_for_frequency(base_geom: Dict[str, Any], base_freq: float, freq_ghz: float,
                           group: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """The Step 1 Gt recipe (``wizard_results.gt_geometry``) re-targeted to another frequency.

    Only the frequency-dependent antenna terms change; the geometry (mode, zeta, delta0, K,
    i / is_max / step) is copied. The physical dish is the same at every frequency:
      * diameter D  = the group's own e_ant diameter, else the Step 1 diameter, else the Step 1
                      D/lambda scaled back to a diameter at ``base_freq``;
      * D/lambda    = D / lambda(f);
      * Gmax        = the group's own e_ant gain when it has one, else the Step 1 Gmax moved by
                      20*log10(f / base_freq) (fixed aperture, gain ~ f^2), else S.465 from D/lambda.
    Returns (geometry copy, notes).
    """
    notes: List[str] = []
    g = dict(base_geom)
    f = float(freq_ghz)
    lam = calculations.wavelength_from_ghz(f)
    grp = group or {}
    grp_d, grp_gain = _pos_num(grp.get("ant_diam_m")), _pos_num(grp.get("ant_gain_dbi"))
    base_dl, base_gmax = _pos_num(base_geom.get("d_over_lambda")), _pos_num(base_geom.get("gmax"))

    diam: Optional[float] = None
    if grp_d is not None:
        diam, d_src = grp_d, "antenna diameter of this group (e_ant)"
    elif _pos_num(base_geom.get("diameter")) is not None:
        diam, d_src = _pos_num(base_geom.get("diameter")), "Step 1 diameter"
    elif base_dl is not None and base_freq > 0:
        diam, d_src = base_dl * calculations.wavelength_from_ghz(base_freq), "Step 1 D/λ scaled with frequency"
    if diam is not None:
        dl = diam / lam
    else:  # no diameter anywhere: ITU defaults (S.465, else 3.7 m) at this frequency
        r = calculations.resolve_antenna_defaults(gmax_dbi=grp_gain, frequency_ghz=f)
        dl, diam, d_src = r["d_over_lambda"], r["diameter_m"], r["note"]

    if grp_gain is not None:
        gmax, g_src = grp_gain, "Gmax of this group (e_ant)"
    elif base_gmax is not None and base_freq > 0:
        gmax = base_gmax + 20.0 * math.log10(f / base_freq)
        g_src = f"Step 1 Gmax {base_gmax:.2f} dBi scaled from {base_freq:g} GHz"
    else:
        gmax, g_src = calculations.gmax_from_d_over_lambda(dl), "estimated from D/λ (S.465)"

    g.update({"d_over_lambda": dl, "gmax": gmax, "wavelength": lam, "diameter": diam})
    notes.append(f"Gt geometry at {f:.4f} GHz: D = {diam:.3f} m ({d_src}), D/λ = {dl:.2f}, "
                 f"Gmax = {gmax:.2f} dBi ({g_src})")
    return g, notes


def params_for_frequency(base: Dict[str, Any], freq_ghz: float, direction: Optional[str],
                         gx: Optional[float] = None) -> Dict[str, Any]:
    """Copy of the accepted parameter set (``ppd_params``) for one frequency.

    Location, zone, N0, horizon profile, p, path composition and rain zone are the station's and
    stay as saved. Changed: the frequency; dmin and dmax1 (frequency-dependent auto-fill values);
    ``is_transmitter`` (Tx group -> True, Rx group -> False; unknown keeps the saved value);
    and Gx (the victim station's gain, used by mode 2) when given.
    """
    p = dict(base)
    p["frequency_GHz"] = float(freq_ghz)
    if direction == "E":
        p["is_transmitter"] = True
    elif direction == "R":
        p["is_transmitter"] = False
    p_pct = _pos_num(p.get("p_percent"))
    try:
        p["dmin"] = pc.dmin(float(freq_ghz), p["latitude"])
        p["dmax1"] = pc.dmax1(float(freq_ghz), p.get("zone"), p_pct)
    except Exception:  # noqa: BLE001 - informational only: the Annex 1 runners compute their own
        pass
    if gx is not None:
        p["gx"] = float(gx)
    return p


def _plan_entry(base_params: Dict[str, Any], pt: float, gt: float, gr: float, pr_p: float,
                gt_geometry: Optional[Dict[str, Any]], group: Optional[Dict[str, Any]],
                freq_ghz: float, is_base: bool) -> Dict[str, Any]:
    """Everything one frequency needs. The BASE frequency (the one saved in the Propagation
    Parameters dialog) uses the Step 1-3 values exactly as saved - the same inputs as a
    single-frequency run, which Step 8's mitigation re-runs. Every other frequency derives
    its own Pt, Gt geometry, Gr / Pr(p) / Gx and is_transmitter."""
    direction = _group_direction(group)
    base_f = float(base_params["frequency_GHz"])
    notes: List[str] = []
    victim: Optional[Dict[str, Any]] = None

    if is_base:
        params, e_pt, e_gt, e_gr, e_pr, geom = dict(base_params), pt, gt, gr, pr_p, gt_geometry
        notes.append("Saved propagation parameters and Step 1-3 values used as entered"
                     + ("" if group is not None else " (frequency is not one of the ITU groups)"))
    else:
        grp_pwr = (group or {}).get("pwr_max_dbw")
        e_pt = float(grp_pwr) if grp_pwr is not None else pt
        notes.append(f"Pt = {e_pt:.2f} dBW (" + ("grp.pwr_max of this group)" if grp_pwr is not None
                                                else "Step 2 value: this group has no pwr_max)"))
        victim = lookup_victim_for_frequency(freq_ghz, direction)
        if victim is not None:
            e_gr, e_pr, gx = victim["gr"], victim["pr"], victim["gr"]
            notes.append(f"Gr = {e_gr:.2f} dBi, Pr(p) = {e_pr:.2f} dBW from {victim['table']}, "
                         f"{victim['band']}")
        else:
            e_gr, e_pr, gx = gr, pr_p, None
            notes.append("No ITU table row with both Gr and Pr(p) at this frequency: "
                         "Step 3 values used")
        params = params_for_frequency(base_params, freq_ghz, direction, gx)
        e_gt, geom = gt, None
        if gt_geometry:
            try:
                geom, gnotes = geometry_for_frequency(gt_geometry, base_f, freq_ghz, group)
                notes.extend(gnotes)
            except Exception as ex:  # noqa: BLE001
                geom = gt_geometry
                notes.append(f"Gt geometry could not be re-targeted ({ex}); Step 1 geometry used")
        else:
            notes.append("Gt entered manually in Step 1: the same Gt is used at every frequency")

    entry: Dict[str, Any] = {
        "key": freq_key(freq_ghz), "frequency_ghz": float(params["frequency_GHz"]),
        "direction": direction, "group": group, "is_base": is_base,
        "params": params, "pt": float(e_pt), "gt": float(e_gt), "gr": float(e_gr),
        "pr_p": float(e_pr), "geometry": geom, "victim": victim, "notes": notes,
        "skip_reason": None,
    }
    lo, hi = FREQ_RANGE_GHZ
    if not lo <= entry["frequency_ghz"] <= hi:
        entry["skip_reason"] = f"{entry['frequency_ghz']:g} GHz is outside the supported {lo:g}-{hi:g} GHz range"
    return entry


def _entry_rank(e: Dict[str, Any]) -> Tuple[Any, ...]:
    """Worst case first, used only when two groups share one dictionary key."""
    gmax = (e["geometry"] or {}).get("gmax")
    return (e["is_base"], e["direction"] == "E", e["pt"] + (gmax if gmax is not None else e["gt"]))


def build_frequency_plan(base_params: Dict[str, Any], pt: float, gt: float, gr: float, pr_p: float,
                         gt_geometry: Optional[Dict[str, Any]],
                         groups: Optional[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """One plan entry per frequency to calculate, plus run-level notes.

    ``groups`` is ``state.itu_import.all_frequencies``. With no groups the plan is the saved
    frequency alone (a single-frequency run, as before). Otherwise it is every group, in table
    order; if none of them is the saved frequency, the saved frequency is calculated too (first),
    so the saved parameters, Step 8 mitigation and the results always agree.

    Results are keyed by frequency, so groups with the same centre frequency collapse into one
    entry: the saved one, else Tx before Rx, else the highest Pt + Gmax (worst case); the dropped
    ones are named in that entry's notes. Entries are returned in first-appearance order.
    Slow only because of the ITU table lookups: call it through ``run.io_bound``.
    """
    f0 = float(base_params["frequency_GHz"])
    usable = [(g, c) for g in (groups or []) for c in [_pos_num(g.get("freq_center_ghz"))] if c is not None]
    run_notes: List[str] = []
    skipped = len(groups or []) - len(usable)
    if skipped:
        run_notes.append(f"{skipped} frequency group(s) without a usable centre frequency were ignored.")

    base_idx: Optional[int] = None
    if usable:
        idx = min(range(len(usable)), key=lambda i: abs(usable[i][1] - f0))
        if abs(usable[idx][1] - f0) <= FREQ_MATCH_TOL_GHZ:
            base_idx = idx

    entries: List[Dict[str, Any]] = []
    if base_idx is None and usable:
        run_notes.append(f"The saved frequency {f0:g} GHz is not one of the ITU groups; it is "
                         "calculated as well so Step 8 stays consistent with the saved parameters.")
    if base_idx is None:
        entries.append(_plan_entry(base_params, pt, gt, gr, pr_p, gt_geometry, None, f0, True))
    for i, (g, c) in enumerate(usable):
        entries.append(_plan_entry(base_params, pt, gt, gr, pr_p, gt_geometry, g,
                                   f0 if i == base_idx else c, i == base_idx))

    merged: Dict[float, Dict[str, Any]] = {}
    for e in entries:
        cur = merged.get(e["key"])
        if cur is None:
            merged[e["key"]] = e
            continue
        win, lose = (e, cur) if _entry_rank(e) > _entry_rank(cur) else (cur, e)
        dropped = (lose["group"] or {}).get("beam_name") or (lose["group"] or {}).get("grp_id") or "other"
        win["notes"].append(f"Another group at the same frequency ({dropped}) was not calculated "
                            "separately (results are keyed by frequency)")
        merged[e["key"]] = win
    return list(merged.values()), run_notes


def _entry_inputs(e: Dict[str, Any]) -> Dict[str, Any]:
    """The inputs a frequency was calculated with (stored in ``mode1_result["inputs"]``)."""
    grp = e["group"] or {}
    v = e["victim"] or {}
    return {
        "frequency_ghz": e["frequency_ghz"], "direction": e["direction"], "is_base": e["is_base"],
        "pt": e["pt"], "gt_default": e["gt"], "gr": e["gr"], "pr_p": e["pr_p"],
        "gx": e["params"].get("gx"), "is_transmitter": e["params"].get("is_transmitter"),
        "grp_id": grp.get("grp_id"), "beam_name": grp.get("beam_name"),
        "table": v.get("table"), "band": v.get("band"),
        "d_over_lambda": (e["geometry"] or {}).get("d_over_lambda"),
        "gmax": (e["geometry"] or {}).get("gmax"),
        "notes": list(e["notes"]),
    }


def diag_to_csv(diag: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    cols = ["azimuth", "eps_h", "dh", "dtm", "dlm", "dc", "land_fraction", "path_source", "gt",
            "lb_p", "km", "ah", "iterations", "terminated_by", "note"]
    w = csv.writer(buf)
    w.writerow(["azimuth_deg", "eps_h_deg", "dh_km", "dtm_km", "dlm_km", "dc_km", "land_fraction",
                "path_composition_source", "Gt_dBi", "Lb_p_dB", "mode1_distance_km",
                "Ah_dB", "iterations", "terminated_by", "note"])
    for row in diag:
        w.writerow([row.get(c) for c in cols])
    return buf.getvalue().encode("utf-8")


def _download(data: bytes, filename: str, media_type: str) -> None:
    try:
        ui.download.content(data, filename, media_type)  # NiceGUI >= 2.x / 3.x
    except AttributeError:  # pragma: no cover
        ui.download(data, filename)


# ===========================================================================
# Propagation Parameters dialog
# ===========================================================================
def _state_number(label: str, obj: Any, attr: str, fmt: str = "%.4f", **kwargs: Any) -> ui.number:
    """A ui.number mirrored into ``obj.attr`` (used for the Step5Inputs auto-fill fields)."""
    el = ui.number(label, value=getattr(obj, attr), format=fmt, **kwargs).props("dense outlined")
    el.on_value_change(lambda e: setattr(obj, attr, _f(e.value)))
    return el


def build_params_dialog(state: AppState, on_saved) -> ui.dialog:
    """Create (closed) the Propagation Parameters dialog. ``on_saved()`` runs after the accepted
    parameter set has been stored in ``state.step5_inputs.ppd_params``."""
    s5 = state.step5_inputs
    wr = state.wizard_results

    with ui.dialog() as dialog, ui.card().style(
            "width: 820px; max-width: 95vw; max-height: 92vh; overflow-y: auto;"):
        ui.label("Propagation Parameters").classes("text-h6")
        ui.label("AP7 main body, Annex 1 (propagation mode 1) and Annex 2 (propagation mode 2)") \
            .classes("text-caption text-grey-7")

        # ---- 1. Earth station location -----------------------------------------------
        _section_title("1. Earth station location")
        ui.label("Shared with the Gt calculator (Step 1); pre-filled from there and from an ITU import.") \
            .classes("text-caption text-grey-7")
        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            lat_in = _drafted_number(state, "Latitude (deg, +N/−S)", "es_lat",
                                     _f(state.step1.es_lat)).classes("col")
            lon_in = _drafted_number(state, "Longitude (deg, +E/−W)", "es_lon",
                                     _f(state.step1.es_lon)).classes("col")
            alt_field = _readonly_field("Altitude (m, from horizon DB)", f"{s5.altitude_val:.2f}") \
                .classes("col")

        # ---- 2. Horizon grid database ------------------------------------------------
        _section_title("2. Horizon grid database")
        db_in = ui.input("Grid DB path override (optional — blank = auto-search)",
                         value=_draft(state, "s5_db_override", "")).props("dense outlined").classes("w-full")
        db_in.on_value_change(lambda e: state.put_draft("s5_db_override", e.value))
        horizon_lbl = ui.label().classes("text-caption text-grey-7")

        def _show_horizon() -> None:
            rec = s5.horizon_profile
            if not rec:
                horizon_lbl.set_text("No horizon profile fetched yet (the manual εh / dh below are used).")
                return
            horizon_lbl.set_text(
                f"Grid point {rec['lat']:.5f}°, {rec['lon']:.5f}° "
                f"({rec.get('offset_km', 0.0):.2f} km from the station); "
                f"{len(rec['profile'])} azimuths, increment {rec['azimuth_increment']}°; "
                f"grid altitude {rec['station_alt']:.2f} m.")
            alt_field.set_value(f"{s5.altitude_val:.2f}")

        async def fetch_horizon() -> None:
            lat, lon = _f(lat_in.value), _f(lon_in.value)
            db_path, searched = _find_horizon_db(db_in.value)
            if db_path is None:
                ui.notify("horizon_grid.db not found (checked: " + ", ".join(searched) + ").",
                          type="negative", multi_line=True)
                return
            busy = ui.notification("Fetching horizon profile…", spinner=True, timeout=None)
            try:
                rec = await run.io_bound(_query_horizon, db_path, lat, lon)  # slow: keep UI alive
            except Exception as ex:  # noqa: BLE001
                ui.notify(f"Horizon lookup failed: {ex}", type="negative", multi_line=True)
                return
            finally:
                busy.dismiss()
            s5.horizon_profile = rec
            s5.altitude_val = _f(rec.get("station_alt"))
            s5.altitude_note = f"from {db_path}"
            s5.fetch_key = (lat, lon, db_path)
            _show_horizon()
            ui.notify(f"Horizon profile loaded ({rec['offset_km']:.2f} km from the nearest grid point).",
                      type="positive")

        ui.button("Fetch horizon profile", icon="terrain", on_click=fetch_horizon) \
            .props("unelevated no-caps color=primary")
        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            eps_in = _drafted_number(state, "Manual εh (deg) — used without a profile", "s5_eps_h", 0.0) \
                .classes("col")
            dh_in = _drafted_number(state, "Manual dh (km) — used without a profile", "s5_dh", 0.5) \
                .classes("col")
        _show_horizon()

        # ---- 3. Common parameters ----------------------------------------------------
        _section_title("3. Common parameters")
        with ui.row().classes("w-full no-wrap q-gutter-sm items-center"):
            # "freq_ghz" is the same draft the ITU import fills with the transmit frequency;
            # the Step 3 selected frequency is the fallback.
            freq_in = _drafted_number(state, "Frequency (GHz)", "freq_ghz",
                                      _f(state.step3.selected_freq), fmt="%.4f", min=0.0).classes("col")
            p_in = _drafted_number(state, "Time percentage p (%)", "p_percent", 0.01,
                                   fmt="%.6f", min=0.0).classes("col")
        is_tx = ui.checkbox("Earth station is a transmitter", value=_draft(state, "s5_is_tx", True),
                            on_change=lambda e: state.put_draft("s5_is_tx", e.value))

        # ---- 4. Radio-climatic zone (Mode 1) ----------------------------------------
        _section_title("4. Radio-climatic zone (propagation mode 1)")
        zone_sel = ui.select(list(pc.ZONES), label="Radio-climatic zone",
                             value=_draft(state, "s5_zone", s5.auto_zone or pc.ZONES[0])) \
            .props("dense outlined").classes("w-full")
        zone_sel.on_value_change(lambda e: state.put_draft("s5_zone", e.value))
        zone_lbl = ui.label(s5.auto_zone_name or "").classes("text-caption text-grey-7")

        # async def detect_zone() -> None:
        #     busy = ui.notification("Detecting radio-climatic zone…", spinner=True, timeout=None)
        #     try:
        #         r = await run.io_bound(_detect_radio_climatic_zone, _f(lat_in.value), _f(lon_in.value))
        #     except Exception as ex:  # noqa: BLE001
        #         ui.notify(f"Zone detection failed: {ex}", type="warning", multi_line=True)
        #         return
        #     finally:
        #         busy.dismiss()


        async def detect_zone() -> None:
            busy = ui.notification("Detecting radio-climatic zone…", spinner=True, timeout=None)
            try:
                # FIX: Pass altitude as the third argument
                r = await run.io_bound(_detect_radio_climatic_zone,
                                    _f(lat_in.value), _f(lon_in.value), _f(s5.altitude_val))
            except Exception as ex:  # noqa: BLE001
                ui.notify(f"Zone detection failed: {ex}", type="warning", multi_line=True)
                return
            finally:
                busy.dismiss()
            
            # Update the UI with the results
            s5.auto_zone, s5.auto_zone_name = r["zone"], r["name"]
            if r["dc"] is not None:
                s5.auto_dc = _f(r["dc"])
                s5.dc_val = s5.auto_dc  # <-- This ensures the dc input box updates!
            zone_sel.value = r["zone"]
            zone_lbl.set_text(f"Detected zone {r['zone']}" + (f" — {r['name']}" if r["name"] else ""))
            ui.notify(f"Radio-climatic zone {r['zone']} detected.", type="positive")


            s5.auto_zone, s5.auto_zone_name = r["zone"], r["name"]
            if r["dc"] is not None:
                s5.auto_dc = _f(r["dc"])
            zone_sel.value = r["zone"]
            zone_lbl.set_text(f"Detected zone {r['zone']}" + (f" — {r['name']}" if r["name"] else ""))
            ui.notify(f"Radio-climatic zone {r['zone']} detected.", type="positive")

        if _rcz is not None:
            ui.button("Detect zone from location", icon="public", on_click=detect_zone) \
                .props("outline no-caps")
        else:
            ui.label("core.radio_climatic_zone is not available — choose the zone manually.") \
                .classes("text-caption text-grey-7")

        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            n0_in = _state_number("N0 (N-units)", s5, "n0_val").classes("col")
            be_field = _readonly_field("βe (deg)", _fmt(s5.be_val)).classes("col")
        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            dm_field = _readonly_field("dmin (km)", _fmt(s5.dm_val)).classes("col")
            dx1_field = _readonly_field("dmax1 (km)", _fmt(s5.dx1_val)).classes("col")

        def _signature() -> Tuple[Any, ...]:
            return (_f(lat_in.value), _f(lon_in.value), s5.altitude_val,
                    _f(freq_in.value), _f(p_in.value))

        def autofill() -> None:
            lat, f, p = _f(lat_in.value), _f(freq_in.value), _f(p_in.value)
            try:
                be = pc.beta_e(lat)
                n0 = pc.N0(lat)
                dm = pc.dmin(f, lat)
                dx1 = pc.dmax1(f, zone_sel.value, p if p > 0 else None)
            except Exception as ex:  # noqa: BLE001
                ui.notify(f"Auto-fill failed: {ex}", type="negative", multi_line=True)
                return
            s5.be_val, s5.dm_val, s5.dx1_val, s5.n0_val = be, dm, dx1, n0
            n0_in.value = n0  # also saved on s5 by the value-change handler
            be_field.set_value(_fmt(be))
            dm_field.set_value(_fmt(dm))
            dx1_field.set_value(_fmt(dx1))
            s5.autofill_sig = _signature()
            ui.notify(f"N0 = {n0:.2f}, βe = {be:.2f}°, dmin = {dm:.2f} km, dmax1 = {dx1:.2f} km.",
                      type="positive")

        ui.button("Auto-fill N0, βe, dmin, dmax1", icon="auto_fix_high", on_click=autofill) \
            .props("outline no-caps")

        # # ---- 5. Path composition (Mode 1) --------------------------------------------
        # _section_title("5. Path composition (propagation mode 1)")
        # gis_available = _pcomp is not None
        # use_gis = ui.checkbox(
        #     "Use GIS-calculated path composition (Recommended)",
        #     value=_draft(state, "s5_use_gis_path", gis_available),
        #     on_change=lambda e: state.put_draft("s5_use_gis_path", e.value))
        # gis_hint = ui.label().classes("text-caption text-grey-7")
        # with ui.row().classes("w-full no-wrap q-gutter-sm"):
        #     inland_in = _drafted_number(state, "Inland threshold (km from coast)", "s5_inland_km",
        #                                 50.0, min=0.0).classes("col") \
        #         .tooltip("Land closer than this to the coastline is coastal; only land beyond it "
        #                  "counts towards dlm. 0 = dlm equals dtm.")
        #     ui.label().classes("col")
        # ui.label("dtm = longest continuous land section, dlm = longest continuous inland section, "
        #          "dc = station-to-coast distance, land fraction = land share of the path.") \
        #     .classes("text-caption text-grey-7")
        # with ui.row().classes("w-full no-wrap q-gutter-sm"):
        #     dtm_in = _drafted_number(state, "dtm (km)", "s5_dtm", 0.0, min=0.0).classes("col") \
        #         .tooltip("Longest continuous land (inland + coastal) section of the path")
        #     dlm_in = _drafted_number(state, "dlm (km)", "s5_dlm", 0.0, min=0.0).classes("col") \
        #         .tooltip("Longest continuous inland section of the path (zone A2)")
        # with ui.row().classes("w-full no-wrap q-gutter-sm"):
        #     dc_in = _state_number("dc (km)", s5, "dc_val", min=0.0).classes("col") \
        #         .tooltip("Distance of the station from the coast")
        #     land_in = _drafted_number(state, "Land fraction", "s5_land_fraction", 0.0, min=0.0).classes("col")
        #     step_in = _drafted_number(state, "Iteration step (km)", "s5_step", 1.0, min=0.0).classes("col")

        # def _sync_gis_mode(_e: Any = None) -> None:
        #     on = bool(use_gis.value)
        #     for el in (dtm_in, dlm_in, dc_in, land_in):
        #         el.set_enabled(not on)
        #     inland_in.set_enabled(on)
        #     if on:
        #         gis_hint.set_text("dtm, dlm, dc and the land fraction are measured along every azimuth "
        #                           "when you press Calculate (rays of length dmax1 over the Natural Earth "
        #                           "data). The manual values below are only a fallback if the GIS "
        #                           "calculation is unavailable.")
        #     else:
        #         gis_hint.set_text("Manual entry. Defaults are the worst-case ITU values: all zero, 1 km "
        #                           "iteration step.")

        # use_gis.on_value_change(_sync_gis_mode)
        # _sync_gis_mode()




            # ---- 5. Path composition (Mode 1) --------------------------------------------
        _section_title("5. Path composition (propagation mode 1)")
        gis_available = _pcomp is not None
        use_gis = ui.checkbox(
            "Use GIS-calculated path composition (Recommended)",
            value=_draft(state, "s5_use_gis_path", gis_available),
            on_change=lambda e: state.put_draft("s5_use_gis_path", e.value))
        gis_hint = ui.label().classes("text-caption text-grey-7")
        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            inland_in = _drafted_number(state, "Inland threshold (km from coast)", "s5_inland_km",
                                        50.0, min=0.0).classes("col") \
                .tooltip("Land closer than this to the coastline is coastal; only land beyond it "
                        "counts towards dlm. 0 = dlm equals dtm.")
            ui.label().classes("col")
        ui.label("dtm = longest continuous land section, dlm = longest continuous inland section, "
                "dc = station-to-coast distance, land fraction = land share of the path.") \
            .classes("text-caption text-grey-7")
        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            dtm_in = _drafted_number(state, "dtm (km)", "s5_dtm", 0.0, min=0.0).classes("col") \
                .tooltip("Longest continuous land (inland + coastal) section of the path")
            dlm_in = _drafted_number(state, "dlm (km)", "s5_dlm", 0.0, min=0.0).classes("col") \
                .tooltip("Longest continuous inland section of the path (zone A2)")
        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            dc_in = _state_number("dc (km)", s5, "dc_val", min=0.0).classes("col") \
                .tooltip("Distance of the station from the coast")
            land_in = _drafted_number(state, "Land fraction", "s5_land_fraction", 0.0, min=0.0).classes("col")
            step_in = _drafted_number(state, "Iteration step (km)", "s5_step", 1.0, min=0.0).classes("col")

        # ---- TEMPORARY: GIS debug button ---------------------------------------------
        async def debug_gis() -> None:
            """Run the GIS path composition for ONE azimuth / 500 km and show the full result or
            the full traceback on screen and in the console (no silent fallback here)."""
            dbg_az, dbg_len = 90.0, 500.0
            lat, lon = _f(lat_in.value), _f(lon_in.value)
            thr = max(0.0, _f(inland_in.value, 50.0))
            header = (f"GIS DEBUG  station=({lat:.5f}, {lon:.5f})  azimuth={dbg_az:g} deg  "
                      f"length={dbg_len:g} km  inland threshold={thr:g} km")
            print("\n" + "=" * 78 + "\n" + header, flush=True)
            if _pcomp is None:
                msg = header + "\nFAILED: core.path_composition could not be imported (see the import error at start-up)."
                print(msg, flush=True)
                ui.notify(msg, type="negative", multi_line=True, timeout=20,
                          close_button=True, classes="multi-line-notification")
                return
            try:
                versions = []
                for mod in ("shapely", "pyproj", "numpy", "geopandas", "shapefile"):
                    try:
                        m = __import__(mod)
                        versions.append(f"{mod} {getattr(m, '__version__', getattr(m, 'VERSION', '?'))}")
                    except Exception as ie:  # noqa: BLE001
                        versions.append(f"{mod} NOT AVAILABLE ({type(ie).__name__}: {ie})")
                ver_txt = "Libraries: " + "; ".join(versions)
                print(ver_txt, flush=True)
                res = await run.io_bound(
                    lambda: _pcomp.calculate_path_params_by_azimuth(
                        lat, lon, [dbg_az], dbg_len, inland_threshold_km=thr))
                print("GIS DEBUG result:", res, flush=True)
                r = res.get(dbg_az) or next(iter(res.values()))
                msg = (f"{header}\nSUCCESS\n"
                       f"dtm = {r['dtm']:.3f} km\n"
                       f"dlm = {r['dlm']:.3f} km\n"
                       f"dc = {r['dc']:.3f} km\n"
                       f"land_fraction = {r['land_fraction']:.4f}\n"
                       f"Full dict: {res}\n{ver_txt}")
                if all(float(r[k]) == 0.0 for k in ("dtm", "dlm", "dc", "land_fraction")):
                    msg += "\nWARNING: all values are 0.0 - see the [path_composition] lines in the terminal."
                ui.notify(msg, type="positive", multi_line=True, timeout=20,
                          close_button=True, classes="multi-line-notification")
            except BaseException as ex:  # noqa: BLE001 - show ANY failure, never fall back silently
                tb = traceback.format_exc()
                msg = (f"{header}\nFAILED: {type(ex).__name__}: {ex}\n\n{tb}\n"
                       f"Python {sys.version.split()[0]}")
                print(msg, flush=True)
                ui.notify(msg, type="negative", multi_line=True, timeout=20,
                          close_button=True, classes="multi-line-notification")

        ui.button("🐞 Debug GIS Calculation", on_click=debug_gis) \
            .props("outline no-caps color=orange dense") \
            .tooltip("Temporary: runs the GIS path composition for azimuth 90 deg / 500 km from the "
                     "station above and shows the result or the full error.")

        def _sync_gis_mode(_e: Any = None) -> None:
            on = bool(use_gis.value)
            for el in (dtm_in, dlm_in, dc_in, land_in):
                el.set_enabled(not on)
            inland_in.set_enabled(on)
            if on:
                gis_hint.set_text("dtm, dlm, dc and the land fraction are measured along every azimuth "
                                "when you press Calculate (rays of length dmax1 over the Natural Earth "
                                "data). The manual values below are only a fallback if the GIS "
                                "calculation is unavailable.")
            else:
                gis_hint.set_text("Manual entry. Defaults are the worst-case ITU values: all zero, 1 km "
                                "iteration step.")
        use_gis.on_value_change(_sync_gis_mode)
        _sync_gis_mode()

        # ---- 6. Rain climatic zone (Mode 2) ------------------------------------------
        _section_title("6. Rain climatic zone (propagation mode 2)")
        r001_in = ui.input("R0.01 map path override (optional — blank = data_sources/R001.TXT)",
                           value=_draft(state, "s5_r001_path", "")).props("dense outlined").classes("w-full")
        r001_in.on_value_change(lambda e: state.put_draft("s5_r001_path", e.value))
        rain_sel = ui.select(list(pc.RAIN_ZONES), label="Rain climatic zone",
                             value=_draft(state, "s5_rain_zone", None), clearable=True) \
            .props("dense outlined").classes("w-full")
        rain_sel.on_value_change(lambda e: state.put_draft("s5_rain_zone", e.value))
        rain_lbl = ui.label().classes("text-caption text-grey-7")

        async def detect_rain() -> None:
            busy = ui.notification("Reading the R0.01 map…", spinner=True, timeout=None)
            try:
                r = await run.io_bound(_detect_rain_zone, _f(lat_in.value), _f(lon_in.value),
                                       (r001_in.value or "").strip() or None)
            except Exception as ex:  # noqa: BLE001
                ui.notify(f"Rain zone detection failed: {ex}", type="negative", multi_line=True)
                return
            finally:
                busy.dismiss()
            if r["zone"] not in pc.RAIN_ZONES:
                ui.notify(f"Detected zone {r['zone']} is not in Table 4 — choose one manually.",
                          type="warning")
                return
            rain_sel.value = r["zone"]
            rain_lbl.set_text(f"R0.01 = {r['r001']:.1f} mm/h → nearest zone {r['zone']} (approximate).")
            ui.notify(f"Rain zone {r['zone']} detected (R0.01 = {r['r001']:.1f} mm/h).", type="positive")

        ui.button("Detect rain zone from location", icon="water_drop", on_click=detect_rain) \
            .props("outline no-caps")
        with ui.row().classes("w-full no-wrap q-gutter-sm"):
            elev_in = _drafted_number(state, "Elevation angle (deg)", "s5_elevation", 0.0).classes("col")
            # Gx is the victim station's gain (Step 3 Gr) unless the user overrides it.
            gx_in = _drafted_number(state, "Gx (dBi)", "s5_gx", _f(wr.gr)).classes("col")

        ui.separator()

        # ---- accept ------------------------------------------------------------------
        def save() -> None:
            lat, lon, f, p = _f(lat_in.value), _f(lon_in.value), _f(freq_in.value), _f(p_in.value)
            step = _f(step_in.value, 1.0)
            problems: List[str] = []
            if not -90.0 <= lat <= 90.0:
                problems.append("latitude must be within −90…90°")
            if not 0.1 <= f <= 105.0:
                problems.append("frequency must be within 0.1…105 GHz")
            if p <= 0:
                problems.append("p must be > 0 %")
            if step <= 0:
                problems.append("iteration step must be > 0 km")
            if f < 60.0 and zone_sel.value not in pc.ZONES:
                problems.append("choose a radio-climatic zone")
            if problems:
                ui.notify("Cannot save: " + "; ".join(problems) + ".", type="negative", multi_line=True)
                return
            if s5.autofill_sig is not None and s5.autofill_sig != _signature():
                ui.notify("Location, frequency or p changed since the last auto-fill — "
                          "N0 / βe / dmin / dmax1 shown may be out of date.", type="warning")
            s5.ppd_params = {
                "latitude": lat, "longitude": lon, "altitude": s5.altitude_val,
                "frequency_GHz": f, "p_percent": p, "is_transmitter": bool(is_tx.value),
                "zone": zone_sel.value, "N0": s5.n0_val,
                "eps_h": _f(eps_in.value), "dh": _f(dh_in.value, 0.5),
                "dtm": _f(dtm_in.value), "dlm": _f(dlm_in.value), "dc": s5.dc_val,
                "land_fraction": _f(land_in.value), "step": step,
                "use_gis_path": bool(use_gis.value),
                "inland_threshold_km": max(0.0, _f(inland_in.value, 50.0)),
                "beta_e": s5.be_val, "dmin": s5.dm_val, "dmax1": s5.dx1_val,
                "horizon_profile": s5.horizon_profile,
                "rain_zone": rain_sel.value, "elevation_angle": _f(elev_in.value),
                "gx": _f(gx_in.value),
            }
            s5.ready = True
            dialog.close()
            ui.notify("Propagation parameters saved.", type="positive")
            on_saved()

        with ui.row().classes("w-full justify-end q-gutter-sm"):
            ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
            ui.button("Save parameters", on_click=save).props("unelevated no-caps color=primary")
    return dialog


# ===========================================================================
# Results
# ===========================================================================
DIAG_COLUMNS = [
    {"name": "azimuth", "label": "Az (°)", "field": "azimuth", "sortable": True, "align": "right"},
    {"name": "eps_h", "label": "εh (°)", "field": "eps_h", "sortable": True, "align": "right"},
    {"name": "dh", "label": "dh (km)", "field": "dh", "align": "right"},
    {"name": "dtm", "label": "dtm (km)", "field": "dtm", "sortable": True, "align": "right"},
    {"name": "dlm", "label": "dlm (km)", "field": "dlm", "sortable": True, "align": "right"},
    {"name": "dc", "label": "dc (km)", "field": "dc", "align": "right"},
    {"name": "land_fraction", "label": "Land frac.", "field": "land_fraction", "align": "right"},
    {"name": "gt", "label": "Gt (dBi)", "field": "gt", "sortable": True, "align": "right"},
    {"name": "lb_p", "label": "Lb(p) (dB)", "field": "lb_p", "sortable": True, "align": "right"},
    {"name": "km", "label": "Distance (km)", "field": "km", "sortable": True, "align": "right"},
    {"name": "ah", "label": "Ah (dB)", "field": "ah", "align": "right"},
    {"name": "iterations", "label": "Iter.", "field": "iterations", "align": "right"},
    {"name": "terminated_by", "label": "Terminated by", "field": "terminated_by", "align": "left"},
    {"name": "note", "label": "Note", "field": "note", "align": "left"},
]


def _round_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for k in ("eps_h", "dh", "dtm", "dlm", "dc", "land_fraction", "gt", "lb_p", "km", "ah"):
        if isinstance(out.get(k), (int, float)):
            out[k] = round(float(out[k]), 3)
    return out


@ui.refreshable
def step5_results(state: AppState) -> None:
    res = state.results
    profile = res.mode1_profile_result
    if not profile:
        if res.mode1_profile_errors:
            ui.label(f"Mode 1 failed for every azimuth ({len(res.mode1_profile_errors)} errors).") \
                .classes("text-negative")
            for az, msg in list(res.mode1_profile_errors)[:5]:
                ui.label(f"{az}°: {msg}").classes("text-caption text-negative")
        return

    kms = [km for _az, km in profile]
    ui.label(f"Propagation mode (1): {len(profile)} azimuths — min {min(kms):.1f} km, "
             f"max {max(kms):.1f} km, mean {sum(kms) / len(kms):.1f} km.").classes("text-positive")

    m2 = res.mode2_result
    if m2:
        ui.label(f"Propagation mode (2): dr = {m2['dr_km']:.1f} km, Δd = {m2['delta_d_km']:.1f} km "
                 f"(zone R(p) = {m2['R_p']:.1f} mm/h, terminated by {m2['terminated_by'].split(' (')[0]}).") \
            .classes("text-positive")
    elif res.mode2_error:
        ui.label(f"Propagation mode (2) not computed: {res.mode2_error}").classes("text-warning")
    combined = res.final_contour_result
    if combined:
        ckms = [km for _az, km in combined]
        ui.label(f"Combined contour max(mode 1, mode 2) (AP7 §1.6): min {min(ckms):.1f} km, "
                 f"max {max(ckms):.1f} km.").classes("text-positive")
    else:
        ui.label("No combined contour: it needs both a mode 1 profile and a mode 2 result.") \
            .classes("text-caption text-grey-7")

    # ---- polar plot -------------------------------------------------------------------
    # if plt is None:
    #     ui.label("matplotlib is not installed, so the polar plot is unavailable.").classes("text-warning")
    # else:
    #     azs = [az for az, _ in profile]
    #     theta = [math.radians(a) for a in azs] + [math.radians(azs[0])]
    #     r = kms + [kms[0]]
    #     with ui.pyplot(figsize=(5, 5)):
    #         ax = plt.subplot(111, projection="polar")
    #         ax.set_theta_zero_location("N")
    #         ax.set_theta_direction(-1)
    #         if combined:   # drawn first and wider, so it stays visible where it equals mode 1
    #             ctheta = [math.radians(a) for a, _ in combined] + [math.radians(combined[0][0])]
    #             cr = [km for _, km in combined] + [combined[0][1]]
    #             ax.plot(ctheta, cr, color="red", linewidth=2.6, label="Combined max(mode 1, mode 2)")
    #             ax.fill(ctheta, cr, color="red", alpha=0.12)
    #         ax.plot(theta, r, color="tab:blue", linewidth=1.3, label="Mode 1 (great circle)")
    #         ax.fill(theta, r, color="tab:blue", alpha=0.15)
    #         if m2:
    #             ring = [math.radians(a) for a in range(0, 361, 2)]
    #             ax.plot(ring, [m2["dr_km"]] * len(ring), "--", color="tab:orange",
    #                     label="Mode 2 (rain scatter) dr")
    #         ax.set_rmin(0)
    #         ax.set_title("Coordination distance (km) by azimuth", pad=14)
    #         ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.2), fontsize=8)


    # ---- polar plot -------------------------------------------------------------------
    if plt is None:
        ui.label("matplotlib is not installed, so the polar plot is unavailable.").classes("text-warning")
    else:
        azs = [az for az, _ in profile]
        theta = [math.radians(a) for a in azs] + [math.radians(azs[0])]
        r = kms + [kms[0]]
        with ui.pyplot(figsize=(5, 5)):
            ax = plt.subplot(111, projection="polar")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            
            # Plot Mode 1
            ax.plot(theta, r, color="tab:blue", linewidth=1.2, label="Mode 1 (great circle)")
            ax.fill(theta, r, color="tab:blue", alpha=0.15)
            
            if m2:
                # Plot Mode 2 as a circle
                dr_km = m2["dr_km"]
                ring = [math.radians(a) for a in range(0, 361, 2)]
                ax.plot(ring, [dr_km] * len(ring), "--", color="tab:orange",
                        linewidth=1.2, label=f"Mode 2 (rain scatter) dr = {dr_km:.1f} km")
                
                # Calculate and plot the combined contour: max(mode 1, mode 2) per azimuth
                finals = [max(d, dr_km) for d in kms]
                finals_closed = finals + [finals[0]]
                theta_closed = theta
                ax.plot(theta_closed, finals_closed, "-", color="tab:red",
                        linewidth=2.0, label="Final = max(mode 1, mode 2)")
                ax.fill(theta_closed, finals_closed, alpha=0.12, color="tab:red")
                
                ax.set_title("Coordination contour: max(Mode 1, Mode 2) per AP7 §1.6", pad=14)
            else:
                ax.set_title("Propagation mode (1) required-distance contour", pad=14)
            
            ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.2), fontsize=8)

    # ---- CSV + errors -------------------------------------------------------------------
    diag = res.mode1_profile_diag or []
    ui.button("Download CSV", icon="download",
              on_click=lambda: _download(diag_to_csv(diag), "coordination_distance_mode1.csv", "text/csv")) \
        .props("outline no-caps")

    if res.mode1_profile_errors:
        with ui.expansion(f"{len(res.mode1_profile_errors)} azimuth(s) could not be computed",
                          icon="warning").classes("w-full text-warning"):
            for az, msg in res.mode1_profile_errors:
                ui.label(f"{az}°: {msg}").classes("text-caption")

    # ---- per-azimuth diagnostics ----------------------------------------------------------
    _section_title("Per-azimuth diagnostics")
    ui.table(columns=DIAG_COLUMNS, rows=[_round_row(d) for d in diag], row_key="azimuth",
             pagination=10).props("dense flat bordered").classes("w-full")


# ===========================================================================
# Step 5 form
# ===========================================================================
@ui.refreshable
def step5_form(state: AppState) -> None:
    s5, wr, res = state.step5_inputs, state.wizard_results, state.results
    ui.label("AP7 Annex 1 (mode 1) and Annex 2 (mode 2)").classes("text-caption text-grey-7")

    @ui.refreshable
    def params_summary() -> None:
        p = s5.ppd_params
        if not (s5.ready and p):
            _status(False, "Propagation parameters not saved yet.")
            return
        hp = p.get("horizon_profile")
        _status(True, f"Parameters saved: {p['latitude']:.4f}°, {p['longitude']:.4f}°, "
                      f"{p['frequency_GHz']:g} GHz, p = {p['p_percent']:g} %, "
                      f"{'transmitter' if p['is_transmitter'] else 'receiver'}.")
        ui.label(
            f"Mode 1 zone {p.get('zone') or '—'}; horizon: "
            f"{'profile with ' + str(len(hp['profile'])) + ' azimuths' if hp else 'manual εh/dh'}; "
            f"rain zone {p.get('rain_zone') or '— (mode 2 skipped)'}; "
            f"path composition {'GIS-calculated per azimuth' if p.get('use_gis_path') else 'manual'}."
        ).classes("text-caption text-grey-7")

    dialog = build_params_dialog(state, on_saved=params_summary.refresh)

    ui.button("Open propagation parameters…", icon="tune", on_click=dialog.open) \
        .props("unelevated no-caps color=primary")
    params_summary()

    # ---- Lb(p) terms -----------------------------------------------------------------------
    missing = [name for name, v in (("Gt (Step 1)", wr.gt), ("Pt (Step 2)", wr.pt),
                                    ("Gr (Step 3)", wr.gr), ("Pr(p) (Step 3)", wr.pr_p)) if v is None]
    if missing:
        ui.label("Still needed for the calculation: " + ", ".join(missing) + ".").classes("text-warning")
    elif wr.gt_geometry:
        ui.label("Gt will be recomputed for each azimuth (Eq. 84-97 geometry from Step 1).") \
            .classes("text-caption text-grey-7")

    calc_btn = ui.button("Calculate coordination distance", icon="calculate") \
        .props("unelevated no-caps color=primary")

    async def calculate() -> None:
        if not (s5.ready and s5.ppd_params):
            ui.notify("Save the propagation parameters first.", type="negative")
            return
        if missing:
            ui.notify("Missing: " + ", ".join(missing) + ".", type="negative")
            return
        base_params = dict(s5.ppd_params)
        groups = [dict(g) for g in (state.itu_import.all_frequencies or [])]
        pt, gt, gr, pr_p = float(wr.pt), float(wr.gt), float(wr.gr), float(wr.pr_p)
        calc_btn.disable()
        busy = ui.notification("Preparing the frequency list…", spinner=True, timeout=None)
        try:
            plan, run_notes = await run.io_bound(build_frequency_plan, base_params, pt, gt, gr,
                                                 pr_p, wr.gt_geometry, groups)
            new_results: Dict[float, FrequencyResults] = {}
            lines: List[str] = []
            n_ok = 0
            for n, e in enumerate(plan, 1):
                tag = f"{e['key']:.4f} GHz ({_dir_label(e['direction'])})"
                busy.message = f"Calculating {tag} — frequency {n} of {len(plan)}…"
                fr = FrequencyResults(frequency_ghz=e["key"], direction=e["direction"])
                new_results[e["key"]] = fr
                if e["skip_reason"]:
                    fr.mode1_profile_errors = [("—", e["skip_reason"])]
                    lines.append(f"{tag}: skipped — {e['skip_reason']}")
                    continue
                try:
                    out = await run.io_bound(run_coordination, e["params"], e["pt"], e["gt"],
                                             e["gr"], e["pr_p"], e["geometry"], e["frequency_ghz"])
                except Exception as ex:  # noqa: BLE001 - one bad frequency must not lose the others
                    fr.mode1_profile_errors = [("—", str(ex))]
                    lines.append(f"{tag}: failed — {ex}")
                    continue
                fr.mode1_profile = out["profile"] or None
                fr.mode1_profile_diag = out["diag"]
                fr.mode1_profile_errors = out["errors"] or None
                kms = [km for _az, km in out["profile"]]
                if kms:
                    inputs = _entry_inputs(e)
                    inputs["path_composition"] = out.get("path_composition")
                    fr.mode1_result = {"max_km": max(kms), "min_km": min(kms),
                                       "n_azimuths": len(kms), "inputs": inputs}
                    n_ok += 1
                    lines.append(f"{tag}: max {max(kms):.1f} km"
                                 + (f" ({len(out['errors'])} azimuth(s) failed)" if out["errors"] else ""))
                else:
                    lines.append(f"{tag}: no azimuth could be computed")
                pc_info = out.get("path_composition") or {}
                if pc_info.get("error"):
                    lines.append(f"{tag}: GIS path composition unavailable - manual dtm/dlm/dc/land "
                                 f"fraction used ({pc_info['error']})")
                fr.mode2_result, fr.mode2_error = out["mode2"], out["mode2_error"]
                # AP7 S1.6 combined contour max(mode 1, mode 2), or None when mode 2 was not computed.
                fr.final_contour = out["final_contour"]
        except Exception as ex:  # noqa: BLE001
            ui.notify(f"Calculation failed: {ex}", type="negative", multi_line=True)
            return
        finally:
            busy.dismiss()
            calc_btn.enable()

        # Replace the previous run as a whole (frequencies that are no longer in the list must
        # not linger), then pick the frequency the single-frequency panels / Step 8 work on:
        # the saved one when it was computed (Step 8 re-runs with the saved parameters), else
        # the first frequency with a mode 1 contour.
        res.results_by_frequency = new_results
        ok_entries = [e for e in plan if new_results[e["key"]].mode1_profile]
        pick = next((e for e in ok_entries if e["is_base"]), None) or (ok_entries or plan)[0]
        res.active_frequency = pick["key"]
        state.border.analysis_rows = None
        state.border.analysis_warning = None
        state.bump_map_version()        # new mode 1 / mode 2 contours -> maps are out of date

        if n_ok:
            ui.notify(f"Done: {n_ok} of {len(plan)} frequenc{'y' if len(plan) == 1 else 'ies'} "
                      f"calculated.\n" + "\n".join(lines + run_notes), type="positive",
                      multi_line=True, classes="multi-line-notification", timeout=10000)
        else:
            ui.notify("No frequency could be computed - see the errors below.\n"
                      + "\n".join(lines + run_notes), type="negative", multi_line=True,
                      classes="multi-line-notification", timeout=10000)
        step5_results.refresh()

    calc_btn.on_click(calculate)

    ui.separator()
    step5_results(state)