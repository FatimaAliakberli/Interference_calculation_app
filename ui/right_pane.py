"""
Right Pane (Phase 6, split view in Part 4): results - polar contour plot, border I/N
table, mitigation.

Public entry point
------------------
``build_right_pane(state, on_map_refresh=None) -> refresh_all``
    Builds three refreshable sections (each wrapped in ``@ui.refreshable``):

    * ``results_grid``     - one cell per selected frequency, laid out by
                             ``middle_pane.build_split_layout`` (1: single column, 2: two
                             columns, 3+: a grid). Each cell holds that frequency's
                             - polar plot: Mode 1 (blue solid), Mode 2 (green dashed ring, it is
                               azimuth independent) and the combined max(mode 1, mode 2)
                               contour (red solid);
                             - per-azimuth Azimuth / Required / Border / I/N / Status table,
                               rows tinted green (PASS) or red (FAIL).
    * ``analysis_panel``   - I/N limit, home country, terrain-diffraction options and the
                             "Analyze borders" button (the NiceGUI port of Streamlit Step 7).
                             One press analyzes EVERY selected frequency that has a contour,
                             each with its own Pt / Gt / Gx / Pr(p) / limit, and stores the rows
                             in ``FrequencyResults.border_rows``.
    * ``mitigation_panel`` - only when some azimuth FAILs: sliders for Pt, satellite
                             longitude and antenna diameter. Mitigation acts on the ACTIVE
                             frequency only (Step 8 re-runs with the saved parameters).

Why the sections refresh by themselves
--------------------------------------
NiceGUI refreshables do not notice state changes on their own, so a light poll
(``ui.timer``) compares a signature of the relevant ``AppState`` fields and refreshes
every section when it changes. The Step 5 pane therefore needs no edits. The poll never
rebuilds the map; the map is rebuilt through ``on_map_refresh`` (the callable returned by
``build_middle_pane``) after "Analyze borders" and after a mitigation.

Mitigation
----------
While a slider is dragged only a cheap preview for the worst azimuth is updated
(same maths as the Streamlit panel). When a slider is RELEASED the mitigation is committed
and everything is recomputed in a worker thread:

    new Pt / satellite longitude / diameter  ->  Mode 1 + Mode 2 (``run_coordination``)
    -> combined contour -> I/N at the already-known border distances (no border re-download)
    -> table, plot and map refresh.

The commit is all-or-nothing: ``wizard_results.pt``, ``wizard_results.gt_geometry``,
``step1.sat_lon`` and ``border.applied_*`` are only changed if the recompute succeeded.
The slider ranges are fixed when the analysis is run, so they do not shift after a commit.

There is no ``state.mitigation``; the mitigation values live in
``state.border.applied_pt / applied_sat_lon / applied_diameter`` (see ui/state.py).
"""
from __future__ import annotations

import asyncio
import inspect
import math
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from nicegui import run, ui

from core import propagation_calculator as pc
from ui.left_pane import _f, _section_title
from ui.middle_pane import (
    border_rows_for,
    build_split_layout,
    effective_frequencies,
    frequency_label,
    result_for_frequency,
)
from ui.state import AppState, FrequencyResults, normalize_dem_path, resolve_dem_path
from ui.step5_pane import _gt_at_azimuth, _profile_dh, params_for_frequency, run_coordination

try:  # matplotlib is only needed for the polar plot
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - depends on the environment
    plt = None

try:  # border-distance lookup (Natural Earth download) - optional, manual km is the fallback
    from core import country_borders as borders
except ImportError:  # pragma: no cover
    borders = None

try:  # geodesic forward calc, only needed for the terrain-diffraction recheck
    from pyproj import Geod
    _GEOD = Geod(ellps="WGS84")
except ImportError:  # pragma: no cover
    _GEOD = None

MaybeAwaitable = Optional[Callable[[], Any]]

# Table columns (name = row key). Row tint comes from the hidden "_bg" field.
TABLE_COLUMNS = [
    {"name": "azimuth", "label": "Azimuth (°)", "field": "azimuth", "align": "right", "sortable": True},
    {"name": "required", "label": "Required dist (km)", "field": "required", "align": "right"},
    {"name": "border", "label": "Border dist (km)", "field": "border", "align": "right"},
    {"name": "in_ratio", "label": "I/N (dB)", "field": "in_ratio", "align": "right"},
    {"name": "status", "label": "Status", "field": "status", "align": "left"},
]
TABLE_ROW_SLOT = r"""
<q-tr :props="props" :style="{ background: props.row._bg }">
  <q-td v-for="col in props.cols" :key="col.name" :props="props">{{ col.value }}</q-td>
</q-tr>
"""
BG_PASS = "rgba(76, 175, 80, 0.22)"
BG_FAIL = "rgba(244, 67, 54, 0.22)"

# Nominal diameter used to back out an implied wavelength when the Gt geometry recorded none
# (same fallback as the Streamlit mitigation panel).
FALLBACK_DIAMETER_M = 3.0


# ===========================================================================
# Pure helpers (no UI; the worker-thread ones touch no AppState either)
# ===========================================================================
def combine_contours(profile: Optional[List[Tuple[float, float]]],
                     mode2: Optional[Dict[str, Any]]) -> Optional[List[Tuple[float, float]]]:
    """AP7 S1.6: required distance per azimuth = max(mode 1, mode 2). Mode 2 is azimuth
    independent, so its single distance ``dr_km`` is compared with every mode 1 value.
    None when either input is missing."""
    if not profile or not mode2:
        return None
    d2 = float(mode2["dr_km"])
    return [(az, max(d1, d2)) for az, d1 in profile]


def sync_combined_contour(state: AppState) -> None:
    """Keep ``results.final_contour_result`` equal to the combined contour. Assigns only on a
    real change, so callers (and the change-detection poll) see a stable object."""
    new = combine_contours(state.results.mode1_profile_result, state.results.mode2_result)
    if new != state.results.final_contour_result:
        state.results.final_contour_result = new
        state.bump_map_version()


def _derived_limit(state: AppState) -> Optional[Dict[str, Any]]:
    """I/N limit = Pr(p) - N with N from the Step 3 Te / B (Streamlit Step 7)."""
    wr = state.wizard_results
    if wr.pr_p is None or wr.final_te is None or wr.final_b is None:
        return None
    return pc.calculate_IN_ratio(wr.pr_p, wr.final_te, wr.final_b)


def _current_limit(state: AppState) -> Optional[float]:
    if state.drafts.get("s8_limit_manual_on"):
        return _f(state.drafts.get("s8_limit_manual"))
    derived = _derived_limit(state)
    return None if derived is None else float(derived["IN_ratio_dB"])


def _missing_inputs(state: AppState) -> List[str]:
    wr = state.wizard_results
    p = state.step5_inputs.ppd_params
    missing = []
    if wr.pt is None:
        missing.append("Pt (Step 2)")
    if wr.gr is None:
        missing.append("Gx / victim antenna gain (Step 3)")
    if wr.pr_p is None:
        missing.append("Pr(p) (Step 3)")
    if wr.final_te is None or wr.final_b is None:
        missing.append("Te and B (Step 3)")
    if not p or p.get("latitude") is None or p.get("longitude") is None:
        missing.append("propagation parameters (Step 5)")
    return missing


def _horizon_at(params: Dict[str, Any], az: float) -> Tuple[float, float]:
    """(eps_h, dh) for an azimuth, resolved exactly as step5_pane.compute_mode1_profile does."""
    horizon = (params.get("horizon_profile") or {}).get("profile") or {}
    entry = horizon.get(int(round(az))) or {}
    eps_h = _f(entry.get("elevation_angle"), _f(params.get("eps_h")))
    dh = _profile_dh(entry) if entry else None
    if dh is None:
        dh = _f(params.get("dh"), 0.5)
    return eps_h, dh


def row_status(row: Dict[str, Any]) -> str:
    """Plain status of one border-analysis row (before any diffraction recheck)."""
    if not row.get("overlap"):
        return "No overlap" if row.get("border_km") is not None else "No border found"
    in_res = row.get("in_result")
    if not in_res or "IN_ratio_dB" not in in_res:
        return "Error"
    return "FAIL" if in_res["IN_ratio_dB"] > row["limit_db"] else "PASS"


def effective_status(row: Dict[str, Any]) -> str:
    """row_status, except that a FAIL turned into a PASS by the ITU-R P.526 diffraction
    recheck shows as 'PASS (diffraction)'. The mitigation panel acts on this status."""
    base = row_status(row)
    diff = row.get("diffraction_result")
    if base == "FAIL" and diff and diff.get("diffraction_available"):
        return "PASS (diffraction)" if diff["adjusted_status"] == "PASS" else "FAIL"
    return base


def _find_borders(lat: float, lon: float, azimuths: List[float], home_country: str,
                  manual_km: Dict[Any, Any]) -> Tuple[Dict[float, Optional[float]], Optional[str]]:
    """Distance to the nearest foreign border per azimuth. Falls back to the user's manual
    km values (and returns the reason) when the border data cannot be loaded."""
    try:
        if borders is None:
            raise ImportError("core.country_borders is not available")
        found = borders.find_border_distances(
            lat=lat, lon=lon, azimuths_deg=azimuths,
            home_country_name=home_country, max_distance_km=2000.0)
        return dict(found), None
    except (ImportError, RuntimeError) as e:
        return {az: manual_km.get(az) for az in azimuths}, str(e)


def _analyze_rows(params: Dict[str, Any], contour: List[Tuple[float, float]],
                  border_km_by_az: Dict[float, Optional[float]], gt_by_az: Dict[float, float],
                  gt_default: Optional[float], pt: float, gx: float, te: float, b: float,
                  limit: float, diff_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """I/N at the border for every azimuth whose required distance reaches the border.

    Same rows as Streamlit Step 7 ("limit_db" and "diffraction_result" included, which the
    map overlay needs). Worker-thread safe.
    """
    rows: List[Dict[str, Any]] = []
    lat, lon = params["latitude"], params["longitude"]
    for az, required_km in contour:
        border_km = border_km_by_az.get(az)
        gt_alpha = gt_by_az.get(float(az), gt_default)
        overlap = border_km is not None and required_km > border_km
        in_row: Optional[Dict[str, Any]] = None
        diffraction_result: Optional[Dict[str, Any]] = None

        if overlap:
            try:
                if gt_alpha is None:
                    raise ValueError("no Gt available for this azimuth")
                eps_h, dh = _horizon_at(params, az)
                in_row = pc.calculate_IN_at_distance(
                    Pt=pt, Gt_dBi=gt_alpha, Gx_dBi=gx, Te_K=te, B_Hz=b,
                    distance_km=border_km, frequency_GHz=params["frequency_GHz"],
                    p_percent=params["p_percent"], zone=params["zone"],
                    dtm_km=params["dtm"], dlm_km=params.get("dlm", 0.0),
                    is_transmitter=params["is_transmitter"], latitude_deg=lat,
                    dc_km=params.get("dc", 0.0),
                    horizon_elevation_deg=eps_h, horizon_distance_km=dh)

                # ITU-R P.526 terrain-diffraction recheck, FAIL azimuths only.
                if in_row["IN_ratio_dB"] > limit and diff_cfg.get("enabled") and _GEOD is not None:
                    b_lon, b_lat, _ = _GEOD.fwd(lon, lat, az, border_km * 1000.0)
                    calc = pc.calculate_path_loss_with_diffraction(
                        frequency_GHz=params["frequency_GHz"], distance_km=border_km,
                        latitude_deg=lat, es_lat=lat, es_lon=lon,
                        es_height_m=diff_cfg["es_height_m"],
                        target_lat=b_lat, target_lon=b_lon,
                        target_height_m=diff_cfg["target_height_m"],
                        dem_path=diff_cfg["dem_path"])
                    adj_I = pt + gt_alpha + gx - calc["L_path_dB"]
                    adj_IN = adj_I - in_row["N_dBW"]
                    diffraction_result = {
                        "diffraction_available": calc["diffraction_available"],
                        "diffraction_loss_dB": calc["diffraction_loss_dB"],
                        "diffraction_warning": calc["diffraction_warning"],
                        "base_loss_dB": calc["base_loss_dB"],
                        "adjusted_L_path_dB": calc["L_path_dB"],
                        "adjusted_I_dBW": adj_I,
                        "adjusted_IN_ratio_dB": adj_IN,
                        "adjusted_status": "PASS" if adj_IN <= limit else "FAIL",
                        "border_lat": b_lat, "border_lon": b_lon,
                    }
            except Exception as e:  # noqa: BLE001 - one bad azimuth must not kill the table
                in_row = {"error": str(e)}

        rows.append({
            "azimuth_deg": az, "required_km": required_km, "border_km": border_km,
            "overlap": overlap, "Gt_alpha_dBi": gt_alpha, "in_result": in_row,
            "limit_db": limit, "diffraction_result": diffraction_result,
        })
    return rows


def _gt_map(diag: Optional[List[Dict[str, Any]]]) -> Dict[float, float]:
    """{azimuth: Gt(alpha)} from the Step 5 per-azimuth diagnostics."""
    return {float(d["azimuth"]): d["gt"] for d in (diag or []) if d.get("gt") is not None}


def _border_analysis_worker(params, contour, gt_by_az, gt_default, pt, gx, te, b, limit,
                            home_country, manual_km, diff_cfg) -> Dict[str, Any]:
    """'Analyze borders': look up the border distances, then build the rows."""
    azimuths = [az for az, _ in contour]
    border_km, warning = _find_borders(params["latitude"], params["longitude"], azimuths,
                                       home_country, manual_km)
    rows = _analyze_rows(params, contour, border_km, gt_by_az, gt_default, pt, gx, te, b,
                         limit, diff_cfg)
    return {"rows": rows, "warning": warning}


def _mitigation_worker(params, pt, gt, gr, pr_p, geometry, gx, te, b, limit,
                       border_km_by_az, diff_cfg) -> Dict[str, Any]:
    """Recompute Mode 1 + Mode 2 for trial parameters, combine, and redo the border rows at
    the already-known border distances. Touches no AppState: the caller commits on success."""
    out = run_coordination(params, pt, gt, gr, pr_p, geometry)
    if not out["profile"]:
        return {"ok": False, "out": out, "rows": None}
    contour = combine_contours(out["profile"], out["mode2"]) or out["profile"]
    rows = _analyze_rows(params, contour, border_km_by_az, _gt_map(out["diag"]), gt, pt, gx,
                         te, b, limit, diff_cfg)
    return {"ok": True, "out": out, "rows": rows}


def _store_coordination(state: AppState, out: Dict[str, Any]) -> None:
    """Write a run_coordination() result into state (same fields Step 5's button sets)."""
    res = state.results
    res.mode1_profile_result = out["profile"] or None
    res.mode1_profile_diag = out["diag"]
    res.mode1_profile_errors = out["errors"] or None
    kms = [km for _az, km in out["profile"]]
    res.mode1_result = ({"max_km": max(kms), "min_km": min(kms), "n_azimuths": len(kms)}
                        if kms else None)
    res.mode2_result, res.mode2_error = out["mode2"], out["mode2_error"]
    res.final_contour_result = None
    res.blind_in_result = None
    sync_combined_contour(state)
    state.bump_map_version()            # mode 1 / mode 2 results replaced


def _diff_cfg(state: AppState) -> Dict[str, Any]:
    d = state.drafts
    return {
        "enabled": bool(d.get("s8_diffraction_enabled", True)) and _GEOD is not None,
        "es_height_m": _f(d.get("s8_es_height_m"), 10.0),
        "target_height_m": _f(d.get("s8_target_height_m"), 0.0),
        "dem_path": state.border.effective_dem_path,   # sentinel -> real bundled path
    }


def _limit_for(state: AppState, pr_p: Optional[float]) -> Optional[float]:
    """I/N limit for one frequency: the manual limit, else Pr(p) - N with that frequency's
    own Pr(p) and the Step 3 Te / B."""
    if state.drafts.get("s8_limit_manual_on"):
        return _f(state.drafts.get("s8_limit_manual"))
    wr = state.wizard_results
    if pr_p is None or wr.final_te is None or wr.final_b is None:
        return None
    return float(pc.calculate_IN_ratio(pr_p, wr.final_te, wr.final_b)["IN_ratio_dB"])


def _frequency_context(state: AppState, fr: FrequencyResults) -> Dict[str, Any]:
    """The inputs the border analysis of one frequency needs (params, Pt, default Gt, Gx,
    Pr(p)). The base frequency (or results without recorded inputs) uses the Step 1-3 values as
    they are now - including any Step 8 mitigation; every other frequency uses the values it
    was calculated with (``mode1_result["inputs"]``, written by Step 5)."""
    wr = state.wizard_results
    base = dict(state.step5_inputs.ppd_params)
    inp = fr.mode1_result.get("inputs") if isinstance(fr.mode1_result, dict) else None
    if not inp or inp.get("is_base"):
        return {"params": base, "pt": float(wr.pt), "gt": wr.gt, "gx": float(wr.gr),
                "pr_p": float(wr.pr_p)}
    gx = inp.get("gx") if inp.get("gx") is not None else inp.get("gr")
    params = params_for_frequency(base, inp.get("frequency_ghz", fr.frequency_ghz),
                                  fr.direction, inp.get("gx"))
    return {"params": params, "pt": float(inp["pt"]), "gt": inp.get("gt_default"),
            "gx": float(gx if gx is not None else wr.gr), "pr_p": float(inp["pr_p"])}


def _multi_border_worker(jobs: List[Dict[str, Any]], home_country: str,
                         manual_km: Dict[Any, Any]) -> Dict[int, Dict[str, Any]]:
    """'Analyze borders' for several frequencies in one worker thread. The border distances
    only depend on the station and the azimuths, so they are looked up once and shared.
    Touches no AppState. Returns {job index: {"rows": [...], "warning": str | None}}."""
    out: Dict[int, Dict[str, Any]] = {}
    borders_cache: Dict[Tuple[Any, ...], Tuple[Dict[float, Optional[float]], Optional[str]]] = {}
    for job in jobs:
        params, contour = job["params"], job["contour"]
        azimuths = [az for az, _ in contour]
        key = (round(float(params["latitude"]), 6), round(float(params["longitude"]), 6), tuple(azimuths))
        if key not in borders_cache:
            borders_cache[key] = _find_borders(params["latitude"], params["longitude"], azimuths,
                                               home_country, manual_km)
        border_km, warning = borders_cache[key]
        rows = _analyze_rows(params, contour, border_km, job["gt_by_az"], job["gt_default"],
                             job["pt"], job["gx"], job["te"], job["b"], job["limit"], job["diff_cfg"])
        out[job["idx"]] = {"rows": rows, "warning": warning}
    return out


def _table_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        status = effective_status(r)
        in_res = r.get("in_result")
        out.append({
            "azimuth": r["azimuth_deg"],
            "required": round(r["required_km"], 2),
            "border": round(r["border_km"], 2) if r["border_km"] is not None else None,
            "in_ratio": (round(in_res["IN_ratio_dB"], 2)
                         if in_res and "IN_ratio_dB" in in_res else None),
            "status": status,
            "_bg": BG_FAIL if status == "FAIL" else BG_PASS if status.startswith("PASS") else "",
        })
    return out


async def _call_maybe_async(cb: MaybeAwaitable) -> None:
    if cb is None:
        return
    result = cb()
    if inspect.isawaitable(result):
        await result


# ===========================================================================
# UI
# ===========================================================================
def build_right_pane(state: AppState, on_map_refresh: MaybeAwaitable = None) -> Callable[[], None]:
    """Build the Right Pane contents. Call inside the Right Pane's scroll area.

    ``on_map_refresh``: callable (sync or async) that rebuilds the map, e.g. the function
    returned by ``build_middle_pane``. Returns ``refresh_all`` to refresh every section.
    """
    res, wr, bd = state.results, state.wizard_results, state.border

    # Mitigation slider ranges, fixed when an analysis is run (cleared when it is invalidated).
    baseline: Dict[str, float] = {}
    busy = {"on": False, "pending": False}

    # ---- per-frequency cell: polar plot + I/N table --------------------------------------
    def render_plot(fr: Optional[FrequencyResults], compact: bool) -> None:
        if fr is None:
            ui.label("No results for this frequency yet - run Step 5 (Calculate coordination distance)."
                     ).classes("text-grey-7")
            return
        mode1 = fr.mode1_profile
        if not mode1:
            errs = fr.mode1_profile_errors
            first = errs[0] if errs else None
            reason = first[1] if isinstance(first, (list, tuple)) and len(first) > 1 else first
            ui.label(f"No contour for this frequency: {reason}" if reason
                     else "Run Step 5 (coordination distance) to see the contour.").classes("text-grey-7")
            return

        m2 = fr.mode2_result
        combined = fr.final_contour or combine_contours(mode1, m2)
        kms = [km for _az, km in mode1]
        summary = f"Mode 1: min {min(kms):.1f} km, max {max(kms):.1f} km."
        if m2:
            summary += f" Mode 2: dr = {float(m2['dr_km']):.1f} km."
        if combined:
            summary += f" Combined max: {max(km for _az, km in combined):.1f} km."
        ui.label(summary).classes("text-caption")
        if fr.mode2_error and not m2:
            ui.label(f"Mode 2 not computed: {fr.mode2_error}").classes("text-caption text-warning")

        if plt is None:
            ui.label("matplotlib is not installed, so the polar plot is unavailable.").classes("text-warning")
            return

        def closed(points: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
            theta = [math.radians(a) for a, _ in points] + [math.radians(points[0][0])]
            return theta, [km for _, km in points] + [points[0][1]]

        size = 3.2 if compact else 4.5
        with ui.pyplot(figsize=(size, size)):
            ax = plt.subplot(111, projection="polar")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            if combined:   # drawn first and wider, so it stays visible where it equals mode 1
                t, r = closed(combined)
                ax.plot(t, r, color="red", linewidth=2.6, label="Combined max(1, 2)")
            t, r = closed(mode1)
            ax.plot(t, r, color="blue", linewidth=1.3, label="Mode 1")
            if m2:
                ring = [math.radians(a) for a in range(0, 361, 2)]
                ax.plot(ring, [float(m2["dr_km"])] * len(ring), "--", color="green",
                        linewidth=1.5, label="Mode 2 (dr)")
            ax.set_rmin(0)
            ax.set_title("Coordination distance (km)" if compact else "Coordination distance (km) by azimuth",
                         pad=14, fontsize=9 if compact else 12)
            ax.legend(loc="lower left", bbox_to_anchor=(-0.2, -0.22), fontsize=7 if compact else 8)

    def render_table(fr: Optional[FrequencyResults], compact: bool) -> None:
        rows = border_rows_for(state, fr)
        if not rows:
            ui.label("Run “Analyze borders” to see the per-azimuth I/N results."
                     if fr is not None and fr.active_contour() else
                     "No I/N table: this frequency has no contour.").classes("text-grey-7")
            return
        stats = [effective_status(r) for r in rows]
        n_fail = stats.count("FAIL")
        n_pass = sum(1 for s_ in stats if s_.startswith("PASS"))
        ui.label(f"{len(rows)} azimuths: {n_fail} FAIL, {n_pass} PASS, "
                 f"{len(rows) - n_fail - n_pass} not applicable.").classes(
            "text-caption " + ("text-negative" if n_fail else "text-positive"))
        table = ui.table(columns=TABLE_COLUMNS, rows=_table_rows(rows), row_key="azimuth",
                         pagination=8 if compact else 12).props("dense flat bordered").classes("w-full")
        table.add_slot("body", TABLE_ROW_SLOT)
        ui.label("Limit: " + f"{rows[0]['limit_db']:.2f} dB. "
                 "“PASS (diffraction)” = FAIL on free space + gases, PASS with terrain diffraction."
                 ).classes("text-caption text-grey-7")

    @ui.refreshable
    def results_grid() -> None:
        freqs = effective_frequencies(state)
        compact = len(freqs) > 1
        for freq, cell in build_split_layout(freqs, state):
            with cell:
                fr = result_for_frequency(state, freq)
                render_plot(fr, compact)
                ui.separator()
                _section_title("I/N at the border")
                render_table(fr, compact)

    def _analysis_targets() -> List[Tuple[Optional[float], FrequencyResults]]:
        """Shown frequencies that have a contour to analyze (each result set at most once)."""
        out: List[Tuple[Optional[float], FrequencyResults]] = []
        for freq in effective_frequencies(state):
            fr = result_for_frequency(state, freq)
            if fr is not None and fr.active_contour() and all(fr is not o for _f_, o in out):
                out.append((freq, fr))
        return out

    # ---- border analysis controls ----------------------------------------------------------
    @ui.refreshable
    def analysis_panel() -> None:
        targets = _analysis_targets()
        if not targets:
            ui.label("Border analysis needs a full azimuth contour from Step 5.").classes("text-grey-7")
            return
        contour = targets[0][1].active_contour()      # same azimuth grid for every frequency
        missing = _missing_inputs(state)
        if missing:
            ui.label("Complete first: " + ", ".join(missing) + ".").classes("text-warning")
            return

        derived = _derived_limit(state)
        d = state.drafts
        d.setdefault("s8_limit_manual_on", False)
        d.setdefault("s8_limit_manual", round(float(derived["IN_ratio_dB"]), 2))
        d.setdefault("s8_diffraction_enabled", True)
        d.setdefault("s8_es_height_m", 10.0)
        d.setdefault("s8_target_height_m", 0.0)

        ui.label(f"Derived I/N limit = Pr(p) − N = {wr.pr_p:.2f} dBW − ({derived['N_dBW']:.2f} dBW) "
                 f"= {derived['IN_ratio_dB']:.2f} dB").classes("text-caption")
        with ui.row().classes("w-full items-center no-wrap"):
            ui.checkbox("Manual limit").bind_value(d, "s8_limit_manual_on")
            ui.number("I/N limit (dB)", format="%.2f").props("dense outlined") \
                .bind_value(d, "s8_limit_manual").bind_enabled_from(d, "s8_limit_manual_on")
        ui.input("Home country (excluded from the border search)").props("dense outlined") \
            .classes("w-full").bind_value(bd, "home_country")

        with ui.expansion("Terrain diffraction recheck (ITU-R P.526)", icon="terrain").classes("w-full"):
            diff_cb = ui.checkbox("Recheck FAIL azimuths against a DEM").bind_value(d, "s8_diffraction_enabled")
            if _GEOD is None:
                diff_cb.disable()
                diff_cb.tooltip("pyproj is not installed")
            dem_seen = {"v": resolve_dem_path(bd.dem_path)}

            def _dem_path_changed(e) -> None:
                # Only a real change: the initial binding sync fires this once with the same value.
                if e.value != dem_seen["v"]:
                    dem_seen["v"] = e.value
                    state.bump_map_version()         # the maps' DEM layer depends on it

            ui.input("DEM GeoTIFF path").props("dense outlined").classes("w-full") \
                .bind_value(bd, "dem_path", forward=normalize_dem_path, backward=resolve_dem_path) \
                .bind_enabled_from(d, "s8_diffraction_enabled") \
                .on_value_change(_dem_path_changed)
            ui.number("Earth station height above ground (m)", min=0, format="%.1f") \
                .props("dense outlined").classes("w-full") \
                .bind_value(d, "s8_es_height_m").bind_enabled_from(d, "s8_diffraction_enabled")
            ui.number("Border point height above ground (m)", min=0, format="%.1f") \
                .props("dense outlined").classes("w-full") \
                .bind_value(d, "s8_target_height_m").bind_enabled_from(d, "s8_diffraction_enabled")

        analyze_btn = ui.button(
            "Analyze borders" if len(targets) == 1 else f"Analyze borders ({len(targets)} frequencies)",
            icon="public").props("unelevated no-caps color=primary")

        async def analyze() -> None:
            targets_now = _analysis_targets()
            if not targets_now or _missing_inputs(state):
                ui.notify("Inputs changed - complete the earlier steps first.", type="negative")
                return
            jobs: List[Dict[str, Any]] = []
            skipped: List[str] = []
            for freq, fr in targets_now:
                ctx = _frequency_context(state, fr)
                limit = _limit_for(state, ctx["pr_p"])
                if limit is None:
                    skipped.append(frequency_label(state, freq))
                    continue
                jobs.append({
                    "idx": len(jobs), "freq": freq, "fr": fr, "params": ctx["params"],
                    "contour": list(fr.active_contour()), "gt_by_az": _gt_map(fr.mode1_profile_diag),
                    "gt_default": ctx["gt"], "pt": ctx["pt"], "gx": ctx["gx"],
                    "te": float(wr.final_te), "b": float(wr.final_b), "limit": limit,
                    "diff_cfg": _diff_cfg(state)})
            if skipped:
                ui.notify("No I/N limit for " + ", ".join(skipped) + " (Te, B and Pr(p) are needed).",
                          type="warning")
            if not jobs:
                return
            analyze_btn.disable()
            note = ui.notification("Analyzing borders…", spinner=True, timeout=None)
            try:
                # Plain-data copies only: the worker thread must not touch AppState.
                plain = [{k: v for k, v in j.items() if k not in ("fr", "freq")} for j in jobs]
                out = await run.io_bound(_multi_border_worker, plain, bd.home_country, dict(bd.manual_km))
            except Exception as ex:  # noqa: BLE001
                ui.notify(f"Border analysis failed: {ex}", type="negative", multi_line=True)
                return
            finally:
                note.dismiss()
                analyze_btn.enable()
            warning: Optional[str] = None
            for j in jobs:
                j["fr"].border_rows = out[j["idx"]]["rows"]
                warning = warning or out[j["idx"]]["warning"]
            bd.analysis_warning = warning
            active = res.active()
            if active is not None and active.border_rows is not None:
                bd.analysis_rows = active.border_rows      # the mitigation panel works on this one
            state.bump_map_version()                       # border rows changed
            baseline.clear()          # fresh analysis -> fresh slider ranges
            refresh_all()
            await _call_maybe_async(on_map_refresh)

        analyze_btn.on_click(analyze)

        if bd.analysis_warning:
            ui.label(f"Automatic border download unavailable: {bd.analysis_warning}. "
                     "Enter border distances below, then analyze again.").classes("text-caption text-warning")
            with ui.expansion("Manual border distances (km)", icon="edit").classes("w-full"):
                with ui.grid(columns=3).classes("w-full"):
                    for az, _km in contour:
                        ui.number(f"{az:g}°", value=bd.manual_km.get(az) or 0.0, min=0, format="%.2f",
                                  on_change=lambda e, a=az: bd.manual_km.__setitem__(
                                      a, e.value if e.value and e.value > 0 else None)
                                  ).props("dense outlined")

    # ---- mitigation ------------------------------------------------------------------------
    @ui.refreshable
    def mitigation_panel() -> None:
        rows = bd.analysis_rows
        if not rows:
            baseline.clear()
            return
        fails = [r for r in rows if effective_status(r) == "FAIL"]
        if not fails:
            ui.label("No azimuth both crosses the border and fails the I/N limit.").classes("text-positive")
            return

        worst = max(fails, key=lambda r: r["in_result"]["IN_ratio_dB"])
        limit = worst["limit_db"]
        bd.mitigation_az = worst["azimuth_deg"]
        _section_title("Mitigation" if res.active_frequency in (None, 0.0) else
                       f"Mitigation — {frequency_label(state, res.active_frequency)} (active frequency)")
        ui.label(f"Worst azimuth {worst['azimuth_deg']:g}°: I/N = {worst['in_result']['IN_ratio_dB']:.2f} dB "
                 f"(limit {limit:.2f} dB) — FAIL by {worst['in_result']['IN_ratio_dB'] - limit:.2f} dB."
                 ).classes("text-negative")

        geometry = wr.gt_geometry
        if not geometry:
            ui.label("Satellite longitude and antenna diameter need Gt from the Step 1 Eq. 84-97 "
                     "calculator; only Pt is available with a manually entered Gt."
                     ).classes("text-caption text-grey-7")

        # Current values, and the wavelength used to scale D/lambda from the diameter.
        pt_now = float(wr.pt)
        es_lon = state.step1.es_lon if state.step1.es_lon is not None else state.step5_inputs.ppd_params["longitude"]
        sat_now = float(state.step1.sat_lon or 0.0)
        dl_now = float((geometry or {}).get("d_over_lambda") or 1.0)
        d_now = (geometry or {}).get("diameter") or FALLBACK_DIAMETER_M
        wavelength = (geometry or {}).get("wavelength") or (d_now / dl_now if dl_now else None)

        if not baseline:
            baseline["pt"] = pt_now
        pt_lo, pt_hi = baseline["pt"] - 20.0, baseline["pt"] + 5.0

        def trial_geometry(sat: float, diameter: float) -> Optional[Dict[str, Any]]:
            if not geometry:
                return None
            g = dict(geometry)
            g["delta0"] = sat - float(es_lon)
            g["d_over_lambda"] = diameter / wavelength if wavelength else dl_now
            g["diameter"] = diameter
            if wavelength:
                g["wavelength"] = wavelength
            return g

        def make_slider(text: Callable[[float], str], **kwargs: Any) -> ui.slider:
            slider = ui.slider(step=0.1, **kwargs).props("label-always")
            ui.label().bind_text_from(slider, "value", text)
            return slider

        with ui.column().classes("w-full gap-0"):
            pt_slider = make_slider(lambda v: f"Transmit power Pt: {v:.1f} dBW", min=pt_lo, max=pt_hi,
                                    value=min(max(pt_now, pt_lo), pt_hi))
            sat_slider = make_slider(lambda v: f"Satellite longitude: {v:.1f}°", min=-180.0, max=180.0,
                                     value=sat_now)
            d_slider = make_slider(lambda v: f"Antenna diameter D: {v:.1f} m", min=1.0, max=15.0,
                                   value=min(max(float(d_now), 1.0), 15.0))
        if not geometry:
            sat_slider.disable()
            d_slider.disable()

        # -- cheap live preview for the worst azimuth (no propagation re-run) ---------------
        @ui.refreshable
        def preview() -> None:
            gt_new, note = worst["Gt_alpha_dBi"], ""
            g = trial_geometry(sat_slider.value, d_slider.value)
            if g is not None:
                try:
                    gt_new = _gt_at_azimuth(g, worst["azimuth_deg"],
                                            _horizon_at(state.step5_inputs.ppd_params, worst["azimuth_deg"])[0])
                except Exception as ex:  # noqa: BLE001
                    note = f"Could not recompute Gt(φ): {ex}"
            # Path loss and N do not depend on Pt / Gt / D, so reuse them.
            i_new = pt_slider.value + gt_new + float(wr.gr) - worst["in_result"]["L_path_dB"]
            in_new = i_new - worst["in_result"]["N_dBW"]
            ui.label(f"Gt(φ) {gt_new:.2f} dBi  ·  I/N {in_new:.2f} dB "
                     f"({in_new - worst['in_result']['IN_ratio_dB']:+.2f})  ·  "
                     f"margin {limit - in_new:.2f} dB").classes("text-caption")
            if note:
                ui.label(note).classes("text-caption text-warning")
            if in_new <= limit:
                ui.label("With these settings the worst azimuth would PASS.").classes("text-caption text-positive")
            else:
                ui.label(f"Still FAILs by {in_new - limit:.2f} dB at the worst azimuth.").classes(
                    "text-caption text-negative")

        preview()
        ui.label("Release a slider to apply it and recompute the contour and table.").classes(
            "text-caption text-grey-7")

        for s in (pt_slider, sat_slider, d_slider):
            # leading_events=False: handle the settled value, not a stale first event.
            s.on("update:model-value", lambda: preview.refresh(), throttle=0.2, leading_events=False)
            s.on("change", lambda: commit(pt_slider.value, sat_slider.value, d_slider.value))

        async def commit(pt_val: float, sat_val: float, d_val: float) -> None:
            await asyncio.sleep(0.1)       # let the final model update arrive before reading values
            pt_val, sat_val, d_val = pt_slider.value, sat_slider.value, d_slider.value
            unchanged = (abs(pt_val - pt_now) < 1e-9
                         and (not geometry or (abs(sat_val - sat_now) < 1e-9 and abs(d_val - float(d_now)) < 1e-9)))
            if unchanged:
                return
            if busy["on"]:                 # a recompute is running: redo once with the latest values
                busy["pending"] = True
                return
            await apply_mitigation(pt_val, sat_val, d_val, trial_geometry(sat_val, d_val))

    async def apply_mitigation(pt_val: float, sat_val: float, d_val: float,
                               geometry: Optional[Dict[str, Any]]) -> None:
        """Commit a mitigation (all-or-nothing) and refresh table, plot and map."""
        if wr.gt is None or wr.gr is None or wr.pr_p is None or not state.step5_inputs.ppd_params:
            ui.notify("Gt, Gx, Pr(p) and the Step 5 parameters are needed first.", type="negative")
            return
        limit = _current_limit(state)
        if limit is None or wr.final_te is None or wr.final_b is None:
            ui.notify("Te, B and Pr(p) (Step 3) are needed to evaluate the I/N limit.", type="negative")
            return
        old_rows = bd.analysis_rows or []
        border_km = {r["azimuth_deg"]: r["border_km"] for r in old_rows}
        busy["on"] = True
        note = ui.notification("Recomputing contour with the mitigated parameters…", spinner=True, timeout=None)
        try:
            out = await run.io_bound(
                _mitigation_worker, dict(state.step5_inputs.ppd_params), float(pt_val), float(wr.gt),
                float(wr.gr), float(wr.pr_p), geometry, float(wr.gr), float(wr.final_te),
                float(wr.final_b), limit, border_km, _diff_cfg(state))
        except Exception as ex:  # noqa: BLE001
            ui.notify(f"Recompute failed: {ex}", type="negative", multi_line=True)
            return
        finally:
            note.dismiss()
            busy["on"] = False

        if not out["ok"]:
            ui.notify("Recompute failed for every azimuth - nothing was changed.", type="negative")
            return

        # Success: commit the mitigated parameters, then the results computed from them.
        wr.pt = float(pt_val)
        wr.lb_p = wr.pt + float(wr.gt) + float(wr.gr) - float(wr.pr_p)
        bd.applied_pt, bd.applied_sat_lon, bd.applied_diameter = pt_val, sat_val, d_val
        if geometry is not None:
            wr.gt_geometry = geometry
            state.step1.sat_lon = sat_val
        _store_coordination(state, out["out"])
        bd.analysis_rows = out["rows"]
        if res.active() is not None:
            res.active().border_rows = out["rows"]      # keep the per-frequency copy in step
        state.bump_map_version()                        # border rows changed
        ui.notify(f"Applied: Pt = {pt_val:.2f} dBW, satellite longitude = {sat_val:.2f}°, "
                  f"D = {d_val:.2f} m.", type="positive")
        refresh_all()
        await _call_maybe_async(on_map_refresh)

        if busy["pending"]:
            busy["pending"] = False
            ui.notify("Sliders moved during the recompute - release one again to apply the latest values.",
                      type="info")

    # ---- refresh plumbing ---------------------------------------------------------------------
    def refresh_all() -> None:
        sync_combined_contour(state)
        results_grid.refresh()
        analysis_panel.refresh()
        mitigation_panel.refresh()

    def _signature() -> Tuple[Any, ...]:
        per_freq = tuple((k, id(v.mode1_profile), id(v.mode2_result), id(v.final_contour), id(v.border_rows))
                         for k, v in sorted(res.results_by_frequency.items()))
        return (tuple(state.selected_frequencies), res.active_frequency, per_freq,
                id(res.mode1_profile_result), id(res.mode2_result), res.mode2_error,
                id(res.final_contour_result), id(bd.analysis_rows), bd.analysis_warning,
                wr.pt, wr.gr, wr.pr_p, wr.final_te, wr.final_b,
                bool(state.step5_inputs.ppd_params))

    # ---- layout --------------------------------------------------------------------------------
    sync_combined_contour(state)
    with ui.column().classes("w-full gap-2"):
        results_grid()
        ui.separator()
        _section_title("Border analysis")
        analysis_panel()
        mitigation_panel()

    last = {"sig": _signature()}

    def poll() -> None:
        sync_combined_contour(state)
        sig = _signature()
        if sig != last["sig"]:
            last["sig"] = sig
            if not bd.analysis_rows:
                baseline.clear()
            refresh_all()
            last["sig"] = _signature()

    ui.timer(0.7, poll)
    return refresh_all