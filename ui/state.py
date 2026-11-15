"""
Application state for the NiceGUI port of the Link Budget Calculator.

Replaces every key the Streamlit app kept in ``st.session_state`` (the
``DEFAULTS`` dict in app.py plus the keys that were only ever written ad hoc,
such as ``ppd_params``, ``final_te`` / ``final_b``, ``ptd_result`` / ``prd_result``,
the ``vd_*`` lookup keys and the ``s8_applied_*`` mitigation keys).

Design notes
------------
* State is split into small dataclass *groups*; ``AppState`` owns one instance of
  each. Groups are **reset in place**, never replaced, so NiceGUI bindings such as
  ``ui.input().bind_value(state.border, "home_country")`` keep pointing at a live
  object after ``clear_all()``.
* Desktop (native) mode is single-user, so one ``AppState`` per process is enough.
* Nothing here imports NiceGUI or does any I/O - it is plain data plus resets, so
  it can be unit-tested on its own.
* ``LEGACY_KEYS`` maps each old ``st.session_state`` key to its new home, as a
  porting reference while the Streamlit code is moved over.
"""
from __future__ import annotations

import json
import os
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.app_paths import get_resource_path

# DEM GeoTIFF for the optional hillshade overlay in the Step 9 map. Bundled with the
# app (read-only); LBC_DEM_PATH overrides it. Only read if the file exists.
#
# The bundled DEM's absolute path differs per machine / per PyInstaller run (sys._MEIPASS is
# a temp folder), so it is NEVER stored in the state, in autosave.json or in a project file.
# BorderState.dem_path holds this sentinel instead and every consumer resolves it at runtime
# with resolve_dem_path().
BUNDLED_DEM_SENTINEL = "__BUNDLED_DEM__"
BUNDLED_DEM_RELPATH = "data_sources/azerbaijan_dem.tif"


def resolve_dem_path(value: Optional[str]) -> str:
    """Real filesystem path for a stored ``dem_path`` value.

    The sentinel (or an empty value) becomes ``$LBC_DEM_PATH`` if set, else
    ``get_resource_path("data_sources/azerbaijan_dem.tif")``; any other value is a
    user-chosen path and is returned unchanged. Not checked for existence.
    """
    if not value or value == BUNDLED_DEM_SENTINEL:
        return os.environ.get("LBC_DEM_PATH") or get_resource_path(BUNDLED_DEM_RELPATH)
    return value


def normalize_dem_path(value: Optional[str]) -> str:
    """Inverse of resolve_dem_path: map the bundled DEM's path (what the UI shows) back to
    the sentinel, so an absolute bundled path is never persisted. Other values pass through.
    Also cleans projects saved by older versions that stored the absolute bundled path."""
    if not value:
        return BUNDLED_DEM_SENTINEL
    if value == BUNDLED_DEM_SENTINEL:
        return value
    try:
        norm = lambda p: os.path.normcase(os.path.abspath(p))
        if norm(value) in {norm(get_resource_path(BUNDLED_DEM_RELPATH)),
                           norm(resolve_dem_path(BUNDLED_DEM_SENTINEL))}:
            return BUNDLED_DEM_SENTINEL
    except (TypeError, ValueError):
        pass
    return value


# Resolved at import time for code that wants a plain path (e.g. the "custom DEM?" check).
DEM_TIFF_PATH = resolve_dem_path(BUNDLED_DEM_SENTINEL)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _reset_fields(obj: Any, only: Optional[Tuple[str, ...]] = None) -> None:
    """Put dataclass fields back to their declared defaults, in place.

    ``only`` limits the reset to the named fields. Mutable defaults (dict/list)
    are rebuilt from their ``default_factory`` so no state is shared between resets.
    """
    for f in fields(obj):
        if only is not None and f.name not in only:
            continue
        if f.default is not MISSING:
            setattr(obj, f.name, f.default)
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            setattr(obj, f.name, f.default_factory())  # type: ignore[misc]


# Project-file format. Bump PROJECT_FORMAT_VERSION if the saved layout ever changes
# in a way old loaders could not read.
PROJECT_FORMAT = "link-budget-calculator-project"
PROJECT_FORMAT_VERSION = 1


def _json_default(obj: Any) -> Any:
    """json.dump fallback for values the stdlib encoder rejects (numpy scalars/arrays,
    sets, paths). Anything else is stored as its string form rather than aborting the save."""
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist") and callable(obj.tolist):      # numpy scalar or array
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _coerce_loaded(f: Any, value: Any) -> Any:
    """Make one loaded JSON value safe for dataclass field ``f``.

    * ``None`` for a field whose default is not None (e.g. ``home_country``,
      ``altitude_val``) falls back to that default instead of poisoning the UI.
    * A wrong container type (null / string where a dict or list belongs) falls back
      to the empty default.
    * An int where a float is expected becomes a float.
    """
    if f.default is not MISSING:
        default = f.default
    elif f.default_factory is not MISSING:  # type: ignore[misc]
        default = f.default_factory()  # type: ignore[misc]
    else:
        return value
    if value is None:
        return default
    if isinstance(default, (dict, list)) and not isinstance(value, type(default)):
        return default
    if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _pairs_to_tuples(value: Any) -> Any:
    """[[az, km], ...] -> [(az, km), ...] (JSON turns tuples into lists)."""
    if isinstance(value, list):
        return [tuple(v) if isinstance(v, list) else v for v in value]
    return value


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------
@dataclass
class WizardResults:
    """The Lb(p) = Pt + Gt + Gr - Pr(p) terms (Steps 1-4) and their provenance."""
    gt: Optional[float] = None
    gt_source: Optional[str] = None
    # Full Annex 3 geometry "recipe" (mode, zeta, delta0, K, i/is_max/step, gmax,
    # d_over_lambda, and after Step 8 mitigation: diameter, wavelength). ONLY set
    # when Gt came from the Eq. 84-97 calculator; None for a manually entered Gt,
    # which Step 5 uses to decide whether Gt(alpha) can be recomputed per azimuth.
    gt_geometry: Optional[Dict[str, Any]] = None
    pt: Optional[float] = None
    pt_source: Optional[str] = None
    gr: Optional[float] = None
    pr_p: Optional[float] = None
    gr_pr_source: Optional[str] = None
    lb_p: Optional[float] = None           # DERIVED: pt + gt + gr - pr_p

    # Results of the Pt / Pr(p) sub-dialogs (st.session_state "ptd_result" / "prd_result").
    ptd_result: Optional[float] = None
    prd_result: Optional[float] = None

    # Victim noise temperature (K) and reference bandwidth (Hz) saved with Pr(p);
    # used by the I/N steps. NOTE: the Streamlit code read these with
    # .get("final_te", 150.0) / .get("final_b", 12500.0), i.e. a default when the key
    # was *missing*. Here they are None until set, so use `state.wizard_results.final_te or 150.0`.
    final_te: Optional[float] = None
    final_b: Optional[float] = None

    def reset(self) -> None:
        _reset_fields(self)


@dataclass
class Step1Context:
    """Step 1 calculator coordinates (auto-populate Step 5) + gain-calculator scratch results."""
    es_lat: Optional[float] = None         # "step1_es_lat"
    es_lon: Optional[float] = None         # "step1_es_lon"
    sat_lon: Optional[float] = None        # "step1_sat_lon"
    # Intermediate results of the Gt / Gr calculators, keyed by namespace
    # ("gw_Gt" / "gw_Gr"). Each namespace holds: phi, gain, gain_branch, dl_used,
    # dl_estimated, auto_diag, dl_source, gmax_used  (the old f"{ns}_{suffix}" keys).
    gain_calc: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    GAIN_SUFFIXES = ("phi", "gain", "gain_branch", "dl_used", "dl_estimated",
                     "auto_diag", "dl_source", "gmax_used")
    GAIN_NAMESPACES = ("gw_Gt", "gw_Gr")

    def reset_gain_results(self, ns: str) -> None:
        """Forget one calculator's intermediates (old ``_reset_gain_results(ns)``)."""
        self.gain_calc.pop(ns, None)

    def reset_all_gain_results(self) -> None:
        for ns in self.GAIN_NAMESPACES:
            self.reset_gain_results(ns)

    def gain_get(self, ns: str, suffix: str, default: Any = None) -> Any:
        return self.gain_calc.get(ns, {}).get(suffix, default)

    def gain_set(self, ns: str, suffix: str, value: Any) -> None:
        self.gain_calc.setdefault(ns, {})[suffix] = value

    def reset(self) -> None:
        _reset_fields(self)


@dataclass
class Step3Selection:
    """Step 3 band / frequency selection (feeds frequency and p auto-fill in Step 5)."""
    workflow_open: bool = False                         # "s3_workflow_open"
    selected_freq: Optional[float] = None               # "step3_selected_freq"
    selected_band: Optional[str] = None                 # "step3_selected_band"
    # Every band the user accepted in the Step 3 picker: {band_str: center_freq_ghz}
    band_freq_map: Dict[str, float] = field(default_factory=dict)
    freq_candidates: List[Any] = field(default_factory=list)

    def reset(self) -> None:
        _reset_fields(self)


@dataclass
class VictimLookup:
    """Step 3 victim-parameter table lookup workflow (the old ``vd_*`` keys)."""
    rows: List[Any] = field(default_factory=list)       # "vd_rows"
    freq: Optional[float] = None                        # "vd_freq"
    freq_hint: Optional[float] = None                   # "vd_freq_hint"
    gr_calculated: Optional[float] = None               # "vd_gr_calculated"
    pr_calculated: Optional[float] = None               # "vd_pr_calculated"

    def reset(self) -> None:
        _reset_fields(self)


@dataclass
class Step5Inputs:
    """Step 5 coordination-distance inputs, auto-fill state and the accepted parameter set."""
    ready: bool = False                                 # "step5_ready"
    # The accepted parameter dict (latitude, longitude, altitude, frequency_GHz,
    # p_percent, is_transmitter, zone, N0, eps_h, dh, dtm, dlm, dc, land_fraction,
    # step, beta_e, dmin, dmax1, horizon_profile, rain_zone, elevation_angle, gx).
    ppd_params: Optional[Dict[str, Any]] = None
    # Raw record from the precomputed horizon grid DB ("ppd_horizon_profile").
    horizon_profile: Optional[Dict[str, Any]] = None

    # Auto-fetched altitude, and the (lat, lon, db_path) it was fetched for so the
    # DB is only re-queried when something relevant changed.
    altitude_val: float = 0.0
    altitude_note: Optional[str] = None
    fetch_key: Optional[Tuple[Any, ...]] = None

    # Climatic auto-fill
    auto_zone: Optional[str] = None
    auto_zone_name: Optional[str] = None
    auto_dc: float = 0.0
    n0_val: float = 0.0
    be_val: float = 0.0
    dm_val: float = 0.0
    dx1_val: float = 0.0
    dc_val: float = 0.0
    # (lat, lon, altitude, freq, p) the auto-fill was last computed for, so user
    # overrides are only reset when one of those inputs actually changes.
    autofill_sig: Optional[Tuple[Any, ...]] = None

    def reset(self) -> None:
        _reset_fields(self)


def freq_key(freq_ghz: Any) -> float:
    """Canonical dict key for a frequency in GHz (rounded to 1 kHz so that 11.725 computed
    two different ways always lands on the same ``results_by_frequency`` entry)."""
    return round(float(freq_ghz), 6)


# Key used when something writes a result before any frequency has been made active
# (the single-frequency code paths that predate multi-frequency analysis).
UNSPECIFIED_FREQ_KEY = 0.0


@dataclass
class FrequencyResults:
    """Computed results (Steps 5-6) for ONE frequency. All DERIVED."""
    frequency_ghz: Optional[float] = None
    direction: Optional[str] = None                     # 'E' (Tx) / 'R' (Rx) / None
    mode1_result: Optional[Any] = None
    mode1_profile: Optional[Any] = None                 # [(azimuth, km), ...]
    mode1_profile_diag: Optional[Any] = None
    mode1_profile_errors: Optional[Any] = None
    # Propagation mode (2), Annex 2 rain scatter
    mode2_result: Optional[Any] = None
    mode2_error: Optional[str] = None
    # AP7 S1.6 combined contour [(azimuth, max(mode1_km, mode2_km)), ...]
    final_contour: Optional[Any] = None
    # Step 6 blind I/N compliance check
    blind_in_result: Optional[Any] = None
    # Step 7 border-analysis rows (per-azimuth result dicts) for THIS frequency; None until
    # "Analyze borders" has run for it. For the ACTIVE frequency border.analysis_rows mirrors it.
    border_rows: Optional[List[Dict[str, Any]]] = None

    def active_contour(self) -> Optional[Any]:
        """max(mode 1, mode 2) when mode 2 was computed, else the mode 1 contour."""
        return self.final_contour or self.mode1_profile

    def reset(self) -> None:
        _reset_fields(self, only=tuple(f.name for f in fields(self)
                                       if f.name not in ("frequency_ghz", "direction")))


def _legacy_proxy(name: str) -> property:
    """Read/write ``name`` on the ACTIVE frequency's FrequencyResults. Keeps code written
    for the old single-frequency fields working until the UI is moved to
    ``results_by_frequency``. Not a dataclass field, so it is never saved or reset on its own."""
    def getter(self: "CoordinationResults") -> Any:
        fr = self.active()
        return getattr(fr, name) if fr is not None else None

    def setter(self: "CoordinationResults", value: Any) -> None:
        setattr(self.active_or_create(), name, value)

    return property(getter, setter)


@dataclass
class CoordinationResults:
    """Computed results (Steps 5-6), one FrequencyResults per frequency.
    All DERIVED - invalidated by reset_derived()."""
    # {frequency_ghz (see freq_key()): FrequencyResults}
    results_by_frequency: Dict[float, FrequencyResults] = field(default_factory=dict)
    # The frequency the single-frequency panels / map currently show (a key of the dict).
    active_frequency: Optional[float] = None

    # -- per-frequency access ------------------------------------------------
    def get(self, freq_ghz: Any) -> Optional[FrequencyResults]:
        if freq_ghz is None:
            return None
        return self.results_by_frequency.get(freq_key(freq_ghz))

    def get_or_create(self, freq_ghz: Any, direction: Optional[str] = None) -> FrequencyResults:
        key = freq_key(freq_ghz)
        fr = self.results_by_frequency.get(key)
        if fr is None:
            fr = FrequencyResults(frequency_ghz=key, direction=direction)
            self.results_by_frequency[key] = fr
        elif direction is not None and fr.direction is None:
            fr.direction = direction
        return fr

    def set_active(self, freq_ghz: Any) -> FrequencyResults:
        """Make ``freq_ghz`` the active frequency (creating its entry if needed)."""
        fr = self.get_or_create(freq_ghz)
        self.active_frequency = fr.frequency_ghz
        return fr

    def active(self) -> Optional[FrequencyResults]:
        return self.get(self.active_frequency)

    def active_or_create(self) -> FrequencyResults:
        if self.active_frequency is None:
            return self.set_active(UNSPECIFIED_FREQ_KEY)
        return self.get_or_create(self.active_frequency)

    def active_contour(self) -> Optional[Any]:
        """The active frequency's max(mode 1, mode 2) contour (the old
        ``_active_contour()`` helper)."""
        fr = self.active()
        return fr.active_contour() if fr is not None else None

    # -- legacy single-frequency names -> active frequency ---------------------
    mode1_result = _legacy_proxy("mode1_result")
    mode1_profile_result = _legacy_proxy("mode1_profile")
    mode1_profile_diag = _legacy_proxy("mode1_profile_diag")
    mode1_profile_errors = _legacy_proxy("mode1_profile_errors")
    mode2_result = _legacy_proxy("mode2_result")
    mode2_error = _legacy_proxy("mode2_error")
    final_contour_result = _legacy_proxy("final_contour")
    blind_in_result = _legacy_proxy("blind_in_result")

    def reset(self) -> None:
        _reset_fields(self)


@dataclass
class BorderAnalysis:
    """Step 7/8 border analysis and mitigation."""
    home_country: str = "Azerbaijan"
    analysis_rows: Optional[List[Dict[str, Any]]] = None    # per-azimuth result dicts (DERIVED)
    analysis_warning: Optional[str] = None                  # dependency/download error (DERIVED)
    manual_km: Dict[Any, float] = field(default_factory=dict)   # {azimuth: manual border km}
    mitigation_az: Optional[float] = None                   # azimuth the sliders target
    # Mitigation values applied by "Apply mitigation and recompute contour" (s8_applied_*)
    applied_pt: Optional[float] = None
    applied_sat_lon: Optional[float] = None
    applied_diameter: Optional[float] = None
    dem_path: str = BUNDLED_DEM_SENTINEL                    # "s8_dem_path"; sentinel = bundled DEM

    @property
    def effective_dem_path(self) -> str:
        """The real path to read: the sentinel resolved at runtime (see resolve_dem_path)."""
        return resolve_dem_path(self.dem_path)

    def reset(self) -> None:
        _reset_fields(self)


@dataclass
class ItuImport:
    """ITU SNS (.mdb) filing-database import (see core/itu_import.py)."""
    file_key: Optional[Tuple[str, int]] = None   # (filename, size) of last parsed file - avoids re-parsing
    # {"e_stn", "geo", "non_geo", "grp", "e_ant"} -> DataFrame (or None)
    mdb_tables: Optional[Dict[str, Any]] = None
    # Station dicts from itu_import.extract_earth_stations()
    stations: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    last_import_msg: Optional[str] = None
    # Tx/Rx frequency groups of the imported station:
    # {"label", "transmit": [...], "receive": [...], "unknown": [...]}
    freqs: Optional[Dict[str, Any]] = None
    # EVERY frequency group (grp row) of the imported station, Tx and Rx, in table order -
    # the dicts of itu_import.extract_all_frequencies(): index, grp_id, freq_min_mhz,
    # freq_max_mhz, freq_center_ghz, emi_rcp, direction ('E'/'R'/None), direction_source,
    # beam_name, noise_t, pwr_max_dbw, ant_gain_dbi, ant_diam_m. Multi-frequency
    # analysis iterates this list; ``freqs`` above is the same data split by direction.
    all_frequencies: List[Dict[str, Any]] = field(default_factory=list)
    # Victim (receive) operating frequency in GHz - set by the ITU import or the Step 3
    # "Use frequencies from imported ITU MDB file" picker; read by the Gr calculator.
    victim_freq_ghz: Optional[float] = None

    def reset(self) -> None:
        _reset_fields(self)


# Old single-frequency CoordinationResults field -> FrequencyResults field (project-file migration).
_LEGACY_RESULT_KEYS: Dict[str, str] = {
    "mode1_result": "mode1_result", "mode1_profile_result": "mode1_profile",
    "mode1_profile_diag": "mode1_profile_diag", "mode1_profile_errors": "mode1_profile_errors",
    "mode2_result": "mode2_result", "mode2_error": "mode2_error",
    "final_contour_result": "final_contour", "blind_in_result": "blind_in_result",
}


# ---------------------------------------------------------------------------
# Root state
# ---------------------------------------------------------------------------
@dataclass
class AppState:
    wizard_results: WizardResults = field(default_factory=WizardResults)
    step1: Step1Context = field(default_factory=Step1Context)
    step3: Step3Selection = field(default_factory=Step3Selection)
    victim_lookup: VictimLookup = field(default_factory=VictimLookup)
    step5_inputs: Step5Inputs = field(default_factory=Step5Inputs)
    results: CoordinationResults = field(default_factory=CoordinationResults)
    border: BorderAnalysis = field(default_factory=BorderAnalysis)
    itu_import: ItuImport = field(default_factory=ItuImport)
    # Wizard navigation (Phase 2): which step the Left Pane is showing, 1-based.
    # UI-only - not in LEGACY_KEYS because Streamlit had no equivalent (it showed
    # every step on one scrolling page).
    current_step: int = 1
    # Frequencies (GHz, canonical freq_key() values) the user ticked in the top bar. The
    # Middle and Right panes show one view per selected frequency. Always a subset of
    # itu_import.all_frequencies once an ITU station is imported; use sanitize_selected_frequencies().
    selected_frequencies: List[float] = field(default_factory=list)
    # Replacement for core/draft_state (ds.get / ds.put / ds.clear_all): remembered
    # dialog/manual-field values keyed by name, e.g. drafts["es_lat"] = 40.4.
    drafts: Dict[str, Any] = field(default_factory=dict)
    # Runtime-only change counter for the maps: bumped (bump_map_version()) whenever something a
    # map is drawn from changes - mode 1/2 results, border rows, the DEM path, the selected
    # frequencies. The Middle Pane polls it and rebuilds the maps when it moves. Never reset,
    # never saved to a project file.
    map_rebuild_version: int = 0

    def bump_map_version(self) -> int:
        """Mark the maps as out of date. Returns the new version."""
        self.map_rebuild_version += 1
        return self.map_rebuild_version

    # -- resets ------------------------------------------------------------
    def reset_derived(self) -> None:
        """Invalidate everything computed from Lb(p) onward, keeping the user's inputs.

        Mirrors the reset chain run after an ITU "Import this earth station":
        Lb(p), the accepted Step 5 parameters (which hold the OLD station's location),
        the contour / mode 2 / blind I/N / border results, and the Step 8 mitigation
        values applied to that old contour.
        """
        self.wizard_results.lb_p = None
        self.step5_inputs.ready = False
        self.step5_inputs.ppd_params = None
        self.results.reset()
        _reset_fields(self.border, only=("analysis_rows", "analysis_warning",
                                         "applied_pt", "applied_sat_lon", "applied_diameter"))
        self.bump_map_version()

    def clear_all(self) -> None:
        """Start from a blank wizard (the "Clear all" button): every group, the gain
        calculators' scratch results, all saved drafts, and the wizard back to Step 1. Groups are reset in place
        so existing UI bindings stay valid."""
        for group in (self.wizard_results, self.step1, self.step3, self.victim_lookup,
                      self.step5_inputs, self.results, self.border, self.itu_import):
            group.reset()
        self.drafts.clear()
        self.current_step = 1
        self.selected_frequencies = []
        self.bump_map_version()

    # -- frequency selection ---------------------------------------------------
    def available_frequencies(self) -> List[float]:
        """Distinct frequencies (GHz) of the imported station, ascending, as freq_key() values."""
        keys = set()
        for g in self.itu_import.all_frequencies or []:
            try:
                keys.add(freq_key(g["freq_center_ghz"]))
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(keys)

    def sanitize_selected_frequencies(self, default_all: bool = False) -> None:
        """Keep only selected frequencies that exist in the imported station (ascending order).
        With ``default_all`` an empty result becomes 'every frequency'."""
        before = list(self.selected_frequencies or [])
        avail = self.available_frequencies()
        chosen = set()
        for f in self.selected_frequencies or []:
            try:
                chosen.add(freq_key(f))
            except (TypeError, ValueError):
                continue
        self.selected_frequencies = [f for f in avail if f in chosen]
        if default_all and not self.selected_frequencies:
            self.selected_frequencies = list(avail)
        if self.selected_frequencies != before:
            self.bump_map_version()

    # -- drafts (core.draft_state replacement, data only) --------------------
    def get_draft(self, key: str, default: Any = None) -> Any:
        return self.drafts.get(key, default)

    def put_draft(self, key: str, value: Any) -> None:
        self.drafts[key] = value

    # -- save / load project -------------------------------------------------
    _GROUP_NAMES = ("wizard_results", "step1", "step3", "victim_lookup",
                    "step5_inputs", "results", "border", "itu_import")

    def save_to_json(self, filepath: Any) -> None:
        """Write the whole state to ``filepath`` as JSON.

        ``itu_import.mdb_tables`` (pandas DataFrames) is NOT saved; ``file_key`` and the
        parsed ``stations`` list are, so the import section can be restored without the
        .mdb file. The file is written to a temp file first and then moved into place,
        so a failed save never leaves a half-written project behind.
        Raises OSError / TypeError / ValueError on failure.
        """
        path = Path(filepath)

        # asdict() deep-copies everything, so detach the DataFrames first (slow, and
        # pointless since they are dropped anyway) and always put them back.
        tables = self.itu_import.mdb_tables
        self.itu_import.mdb_tables = None
        try:
            data = asdict(self)
        finally:
            self.itu_import.mdb_tables = tables
        data["itu_import"].pop("mdb_tables", None)
        data.pop("map_rebuild_version", None)       # runtime-only, not part of a project

        payload = {
            "format": PROJECT_FORMAT,
            "version": PROJECT_FORMAT_VERSION,
            "state": data,
        }

        if str(path.parent):
            path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=4, default=_json_default)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def load_from_json(self, source: Any) -> None:
        """Replace the current state with a saved project.

        ``source`` is a file path, or an already-parsed dict (e.g. from an uploaded file).

        The data is fully read and validated BEFORE anything is changed, so a bad file
        raises (OSError / json.JSONDecodeError / ValueError) and leaves the current state
        untouched. Groups are filled in place (bindings stay valid). Missing or unknown
        keys, None values and wrong-typed values fall back to the field defaults, so
        projects saved by older/newer versions still load. ``itu_import.mdb_tables`` is
        always None afterwards.
        """
        if isinstance(source, dict):
            payload = source
        else:
            with open(Path(source), "r", encoding="utf-8-sig") as fh:
                payload = json.load(fh)

        if not isinstance(payload, dict):
            raise ValueError("Not a Link Budget Calculator project file.")
        if "state" in payload:
            if payload.get("format") not in (None, PROJECT_FORMAT):
                raise ValueError("Not a Link Budget Calculator project file.")
            version = payload.get("version", 1)
            if not isinstance(version, int) or version > PROJECT_FORMAT_VERSION:
                raise ValueError(
                    f"Project file version {version!r} is newer than this app supports "
                    f"(max {PROJECT_FORMAT_VERSION})."
                )
            data = payload["state"]
        else:
            data = payload          # tolerate a bare asdict() dump
        if not isinstance(data, dict):
            raise ValueError("Project file has no valid state section.")

        # ---- validated: from here on, apply ----
        self.clear_all()            # defaults for anything the file omits

        for name in self._GROUP_NAMES:
            saved = data.get(name)
            if not isinstance(saved, dict):
                continue
            group = getattr(self, name)
            for f in fields(group):
                if f.name == "mdb_tables" or f.name not in saved:
                    continue
                setattr(group, f.name, _coerce_loaded(f, saved[f.name]))

        # Projects saved before multi-frequency support kept ONE set of results.
        saved_results = data.get("results")
        if (isinstance(saved_results, dict) and "results_by_frequency" not in saved_results
                and any(k in saved_results for k in _LEGACY_RESULT_KEYS)):
            self.results.results_by_frequency = {UNSPECIFIED_FREQ_KEY: {
                new: saved_results.get(old) for old, new in _LEGACY_RESULT_KEYS.items()}}
            self.results.active_frequency = UNSPECIFIED_FREQ_KEY

        # DataFrames are never saved; the UI restores from file_key + stations.
        self.itu_import.mdb_tables = None

        drafts = data.get("drafts")
        if isinstance(drafts, dict):
            self.drafts.update(drafts)
        try:
            self.current_step = max(1, int(data.get("current_step", 1)))
        except (TypeError, ValueError):
            self.current_step = 1

        saved_sel = data.get("selected_frequencies")
        self.selected_frequencies = list(saved_sel) if isinstance(saved_sel, list) else []
        self.sanitize_selected_frequencies(default_all=bool(self.itu_import.all_frequencies))

        self._restore_json_types()
        self.bump_map_version()         # everything the maps are drawn from was replaced

    def _restore_json_types(self) -> None:
        """Undo the things JSON changes: tuples come back as lists, and numeric dict
        keys come back as strings. Code compares/unpacks these, so put them back."""
        # Never keep the bundled DEM's absolute path (older projects / autosaves stored it).
        self.border.dem_path = normalize_dem_path(self.border.dem_path)

        # Keys compared with == against freshly built tuples.
        ii, s5 = self.itu_import, self.step5_inputs
        if isinstance(ii.file_key, list):
            ii.file_key = tuple(ii.file_key)
        if isinstance(s5.fetch_key, list):
            s5.fetch_key = tuple(s5.fetch_key)
        if isinstance(s5.autofill_sig, list):
            s5.autofill_sig = tuple(s5.autofill_sig)

        # {frequency: FrequencyResults}: JSON turned the float keys into strings and the
        # dataclasses into plain dicts; contours are [(azimuth, km), ...] tuples again.
        rebuilt: Dict[float, FrequencyResults] = {}
        for k, v in (self.results.results_by_frequency or {}).items():
            try:
                key = freq_key(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, dict):
                v = FrequencyResults(**{f.name: _coerce_loaded(f, v[f.name])
                                        for f in fields(FrequencyResults) if f.name in v})
            if not isinstance(v, FrequencyResults):
                continue
            if v.frequency_ghz is None:
                v.frequency_ghz = key
            v.mode1_profile = _pairs_to_tuples(v.mode1_profile)
            v.final_contour = _pairs_to_tuples(v.final_contour)
            rebuilt[key] = v
        self.results.results_by_frequency = rebuilt
        af = self.results.active_frequency
        try:
            self.results.active_frequency = freq_key(af) if af is not None else None
        except (TypeError, ValueError):
            self.results.active_frequency = None
        if self.results.active_frequency not in rebuilt:
            self.results.active_frequency = None

        # {azimuth: km} - azimuth keys were floats before JSON stringified them.
        fixed: Dict[Any, float] = {}
        for k, v in (self.border.manual_km or {}).items():
            try:
                fixed[float(k)] = v
            except (TypeError, ValueError):
                fixed[k] = v
        self.border.manual_km = fixed


# ---------------------------------------------------------------------------
# Porting reference: old st.session_state key -> (group attribute, field)
# Dynamic keys f"{ns}_{suffix}" (ns = gw_Gt / gw_Gr) live in step1.gain_calc.
# ---------------------------------------------------------------------------
LEGACY_KEYS: Dict[str, Tuple[str, str]] = {
    # wizard_results
    "gt": ("wizard_results", "gt"), "gt_source": ("wizard_results", "gt_source"),
    "gt_geometry": ("wizard_results", "gt_geometry"),
    "pt": ("wizard_results", "pt"), "pt_source": ("wizard_results", "pt_source"),
    "gr": ("wizard_results", "gr"), "pr_p": ("wizard_results", "pr_p"),
    "gr_pr_source": ("wizard_results", "gr_pr_source"), "lb_p": ("wizard_results", "lb_p"),
    "ptd_result": ("wizard_results", "ptd_result"), "prd_result": ("wizard_results", "prd_result"),
    "final_te": ("wizard_results", "final_te"), "final_b": ("wizard_results", "final_b"),
    # step1
    "step1_es_lat": ("step1", "es_lat"), "step1_es_lon": ("step1", "es_lon"),
    "step1_sat_lon": ("step1", "sat_lon"),
    # step3
    "s3_workflow_open": ("step3", "workflow_open"),
    "step3_selected_freq": ("step3", "selected_freq"),
    "step3_selected_band": ("step3", "selected_band"),
    "step3_band_freq_map": ("step3", "band_freq_map"),
    "step3_freq_candidates": ("step3", "freq_candidates"),
    # victim_lookup
    "vd_rows": ("victim_lookup", "rows"), "vd_freq": ("victim_lookup", "freq"),
    "vd_freq_hint": ("victim_lookup", "freq_hint"),
    "vd_gr_calculated": ("victim_lookup", "gr_calculated"),
    "vd_pr_calculated": ("victim_lookup", "pr_calculated"),
    # step5_inputs
    "step5_ready": ("step5_inputs", "ready"), "ppd_params": ("step5_inputs", "ppd_params"),
    "ppd_horizon_profile": ("step5_inputs", "horizon_profile"),
    "ppd_altitude_val": ("step5_inputs", "altitude_val"),
    "ppd_altitude_note": ("step5_inputs", "altitude_note"),
    "ppd_fetch_key": ("step5_inputs", "fetch_key"),
    "ppd_auto_zone": ("step5_inputs", "auto_zone"),
    "ppd_auto_zone_name": ("step5_inputs", "auto_zone_name"),
    "ppd_auto_dc": ("step5_inputs", "auto_dc"), "ppd_n0_val": ("step5_inputs", "n0_val"),
    "ppd_be_val": ("step5_inputs", "be_val"), "ppd_dm_val": ("step5_inputs", "dm_val"),
    "ppd_dx1_val": ("step5_inputs", "dx1_val"), "ppd_dc_val": ("step5_inputs", "dc_val"),
    "ppd_autofill_sig": ("step5_inputs", "autofill_sig"),
    # results
    "mode1_result": ("results", "mode1_result"),
    "mode1_profile_result": ("results", "mode1_profile_result"),
    "mode1_profile_diag": ("results", "mode1_profile_diag"),
    "mode1_profile_errors": ("results", "mode1_profile_errors"),
    "mode2_result": ("results", "mode2_result"), "mode2_error": ("results", "mode2_error"),
    "final_contour_result": ("results", "final_contour_result"),
    "blind_in_result": ("results", "blind_in_result"),
    # border
    "border_home_country": ("border", "home_country"),
    "border_analysis_rows": ("border", "analysis_rows"),
    "border_analysis_warning": ("border", "analysis_warning"),
    "border_manual_km": ("border", "manual_km"),
    "border_mitigation_az": ("border", "mitigation_az"),
    "s8_applied_pt": ("border", "applied_pt"),
    "s8_applied_sat_lon": ("border", "applied_sat_lon"),
    "s8_applied_diameter": ("border", "applied_diameter"),
    "s8_dem_path": ("border", "dem_path"),
    # itu_import
    "itu_mdb_file_key": ("itu_import", "file_key"), "itu_mdb_tables": ("itu_import", "mdb_tables"),
    "itu_mdb_stations": ("itu_import", "stations"), "itu_mdb_error": ("itu_import", "error"),
    "itu_mdb_last_import_msg": ("itu_import", "last_import_msg"),
    "itu_mdb_freqs": ("itu_import", "freqs"),
    "itu_mdb_all_frequencies": ("itu_import", "all_frequencies"),   # new (multi-frequency)
    "victim_freq_ghz": ("itu_import", "victim_freq_ghz"),
}