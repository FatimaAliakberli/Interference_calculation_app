"""
core/path_composition.py - GIS-based path composition for AP7 Annex 1 (propagation mode 1).

Replaces the worst-case defaults (dtm = dlm = land_fraction = 0) with values measured along
the great-circle path from the earth station, using the Natural Earth data bundled in
``data_sources/``:

* LAND / WATER  - ``ne_10m_admin_0_countries`` (.shp or .geojson). The country polygons are
  the land; everything outside them (sea, Caspian Sea, ...) is water.
* COASTLINE     - ``ne_10m_coastline.shp``. Used for ``dc`` (station -> nearest coast) and to
  split land into coastal / inland for ``dlm``.

Public API
----------
``calculate_path_params(lat_start, lon_start, lat_end, lon_end)``
    -> ``{"dtm": km, "dlm": km, "dc": km, "land_fraction": 0..1}`` for ONE path.

``calculate_path_params_by_azimuth(lat, lon, azimuths_deg, length_km)``
    -> ``{azimuth: {same four keys}}`` for MANY azimuths from one station. Much faster than
    calling the function above in a loop: the data is indexed once and ``dc`` is shared.

Definitions (as used by ``core.propagation_calculator``)
--------------------------------------------------------
dtm            longest continuous LAND section (inland + coastal) of the path, km
dlm            longest continuous INLAND section (zone A2) of the path, km. Inland = land
               farther than ``inland_threshold_km`` (default 50 km) from the coastline
dc             distance from the START point to the nearest coastline, km
land_fraction  land samples / all samples (0..1)

How lengths are measured
------------------------
The path is sampled every ``step_km`` (default 1 km) with ``pyproj.Geod.npts``. Each sample owns
the half-segments on either side of it (the path ends own one half), so an all-land path gives
dtm == path length and an isolated land sample counts as about one step. Segment lengths come
from ``Geod.inv`` (WGS84, metres).

Limitations (screening-level, like the rest of the border/zone helpers)
-----------------------------------------------------------------------
* "Inland" is a distance-from-coast approximation (the Recommendation also uses terrain height).
* Natural Earth is a cartographic dataset, not a legal one; small lakes/rivers count as land.
* Coastline lookups assume the path does not cross the antimeridian within the search window.

Dependencies: shapely >= 2.0, pyproj, numpy. geopandas is used to read files when present;
otherwise pyshp (.shp) or plain json (.geojson) is used.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from core.app_paths import get_resource_path

try:
    import shapely
    from shapely import STRtree
    from shapely.geometry import shape as _shape
    _SHAPELY_ERR: Optional[Exception] = None
except Exception as _e:  # noqa: BLE001 - reported lazily by _require_libs()
    shapely = STRtree = _shape = None  # type: ignore[assignment]
    _SHAPELY_ERR = _e

try:
    from pyproj import Geod, Transformer
    _PYPROJ_ERR: Optional[Exception] = None
except Exception as _e:  # noqa: BLE001
    Geod = Transformer = None  # type: ignore[assignment]
    _PYPROJ_ERR = _e

logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    # Make sure the DEBUG output is visible in the terminal even if the app never configured logging.
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)


def _dbg(msg: str, level: int = logging.INFO) -> None:
    """Log AND print: the print() guarantees the line shows up in the console even when the
    logging configuration of the host application swallows it."""
    logger.log(level, msg)
    print(f"[path_composition] {msg}", flush=True)


__all__ = [
    "PathCompositionError",
    "calculate_path_params",
    "calculate_path_params_by_azimuth",
    "set_data_paths",
    "clear_cache",
]

DEFAULT_STEP_KM = 1.0
DEFAULT_INLAND_THRESHOLD_KM = 50.0
DEFAULT_MAX_POINTS = 3000          # per path; the step grows for very long paths
_KM_PER_DEG = 111.2                # only used to size search windows, never for results

_LAND_NAMES = ("ne_10m_admin_0_countries.shp", "ne_10m_admin_0_countries.geojson")
_COAST_NAMES = ("ne_10m_coastline.shp", "ne_10m_coastline.geojson")


class PathCompositionError(RuntimeError):
    """Libraries or data files are missing / unreadable. Callers should fall back to manual
    dtm / dlm / dc / land-fraction entry."""


# ===========================================================================
# Data loading (cached for the process)
# ===========================================================================
_lock = threading.Lock()
_override_paths: Dict[str, Optional[str]] = {"land": None, "coast": None}
_cache: Dict[str, Any] = {}


def set_data_paths(land: Optional[str] = None, coast: Optional[str] = None) -> None:
    """Use explicit files instead of searching ``data_sources/`` (tests, custom installs).
    ``None`` keeps the automatic search for that layer. Clears the cache."""
    _override_paths["land"], _override_paths["coast"] = land, coast
    clear_cache()


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _require_libs() -> None:
    if _SHAPELY_ERR is not None or _PYPROJ_ERR is not None:
        raise PathCompositionError(
            "Path composition needs shapely>=2, pyproj and numpy (pip install shapely pyproj). "
            f"Import error: {_SHAPELY_ERR or _PYPROJ_ERR}")
    if not hasattr(shapely, "intersects_xy") or not hasattr(STRtree, "query_nearest"):
        raise PathCompositionError(
            f"shapely >= 2.0 is required (found {getattr(shapely, '__version__', '?')}).")


def _find_file(names: Tuple[str, ...], override: Optional[str]) -> str:
    """Locate a data file independent of the working directory (also inside a PyInstaller build)."""
    _dbg(f"_find_file: looking for {' or '.join(names)} (override={override!r}, cwd={os.getcwd()})")
    if override:
        _dbg(f"_find_file: override path (absolute) = {os.path.abspath(override)}")
        if os.path.isfile(override):
            _dbg(f"_find_file: FOUND override file: {os.path.abspath(override)}")
            return override
        msg = f"Data file not found: {override}"
        _dbg(f"_find_file: ERROR {msg}", logging.ERROR)
        raise PathCompositionError(msg)
    root = os.path.abspath(get_resource_path("data_sources"))
    _dbg(f"_find_file: searching root folder (absolute) = {root}  "
         f"[exists={os.path.exists(root)}, is_dir={os.path.isdir(root)}]")
    wanted = [n.lower() for n in names]
    for n in names:                                    # fast path: directly under data_sources/
        p = os.path.join(root, n)
        _dbg(f"_find_file: trying exact path {os.path.abspath(p)}  [is_file={os.path.isfile(p)}]")
        if os.path.isfile(p):
            _dbg(f"_find_file: FOUND {os.path.abspath(p)}")
            return p
    if os.path.isdir(root):                            # then any sub-folder (coastline/, costline/, ...)
        try:
            top = sorted(os.listdir(root))
            _dbg(f"_find_file: contents of {root}: {top[:60]}{' ...' if len(top) > 60 else ''}")
        except OSError as ex:
            _dbg(f"_find_file: could not list {root}: {ex}", logging.ERROR)
        for dirpath, _dirs, files in os.walk(root):
            by_lower = {f.lower(): f for f in files}
            for w in wanted:                           # keep the preference order of ``names``
                if w in by_lower:
                    found = os.path.join(dirpath, by_lower[w])
                    _dbg(f"_find_file: FOUND (sub-folder search) {os.path.abspath(found)}")
                    return found
    msg = (f"Could not find {' or '.join(names)} under {root}. Place the Natural Earth files in "
           "data_sources/ (any sub-folder).")
    _dbg(f"_find_file: ERROR {msg}", logging.ERROR)
    raise PathCompositionError(msg)


def _read_geometries(path: str) -> List[Any]:
    """All non-empty geometries of a vector file, as shapely objects in lon/lat (EPSG:4326)."""
    _dbg(f"_read_geometries: reading {os.path.abspath(path)}  "
         f"[size={os.path.getsize(path) if os.path.isfile(path) else 'MISSING'} bytes]")
    if path.lower().endswith(".shp"):
        base = os.path.splitext(path)[0]
        for ext in (".shx", ".dbf", ".prj"):
            sib = [base + ext, base + ext.upper()]
            ok = any(os.path.isfile(x) for x in sib)
            _dbg(f"_read_geometries: companion file {os.path.basename(base)}{ext}: "
                 f"{'present' if ok else 'MISSING'}", logging.INFO if ok else logging.WARNING)
    try:
        import geopandas as gpd
        gdf = gpd.read_file(path)
        _dbg(f"_read_geometries: geopandas read {len(gdf)} rows, crs={gdf.crs}")
        if gdf.crs is not None and gdf.crs.to_epsg() not in (None, 4326):
            _dbg(f"_read_geometries: re-projecting from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs(epsg=4326)
        geoms = [g for g in gdf.geometry.values if g is not None and not g.is_empty]
        _dbg(f"_read_geometries: geopandas -> {len(geoms)} non-empty geometries loaded from {path}")
        return geoms
    except ImportError as ex:
        _dbg(f"_read_geometries: geopandas is NOT installed ({ex}); "
             "FALLING BACK to the pure-python reader", logging.WARNING)
    except Exception as primary:  # noqa: BLE001 - broken GDAL backend etc.: try the pure-python readers
        _dbg(f"_read_geometries: geopandas FAILED on {path}: {type(primary).__name__}: {primary}; "
             "FALLING BACK to the pure-python reader", logging.ERROR)
        try:
            geoms = _read_geometries_pure_python(path)
            _dbg(f"_read_geometries: pure-python fallback -> {len(geoms)} geometries loaded")
            return geoms
        except Exception as fallback:  # noqa: BLE001
            _dbg(f"_read_geometries: pure-python fallback ALSO FAILED: "
                 f"{type(fallback).__name__}: {fallback}", logging.ERROR)
            raise PathCompositionError(
                f"Could not read {path}: {primary}; pure-python fallback also failed: {fallback}") from fallback
    try:
        geoms = _read_geometries_pure_python(path)
        _dbg(f"_read_geometries: pure-python reader -> {len(geoms)} geometries loaded")
        return geoms
    except Exception as ex:  # noqa: BLE001
        _dbg(f"_read_geometries: pure-python reader FAILED: {type(ex).__name__}: {ex}", logging.ERROR)
        raise PathCompositionError(f"Could not read {path}: {ex}") from ex


def _read_geometries_pure_python(path: str) -> List[Any]:
    if path.lower().endswith((".geojson", ".json")):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        feats = data.get("features", [data])
        geoms = [_shape(f["geometry"]) for f in feats if f.get("geometry")]
    else:
        import shapefile                                  # pip install pyshp
        geoms = []
        with shapefile.Reader(path) as reader:
            for s in reader.iterShapes():
                if s.shapeType != shapefile.NULL:
                    geoms.append(_shape(s.__geo_interface__))
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        raise ValueError("the file contains no geometries")
    return geoms


def _land() -> Tuple[Any, np.ndarray]:
    """(STRtree over land polygons, geometry array). Polygons are prepared for fast point tests."""
    with _lock:
        if "land" not in _cache:
            path = _find_file(_LAND_NAMES, _override_paths["land"])
            all_geoms = _read_geometries(path)
            type_counts: Dict[str, int] = {}
            for g in all_geoms:
                type_counts[g.geom_type] = type_counts.get(g.geom_type, 0) + 1
            _dbg(f"_land: {len(all_geoms)} geometries read; types = {type_counts}")
            geoms = [g for g in all_geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            _dbg(f"_land: {len(geoms)} valid Polygon/MultiPolygon geometries found")
            if not geoms:
                _dbg(f"_land: WARNING - 0 valid polygons in {path}; every point will be WATER "
                     "(dtm = dlm = land_fraction = 0)", logging.WARNING)
                raise PathCompositionError(f"{path} contains no polygons.")
            arr = np.array(geoms, dtype=object)
            shapely.prepare(arr)
            _cache["land"] = (STRtree(arr), arr)
        return _cache["land"]


def _coast() -> Tuple[Any, np.ndarray]:
    """(STRtree over coastline lines, geometry array)."""
    with _lock:
        if "coast" not in _cache:
            path = _find_file(_COAST_NAMES, _override_paths["coast"])
            geoms = [g for g in _read_geometries(path)
                     if g.geom_type in ("LineString", "MultiLineString")]
            if not geoms:
                raise PathCompositionError(f"{path} contains no lines.")
            arr = np.array(geoms, dtype=object)
            _cache["coast"] = (STRtree(arr), arr)
        return _cache["coast"]


# ===========================================================================
# Land / water test
# ===========================================================================
def _land_mask(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """True where the point lies on (or on the edge of) a land polygon."""
    tree, geoms = _land()
    pts = shapely.points(lons, lats)
    inp, tgt = tree.query(pts)                      # bounding-box candidates only
    mask = np.zeros(len(lons), dtype=bool)
    if inp.size == 0:
        return mask
    order = np.argsort(tgt, kind="stable")
    inp, tgt = inp[order], tgt[order]
    bounds = np.flatnonzero(np.diff(tgt)) + 1
    for idx_chunk, poly_idx in zip(np.split(inp, bounds), tgt[np.r_[0, bounds]]):
        todo = idx_chunk[~mask[idx_chunk]]          # skip points already found on land
        if todo.size:
            mask[todo] = shapely.intersects_xy(geoms[poly_idx], lons[todo], lats[todo])
    return mask


# ===========================================================================
# Coastline distances
# ===========================================================================
class _CoastIndex:
    """Coastline pieces within ``radius_km`` of a centre, projected to an azimuthal-equidistant
    plane centred there (distances from the centre are exact; elsewhere error is a fraction of
    a percent over the few-hundred-km scales used here) and indexed for nearest-line queries."""

    def __init__(self, lon0: float, lat0: float, radius_km: float):
        tree, geoms = _coast()
        dlat = radius_km / _KM_PER_DEG
        lat_lo, lat_hi = max(-90.0, lat0 - dlat), min(90.0, lat0 + dlat)
        worst_lat = max(abs(lat_lo), abs(lat_hi))
        if worst_lat >= 89.0 or dlat >= 90.0:
            lon_lo, lon_hi = -180.0, 180.0
        else:
            dlon = min(180.0, dlat / math.cos(math.radians(worst_lat)))
            lon_lo, lon_hi = lon0 - dlon, lon0 + dlon
        # Window across the antimeridian: query both sides.
        windows = [(lon_lo, lon_hi)]
        if lon_lo < -180.0:
            windows = [(-180.0, lon_hi), (lon_lo + 360.0, 180.0)]
        elif lon_hi > 180.0:
            windows = [(lon_lo, 180.0), (-180.0, lon_hi - 360.0)]

        self._proj = Transformer.from_crs(
            "EPSG:4326",
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
            always_xy=True)
        pieces: List[Any] = []
        for x0, x1 in windows:
            cand = tree.query(shapely.box(x0, lat_lo, x1, lat_hi))
            for g in geoms[cand]:
                clipped = shapely.clip_by_rect(g, x0, lat_lo, x1, lat_hi)
                if not clipped.is_empty:
                    pieces.append(clipped)
        self.empty = not pieces
        self._tree = None
        if pieces:
            arr = shapely.transform(np.array(pieces, dtype=object), self._xy)
            self._tree = STRtree(arr)

    def _xy(self, coords: np.ndarray) -> np.ndarray:
        x, y = self._proj.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    def nearest_km(self, lons: np.ndarray, lats: np.ndarray, max_km: float) -> np.ndarray:
        """Distance (km) from each point to the nearest coastline; ``inf`` when none within ``max_km``."""
        out = np.full(len(lons), np.inf)
        if self._tree is None or len(lons) == 0:
            return out
        x, y = self._proj.transform(np.asarray(lons), np.asarray(lats))
        # For an ARRAY of input points query_nearest returns ``indices`` of shape (2, n_matches):
        # row 0 = index into the input points, row 1 = index into the tree. Only row 0 may be
        # used to address ``out`` (using the whole array caused IndexError: index N is out of
        # bounds). Points with no coastline within max_distance are simply absent -> stay inf.
        idx, d = self._tree.query_nearest(shapely.points(x, y), max_distance=max_km * 1000.0,
                                          return_distance=True, all_matches=False)
        idx = np.asarray(idx)
        inp = idx[0] if idx.ndim == 2 else idx
        out[inp] = np.asarray(d, dtype=float) / 1000.0
        return out


def _distance_to_coast_km(lon: float, lat: float, first_radius_km: float = 150.0) -> float:
    """Distance (km) from a point to the nearest coastline, searching in growing windows."""
    r = max(first_radius_km, 10.0)
    _dbg(f"_distance_to_coast_km: start point lat={lat:.5f}, lon={lon:.5f}, first radius={r:.0f} km")
    for _ in range(8):                                   # 150 ... ~19 000 km
        idx = _CoastIndex(lon, lat, r)
        d = idx.nearest_km(np.array([lon]), np.array([lat]), r)[0]
        _dbg(f"_distance_to_coast_km: search radius {r:.0f} km -> "
             f"{'no coastline pieces in window' if idx.empty else 'coastline pieces present'}, "
             f"nearest = {d}")
        if np.isfinite(d):
            _dbg(f"_distance_to_coast_km: RESULT dc = {float(d):.3f} km")
            return float(d)
        r *= 2.5
        if r > 20000.0:
            break
    _dbg("_distance_to_coast_km: ERROR no coastline found anywhere near the start point", logging.ERROR)
    raise PathCompositionError("No coastline found anywhere near the start point.")


# ===========================================================================
# Path sampling and run-length metrics
# ===========================================================================
def _sample_path(geod: "Geod", lat1: float, lon1: float, lat2: float, lon2: float,
                 step_km: float, max_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Points along the geodesic, endpoints included, evenly spaced.
    Returns (lons, lats, cum_m, total_m); ``cum_m[i]`` = geodesic length from the start to point i."""
    _az12, _az21, total_m = geod.inv(lon1, lat1, lon2, lat2)
    if total_m < 1e-3:
        return (np.array([lon1]), np.array([lat1]), np.array([0.0]), 0.0)
    n_seg = int(min(max(1, math.ceil(total_m / (step_km * 1000.0))), max(1, max_points - 1)))
    inner = geod.npts(lon1, lat1, lon2, lat2, n_seg - 1) if n_seg > 1 else []
    pts = [(lon1, lat1)] + list(inner) + [(lon2, lat2)]
    lons = np.array([p[0] for p in pts], dtype=float)
    lats = np.array([p[1] for p in pts], dtype=float)
    # Segment lengths from Geod.inv (vectorised), not from the nominal step.
    _a, _b, seg_m = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(seg_m, dtype=float))])
    return lons, lats, cum, float(cum[-1])


def _longest_run_km(flags: np.ndarray, cum_m: np.ndarray) -> float:
    """Longest run of True samples, in km. A sample owns the half-segments on both sides of it."""
    n = len(flags)
    if n == 0 or not flags.any():
        return 0.0
    # Cell boundaries: start, midpoints between samples, end.
    edges = np.concatenate([[cum_m[0]], 0.5 * (cum_m[:-1] + cum_m[1:]), [cum_m[-1]]])
    padded = np.concatenate([[False], flags, [False]])
    change = np.flatnonzero(padded[1:] != padded[:-1])      # alternating run starts / ends
    starts, ends = change[0::2], change[1::2]               # run = samples [start, end)
    return float((edges[ends] - edges[starts]).max() / 1000.0)


class _Station:
    """Everything shared by all paths that start at one earth station."""

    def __init__(self, lat: float, lon: float, reach_km: float, inland_threshold_km: float,
                 need_inland: bool):
        self.lat, self.lon = lat, lon
        self.threshold = inland_threshold_km
        self.coast: Optional[_CoastIndex] = None
        # dc: exact search, independent of the window below.
        self.dc = _distance_to_coast_km(lon, lat)
        if need_inland:
            self.coast = _CoastIndex(lon, lat, reach_km + inland_threshold_km + 25.0)

    def path(self, geod: "Geod", lat2: float, lon2: float, step_km: float,
             max_points: int) -> Dict[str, float]:
        lons, lats, cum, total_m = _sample_path(geod, self.lat, self.lon, lat2, lon2,
                                                step_km, max_points)
        land = _land_mask(lons, lats)
        dtm = _longest_run_km(land, cum)
        if self.coast is None:
            dlm = dtm
        else:
            near = self.coast.nearest_km(lons, lats, self.threshold) < np.inf
            dlm = _longest_run_km(land & ~near, cum)
        result = {"dtm": dtm, "dlm": dlm, "dc": float(self.dc),
                  "land_fraction": float(land.sum()) / float(len(land))}
        # Log every path when it is a handful, otherwise only the first one (a sample).
        self._n_logged = getattr(self, "_n_logged", 0) + 1
        if self._n_logged <= 3:
            _dbg(f"_Station.path: sample path #{self._n_logged}: start=({self.lat:.5f}, {self.lon:.5f}) "
                 f"end=({lat2:.5f}, {lon2:.5f}) length={total_m / 1000.0:.1f} km, "
                 f"{len(land)} samples, {int(land.sum())} on land, "
                 f"inland check={'ON' if self.coast is not None else 'OFF'} -> "
                 f"dtm={dtm:.3f} km, dlm={dlm:.3f} km, dc={self.dc:.3f} km, "
                 f"land_fraction={result['land_fraction']:.4f}")
            if not land.any():
                _dbg("_Station.path: WARNING - not a single sample of this path is on land. "
                     "Either the path is really over water, or the land polygons/lon-lat order is wrong.",
                     logging.WARNING)
        return result


def _check_coords(lat: float, lon: float, what: str) -> None:
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError(f"{what}: latitude/longitude must be finite numbers")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"{what}: latitude {lat} is outside -90..90")


# ===========================================================================
# Public API
# ===========================================================================
def calculate_path_params(lat_start: float, lon_start: float, lat_end: float, lon_end: float, *,
                          step_km: float = DEFAULT_STEP_KM,
                          inland_threshold_km: Optional[float] = DEFAULT_INLAND_THRESHOLD_KM,
                          max_points: int = DEFAULT_MAX_POINTS) -> Dict[str, float]:
    """Path composition of the great-circle path start -> end.

    Returns ``{"dtm": km, "dlm": km, "dc": km, "land_fraction": 0..1}``.

    ``inland_threshold_km``: land closer than this to the coast is "coastal" and does not count
    towards dlm. Pass ``None`` (or 0) for the simple variant dlm == dtm.
    ``step_km`` / ``max_points``: sampling interval and the cap on samples per path (the step is
    enlarged automatically for very long paths).

    Raises PathCompositionError when libraries or data files are missing (callers fall back to
    manual entry) and ValueError for invalid coordinates.
    """
    _require_libs()
    _check_coords(lat_start, lon_start, "start")
    _check_coords(lat_end, lon_end, "end")
    if step_km <= 0:
        raise ValueError("step_km must be > 0")
    geod = Geod(ellps="WGS84")
    reach_km = geod.inv(lon_start, lat_start, lon_end, lat_end)[2] / 1000.0
    need_inland = bool(inland_threshold_km and inland_threshold_km > 0)
    st = _Station(lat_start, lon_start, reach_km, float(inland_threshold_km or 0.0), need_inland)
    return st.path(geod, lat_end, lon_end, step_km, max_points)


def calculate_path_params_by_azimuth(lat: float, lon: float, azimuths_deg: Iterable[float],
                                     length_km: float, *,
                                     step_km: float = DEFAULT_STEP_KM,
                                     inland_threshold_km: Optional[float] = DEFAULT_INLAND_THRESHOLD_KM,
                                     max_points: int = DEFAULT_MAX_POINTS,
                                     progress: Optional[Callable[[int, int], None]] = None
                                     ) -> Dict[float, Dict[str, float]]:
    """Path composition along a ray of ``length_km`` for every azimuth (degrees, 0 = North,
    clockwise) from one station. ``{azimuth: {"dtm", "dlm", "dc", "land_fraction"}}``.

    The land / coastline data is indexed once and ``dc`` is computed once, so this is far
    cheaper than ``calculate_path_params`` in a loop. ``progress(done, total)`` is optional and
    is called from the calling thread.
    """
    _require_libs()
    _check_coords(lat, lon, "station")
    if length_km <= 0:
        raise ValueError("length_km must be > 0")
    if step_km <= 0:
        raise ValueError("step_km must be > 0")
    az_list = [float(a) for a in azimuths_deg]
    geod = Geod(ellps="WGS84")
    need_inland = bool(inland_threshold_km and inland_threshold_km > 0)
    st = _Station(lat, lon, length_km, float(inland_threshold_km or 0.0), need_inland)
    _land()                                                  # load now so errors surface early
    out: Dict[float, Dict[str, float]] = {}
    for i, az in enumerate(az_list, 1):
        end_lon, end_lat, _ = geod.fwd(lon, lat, az, length_km * 1000.0)
        out[az] = st.path(geod, end_lat, end_lon, step_km, max_points)
        if progress is not None:
            progress(i, len(az_list))
    return out