"""
map_visualization.py

Utilities for building an interactive folium map centered on an earth
station, with support for overlaying country border polygons and adding
customizable earth-station markers.

Also includes utilities for drawing satellite-coordination-style overlays:
variable-distance coordination contours (polygon built from per-azimuth
distances) and constant-distance circles, both computed with accurate
geodesic math via pyproj; and for overlaying Digital Elevation Model (DEM)
terrain (as a hillshade) read from a GeoTIFF via rasterio, including
point-elevation lookups; and for visualizing per-azimuth interference /
coordination-analysis results (PASS/FAIL sectors, detail popups, and a
map legend).

Dependencies:
    pip install folium requests pyproj rasterio numpy matplotlib
    (optional, for local shapefile/geopackage support) pip install geopandas
"""

import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import folium

try:
    import requests
except ImportError:  # pragma: no cover - requests is an optional convenience dep
    requests = None

try:
    from pyproj import Geod
except ImportError:  # pragma: no cover - surfaced as a clear error at call time
    Geod = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - surfaced as a clear error at call time
    np = None

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, transform as warp_transform
    from rasterio.enums import Resampling
    from rasterio.transform import array_bounds
except ImportError:  # pragma: no cover - surfaced as a clear error at call time
    rasterio = None

try:
    from matplotlib.colors import LightSource
except ImportError:  # pragma: no cover - surfaced as a clear error at call time
    LightSource = None

from core.app_paths import get_resource_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Public Natural Earth country-borders GeoJSON, used as a fallback when the
# caller doesn't supply a local file. This mirrors the "existing
# country_borders.py logic" pattern of fetching a world borders dataset.
DEFAULT_BORDERS_URL = (
    "https://raw.githubusercontent.com/johan/world.geo.json/master/"
    "countries.geo.json"
)


# ---------------------------------------------------------------------------
# Layer status contract, Leaflet panes and the single LayerControl
# ---------------------------------------------------------------------------
# Leaflet default panes: tilePane z=200, overlayPane z=400, shadowPane z=500,
# markerPane z=600.  The DEM hillshade lives in a custom pane ABOVE the tiles
# but BELOW the default overlay pane; the coordination contour and the
# PASS/FAIL sectors live in a custom pane ABOVE the default overlay pane, so
# they are always drawn on top of the DEM (whatever order they were added in).
PANE_DEM = "dem_pane"
Z_INDEX_DEM = 250
PANE_COORD = "coord_pane"
Z_INDEX_COORD = 450


def layer_status(
    ok: bool,
    reason: str = "",
    count: int = 0,
    severity: Optional[str] = None,
    detail: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    """
    Build the status dict returned by the layer-adding functions.

    Keys:
        ok:       True if the layer was added to the map.
        reason:   Short human-readable reason (suitable for a badge), e.g.
                  "OK", "Offline", "File not found".
        count:    Number of items drawn (features, sectors, ...).
        severity: "ok" | "warn" | "error" | "off" (defaults to "ok" / "error"
                  from ``ok``). "warn" = degraded but not broken (e.g. the
                  borders layer is unavailable because we are offline).
        detail:   Optional long message for tooltips / logs.
    """
    if severity is None:
        severity = "ok" if ok else "error"
    status: Dict[str, Any] = {
        "ok": bool(ok),
        "reason": str(reason),
        "count": int(count),
        "severity": severity,
        "detail": str(detail),
    }
    status.update(extra)
    return status


def _ensure_pane(map_obj: folium.Map, name: str, z_index: int) -> str:
    """Create the custom Leaflet pane ``name`` on ``map_obj`` once (idempotent)
    and return its name. Must run before any layer that uses the pane is
    added, so the ``createPane`` call precedes the layer in the page script."""
    created = getattr(map_obj, "_lbc_panes", None)
    if created is None:
        created = set()
        map_obj._lbc_panes = created
    if name not in created:
        folium.map.CustomPane(name, z_index=z_index).add_to(map_obj)
        created.add(name)
    return name


def _in_pane(layer: Any, pane: str) -> Any:
    """Put a folium vector layer (Polygon / PolyLine) into a Leaflet pane.
    folium silently drops a ``pane=`` kwarg on vector layers, so the option is
    set on the layer's options dict after construction."""
    layer.options["pane"] = pane
    return layer


def ensure_layer_control(map_obj: folium.Map, **kwargs: Any) -> bool:
    """
    Make sure ``map_obj`` has exactly ONE folium.LayerControl.

    The layer-adding functions never add a control themselves; call this once,
    after all layers have been added. Safe to call repeatedly.

    Returns:
        True if a control was added by this call, False if one already existed.
    """
    for child in map_obj._children.values():
        if isinstance(child, folium.LayerControl):
            return False
    folium.LayerControl(**kwargs).add_to(map_obj)
    return True


def create_base_map(
    earth_station_lat: float,
    earth_station_lon: float,
    zoom: int = 8,
) -> folium.Map:
    """
    Create an interactive folium map centered on an earth station location.

    Args:
        earth_station_lat: Latitude of the earth station, in decimal degrees
            (-90 to 90).
        earth_station_lon: Longitude of the earth station, in decimal
            degrees (-180 to 180).
        zoom: Initial zoom level for the map (higher = more zoomed in).
            Defaults to 8.

    Returns:
        A folium.Map object centered on the earth station, with a marker
        already placed at the earth station's location.

    Raises:
        ValueError: If the latitude/longitude are out of valid range, or
            zoom is not a positive integer.
        TypeError: If inputs are not numeric.
    """
    try:
        earth_station_lat = float(earth_station_lat)
        earth_station_lon = float(earth_station_lon)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "earth_station_lat and earth_station_lon must be numeric"
        ) from exc

    if not (-90.0 <= earth_station_lat <= 90.0):
        raise ValueError(
            f"earth_station_lat must be between -90 and 90, got {earth_station_lat}"
        )
    if not (-180.0 <= earth_station_lon <= 180.0):
        raise ValueError(
            f"earth_station_lon must be between -180 and 180, got {earth_station_lon}"
        )
    if not isinstance(zoom, int) or zoom <= 0:
        raise ValueError(f"zoom must be a positive integer, got {zoom!r}")

    try:
        map_obj = folium.Map(
            location=[earth_station_lat, earth_station_lon],
            zoom_start=zoom,
            tiles="OpenStreetMap",
            control_scale=True,
        )
    except Exception as exc:
        logger.error("Failed to create base map: %s", exc)
        raise RuntimeError(f"Could not create base map: {exc}") from exc

    # Place the initial earth station marker on the freshly created map.
    map_obj = add_earth_station_marker(
        map_obj, earth_station_lat, earth_station_lon, station_name="Earth Station"
    )

    logger.info(
        "Base map created centered at (%s, %s), zoom=%s",
        earth_station_lat,
        earth_station_lon,
        zoom,
    )
    return map_obj


def add_country_borders(
    map_obj: folium.Map,
    geojson_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Add country border polygons to an existing folium map as a GeoJSON layer.

    Loads country border data either from a local GeoJSON/shapefile (if
    geojson_path is provided) or by downloading a public world-countries
    GeoJSON dataset as a fallback.

    Args:
        map_obj: An existing folium.Map object to add the layer to (modified
            in place).
        geojson_path: Optional path to a local GeoJSON (or .shp) file
            containing country border geometries. If None or unusable,
            borders are fetched from a public dataset over the network.

    Returns:
        A status dict ``{"ok": bool, "reason": str, "count": int, ...}`` (see
        :func:`layer_status`). ``count`` is the number of border features
        added. On failure the map is left unmodified, ``ok`` is False and
        ``severity`` is "warn" when the layer is merely unavailable (offline)
        or "error" when it could not be rendered. This function does not add
        a LayerControl; see :func:`ensure_layer_control`.

    Raises:
        TypeError: If map_obj is not a folium.Map instance.
    """
    if not isinstance(map_obj, folium.Map):
        raise TypeError("map_obj must be a folium.Map instance")

    borders_data = None
    source_desc = None
    local_problem = ""

    # 1. Try loading from a local file (supports both .geojson and .shp).
    if geojson_path:
        source_desc = geojson_path
        try:
            if not os.path.isfile(geojson_path):
                raise FileNotFoundError(f"No such file: {geojson_path}")

            if geojson_path.lower().endswith(".shp"):
                # Shapefile: use geopandas to read and convert to GeoJSON in memory.
                import geopandas as gpd
                gdf = gpd.read_file(geojson_path)
                borders_data = json.loads(gdf.to_json())
            else:
                with open(geojson_path, "r", encoding="utf-8") as f:
                    borders_data = json.load(f)
        except FileNotFoundError as exc:
            local_problem = f"local file not found: {exc}"
            logger.warning("Country borders file not found: %s", exc)
        except json.JSONDecodeError as exc:
            local_problem = f"invalid GeoJSON in {geojson_path}: {exc}"
            logger.warning("Invalid GeoJSON in %s: %s", geojson_path, exc)
        except OSError as exc:
            local_problem = f"could not read {geojson_path}: {exc}"
            logger.warning("Could not read %s: %s", geojson_path, exc)
        except Exception as exc:  # e.g. geopandas missing / unreadable shapefile
            local_problem = f"could not load {geojson_path}: {exc}"
            logger.warning("Could not load country borders from %s: %s", geojson_path, exc)

    # 2. Fall back to downloading a public borders dataset.
    download_problem = ""
    if borders_data is None:
        source_desc = DEFAULT_BORDERS_URL
        if requests is None:
            download_problem = "'requests' is not installed"
            logger.warning(
                "'requests' is not installed and no valid geojson_path was "
                "given; skipping country borders layer."
            )
        else:
            try:
                response = requests.get(DEFAULT_BORDERS_URL, timeout=15)
                response.raise_for_status()
                borders_data = response.json()
            except requests.exceptions.RequestException as exc:
                download_problem = f"download failed: {exc}"
                logger.warning("Failed to download country borders: %s", exc)
            except json.JSONDecodeError as exc:
                download_problem = f"downloaded data was not valid JSON: {exc}"
                logger.warning("Downloaded borders data was not valid JSON: %s", exc)

    if borders_data is None:
        detail = "; ".join(p for p in (local_problem, download_problem) if p)
        logger.warning(
            "No country borders could be loaded from %s; map left unchanged.",
            source_desc,
        )
        return layer_status(False, "Offline", count=0, severity="warn", detail=detail)

    features = borders_data.get("features") if isinstance(borders_data, dict) else None
    n_features = len(features) if isinstance(features, list) else 0

    try:
        folium.GeoJson(
            borders_data,
            name="Country Borders",
            style_function=lambda feature: {
                "fillColor": "#00000000",
                "color": "#3388ff",
                "weight": 1,
                "fillOpacity": 0.0,
            },
            highlight_function=lambda feature: {
                "weight": 2,
                "color": "#ff7800",
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    field
                    for field in ("name", "NAME", "ADMIN")
                    if borders_data.get("features")
                    and field in borders_data["features"][0].get("properties", {})
                ]
                or None
            ),
        ).add_to(map_obj)
    except Exception as exc:
        logger.error("Failed to add country borders layer: %s", exc)
        return layer_status(False, "Render error", count=0, severity="error", detail=str(exc))

    logger.info("Country borders added from %s (%d features)", source_desc, n_features)
    note = "OK" if not local_problem else "OK (public dataset)"
    return layer_status(True, note, count=n_features, detail=local_problem)


def add_earth_station_marker(
    map_obj: folium.Map,
    lat: float,
    lon: float,
    station_name: str = "ES",
) -> folium.Map:
    """
    Add a customizable marker representing an earth station to a folium map.

    Args:
        map_obj: An existing folium.Map object to add the marker to.
        lat: Latitude of the earth station, in decimal degrees (-90 to 90).
        lon: Longitude of the earth station, in decimal degrees
            (-180 to 180).
        station_name: Label shown in the marker's popup/tooltip. Defaults
            to "ES".

    Returns:
        The updated folium.Map object with the new marker added.

    Raises:
        TypeError: If map_obj is not a folium.Map instance, or lat/lon are
            not numeric.
        ValueError: If lat/lon are out of valid range.
    """
    if not isinstance(map_obj, folium.Map):
        raise TypeError("map_obj must be a folium.Map instance")

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError) as exc:
        raise TypeError("lat and lon must be numeric") from exc

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"lat must be between -90 and 90, got {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"lon must be between -180 and 180, got {lon}")

    try:
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(f"{station_name}<br>({lat:.5f}, {lon:.5f})", max_width=250),
            tooltip=station_name,
            icon=folium.Icon(color="red", icon="satellite-dish", prefix="fa"),
        ).add_to(map_obj)
    except Exception as exc:
        logger.error("Failed to add earth station marker: %s", exc)
        raise RuntimeError(f"Could not add marker for {station_name}: {exc}") from exc

    logger.info("Marker '%s' added at (%s, %s)", station_name, lat, lon)
    return map_obj


def _polar_to_latlon(
    origin_lat: float,
    origin_lon: float,
    azimuth_deg: float,
    distance_km: float,
    geod: Optional["Geod"] = None,
) -> Tuple[float, float]:
    """
    Convert a polar (azimuth, distance) offset from an origin point into an
    absolute (lat, lon) coordinate, using accurate geodesic (ellipsoidal)
    math rather than a flat-earth approximation.

    Args:
        origin_lat: Latitude of the origin point, in decimal degrees.
        origin_lon: Longitude of the origin point, in decimal degrees.
        azimuth_deg: Bearing from the origin, in degrees clockwise from true
            north (0 = north, 90 = east, 180 = south, 270 = west).
        distance_km: Distance from the origin, in kilometers. Must be
            non-negative.
        geod: Optional pre-built pyproj.Geod instance to reuse across many
            calls (avoids re-constructing it for every point). If omitted,
            a WGS84 Geod is created.

    Returns:
        A (lat, lon) tuple for the destination point, in decimal degrees.

    Raises:
        ImportError: If pyproj is not installed.
        ValueError: If distance_km is negative.
    """
    if Geod is None:
        raise ImportError(
            "pyproj is required for geodesic calculations. Install it with "
            "'pip install pyproj'."
        )
    if distance_km < 0:
        raise ValueError(f"distance_km must be non-negative, got {distance_km}")

    geod = geod or Geod(ellps="WGS84")
    distance_m = distance_km * 1000.0
    dest_lon, dest_lat, _back_azimuth = geod.fwd(
        origin_lon, origin_lat, azimuth_deg, distance_m
    )
    return dest_lat, dest_lon


def calculate_contour_points(
    earth_station_lat: float,
    earth_station_lon: float,
    distances_by_azimuth: Dict[float, float],
) -> List[Tuple[float, float]]:
    """
    Convert a set of (azimuth, distance) pairs around an earth station into
    absolute (lat, lon) coordinates, suitable for drawing a coordination
    contour polygon.

    Uses pyproj.Geod (WGS84 ellipsoid) for accurate geodesic forward
    calculations rather than a planar approximation, which matters at the
    distances typically involved in earth-station coordination contours
    (tens to hundreds of km).

    Args:
        earth_station_lat: Latitude of the earth station, in decimal
            degrees (-90 to 90).
        earth_station_lon: Longitude of the earth station, in decimal
            degrees (-180 to 180).
        distances_by_azimuth: Dict mapping azimuth in degrees (0-360,
            clockwise from true north) to coordination distance in
            kilometers at that azimuth. Order is preserved from the dict as
            given, so pass an already azimuth-sorted dict (e.g. via
            `dict(sorted(...))`) if a clean, non-self-intersecting polygon
            is required.

    Returns:
        A list of (lat, lon) tuples, one per entry in distances_by_azimuth,
        in the same order as the input dict.

    Raises:
        ImportError: If pyproj is not installed.
        TypeError: If earth_station_lat/lon are not numeric, or
            distances_by_azimuth is not a dict.
        ValueError: If distances_by_azimuth is empty, or contains an
            invalid azimuth or a negative distance.
    """
    if Geod is None:
        raise ImportError(
            "pyproj is required for geodesic calculations. Install it with "
            "'pip install pyproj'."
        )
    try:
        earth_station_lat = float(earth_station_lat)
        earth_station_lon = float(earth_station_lon)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "earth_station_lat and earth_station_lon must be numeric"
        ) from exc

    if not (-90.0 <= earth_station_lat <= 90.0):
        raise ValueError(
            f"earth_station_lat must be between -90 and 90, got {earth_station_lat}"
        )
    if not (-180.0 <= earth_station_lon <= 180.0):
        raise ValueError(
            f"earth_station_lon must be between -180 and 180, got {earth_station_lon}"
        )
    if not isinstance(distances_by_azimuth, dict):
        raise TypeError("distances_by_azimuth must be a dict")
    if not distances_by_azimuth:
        raise ValueError("distances_by_azimuth must not be empty")

    geod = Geod(ellps="WGS84")
    points: List[Tuple[float, float]] = []

    for azimuth_deg, distance_km in distances_by_azimuth.items():
        try:
            azimuth_deg = float(azimuth_deg)
            distance_km = float(distance_km)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Azimuth/distance pair ({azimuth_deg!r}, {distance_km!r}) "
                "must be numeric"
            ) from exc

        if not (0.0 <= azimuth_deg <= 360.0):
            raise ValueError(
                f"azimuth must be between 0 and 360 degrees, got {azimuth_deg}"
            )
        if distance_km < 0:
            raise ValueError(
                f"distance_km must be non-negative, got {distance_km} "
                f"(at azimuth {azimuth_deg})"
            )

        lat, lon = _polar_to_latlon(
            earth_station_lat, earth_station_lon, azimuth_deg, distance_km, geod=geod
        )
        points.append((lat, lon))

    logger.info(
        "Calculated %d contour points around (%s, %s)",
        len(points),
        earth_station_lat,
        earth_station_lon,
    )
    return points


def add_coordination_contour(
    map_obj: folium.Map,
    earth_station_lat: float,
    earth_station_lon: float,
    distances_by_azimuth: Dict[float, float],
    color: str = "red",
    line_style: str = "dashed",
) -> folium.Map:
    """
    Draw a coordination contour (a polygon showing variable coordination
    distance by azimuth) around an earth station on a folium map.

    Args:
        map_obj: An existing folium.Map object to add the contour to.
        earth_station_lat: Latitude of the earth station, in decimal
            degrees.
        earth_station_lon: Longitude of the earth station, in decimal
            degrees.
        distances_by_azimuth: Dict mapping azimuth in degrees (0-360) to
            coordination distance in kilometers at that azimuth. For a
            clean polygon outline, supply azimuths in sorted order (e.g.
            `dict(sorted(distances_by_azimuth.items()))`).
        color: Line (and light fill) color for the contour. Any folium/CSS
            color string. Defaults to 'red'.
        line_style: One of 'solid', 'dashed', or 'dotted'. Defaults to
            'dashed'.

    Returns:
        The updated folium.Map object with the coordination contour
        polygon added.

    Raises:
        ImportError: If pyproj is not installed.
        TypeError: If map_obj is not a folium.Map instance, or inputs are
            not numeric/dict as required.
        ValueError: If distances_by_azimuth is empty or contains invalid
            values, or line_style is not recognized.
    """
    if not isinstance(map_obj, folium.Map):
        raise TypeError("map_obj must be a folium.Map instance")

    dash_arrays = {"solid": None, "dashed": "8, 6", "dotted": "1, 6"}
    if line_style not in dash_arrays:
        raise ValueError(
            f"line_style must be one of {list(dash_arrays)}, got {line_style!r}"
        )

    # Sort by azimuth so the polygon traces a clean loop rather than
    # jumping around based on dict insertion order.
    sorted_distances = dict(sorted(distances_by_azimuth.items(), key=lambda kv: kv[0]))
    contour_points = calculate_contour_points(
        earth_station_lat, earth_station_lon, sorted_distances
    )

    try:
        # Contour + sectors share a pane that sits above the DEM hillshade.
        _ensure_pane(map_obj, PANE_COORD, Z_INDEX_COORD)
        contour_polygon = folium.Polygon(
            locations=contour_points,
            color=color,
            weight=2,
            dash_array=dash_arrays[line_style],
            fill=True,
            fill_color=color,
            fill_opacity=0.08,
            popup="Coordination Contour",
            tooltip="Coordination Contour",
        )
        _in_pane(contour_polygon, PANE_COORD)
        contour_group = folium.FeatureGroup(name="Coordination Contour")
        contour_group.add_to(map_obj)
        contour_polygon.add_to(contour_group)
    except Exception as exc:
        logger.error("Failed to add coordination contour: %s", exc)
        raise RuntimeError(f"Could not add coordination contour: {exc}") from exc

    logger.info(
        "Coordination contour added with %d points (color=%s, style=%s)",
        len(contour_points),
        color,
        line_style,
    )
    return map_obj


def add_distance_circle(
    map_obj: folium.Map,
    lat: float,
    lon: float,
    radius_km: float,
    color: str = "blue",
    fill: bool = False,
    label: str = "",
) -> folium.Map:
    """
    Draw a constant-distance circle (e.g. a fixed coordination or exclusion
    radius) around a point on a folium map.

    Note: folium.Circle renders a true geodesic-radius circle (via Leaflet),
    so this is accurate even at large radii, unlike a naive lat/lon-degree
    circle.

    Args:
        map_obj: An existing folium.Map object to add the circle to.
        lat: Latitude of the circle's center, in decimal degrees.
        lon: Longitude of the circle's center, in decimal degrees.
        radius_km: Circle radius in kilometers. Must be positive.
        color: Circle outline (and fill, if enabled) color. Defaults to
            'blue'.
        fill: Whether to fill the circle. Defaults to False.
        label: Optional text shown in the circle's tooltip/popup.

    Returns:
        The updated folium.Map object with the distance circle added.

    Raises:
        TypeError: If map_obj is not a folium.Map instance, or lat/lon/
            radius_km are not numeric.
        ValueError: If lat/lon are out of range or radius_km is not
            positive.
    """
    if not isinstance(map_obj, folium.Map):
        raise TypeError("map_obj must be a folium.Map instance")

    try:
        lat = float(lat)
        lon = float(lon)
        radius_km = float(radius_km)
    except (TypeError, ValueError) as exc:
        raise TypeError("lat, lon, and radius_km must be numeric") from exc

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"lat must be between -90 and 90, got {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"lon must be between -180 and 180, got {lon}")
    if radius_km <= 0:
        raise ValueError(f"radius_km must be positive, got {radius_km}")

    tooltip_text = label if label else f"{radius_km:g} km radius"

    try:
        folium.Circle(
            location=[lat, lon],
            radius=radius_km * 1000.0,  # folium.Circle expects meters
            color=color,
            weight=2,
            fill=fill,
            fill_color=color if fill else None,
            fill_opacity=0.15 if fill else 0.0,
            popup=tooltip_text,
            tooltip=tooltip_text,
        ).add_to(map_obj)
    except Exception as exc:
        logger.error("Failed to add distance circle: %s", exc)
        raise RuntimeError(f"Could not add distance circle: {exc}") from exc

    logger.info(
        "Distance circle added at (%s, %s), radius=%s km, label=%r",
        lat,
        lon,
        radius_km,
        label,
    )
    return map_obj


def _require_rasterio() -> None:
    """Raise a clear ImportError if rasterio is unavailable."""
    if rasterio is None:
        raise ImportError(
            "rasterio is required for DEM operations. Install it with "
            "'pip install rasterio'."
        )


# def load_dem_data(dem_tiff_path: str) -> Tuple["np.ndarray", Dict[str, Any]]:
#     """
#     Read a Digital Elevation Model (DEM) GeoTIFF using rasterio.

#     Args:
#         dem_tiff_path: Path to a single-band DEM GeoTIFF file.

#     Returns:
#         A tuple of (elevation, info):
#             elevation: A 2D numpy array of elevation values (band 1), in
#                 the file's native units (typically meters).
#             info: A dict with:
#                 - 'transform': the affine rasterio.Affine transform
#                   (pixel -> source CRS coordinates)
#                 - 'crs': the source rasterio.crs.CRS
#                 - 'bounds': (west, south, east, north) in the source CRS
#                 - 'width' / 'height': raster dimensions in pixels
#                 - 'nodata': the nodata value (may be None)

#     Raises:
#         ImportError: If rasterio is not installed.
#         FileNotFoundError: If dem_tiff_path does not exist.
#         RuntimeError: If the file cannot be read or opened as a raster.
#     """
#     _require_rasterio()

#     if not os.path.isfile(dem_tiff_path):
#         raise FileNotFoundError(f"DEM file not found: {dem_tiff_path}")

#     try:
#         with rasterio.open(dem_tiff_path) as dataset:
#             elevation = dataset.read(1).astype("float64")
#             info = {
#                 "transform": dataset.transform,
#                 "crs": dataset.crs,
#                 "bounds": tuple(dataset.bounds),
#                 "width": dataset.width,
#                 "height": dataset.height,
#                 "nodata": dataset.nodata,
#             }
#     except rasterio.errors.RasterioIOError as exc:
#         raise RuntimeError(f"Could not open DEM file {dem_tiff_path}: {exc}") from exc
#     except Exception as exc:
#         raise RuntimeError(f"Failed to read DEM file {dem_tiff_path}: {exc}") from exc

#     if info["nodata"] is not None:
#         elevation = np.where(elevation == info["nodata"], np.nan, elevation)

#     logger.info(
#         "Loaded DEM %s: %dx%d px, CRS=%s, bounds=%s",
#         dem_tiff_path,
#         info["width"],
#         info["height"],
#         info["crs"],
#         info["bounds"],
#     )
#     return elevation, info



def load_dem_data(dem_tiff_path: str, max_dim: int = 4096) -> Tuple["np.ndarray", Dict[str, Any]]:
    _require_rasterio()

    if not os.path.isfile(dem_tiff_path):
        raise FileNotFoundError(f"DEM file not found: {dem_tiff_path}")

    try:
        with rasterio.open(dem_tiff_path) as dataset:
            # Downsample on read so the longest side is <= max_dim pixels.
            scale = min(1.0, max_dim / max(dataset.width, dataset.height))
            out_h = max(1, int(round(dataset.height * scale)))
            out_w = max(1, int(round(dataset.width * scale)))

            masked = dataset.read(
                1,
                out_shape=(out_h, out_w),
                resampling=Resampling.average,
                masked=True,               # respects nodata during averaging
            )
            elevation = masked.astype("float32").filled(np.nan)
            del masked

            info = {
                # Adjust the transform for the coarser pixel size
                "transform": dataset.transform * dataset.transform.scale(
                    dataset.width / out_w, dataset.height / out_h
                ),
                "crs": dataset.crs,
                "bounds": tuple(dataset.bounds),
                "width": out_w,
                "height": out_h,
                "nodata": dataset.nodata,
            }
    except rasterio.errors.RasterioIOError as exc:
        raise RuntimeError(f"Could not open DEM file {dem_tiff_path}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read DEM file {dem_tiff_path}: {exc}") from exc

    # (the old `np.where(elevation == nodata, nan, ...)` block is no longer needed: delete it)
    logger.info("Loaded DEM %s downsampled to %dx%d px", dem_tiff_path, out_w, out_h)
    return elevation, info



def get_elevation_at_point(dem_path: str, lat: float, lon: float) -> Optional[float]:
    """
    Look up the elevation value from a DEM GeoTIFF at a specific lat/lon.

    Useful for validating a DEM overlay (e.g. checking the elevation under
    the earth station itself).

    Args:
        dem_path: Path to a single-band DEM GeoTIFF file.
        lat: Latitude of the query point, in decimal degrees (WGS84).
        lon: Longitude of the query point, in decimal degrees (WGS84).

    Returns:
        The elevation value (in the DEM's native units, typically meters)
        at the given point, or None if the point falls on a nodata pixel.

    Raises:
        ImportError: If rasterio is not installed.
        FileNotFoundError: If dem_path does not exist.
        TypeError: If lat/lon are not numeric.
        ValueError: If lat/lon are out of range, or the point falls outside
            the DEM's coverage.
        RuntimeError: If the file cannot be read.
    """
    _require_rasterio()

    if not os.path.isfile(dem_path):
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError) as exc:
        raise TypeError("lat and lon must be numeric") from exc

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"lat must be between -90 and 90, got {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"lon must be between -180 and 180, got {lon}")

    try:
        with rasterio.open(dem_path) as dataset:
            # Reproject the query point from WGS84 into the DEM's native CRS.
            if dataset.crs is not None and dataset.crs.to_epsg() != 4326:
                xs, ys = warp_transform("EPSG:4326", dataset.crs, [lon], [lat])
                x, y = xs[0], ys[0]
            else:
                x, y = lon, lat

            row, col = dataset.index(x, y)

            if not (0 <= row < dataset.height and 0 <= col < dataset.width):
                raise ValueError(
                    f"Point ({lat}, {lon}) falls outside the DEM's coverage "
                    f"(bounds={tuple(dataset.bounds)})"
                )

            value = dataset.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
            value = float(value)

            if dataset.nodata is not None and value == dataset.nodata:
                logger.warning(
                    "Elevation at (%s, %s) is nodata in %s", lat, lon, dem_path
                )
                return None
    except rasterio.errors.RasterioIOError as exc:
        raise RuntimeError(f"Could not open DEM file {dem_path}: {exc}") from exc

    logger.info("Elevation at (%s, %s) = %s", lat, lon, value)
    return value


def _reproject_dem_to_wgs84(
    elevation: "np.ndarray", info: Dict[str, Any]
) -> Tuple["np.ndarray", Any, int, int]:
    """
    Reproject a DEM array (in its native CRS) into EPSG:4326 (WGS84).

    Args:
        elevation: 2D elevation array as returned by load_dem_data.
        info: The corresponding info dict from load_dem_data.

    Returns:
        A tuple of (elevation_wgs84, dst_transform, dst_width, dst_height),
        where nodata/out-of-coverage pixels are np.nan.
    """
    src_crs = info["crs"]
    if src_crs is None:
        raise ValueError(
            "DEM has no CRS defined; cannot reproject to WGS84 for overlay."
        )

    dst_crs = "EPSG:4326"
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, info["width"], info["height"], *info["bounds"]
    )

    # dem_wgs84 = np.full((dst_height, dst_width), np.nan, dtype="float64")

    dem_wgs84 = np.full((dst_height, dst_width), np.nan, dtype="float32")

    reproject(
        source=elevation,
        destination=dem_wgs84,
        src_transform=info["transform"],
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    return dem_wgs84, dst_transform, dst_width, dst_height


# ---------------------------------------------------------------------------
# Static (prebuilt) DEM hillshade + local web assets
# ---------------------------------------------------------------------------
# tools/prepare_dem_overlay.py writes these two files once, at build time. When both exist
# the map uses them (a plain Leaflet ImageOverlay) and never touches rasterio.
BUNDLED_DEM_RELPATH = "data_sources/azerbaijan_dem.tif"
HILLSHADE_PNG_RELPATH = "data_sources/dem_hillshade.png"
HILLSHADE_BOUNDS_RELPATH = "data_sources/dem_bounds.json"

# Optional URL under which the host app serves the PNG (desktop_app registers it). If None,
# the PNG is embedded in the map HTML as a data: URI (works anywhere, but larger files).
_STATIC_HILLSHADE_URL: Optional[str] = None

# Where desktop_app serves ui/static/web_assets (Leaflet, Bootstrap, FontAwesome, ...).
WEB_ASSETS_RELPATH = "ui/static/web_assets"
WEB_ASSETS_URL = "/web_assets"


def configure_static_hillshade_url(url: Optional[str]) -> None:
    """Tell this module the URL at which the host app serves dem_hillshade.png."""
    global _STATIC_HILLSHADE_URL
    _STATIC_HILLSHADE_URL = url


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def static_hillshade_available() -> bool:
    return (os.path.isfile(get_resource_path(HILLSHADE_PNG_RELPATH))
            and os.path.isfile(get_resource_path(HILLSHADE_BOUNDS_RELPATH)))


def _is_bundled_dem(dem_path: Optional[str]) -> bool:
    """True if ``dem_path`` is the bundled DEM (the prebuilt PNG only describes that one)."""
    return (not dem_path) or _same_path(dem_path, get_resource_path(BUNDLED_DEM_RELPATH))


def dem_overlay_available(dem_path: Optional[str]) -> bool:
    """Can a DEM layer be drawn for ``dem_path``? Either the GeoTIFF exists, or it is the
    bundled DEM and its prebuilt PNG hillshade is shipped."""
    if dem_path and os.path.isfile(dem_path):
        return True
    return _is_bundled_dem(dem_path) and static_hillshade_available()


def _add_static_hillshade(
    map_obj: folium.Map, earth_station_lat: float, earth_station_lon: float, opacity: float
) -> Optional[Dict[str, Any]]:
    """Add the prebuilt PNG as a Folium ImageOverlay. Returns a status dict, or None if the
    static files are missing/unusable (the caller then falls back to rasterio).

    The PNG is already in Web Mercator (EPSG:3857), which is exactly how Leaflet stretches an
    ImageOverlay between its corner points, so no reprojection happens here."""
    png = get_resource_path(HILLSHADE_PNG_RELPATH)
    meta = get_resource_path(HILLSHADE_BOUNDS_RELPATH)
    if not (os.path.isfile(png) and os.path.isfile(meta)):
        return None
    try:
        with open(meta, "r", encoding="utf-8") as fh:
            (south, west), (north, east) = [[float(v) for v in pt] for pt in json.load(fh)["bounds"]]
    except Exception as exc:
        logger.warning("Static DEM bounds unreadable (%s): %s; falling back to rasterio.", meta, exc)
        return None

    try:
        _ensure_pane(map_obj, PANE_DEM, Z_INDEX_DEM)
        overlay = folium.raster_layers.ImageOverlay(
            image=png,                      # a file path -> folium embeds it as a data: URI ...
            bounds=[[south, west], [north, east]],
            opacity=float(opacity),         # the PNG's alpha only marks "no data"; opacity is live
            name="DEM Hillshade",
            interactive=False,
            cross_origin=False,
            pane=PANE_DEM,
        )
        if _STATIC_HILLSHADE_URL:
            overlay.url = _STATIC_HILLSHADE_URL   # ... unless the app serves it: keeps map HTML small
        overlay.add_to(map_obj)
    except Exception as exc:
        logger.warning("Static DEM overlay failed (%s); falling back to rasterio.", exc)
        return None

    logger.info("Static DEM hillshade overlay added from %s (bounds=[%s,%s]-[%s,%s])",
                png, south, west, north, east)
    if not (south <= earth_station_lat <= north and west <= earth_station_lon <= east):
        note = (f"Earth station ({earth_station_lat}, {earth_station_lon}) is outside the DEM "
                f"coverage (west={west}, south={south}, east={east}, north={north})")
        logger.warning("%s; overlay was still added.", note)
        return layer_status(True, "Station outside DEM", count=1, severity="warn", detail=note)
    return layer_status(True, "OK (static)", count=1)


# --- local web assets (Leaflet / Bootstrap / FontAwesome ...) ----------------------------
# Folium's HTML links these from public CDNs. After the map is saved, every known CDN URL
# is rewritten to /web_assets/... IF the matching file exists in ui/static/web_assets, so a
# partially populated folder never breaks a map. Layout is created by tools/fetch_web_assets.py.
_CDN_URL_RE = re.compile(
    r"""(?P<url>(?:https?:)?//(?:cdn\.jsdelivr\.net|cdnjs\.cloudflare\.com|unpkg\.com|"""
    r"""code\.jquery\.com|netdna\.bootstrapcdn\.com|maxcdn\.bootstrapcdn\.com|"""
    r"""stackpath\.bootstrapcdn\.com)/[^"'\s)<>]+)"""
)


def _local_asset_for(url: str) -> Optional[str]:
    """Relative path (under WEB_ASSETS_RELPATH) that replaces a CDN ``url``, or None."""
    base = url.split("?", 1)[0].rsplit("/", 1)[-1]
    low = url.lower()
    if base == "leaflet.js":
        return "leaflet/leaflet.js"
    if base == "leaflet.css":
        return "leaflet/leaflet.css"
    if re.fullmatch(r"jquery[-.\d]*(?:min\.)?js", base):
        return "jquery/jquery.min.js"
    if base in ("bootstrap.bundle.min.js", "bootstrap.min.css", "bootstrap.min.js"):
        # Only Bootstrap 5 (what Folium >= 0.14 uses); never swap in a different major version.
        return f"bootstrap/{base}" if ("bootstrap@5" in low or "/bootstrap/5." in low) else None
    if "glyphicons" in base:
        return "glyphicons/bootstrap-glyphicons.css"      # stub: the app only uses fa icons
    if base in ("all.min.css", "all.css") and "fontawesome" in low:
        return "fontawesome/css/all.min.css"
    if re.fullmatch(r"leaflet\.awesome-markers(?:\.min)?\.js", base):
        return "awesome-markers/leaflet.awesome-markers.js"
    if re.fullmatch(r"leaflet\.awesome-markers(?:\.min)?\.css", base):
        return "awesome-markers/leaflet.awesome-markers.css"
    if base.startswith("leaflet.awesome.rotate"):
        return "awesome-markers/leaflet.awesome.rotate.min.css"
    return None


def localize_web_assets_html(html: str, assets_dir: Optional[str] = None,
                             base_url: str = WEB_ASSETS_URL) -> Tuple[str, List[str]]:
    """Rewrite known CDN URLs in ``html`` to ``base_url``/<file>. Returns
    ``(new_html, still_remote_urls)``; URLs whose local file is missing are left unchanged."""
    root = assets_dir or get_resource_path(WEB_ASSETS_RELPATH)
    remaining: List[str] = []

    def repl(m: "re.Match[str]") -> str:
        url = m.group("url")
        rel = _local_asset_for(url)
        if rel and os.path.isfile(os.path.join(root, *rel.split("/"))):
            return f"{base_url.rstrip('/')}/{rel}"
        remaining.append(url)
        return url

    return _CDN_URL_RE.sub(repl, html), remaining


def localize_map_file(path: Any, assets_dir: Optional[str] = None) -> bool:
    """Post-process a saved Folium map file in place (see localize_web_assets_html).
    Returns True if anything was rewritten. Never raises for a missing assets folder."""
    root = assets_dir or get_resource_path(WEB_ASSETS_RELPATH)
    if not os.path.isdir(root):
        logger.warning("Local web assets folder not found (%s); map still uses CDN links.", root)
        return False
    with open(path, "r", encoding="utf-8") as fh:
        html = fh.read()
    new_html, remaining = localize_web_assets_html(html, root)
    if remaining:
        logger.warning("Map still references %d remote asset(s) (not bundled): %s",
                       len(remaining), sorted(set(remaining)))
    if new_html == html:
        return False
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(new_html)
    return True


def add_dem_overlay(
    map_obj: folium.Map,
    dem_path: str,
    earth_station_lat: float,
    earth_station_lon: float,
    opacity: float = 0.5,
) -> Dict[str, Any]:
    """
    Overlay a hillshaded Digital Elevation Model (DEM) on a folium map.

    Loads the DEM, reprojects it to WGS84 (lat/lon) so it can be placed on
    the map, computes a hillshade (a shaded-relief grayscale rendering of
    the terrain), and adds it as a semi-transparent image overlay in a custom
    Leaflet pane that sits above the base tiles but BELOW the default overlay
    pane (so contour / sectors / borders always draw on top of it). The earth
    station coordinates are used to confirm the DEM actually covers the
    station and to log the elevation there for a quick sanity check.

    Args:
        map_obj: An existing folium.Map object to add the overlay to (modified
            in place).
        dem_path: Path to a single-band DEM GeoTIFF file.
        earth_station_lat: Latitude of the earth station, in decimal degrees.
        earth_station_lon: Longitude of the earth station, in decimal degrees.
        opacity: Overlay opacity from 0 (fully transparent) to 1 (fully
            opaque). Defaults to 0.5.

    Returns:
        A status dict ``{"ok": bool, "reason": str, "count": int, ...}`` (see
        :func:`layer_status`). On any failure to load / process / add the DEM
        the map is left unmodified and ``ok`` is False, with ``reason`` one of
        e.g. "File not found", "Missing dependency", "Unreadable",
        "Reprojection failed", "No valid data", "Hillshade failed",
        "Render error". If the DEM does not cover the earth station the
        overlay is still added: ``ok`` is True with ``severity`` "warn". This
        function does not add a LayerControl; see :func:`ensure_layer_control`.

    Raises:
        TypeError: If map_obj is not a folium.Map instance.
        ValueError: If opacity is out of the 0-1 range.
    """
    if not isinstance(map_obj, folium.Map):
        raise TypeError("map_obj must be a folium.Map instance")
    if not (0.0 <= opacity <= 1.0):
        raise ValueError(f"opacity must be between 0 and 1, got {opacity}")

    # Preferred path: the prebuilt static hillshade (tools/prepare_dem_overlay.py). Only valid
    # for the bundled DEM; a user-chosen GeoTIFF always takes the rasterio path below.
    if _is_bundled_dem(dem_path):
        static_status = _add_static_hillshade(map_obj, earth_station_lat, earth_station_lon, opacity)
        if static_status is not None:
            return static_status

    try:
        _require_rasterio()
        if np is None:
            raise ImportError("numpy is required for DEM operations. Install it with 'pip install numpy'.")
        if LightSource is None:
            raise ImportError(
                "matplotlib is required for hillshading. Install it with 'pip install matplotlib'."
            )
    except ImportError as exc:
        logger.warning("DEM overlay unavailable: %s", exc)
        return layer_status(False, "Missing dependency", detail=str(exc))

    if not dem_path or not os.path.isfile(dem_path):
        logger.warning("Could not load DEM %s: file not found", dem_path)
        return layer_status(False, "File not found", detail=f"DEM file not found: {dem_path!r}")

    try:
        elevation, info = load_dem_data(dem_path)
    except FileNotFoundError as exc:
        logger.warning("Could not load DEM %s: %s", dem_path, exc)
        return layer_status(False, "File not found", detail=str(exc))
    except Exception as exc:
        logger.warning("Could not load DEM %s: %s", dem_path, exc)
        return layer_status(False, "Unreadable", detail=str(exc))

    try:
        dem_wgs84, dst_transform, width, height = _reproject_dem_to_wgs84(elevation, info)
    except Exception as exc:
        logger.warning("Could not reproject DEM %s to WGS84: %s", dem_path, exc)
        return layer_status(False, "Reprojection failed", detail=str(exc))

    west, south, east, north = array_bounds(height, width, dst_transform)

    outside_note = ""
    if not (south <= earth_station_lat <= north and west <= earth_station_lon <= east):
        outside_note = (
            f"Earth station ({earth_station_lat}, {earth_station_lon}) is outside the DEM "
            f"coverage (west={west}, south={south}, east={east}, north={north})"
        )
        logger.warning("%s; overlay will still be added.", outside_note)
    else:
        try:
            es_elevation = get_elevation_at_point(dem_path, earth_station_lat, earth_station_lon)
            logger.info("Elevation at earth station: %s", es_elevation)
        except Exception as exc:
            logger.warning("Could not determine elevation at earth station: %s", exc)

    # Approximate meters-per-pixel spacing for the hillshade light model,
    # since the reprojected grid is in degrees, not meters.
    mean_lat_rad = np.deg2rad((south + north) / 2.0)
    dx_m = abs(dst_transform.a) * 111_320.0 * max(np.cos(mean_lat_rad), 1e-6)
    dy_m = abs(dst_transform.e) * 110_540.0

    valid_mask = ~np.isnan(dem_wgs84)
    if not np.any(valid_mask):
        logger.warning("DEM %s has no valid elevation data after reprojection.", dem_path)
        return layer_status(False, "No valid data", detail=f"No valid elevation data in {dem_path}")

    filled = np.where(valid_mask, dem_wgs84, np.nanmin(dem_wgs84[valid_mask]))

    try:
        light_source = LightSource(azdeg=315, altdeg=45)
        hillshade = light_source.hillshade(filled, vert_exag=1, dx=dx_m, dy=dy_m)
    except Exception as exc:
        logger.error("Failed to compute hillshade: %s", exc)
        return layer_status(False, "Hillshade failed", detail=str(exc))

    gray = (np.clip(hillshade, 0, 1) * 255).astype("uint8")
    rgba = np.zeros((height, width, 4), dtype="uint8")
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = np.where(valid_mask, int(round(opacity * 255)), 0).astype("uint8")

    try:
        _ensure_pane(map_obj, PANE_DEM, Z_INDEX_DEM)
        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[south, west], [north, east]],
            opacity=1.0,  # opacity is already baked into the RGBA alpha channel
            name="DEM Hillshade",
            interactive=False,
            cross_origin=False,
            pane=PANE_DEM,
        ).add_to(map_obj)
    except Exception as exc:
        logger.error("Failed to add DEM overlay: %s", exc)
        return layer_status(False, "Render error", detail=str(exc))

    logger.info(
        "DEM hillshade overlay added from %s (bounds=[%s,%s]-[%s,%s], opacity=%s)",
        dem_path, south, west, north, east, opacity,
    )
    if outside_note:
        return layer_status(True, "Station outside DEM", count=1, severity="warn", detail=outside_note)
    return layer_status(True, "OK", count=1)


_REQUIRED_INTERFERENCE_KEYS = (
    "azimuth",
    "border_distance_km",
    "required_distance_km",
    "in_ratio_db",
    "limit_db",
    "status",
)


def _validate_interference_result(result: Dict[str, Any], index: int) -> None:
    """Validate a single interference-analysis result dict, raising with a
    helpful message (including its index) if something is missing or of
    the wrong type."""
    if not isinstance(result, dict):
        raise TypeError(f"analysis_results[{index}] must be a dict, got {type(result)}")

    missing = [k for k in _REQUIRED_INTERFERENCE_KEYS if k not in result]
    if missing:
        raise ValueError(
            f"analysis_results[{index}] is missing required key(s): {missing}"
        )

    for key in ("azimuth", "border_distance_km", "required_distance_km", "in_ratio_db", "limit_db"):
        try:
            float(result[key])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"analysis_results[{index}]['{key}'] must be numeric, got {result[key]!r}"
            ) from exc

    status = str(result["status"]).upper()
    if status not in ("PASS", "FAIL"):
        raise ValueError(
            f"analysis_results[{index}]['status'] must be 'PASS' or 'FAIL', "
            f"got {result['status']!r}"
        )


def _get_map_center(map_obj: folium.Map) -> Tuple[float, float]:
    """
    Best-effort retrieval of a folium.Map's center (as set via its
    `location` argument, e.g. by create_base_map). Used as the implicit
    earth-station origin for functions that don't take explicit
    lat/lon arguments.

    Raises:
        ValueError: If the map has no discoverable center location.
    """
    location = getattr(map_obj, "location", None)
    if not location or len(location) != 2:
        raise ValueError(
            "Could not determine the earth station location from map_obj. "
            "Build the map with create_base_map(earth_station_lat, "
            "earth_station_lon, ...) first, or pass earth_station_lat/"
            "earth_station_lon explicitly."
        )
    return float(location[0]), float(location[1])


def _build_sector_polygon(
    earth_station_lat: float,
    earth_station_lon: float,
    azimuth_deg: float,
    half_width_deg: float,
    radius_km: float,
    geod: Optional["Geod"] = None,
    arc_points: int = 8,
) -> List[Tuple[float, float]]:
    """
    Build a closed pie-slice ("sector") polygon centered on a given azimuth,
    from the earth station out to radius_km, using accurate geodesic math.

    Args:
        earth_station_lat / earth_station_lon: Origin point.
        azimuth_deg: Center bearing of the sector, in degrees from true
            north.
        half_width_deg: Angular half-width of the sector, in degrees (the
            sector spans azimuth_deg - half_width_deg to
            azimuth_deg + half_width_deg).
        radius_km: Sector radius in kilometers.
        geod: Optional pre-built pyproj.Geod to reuse.
        arc_points: Number of points used to approximate the outer arc
            (higher = smoother curve).

    Returns:
        A list of (lat, lon) tuples forming a closed polygon: the earth
        station, the outer arc, and back to the earth station.
    """
    if Geod is None:
        raise ImportError(
            "pyproj is required for geodesic calculations. Install it with "
            "'pip install pyproj'."
        )
    geod = geod or Geod(ellps="WGS84")

    start_az = azimuth_deg - half_width_deg
    end_az = azimuth_deg + half_width_deg
    arc = [
        _polar_to_latlon(
            earth_station_lat, earth_station_lon,
            start_az + (end_az - start_az) * i / max(arc_points - 1, 1),
            radius_km, geod=geod,
        )
        for i in range(arc_points)
    ]
    return [(earth_station_lat, earth_station_lon)] + arc + [(earth_station_lat, earth_station_lon)]


def interference_half_width(analysis_results: List[Dict[str, Any]]) -> float:
    """
    Angular half-width (degrees) for the sectors of ``analysis_results``,
    estimated from the spacing between consecutive azimuths so adjacent
    sectors tile without large gaps or overlaps. Rows with a missing /
    non-finite azimuth are ignored; a single usable azimuth gives 15 deg.
    """
    azimuths: List[float] = []
    for r in analysis_results:
        try:
            az = float(r["azimuth"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(az):
            azimuths.append(az % 360)
    azimuths.sort()
    if len(azimuths) > 1:
        diffs = [
            (azimuths[(i + 1) % len(azimuths)] - azimuths[i]) % 360
            for i in range(len(azimuths))
        ]
        return (sum(diffs) / len(diffs)) / 2.0
    return 15.0  # default: a 30-degree-wide sector


def create_interference_group(map_obj: folium.Map, name: str = "PASS/FAIL Sectors") -> folium.FeatureGroup:
    """
    Create the (layer-control-visible) feature group the sectors are drawn
    into, and make sure the coordination pane (above the DEM) exists. Pass the
    returned group to :func:`draw_interference_sector`.
    """
    _ensure_pane(map_obj, PANE_COORD, Z_INDEX_COORD)
    group = folium.FeatureGroup(name=name)
    group.add_to(map_obj)
    return group


def draw_interference_sector(
    target: Any,
    result: Dict[str, Any],
    earth_station_lat: float,
    earth_station_lon: float,
    half_width: float,
    geod: Optional["Geod"] = None,
    index: int = 0,
) -> None:
    """
    Draw ONE PASS/FAIL sector (plus its dashed required-distance arc) into
    ``target`` (a folium.FeatureGroup from :func:`create_interference_group`,
    or a folium.Map). Both shapes are built before either is added, so a
    failure never leaves half a sector on the map.

    Raises:
        TypeError / ValueError: invalid row (missing keys, non-numeric,
            NaN / inf values, bad status).
        Exception: any geodesic / folium failure for this row. The caller is
            expected to catch it, log it and skip the row.
    """
    _validate_interference_result(result, index)

    azimuth = float(result["azimuth"])
    border_km = float(result["border_distance_km"])
    required_km = float(result["required_distance_km"])
    in_ratio = float(result["in_ratio_db"])
    limit = float(result["limit_db"])
    for name, value in (
        ("azimuth", azimuth), ("border_distance_km", border_km),
        ("required_distance_km", required_km), ("in_ratio_db", in_ratio),
        ("limit_db", limit),
    ):
        if not math.isfinite(value):
            raise ValueError(f"analysis_results[{index}]['{name}'] is not finite: {value!r}")

    status = str(result["status"]).upper()
    color = "green" if status == "PASS" else "red"
    margin_db = limit - in_ratio

    popup_html = (
        f"<b>Azimuth:</b> {azimuth:.1f}&deg;<br>"
        f"<b>Status:</b> <span style='color:{color}; font-weight:bold'>{status}</span><br>"
        f"<b>Border distance:</b> {border_km:.1f} km<br>"
        f"<b>Required distance:</b> {required_km:.1f} km<br>"
        f"<b>I/N ratio:</b> {in_ratio:.2f} dB<br>"
        f"<b>Limit:</b> {limit:.2f} dB<br>"
        f"<b>Margin:</b> {margin_db:+.2f} dB"
    )

    # Optional ITU-R P.526 terrain-diffraction recheck (see
    # streamlit_map_component._build_zone_results / app.py Step 7).
    # Only present for azimuths that FAILed the plain I/N check and had
    # the recheck run against a DEM; shown as an annotation alongside
    # the sector's original (still authoritative) color/status.
    diffraction_loss = result.get("diffraction_loss_db")
    if diffraction_loss is not None:
        diff_status = str(result.get("diffraction_status", "")).upper()
        diff_in_ratio = result.get("diffraction_in_ratio_db")
        diff_color = "green" if diff_status == "PASS" else "red"
        diff_note = (
            "PASSes with diffraction" if diff_status == "PASS" else "still FAILs with diffraction"
        )
        popup_html += (
            "<hr style='margin:4px 0'>"
            "<b>Terrain diffraction (ITU-R P.526):</b><br>"
            f"Diffraction loss: {diffraction_loss:.2f} dB<br>"
            + (f"I/N with diffraction: {diff_in_ratio:.2f} dB<br>" if diff_in_ratio is not None else "")
            + f"<span style='color:{diff_color}; font-weight:bold'>{diff_note}</span>"
        )

    sector_pts = _build_sector_polygon(
        earth_station_lat, earth_station_lon, azimuth, half_width,
        border_km, geod=geod,
    )
    sector = folium.Polygon(
        locations=sector_pts,
        color=color,
        weight=1,
        fill=True,
        fill_color=color,
        fill_opacity=0.35,
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=f"Az {azimuth:.0f}\u00b0: {status}",
    )
    _in_pane(sector, PANE_COORD)

    # Reference arc at the required coordination distance, drawn
    # as a dashed line, for a quick actual-vs-required comparison.
    required_arc = _build_sector_polygon(
        earth_station_lat, earth_station_lon, azimuth, half_width,
        required_km, geod=geod,
    )[1:-1]  # drop the ES-origin start/end points, keep just the arc
    arc = folium.PolyLine(
        locations=required_arc,
        color="black",
        weight=1.5,
        dash_array="4, 4",
        opacity=0.7,
        tooltip=f"Required distance: {required_km:.1f} km",
    )
    _in_pane(arc, PANE_COORD)

    sector.add_to(target)
    arc.add_to(target)


def add_interference_zones(
    map_obj: folium.Map,
    analysis_results: List[Dict[str, Any]],
    earth_station_lat: Optional[float] = None,
    earth_station_lon: Optional[float] = None,
) -> folium.Map:
    """
    Draw color-coded interference/coordination-analysis sectors on a folium
    map, one per azimuth entry in analysis_results.

    Each result is rendered as a filled pie-slice ("sector") pointing along
    its azimuth, extending out to `border_distance_km`, colored green if
    status == 'PASS' or red if status == 'FAIL', with a dashed arc at
    `required_distance_km` and a detail popup (see
    :func:`draw_interference_sector`). Sectors are drawn into a layer group in
    the coordination pane (above the DEM).

    A bad row (missing keys, NaN / inf, geodesic failure, ...) is logged and
    SKIPPED; it never prevents the other sectors from being drawn.

    Args:
        map_obj: An existing folium.Map object to add the sectors to.
        analysis_results: List of dicts with keys 'azimuth',
            'border_distance_km', 'required_distance_km', 'in_ratio_db',
            'limit_db', 'status' ('PASS' / 'FAIL').
        earth_station_lat / earth_station_lon: Optional explicit origin;
            defaults to the map's center.

    Returns:
        The folium.Map (sectors added in place).

    Raises:
        ImportError: If pyproj is not installed.
        TypeError: If map_obj is not a folium.Map, or analysis_results is
            not a list.
        ValueError: If analysis_results is empty, or the earth station
            location cannot be determined.
    """
    if not isinstance(map_obj, folium.Map):
        raise TypeError("map_obj must be a folium.Map instance")
    if not isinstance(analysis_results, list) or not analysis_results:
        raise ValueError("analysis_results must be a non-empty list of dicts")

    if earth_station_lat is None or earth_station_lon is None:
        earth_station_lat, earth_station_lon = _get_map_center(map_obj)

    if Geod is None:
        raise ImportError(
            "pyproj is required for geodesic calculations. Install it with "
            "'pip install pyproj'."
        )
    geod = Geod(ellps="WGS84")

    half_width = interference_half_width(analysis_results)
    group = create_interference_group(map_obj)

    drawn = 0
    fail_count = 0
    for i, result in enumerate(analysis_results):
        try:
            draw_interference_sector(
                group, result, earth_station_lat, earth_station_lon,
                half_width, geod=geod, index=i,
            )
        except Exception as exc:
            az = result.get("azimuth") if isinstance(result, dict) else None
            logger.error("Skipping interference sector %d (azimuth %s): %s", i, az, exc)
            continue
        drawn += 1
        if str(result["status"]).upper() == "FAIL":
            fail_count += 1

    logger.info(
        "Added %d/%d interference sectors (%d FAIL, %d PASS)",
        drawn, len(analysis_results), fail_count, drawn - fail_count,
    )
    return map_obj


def add_interference_popup(
    map_obj: folium.Map,
    lat: float,
    lon: float,
    result_dict: Dict[str, Any],
) -> folium.Map:
    """
    Add a clickable marker with a detailed interference-analysis popup at a
    specific location on a folium map.

    Args:
        map_obj: An existing folium.Map object to add the marker to.
        lat: Latitude for the marker, in decimal degrees.
        lon: Longitude for the marker, in decimal degrees.
        result_dict: Dict describing the interference analysis at this
            point. Expected keys (all optional except 'status', 'in_ratio_db',
            and 'limit_db', which are needed to compute the margin):
                - 'status' (str): 'PASS' or 'FAIL'
                - 'in_ratio_db' (float): computed I/N ratio, in dB
                - 'limit_db' (float): the I/N limit/threshold, in dB
                - 'azimuth' (float, optional): bearing in degrees
                - 'border_distance_km' (float, optional)
                - 'required_distance_km' (float, optional)
                - any other keys are ignored

    Returns:
        The updated folium.Map object with the popup marker added.

    Raises:
        TypeError: If map_obj is not a folium.Map, or lat/lon are not
            numeric, or result_dict is not a dict.
        ValueError: If lat/lon are out of range, or result_dict is missing
            'status', 'in_ratio_db', or 'limit_db'.
    """
    if not isinstance(map_obj, folium.Map):
        raise TypeError("map_obj must be a folium.Map instance")
    if not isinstance(result_dict, dict):
        raise TypeError("result_dict must be a dict")

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError) as exc:
        raise TypeError("lat and lon must be numeric") from exc
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"lat must be between -90 and 90, got {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"lon must be between -180 and 180, got {lon}")

    for key in ("status", "in_ratio_db", "limit_db"):
        if key not in result_dict:
            raise ValueError(f"result_dict is missing required key: {key!r}")

    status = str(result_dict["status"]).upper()
    if status not in ("PASS", "FAIL"):
        raise ValueError(f"result_dict['status'] must be 'PASS' or 'FAIL', got {result_dict['status']!r}")

    try:
        in_ratio = float(result_dict["in_ratio_db"])
        limit = float(result_dict["limit_db"])
    except (TypeError, ValueError) as exc:
        raise TypeError("result_dict['in_ratio_db'] and ['limit_db'] must be numeric") from exc

    margin_db = limit - in_ratio
    color = "green" if status == "PASS" else "red"

    rows = [
        f"<b>Status:</b> <span style='color:{color}; font-weight:bold'>{status}</span>",
        f"<b>I/N ratio:</b> {in_ratio:.2f} dB",
        f"<b>Limit:</b> {limit:.2f} dB",
        f"<b>Margin:</b> {margin_db:+.2f} dB",
    ]
    if "azimuth" in result_dict:
        rows.insert(0, f"<b>Azimuth:</b> {float(result_dict['azimuth']):.1f}&deg;")
    if "border_distance_km" in result_dict:
        rows.append(f"<b>Border distance:</b> {float(result_dict['border_distance_km']):.1f} km")
    if "required_distance_km" in result_dict:
        rows.append(f"<b>Required distance:</b> {float(result_dict['required_distance_km']):.1f} km")

    popup_html = "<br>".join(rows)

    try:
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{status}: click for details",
        ).add_to(map_obj)
    except Exception as exc:
        logger.error("Failed to add interference popup at (%s, %s): %s", lat, lon, exc)
        raise RuntimeError(f"Could not add interference popup: {exc}") from exc

    logger.info("Interference popup added at (%s, %s), status=%s", lat, lon, status)
    return map_obj


def create_interference_legend() -> str:
    """
    Build an HTML legend explaining the interference-visualization color
    coding (green = PASS, red = FAIL), styled as a fixed-position box.

    Returns:
        An HTML string. To attach it to a map, add it as a child element,
        e.g.:
            legend_html = create_interference_legend()
            map_obj.get_root().html.add_child(folium.Element(legend_html))

    Raises: None. This function performs no I/O or external calls.
    """
    return """
<div style="
    position: fixed;
    bottom: 30px; left: 30px;
    z-index: 9999;
    background-color: white;
    padding: 10px 14px;
    border: 2px solid #666;
    border-radius: 6px;
    font-size: 13px;
    font-family: Arial, sans-serif;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">
    <div style="font-weight:bold; margin-bottom:6px;">Interference Analysis</div>
    <div style="margin-bottom:4px;">
        <span style="display:inline-block; width:14px; height:14px; background:green;
                     opacity:0.6; border:1px solid green; margin-right:6px;"></span>
        PASS &mdash; within I/N limit
    </div>
    <div style="margin-bottom:4px;">
        <span style="display:inline-block; width:14px; height:14px; background:red;
                     opacity:0.6; border:1px solid red; margin-right:6px;"></span>
        FAIL &mdash; exceeds I/N limit
    </div>
    <div>
        <span style="display:inline-block; width:14px; height:2px; background:black;
                     border-top:2px dashed black; margin-right:6px; margin-bottom:3px;"></span>
        Required coordination distance
    </div>
</div>
"""


if __name__ == "__main__":
    # Simple smoke test / usage example.
    es_lat, es_lon = 40.7128, -74.0060

    m = create_base_map(es_lat, es_lon, zoom=6)
    add_country_borders(m)  # returns a status dict; the map is modified in place
    m = add_earth_station_marker(m, 51.5074, -0.1278, station_name="London ES")
    m = add_distance_circle(m, es_lat, es_lon, radius_km=100, fill=True, label="100 km")
    sample_contour = {
        az: 150 + 50 * (az % 90) / 90 for az in range(0, 360, 15)
    }
    m = add_coordination_contour(m, es_lat, es_lon, sample_contour)

    # --- DEM example ---------------------------------------------------
    # Uses a small synthetic DEM GeoTIFF so this example runs standalone.
    # In practice, pass the path to a real DEM (e.g. SRTM/USGS 3DEP tile)
    # that covers your earth station's coordination area.
    dem_path = "example_dem.tif"
    try:
        if rasterio is not None and np is not None:
            from rasterio.transform import from_origin

            size = 200
            xs, ys = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
            synthetic_elevation = (
                200 + 300 * np.exp(-(xs**2 + ys**2) * 3) + 20 * np.sin(xs * 10)
            ).astype("float32")
            pixel_deg = 2.0 / size
            transform = from_origin(
                es_lon - 1.0, es_lat + 1.0, pixel_deg, pixel_deg
            )
            with rasterio.open(
                dem_path, "w", driver="GTiff", height=size, width=size, count=1,
                dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999,
            ) as dst:
                dst.write(synthetic_elevation, 1)

            print("DEM status:", add_dem_overlay(m, dem_path, es_lat, es_lon, opacity=0.5))
            elevation_here = get_elevation_at_point(dem_path, es_lat, es_lon)
            print(f"Elevation at earth station: {elevation_here} m")
        else:
            print("Skipping DEM example: rasterio/numpy not available.")
    except Exception as exc:
        print(f"Skipping DEM example due to error: {exc}")

    # --- Interference visualization example -----------------------------
    # This is the pattern an app.py would follow after running its own
    # coordination/interference analysis (e.g. ITU-R or FCC methodology)
    # to produce a list of per-azimuth results:
    #
    #     from map_visualization import (
    #         create_base_map, add_interference_zones,
    #         add_interference_popup, create_interference_legend,
    #     )
    #
    #     analysis_results = run_coordination_analysis(es_lat, es_lon)  # your code
    #     m = create_base_map(es_lat, es_lon, zoom=7)
    #     m = add_interference_zones(m, analysis_results)
    #     m = add_interference_popup(
    #         m, other_station_lat, other_station_lon, analysis_results[0]
    #     )
    #     m.get_root().html.add_child(folium.Element(create_interference_legend()))
    #     m.save("coordination_report.html")
    sample_analysis_results = [
        {
            "azimuth": az,
            "border_distance_km": 120 + 40 * (az % 120) / 120,
            "required_distance_km": 140,
            "in_ratio_db": -12.5 if az % 90 else -6.0,
            "limit_db": -10.0,
            "status": "FAIL" if az % 90 == 0 else "PASS",
        }
        for az in range(0, 360, 30)
    ]
    m = add_interference_zones(m, sample_analysis_results)
    m = add_interference_popup(
        m, es_lat + 0.5, es_lon + 0.5, sample_analysis_results[0]
    )
    m.get_root().html.add_child(folium.Element(create_interference_legend()))

    ensure_layer_control(m)
    output_path = "example_map.html"
    m.save(output_path)
    print(f"Map saved to {output_path}")