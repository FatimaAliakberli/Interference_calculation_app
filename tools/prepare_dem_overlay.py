#!/usr/bin/env python
"""
tools/prepare_dem_overlay.py - one-off build step: turn the bundled DEM GeoTIFF into a static
hillshade PNG that the map can show with a plain Leaflet ImageOverlay (no rasterio at runtime).

Pipeline
  1. read the source DEM GeoTIFF (default: data_sources/azerbaijan_dem.tif)
  2. reproject it to Web Mercator (EPSG:3857) - the projection Leaflet stretches an
     ImageOverlay in, so the PNG lines up with the tiles exactly
  3. downsample so the longest side is <= --max-dim pixels (default 2000); the DEM is
     resampled straight into that grid, so a huge GeoTIFF is never held in memory
  4. compute a hillshade (matplotlib LightSource, azimuth 315 / altitude 45 - the same light
     the on-the-fly rasterio overlay uses)
  5. write  data_sources/dem_hillshade.png  (RGBA, alpha = 0 where there is no data)  and
            data_sources/dem_bounds.json     (lat/lon corners for Leaflet + metadata)

Usage (from the project root, with rasterio, numpy, matplotlib and Pillow installed):
    python tools/prepare_dem_overlay.py
    python tools/prepare_dem_overlay.py --dem path/to/other.tif --max-dim 1600 --out-dir data_sources

The map looks for the two output files via get_resource_path("data_sources/..."), so ship them
with the app (PyInstaller `datas`).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Make `core.*` importable when run as `python tools/prepare_dem_overlay.py`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.app_paths import get_resource_path  # noqa: E402

PNG_NAME = "dem_hillshade.png"
BOUNDS_NAME = "dem_bounds.json"
DST_CRS = "EPSG:3857"
MAX_MERCATOR_LAT = 85.0511287798  # Web Mercator's northern/southern limit


def _require(module: str, pip_name: str | None = None):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:
        raise SystemExit(f"Missing dependency '{module}': pip install {pip_name or module}") from exc


def prepare(dem_path: Path, out_dir: Path, max_dim: int = 2000,
            azimuth: float = 315.0, altitude: float = 45.0, vert_exag: float = 1.0) -> dict:
    np = _require("numpy")
    rasterio = _require("rasterio")
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject, transform_bounds
    LightSource = _require("matplotlib.colors").LightSource
    Image = _require("PIL.Image", "Pillow")

    if max_dim < 16:
        raise SystemExit("--max-dim must be >= 16")
    if not dem_path.is_file():
        raise SystemExit(f"DEM not found: {dem_path}")

    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise SystemExit("The DEM has no CRS; cannot reproject it.")
        print(f"[1/5] Source DEM: {dem_path}  ({src.width}x{src.height}px, {src.crs})")

        # Clamp the extent to Web Mercator's valid latitude range (matters for polar DEMs only).
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        if s < -MAX_MERCATOR_LAT or n > MAX_MERCATOR_LAT:
            raise SystemExit("The DEM extends beyond +-85.05 deg latitude, outside Web Mercator.")

        # [2/5] natural 3857 grid, then scale its resolution so max(width, height) <= max_dim.
        transform, width, height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds)
        scale = max(width, height) / float(max_dim)
        if scale > 1.0:
            transform, width, height = calculate_default_transform(
                src.crs, DST_CRS, src.width, src.height, *src.bounds,
                resolution=(abs(transform.a) * scale, abs(transform.e) * scale))
        # Guard against rounding giving max_dim + 1.
        if max(width, height) > max_dim:
            f = max_dim / float(max(width, height))
            transform, width, height = calculate_default_transform(
                src.crs, DST_CRS, src.width, src.height, *src.bounds,
                dst_width=max(1, int(width * f)), dst_height=max(1, int(height * f)))
        print(f"[2/5] Reprojecting to {DST_CRS} and downsampling to {width}x{height}px")

        dem = np.full((height, width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dem,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=DST_CRS,
            dst_nodata=np.nan,
            # 'average' is the right filter for shrinking (bilinear would alias); with
            # nearly 1:1 pixels it is simply a mean of the covered cells.
            resampling=Resampling.average,
        )

    valid = np.isfinite(dem)
    if not valid.any():
        raise SystemExit("No valid elevation data after reprojection.")

    # [3/5] hillshade. Mercator pixels are stretched by 1/cos(lat), so true ground spacing is
    # pixel_size * cos(lat). Use the mid-latitude of the DEM (the area is small).
    mid_lat = math.radians((s + n) / 2.0)
    dx = abs(transform.a) * math.cos(mid_lat)
    dy = abs(transform.e) * math.cos(mid_lat)
    print(f"[3/5] Hillshade (azimuth={azimuth}, altitude={altitude}, vert_exag={vert_exag}, "
          f"ground pixel ~{dx:.0f} m)")
    filled = np.where(valid, dem, float(np.nanmin(dem)))
    shade = LightSource(azdeg=azimuth, altdeg=altitude).hillshade(
        filled, vert_exag=vert_exag, dx=dx, dy=dy)

    # [4/5] RGBA: grey hillshade, fully transparent outside the DEM. Overall opacity is applied
    # live by the ImageOverlay (the app's DEM opacity slider), so alpha is only 0 or 255 here.
    gray = (np.clip(shade, 0.0, 1.0) * 255).astype("uint8")
    rgba = np.dstack([gray, gray, gray, np.where(valid, 255, 0).astype("uint8")])

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / PNG_NAME
    Image.fromarray(rgba, "RGBA").save(png_path, optimize=True)
    print(f"[4/5] Wrote {png_path}  ({png_path.stat().st_size / 1e6:.2f} MB)")

    # [5/5] corners. Leaflet wants lat/lon: [[south, west], [north, east]] - taken from the
    # *reprojected raster's* own extent so the image and its bounds agree to the pixel.
    left, top = transform.c, transform.f
    right, bottom = left + transform.a * width, top + transform.e * height
    west, south, east, north = transform_bounds(DST_CRS, "EPSG:4326", left, bottom, right, top)
    meta = {
        "bounds": [[south, west], [north, east]],           # Leaflet / Folium order
        "bounds_epsg3857": [left, bottom, right, top],       # minx, miny, maxx, maxy (metres)
        "crs": DST_CRS,
        "width": int(width),
        "height": int(height),
        "source_dem": dem_path.name,
        "hillshade": {"azimuth": azimuth, "altitude": altitude, "vert_exag": vert_exag},
    }
    bounds_path = out_dir / BOUNDS_NAME
    bounds_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[5/5] Wrote {bounds_path}")
    print(f"      bounds (S,W / N,E): {south:.5f},{west:.5f} / {north:.5f},{east:.5f}")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the static DEM hillshade PNG + bounds JSON.")
    ap.add_argument("--dem", type=Path, default=Path(get_resource_path("data_sources/azerbaijan_dem.tif")),
                    help="source DEM GeoTIFF (default: data_sources/azerbaijan_dem.tif)")
    ap.add_argument("--out-dir", type=Path, default=Path(get_resource_path("data_sources")),
                    help="output folder (default: data_sources)")
    ap.add_argument("--max-dim", type=int, default=2000, help="max width/height in px (default 2000)")
    ap.add_argument("--azimuth", type=float, default=315.0, help="light azimuth, degrees")
    ap.add_argument("--altitude", type=float, default=45.0, help="light altitude, degrees")
    ap.add_argument("--vert-exag", type=float, default=1.0, help="vertical exaggeration")
    a = ap.parse_args(argv)
    prepare(a.dem, a.out_dir, a.max_dim, a.azimuth, a.altitude, a.vert_exag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())