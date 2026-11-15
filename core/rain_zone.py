#!/usr/bin/env python3
"""
core/rain_zone.py - Automatic rain climatic zone (A-Q) detection from a
latitude/longitude, using the ITU-R P.837-8 pre-computed R0.01 map.

R0.01 is the 1-minute rainfall rate (mm/h) exceeded for 0.01 % of an average
year. Rec. ITU-R P.837-8 provides it as a pre-computed digital map
(0.125 deg x 0.125 deg, latitude -90..+90, longitude -180..+180) in ASCII
format (R001.TXT), so the full Annex 1 algorithm does not have to be run.

Two functions:
- get_r001_from_txt(lat, lon, txt_path=...) -> R0.01 (mm/h)
- get_rain_zone_from_r001(r001_mm_h) -> "A".."Q"
"""

import math
import numpy as np
from pathlib import Path
import functools

DEFAULT_R001_PATH = "data_sources/R001.TXT"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Median rainfall rate (mm/h) exceeded for 0.01 % of the time, per rain
# climatic zone - Rec. ITU-R P.837-1, Table 1 (zones I and O do not exist).
ZONE_R001_MM_H = {
    "A": 8.0, "B": 12.0, "C": 15.0, "D": 19.0, "E": 22.0,
    "F": 28.0, "G": 30.0, "H": 32.0, "J": 35.0, "K": 42.0,
    "L": 60.0, "M": 63.0, "N": 95.0, "P": 145.0, "Q": 115.0,
}

_MAX_OFFSET_DEG = 0.125  # half the grid spacing

def _resolve_path(path):
    if path is None:
        path = DEFAULT_R001_PATH
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"R0.01 map file not found: {p}")
    return p

def get_r001_from_txt(lat, lon, txt_path=None):
    """
    Read the R0.01 value (mm/h) from the ITU-R P.837-8 R001.TXT map.
    
    The file is a 1441 x 2881 grid (lat x lon) with 0.125 deg spacing.
    Latitude: -90 to +90 deg (1441 points)
    Longitude: -180 to +180 deg (2881 points)
    
    Parameters
    ----------
    lat : float
        Latitude in degrees (+N / -S), range -90 to +90
    lon : float
        Longitude in degrees (+E / -W), any range (will be wrapped)
    txt_path : str or Path, optional
        Path to R001.TXT file. Default: "data_sources/R001.TXT"
    
    Returns
    -------
    float
        R0.01 (mm/h) at the grid point nearest to (lat, lon).
    
    Raises
    ------
    FileNotFoundError
        If the map file cannot be found
    ValueError
        Bad coordinates or query outside the map's coverage
    """
    import numpy as np
    
    lat, lon = float(lat), float(lon)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("Latitude and longitude must be finite numbers.")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude {lat} is outside -90..90 degrees.")
    
    path = _resolve_path(txt_path)
    lats, lons, grid = _load_grid(str(path), path.stat().st_mtime_ns)
    
    # Wrap the query into the map's longitude convention (-180 to +180)
    lon_q = ((lon + 180.0) % 360.0) - 180.0
    
    # Find nearest grid point
    i = int(np.abs(lats - lat).argmin())
    j = int(np.abs(lons - lon_q).argmin())
    
    if abs(lats[i] - lat) > _MAX_OFFSET_DEG or abs(lons[j] - lon_q) > _MAX_OFFSET_DEG:
        raise ValueError(f"({lat}, {lon}) is outside the area covered by {path.name}.")
    
    value = float(grid[i, j])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{path.name} has no valid R0.01 data at ({lat}, {lon}).")
    
    return value

def get_rain_zone_from_r001(r001_mm_h):
    """
    Rain climatic zone letter (A-Q; no I or O) whose reference R0.01
    (ZONE_R001_MM_H) is nearest, in log space, to r001_mm_h.
    
    An approximation - see the module docstring. Values <= 0 are rejected;
    very small values map to zone A.
    """
    r = float(r001_mm_h)
    if not math.isfinite(r) or r < 0.0:
        raise ValueError(f"R0.01 must be a finite value >= 0 mm/h, got {r001_mm_h!r}.")
    
    x = math.log(max(r, 1e-6))
    
    def _dist(z):
        return abs(math.log(ZONE_R001_MM_H[z]) - x)
    
    return min(ZONE_R001_MM_H.keys(), key=_dist)

@functools.lru_cache(maxsize=2)
def _load_grid(path_str, mtime_ns):
    """
    Load the R001.TXT grid file.
    
    The file format is:
    - 1441 rows (latitude -90 to +90 in 0.125 deg steps)
    - 2881 columns (longitude -180 to +180 in 0.125 deg steps)
    - Values are space-separated floats (mm/h)
    """
    import numpy as np
    
    # Grid dimensions per ITU-R P.837-8
    n_lat = 1441  # -90 to +90 in 0.125 deg steps
    n_lon = 2881  # -180 to +180 in 0.125 deg steps
    
    try:
        # Load the ASCII grid (space-separated values)
        grid = np.loadtxt(path_str)
        
        if grid.shape != (n_lat, n_lon):
            raise ValueError(
                f"R001.TXT has shape {grid.shape}; expected ({n_lat}, {n_lon}) "
                "(latitude -90..90 x longitude -180..180, 0.125 deg)."
            )
        
        # Create coordinate arrays
        lats = np.linspace(-90.0, 90.0, n_lat)
        lons = np.linspace(-180.0, 180.0, n_lon)
        
        return lats, lons, grid
        
    except Exception as e:
        raise RuntimeError(f"Failed to load R001.TXT from {path_str}: {e}")

def _match_name(obj, candidates):
    """Case-insensitive lookup of a name in a dict or list."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        keys = obj.keys()
    else:
        keys = obj
    lower_map = {str(k).lower(): k for k in keys}
    for cand in candidates:
        real = lower_map.get(cand.lower())
        if real is not None:
            return real
    return None