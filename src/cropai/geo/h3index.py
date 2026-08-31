"""H3 cell indexing with an explicit grid fallback when the h3 package is absent."""

from __future__ import annotations

import math
from typing import Any


# Approximate H3 hex edge length (m) used only for the non-H3 grid fallback.
_FALLBACK_EDGE_M = {
    0: 1107_000,
    1: 418_000,
    2: 158_000,
    3: 59_800,
    4: 22_600,
    5: 8_540,
    6: 3_230,
    7: 1_220,
    8: 461,
    9: 174,
    10: 66,
    11: 25,
    12: 9.4,
    13: 3.5,
    14: 1.3,
    15: 0.5,
}


def h3_available() -> bool:
    try:
        import h3  # noqa: F401

        return True
    except Exception:
        return False


def describe_h3() -> dict[str, Any]:
    ok = h3_available()
    version = None
    if ok:
        try:
            import h3

            version = getattr(h3, "__version__", None)
        except Exception:
            version = None
    return {"h3_installed": ok, "h3_version": version, "fallback": "latlon_grid" if not ok else None}


def cell_index(lat: float, lon: float, resolution: int) -> tuple[str, str]:
    """Return (index, backend) where backend is 'h3' or 'grid_fallback'."""
    if h3_available():
        try:
            import h3

            if hasattr(h3, "latlng_to_cell"):
                idx = str(h3.latlng_to_cell(float(lat), float(lon), int(resolution)))
            else:
                idx = str(h3.geo_to_h3(float(lat), float(lon), int(resolution)))
            return idx, "h3"
        except Exception:
            pass
    return _grid_cell(float(lat), float(lon), int(resolution)), "grid_fallback"


def _grid_cell(lat: float, lon: float, resolution: int) -> str:
    meters = float(_FALLBACK_EDGE_M.get(int(resolution), 461.0))
    dlat = meters / 111_320.0
    cos_lat = math.cos(math.radians(lat))
    dlon = meters / (111_320.0 * max(0.2, abs(cos_lat)))
    i = math.floor(lat / dlat)
    j = math.floor(lon / dlon)
    return f"grid{int(resolution)}:{i}:{j}"


def neighbor_cells(index: str, resolution: int) -> list[str]:
    if index.startswith("grid") and ":" in index:
        try:
            _, i_s, j_s = index.split(":")
            i, j = int(i_s), int(j_s)
        except ValueError:
            return [index]
        out = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                out.append(f"grid{int(resolution)}:{i + di}:{j + dj}")
        return out
    if h3_available():
        try:
            import h3

            if hasattr(h3, "grid_disk"):
                return [str(x) for x in h3.grid_disk(index, 1)]
            return [str(x) for x in h3.k_ring(index, 1)]
        except Exception:
            return [index]
    return [index]
