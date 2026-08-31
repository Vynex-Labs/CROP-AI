"""Geospatial indexing and hotspot detection (Phase 4)."""

from cropai.geo.hotspots import detect_hotspots
from cropai.geo.schema import Hotspot, MODEL_VERSION

__all__ = ["detect_hotspots", "Hotspot", "MODEL_VERSION"]
