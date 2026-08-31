"""Computer-vision perception (Phase 2)."""

from .schema import VisionOutput, Detection, QualityReport
from .pipeline import PerceptionPipeline, run_perception

__all__ = [
    "VisionOutput",
    "Detection",
    "QualityReport",
    "PerceptionPipeline",
    "run_perception",
]
