"""Runtime profile records. Missing hardware metrics stay NOT MEASURED."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MODEL_VERSION = "cropai-runtime-0.5.0-untrained"

NOT_MEASURED = "NOT MEASURED"


@dataclass
class StageTiming:
    name: str
    ms: float
    backend: str
    n: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProfileReport:
    timestamp: str
    backend: str
    precision_policy: str
    n_repeats: int
    timings_ms: dict[str, float]
    fps: float | None
    rss_bytes: int | None
    gpu_util: str = NOT_MEASURED
    vram_bytes: str | int | None = NOT_MEASURED
    warmup_done: bool = False
    int8_adopted: bool = False
    tensorrt: bool = False
    onnxruntime: bool = False
    pytorch: bool = False
    dummy: bool = True
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.fps is None:
            data["fps"] = NOT_MEASURED
        return data
