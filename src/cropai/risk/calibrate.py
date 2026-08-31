"""Calibration. Identity until evaluated on real outbreak labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class IdentityCalibrator:
    method: str = "none"

    def apply(self, scores: Mapping[str, float]) -> dict[str, float]:
        return {k: float(v) for k, v in scores.items()}


def load_calibrator(method: str = "none") -> IdentityCalibrator:
    # Platt / isotonic require labeled forecasts. Do not fake them.
    if method not in {"none", "identity", ""}:
        return IdentityCalibrator(method="none")
    return IdentityCalibrator(method="none")
