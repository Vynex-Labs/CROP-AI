"""Risk backends. LightGBM/XGBoost optional; heuristic always available."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cropai.risk.features import feature_vector, numeric_feature_names
from cropai.risk.rules import score_all


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except Exception:
        return False


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:
        return False


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


def describe_stack() -> dict[str, bool]:
    return {
        "lightgbm": lightgbm_available(),
        "xgboost": xgboost_available(),
        "sklearn": sklearn_available(),
    }


@dataclass
class BackendInfo:
    kind: str
    trained: bool
    weights: str = ""


class RiskBackend(Protocol):
    info: BackendInfo

    def predict(self, features: dict[str, float | None]) -> dict[str, float]: ...


class HeuristicBackend:
    def __init__(self, heuristic_cfg: dict[str, Any]) -> None:
        self.heuristic_cfg = heuristic_cfg
        self.info = BackendInfo(kind="heuristic_unvalidated", trained=False)

    def predict(self, features: dict[str, float | None]) -> dict[str, float]:
        return score_all(features, self.heuristic_cfg)


class DummyBackend:
    def __init__(self) -> None:
        self.info = BackendInfo(kind="dummy", trained=False)

    def predict(self, features: dict[str, float | None]) -> dict[str, float]:
        return {
            "disease_1": 0.5,
            "disease_3": 0.5,
            "disease_7": 0.5,
            "pest_1": 0.5,
            "pest_3": 0.5,
            "pest_7": 0.5,
        }


class LightGBMBackend:
    def __init__(self, booster: Any, feature_names: list[str] | None = None) -> None:
        self.booster = booster
        self.feature_names = feature_names or numeric_feature_names()
        path = getattr(booster, "model_file", "") or ""
        self.info = BackendInfo(kind="lightgbm", trained=True, weights=str(path))

    def predict(self, features: dict[str, float | None]) -> dict[str, float]:
        vec = [feature_vector(features, self.feature_names)]
        raw = self.booster.predict(vec)
        # Expect 6 outputs or a single disease score we replicate honestly as uncalibrated.
        row = list(raw[0]) if hasattr(raw[0], "__iter__") else [float(raw[0])] * 6
        while len(row) < 6:
            row.append(row[-1] if row else 0.5)
        keys = ["disease_1", "disease_3", "disease_7", "pest_1", "pest_3", "pest_7"]
        return {k: max(0.0, min(1.0, float(v))) for k, v in zip(keys, row[:6], strict=False)}


def load_backend(
    *,
    kind: str = "auto",
    heuristic_cfg: dict[str, Any] | None = None,
    weights: Path | None = None,
) -> RiskBackend:
    heuristic_cfg = heuristic_cfg or {}
    if kind in {"heuristic", "heuristic_unvalidated", "auto"}:
        if kind != "auto":
            return HeuristicBackend(heuristic_cfg)
        if weights and Path(weights).exists() and lightgbm_available():
            try:
                import lightgbm as lgb

                booster = lgb.Booster(model_file=str(weights))
                return LightGBMBackend(booster)
            except Exception:
                return HeuristicBackend(heuristic_cfg)
        return HeuristicBackend(heuristic_cfg)
    if kind == "dummy":
        return DummyBackend()
    if kind == "lightgbm":
        if weights and Path(weights).exists() and lightgbm_available():
            import lightgbm as lgb

            return LightGBMBackend(lgb.Booster(model_file=str(weights)))
        return HeuristicBackend(heuristic_cfg)
    return HeuristicBackend(heuristic_cfg)
