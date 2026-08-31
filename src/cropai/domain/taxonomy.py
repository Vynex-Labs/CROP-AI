"""Data-driven crop / disease / pest taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from cropai.config.loader import load_crop_config


@dataclass(frozen=True)
class SeverityBin:
    id: str
    label: str
    min_pct: float
    max_pct: float
    score: int


class Taxonomy:
    """Read-only view over crop_config.yaml."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_crop_config()
        self.crops: dict[str, dict[str, Any]] = dict(self.config.get("crops") or {})
        self.diseases: dict[str, dict[str, Any]] = dict(self.config.get("diseases") or {})
        self.pests: dict[str, dict[str, Any]] = dict(self.config.get("pests") or {})
        self.detection_classes: list[str] = list(self.config.get("detection_classes") or [])
        self.symptom_regions: list[str] = list(self.config.get("symptom_regions") or [])
        self.crop_stages: list[str] = [
            str(item["id"]) for item in (self.config.get("crop_stages") or [])
        ]
        bins = (self.config.get("severity") or {}).get("bins") or []
        self.severity_bins = [
            SeverityBin(
                id=str(b["id"]),
                label=str(b.get("label", b["id"])),
                min_pct=float(b["min_pct"]),
                max_pct=float(b["max_pct"]),
                score=int(b["score"]),
            )
            for b in bins
        ]
        self.confidence = dict(self.config.get("confidence_thresholds") or {})

    def crop_ids(self) -> list[str]:
        return sorted(self.crops)

    def disease_ids(self, crop_id: str | None = None) -> list[str]:
        if crop_id is None:
            return sorted(self.diseases)
        return sorted(d for d, meta in self.diseases.items() if meta.get("crop") == crop_id)

    def healthy_ids(self) -> list[str]:
        return sorted(
            d
            for d, meta in self.diseases.items()
            if str(meta.get("type", "")).lower() == "healthy" or d.endswith("_healthy")
        )

    def pest_ids(self, crop_id: str | None = None) -> list[str]:
        if crop_id is None:
            return sorted(self.pests)
        return sorted(
            p for p, meta in self.pests.items() if crop_id in list(meta.get("crops") or [])
        )

    def classifier_classes(self) -> list[str]:
        return self.disease_ids()

    def crop_set(self, name: str) -> list[str]:
        sets = dict(self.config.get("crop_sets") or {})
        return [str(x) for x in (sets.get(name) or [])]

    def top10_india(self) -> list[str]:
        return self.crop_set("top10_india")

    def horticulture_required(self) -> list[str]:
        return self.crop_set("horticulture_required")

    def operational_crops(self) -> list[str]:
        return sorted(
            cid
            for cid, meta in self.crops.items()
            if "operational" in list(meta.get("tiers") or [])
        )

    def training_public_crops(self) -> list[str]:
        return sorted(
            cid
            for cid, meta in self.crops.items()
            if "training_public" in list(meta.get("tiers") or [])
        )

    def crops_missing_local_field(self) -> list[str]:
        missing = []
        for cid, meta in self.crops.items():
            coverage = dict(meta.get("image_coverage") or {})
            if str(coverage.get("local_field", "none")).lower() in {"none", "no", ""}:
                missing.append(cid)
        return sorted(missing)

    def severity_from_area(self, affected_area_pct: float) -> SeverityBin:
        pct = max(0.0, min(100.0, float(affected_area_pct)))
        for bin_ in self.severity_bins:
            if bin_.min_pct <= pct < bin_.max_pct or (
                bin_.id == self.severity_bins[-1].id and pct <= bin_.max_pct
            ):
                return bin_
        return self.severity_bins[-1]

    def validate_integrity(self) -> list[str]:
        """Return human-readable integrity errors (empty means OK)."""
        errors: list[str] = []
        for crop_id, meta in self.crops.items():
            for disease_id in meta.get("diseases") or []:
                if disease_id not in self.diseases:
                    errors.append(f"crop {crop_id} references unknown disease {disease_id}")
            for pest_id in meta.get("pests") or []:
                if pest_id not in self.pests:
                    errors.append(f"crop {crop_id} references unknown pest {pest_id}")
        for disease_id, meta in self.diseases.items():
            crop_id = meta.get("crop")
            if crop_id and crop_id not in self.crops:
                errors.append(f"disease {disease_id} references unknown crop {crop_id}")
            elif crop_id and disease_id not in list(self.crops[crop_id].get("diseases") or []):
                errors.append(f"disease {disease_id} not listed on crop {crop_id}")
        for pest_id, meta in self.pests.items():
            for crop_id in meta.get("crops") or []:
                if crop_id not in self.crops:
                    errors.append(f"pest {pest_id} references unknown crop {crop_id}")
        if not self.severity_bins:
            errors.append("severity bins missing")
        for set_name, members in (self.config.get("crop_sets") or {}).items():
            for cid in members or []:
                if cid not in self.crops:
                    errors.append(f"crop_set {set_name} references unknown crop {cid}")
        return errors

    def plantvillage_alias_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for disease_id, meta in self.diseases.items():
            alias = meta.get("plantvillage")
            if alias:
                mapping[str(alias)] = disease_id
        return mapping

    def iter_classes(self) -> Iterable[tuple[str, dict[str, Any]]]:
        yield from self.diseases.items()
