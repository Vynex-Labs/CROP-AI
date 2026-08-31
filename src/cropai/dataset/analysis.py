"""Quantitative dataset analysis for the Phase 1.5 gate."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cropai.dataset.schema import ImageRecord
from cropai.domain.taxonomy import Taxonomy
from cropai.utils.logging import utc_now_iso


def analyze_manifest(
    records: list[ImageRecord],
    *,
    taxonomy: Taxonomy | None = None,
) -> dict[str, Any]:
    taxonomy = taxonomy or Taxonomy()
    class_counts: Counter[str] = Counter(r.class_id for r in records)
    crop_counts: Counter[str] = Counter(r.crop_id for r in records)
    source_counts: Counter[str] = Counter(r.source for r in records)
    split_counts: Counter[str] = Counter(r.split for r in records)
    stage_counts: Counter[str] = Counter(r.growth_stage or "unknown" for r in records)
    severity_counts: Counter[str] = Counter(r.severity_bin or "unlabeled" for r in records)

    n = len(records)
    n_syn = sum(1 for r in records if r.is_synthetic)
    n_field = sum(1 for r in records if r.is_field and not r.is_synthetic)
    n_lab = sum(1 for r in records if (not r.is_field) and (not r.is_synthetic))

    coverage = {}
    for crop_id, meta in taxonomy.crops.items():
        expected = list(meta.get("diseases") or [])
        present = [d for d in expected if class_counts.get(d, 0) > 0]
        coverage[crop_id] = {
            "n_images": crop_counts.get(crop_id, 0),
            "diseases_defined": len(expected),
            "diseases_with_images": len(present),
            "missing_disease_classes": [d for d in expected if class_counts.get(d, 0) == 0],
            "pests_defined": len(list(meta.get("pests") or [])),
            "image_coverage_declared": meta.get("image_coverage"),
            "local_field_images": 0,  # none unless a field_maharashtra source appears
        }

    pest_image_classes = [c for c in class_counts if c.startswith("ip102_")]
    geo = {
        "n_with_latlon": sum(1 for r in records if r.lat is not None and r.lon is not None),
        "unique_locations": len({r.location_id for r in records if r.location_id}),
        "unique_farmers": len({r.farmer_id for r in records if r.farmer_id}),
    }

    balance = {}
    if class_counts:
        most_cls, most_n = class_counts.most_common(1)[0]
        least_cls, least_n = min(class_counts.items(), key=lambda kv: kv[1])
        balance = {
            "n_classes_present": len(class_counts),
            "n_classes_defined": len(taxonomy.classifier_classes()),
            "most_common": {most_cls: most_n},
            "least_common": {least_cls: least_n},
            "ratio_most_least": (most_n / least_n) if least_n else None,
        }

    return {
        "created_at": utc_now_iso(),
        "n_records": n,
        "n_synthetic": n_syn,
        "n_lab_real": n_lab,
        "n_field_real": n_field,
        "class_counts": dict(class_counts),
        "crop_counts": dict(crop_counts),
        "source_counts": dict(source_counts),
        "split_counts": dict(split_counts),
        "stage_counts": dict(stage_counts),
        "severity_counts": dict(severity_counts),
        "crop_coverage": coverage,
        "n_ip102_pest_classes_seen": len(pest_image_classes),
        "geography": geo,
        "balance": balance,
        "gate": {
            "public_plus_field_sufficient_for_transfer_learning_start": bool(n_lab >= 1000),
            "public_plus_field_sufficient_for_maharashtra_deployment": False,
            "additional_field_images_required": True,
            "real_field_images_in_repo": n_field,
            "fabricated_field_data": False,
            "notes": [
                "No Maharashtra field images are in this repository.",
                "Synthetic images exist only for pipeline tests.",
                "PlantVillage (if downloaded) is lab-captured, not field-representative.",
                "PlantDoc / FieldPlant (if downloaded) are not Maharashtra CROPSAP surveys.",
                "Pest-trap, weather, and soil observations are not present as real data.",
            ],
        },
    }


def write_analysis(analysis: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def crosstab_split_group(records: list[ImageRecord]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in records:
        table[rec.group_key()][rec.split] += 1
    return {g: dict(splits) for g, splits in table.items()}
