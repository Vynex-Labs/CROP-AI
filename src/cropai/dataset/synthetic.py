"""Synthetic smoke-test data. Always labeled is_synthetic=true. Never field claims."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

from cropai.dataset.coco import write_coco_detection
from cropai.dataset.schema import (
    BBox,
    DetectionAnnotation,
    ImageRecord,
    ObservationRecord,
    PolygonMask,
    TrapRecord,
    WeatherRecord,
    write_jsonl,
)
from cropai.dataset.yolo import write_yolo_detection, write_yolo_segmentation
from cropai.domain.taxonomy import Taxonomy
from cropai.utils.hashing import sha256_file
from cropai.utils.logging import utc_now_iso
from cropai.utils.paths import data_dir


DIFFICULTY_TAGS = [
    "healthy",
    "early_disease",
    "advanced_disease",
    "multiple_diseases",
    "multiple_pests",
    "partial_symptoms",
    "occluded_symptoms",
    "poor_lighting",
    "blurred",
    "cluttered_background",
    "different_cultivar",
    "different_growth_stage",
]


def _palette(index: int) -> tuple[int, int, int]:
    rng = random.Random(index + 17)
    return rng.randint(20, 220), rng.randint(40, 200), rng.randint(20, 180)


def _draw_smoke_image(
    path: Path,
    *,
    crop_id: str,
    class_id: str,
    difficulty: str,
    seed: int,
    size: int = 96,
) -> tuple[int, int, list[BBox], list[PolygonMask]]:
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), color=_palette(seed))
    draw = ImageDraw.Draw(img)
    # Plant/leaf rectangle
    lx, ly = rng.randint(8, 24), rng.randint(8, 24)
    lw, lh = rng.randint(40, 64), rng.randint(40, 64)
    leaf_color = (34, 120 + rng.randint(0, 80), 34)
    if difficulty == "poor_lighting":
        leaf_color = tuple(max(0, c - 70) for c in leaf_color)
    draw.ellipse([lx, ly, lx + lw, ly + lh], fill=leaf_color)
    boxes = [BBox(x=float(lx), y=float(ly), w=float(lw), h=float(lh), class_id="leaf")]
    polygons: list[PolygonMask] = []

    if not class_id.endswith("_healthy") and difficulty not in {"healthy"}:
        # Lesion blob
        cx = lx + rng.randint(8, max(9, lw - 8))
        cy = ly + rng.randint(8, max(9, lh - 8))
        rw, rh = rng.randint(8, 18), rng.randint(8, 18)
        lesion = (160, 40, 40) if "blight" in class_id or "blast" in class_id else (180, 140, 40)
        if difficulty == "early_disease":
            rw, rh = 6, 6
        if difficulty == "advanced_disease":
            rw, rh = 22, 20
        draw.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=lesion)
        boxes.append(
            BBox(
                x=float(max(0, cx - rw)),
                y=float(max(0, cy - rh)),
                w=float(rw * 2),
                h=float(rh * 2),
                class_id="lesion",
            )
        )
        polygons.append(
            PolygonMask(
                class_id="lesion",
                points=[
                    (float(cx - rw), float(cy)),
                    (float(cx), float(cy - rh)),
                    (float(cx + rw), float(cy)),
                    (float(cx), float(cy + rh)),
                ],
            )
        )
    if difficulty in {"multiple_pests", "multiple_diseases"}:
        px, py = rng.randint(10, size - 16), rng.randint(10, size - 16)
        draw.rectangle([px, py, px + 10, py + 8], fill=(20, 20, 20))
        boxes.append(BBox(x=float(px), y=float(py), w=10.0, h=8.0, class_id="pest"))
    if difficulty == "blurred":
        img = img.resize((size // 3, size // 3)).resize((size, size))
        draw = ImageDraw.Draw(img)
    # Watermark so synthetic data cannot be mistaken for photographs.
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), "SYN", fill=(255, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return size, size, boxes, polygons


def generate_smoke_dataset(
    out_dir: Path | None = None,
    *,
    n_per_class: int = 4,
    seed: int = 42,
    taxonomy: Taxonomy | None = None,
    dataset_version: str = "v0.1.0-phase1",
) -> list[ImageRecord]:
    """Create a tiny labeled synthetic set covering taxonomy + difficulty tags."""
    taxonomy = taxonomy or Taxonomy()
    out_dir = out_dir or (data_dir() / "synthetic" / "smoke")
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Keep smoke set small: one healthy + one disease per training-public crop, plus a few operational-only.
    selected: list[tuple[str, str]] = []
    for crop_id in taxonomy.training_public_crops():
        diseases = taxonomy.disease_ids(crop_id)
        healthy = [d for d in diseases if d.endswith("_healthy")]
        other = [d for d in diseases if not d.endswith("_healthy")]
        if healthy:
            selected.append((crop_id, healthy[0]))
        if other:
            selected.append((crop_id, other[0]))
    extra = ["cotton", "sugarcane", "onion", "groundnut", "mustard", "pearl_millet"]
    already = {c for c, _ in selected}
    for crop_id in extra:
        if crop_id in already:
            continue
        diseases = taxonomy.disease_ids(crop_id)
        if diseases:
            selected.append((crop_id, diseases[0]))

    records: list[ImageRecord] = []
    detections: list[DetectionAnnotation] = []
    observations: list[ObservationRecord] = []
    traps: list[TrapRecord] = []
    weather: list[WeatherRecord] = []

    # Groups for leakage tests: some farmers contribute multiple images.
    farmers = [f"syn_farmer_{i:02d}" for i in range(8)]
    locations = [f"syn_loc_{i:02d}" for i in range(6)]

    idx = 0
    for crop_id, class_id in selected:
        for k in range(n_per_class):
            difficulty = DIFFICULTY_TAGS[idx % len(DIFFICULTY_TAGS)]
            sample_id = f"syn_{crop_id}_{class_id}_{k:03d}"
            rel = Path("synthetic") / "smoke" / "images" / f"{sample_id}.png"
            abs_path = data_dir() / rel if out_dir == data_dir() / "synthetic" / "smoke" else images_dir / f"{sample_id}.png"
            # Always write under out_dir/images regardless of data_dir layout.
            abs_path = images_dir / f"{sample_id}.png"
            w, h, boxes, polygons = _draw_smoke_image(
                abs_path,
                crop_id=crop_id,
                class_id=class_id,
                difficulty=difficulty,
                seed=seed + idx,
            )
            farmer = farmers[idx % len(farmers)]
            location = locations[idx % len(locations)]
            area = None
            severity = ""
            if polygons:
                # Approximate ellipse area vs image.
                bw, bh = boxes[-1].w if boxes else 8, boxes[-1].h if boxes else 8
                area = round(100.0 * math.pi * (bw / 2) * (bh / 2) / (w * h), 3)
                severity = taxonomy.severity_from_area(area).id
            rec = ImageRecord(
                sample_id=sample_id,
                relpath=str(Path("synthetic") / "smoke" / "images" / f"{sample_id}.png"),
                crop_id=crop_id,
                class_id=class_id,
                task="classification",
                split="unassigned",
                source="synthetic_smoke",
                license="generated_in_repo",
                dataset_version=dataset_version,
                is_synthetic=True,
                is_field=False,
                farmer_id=farmer,
                location_id=location,
                capture_session_id=f"{farmer}_sess_{k // 2}",
                source_group=farmer,
                width=w,
                height=h,
                sha256=sha256_file(abs_path),
                captured_at=utc_now_iso(),
                lat=18.5 + (idx % 5) * 0.2,
                lon=73.8 + (idx % 4) * 0.2,
                growth_stage=taxonomy.crop_stages[idx % max(1, len(taxonomy.crop_stages) - 1)],
                variety=f"{crop_id}_var_{idx % 3}",
                severity_bin=severity,
                affected_area_pct=area,
                symptom_region="leaf",
                notes=f"synthetic difficulty={difficulty}",
            )
            records.append(rec)
            detections.append(
                DetectionAnnotation(
                    sample_id=sample_id,
                    width=w,
                    height=h,
                    boxes=boxes,
                    polygons=polygons,
                    is_synthetic=True,
                )
            )
            observations.append(
                ObservationRecord(
                    observation_id=f"obs_{sample_id}",
                    timestamp=utc_now_iso(),
                    crop_id=crop_id,
                    farm_id=farmer,
                    field_id=location,
                    farmer_id=farmer,
                    lat=rec.lat,
                    lon=rec.lon,
                    disease_id=class_id,
                    severity_bin=severity,
                    affected_area_pct=area,
                    confidence=0.5,
                    source="synthetic",
                    is_synthetic=True,
                    notes=difficulty,
                )
            )
            idx += 1

    # Synthetic trap + weather rows (clearly labeled). Values are placeholders.
    pest_ids = taxonomy.pest_ids()[:8] or ["aphid"]
    for i, pest_id in enumerate(pest_ids):
        traps.append(
            TrapRecord(
                trap_id=f"syn_trap_{i:02d}",
                timestamp=utc_now_iso(),
                trap_type="pheromone",
                pest_id=pest_id,
                count=int(rng.randint(0, 12)),
                crop_id=taxonomy.crops.get(next(iter(taxonomy.crops))) and list(taxonomy.crops)[i % len(taxonomy.crops)],
                lat=19.0,
                lon=74.0,
                is_synthetic=True,
                source="synthetic_smoke",
            )
        )
        weather.append(
            WeatherRecord(
                station_id=f"syn_wx_{i:02d}",
                timestamp=utc_now_iso(),
                lat=19.0,
                lon=74.0,
                temperature_c=round(24 + rng.random() * 8, 2),
                humidity_pct=round(55 + rng.random() * 30, 2),
                rainfall_mm=round(rng.random() * 12, 2),
                wind_ms=round(rng.random() * 4, 2),
                soil_moisture=None,
                is_synthetic=True,
                source="synthetic_smoke",
                missing_fields="soil_moisture",
            )
        )

    write_yolo_detection(out_dir / "yolo" / "detect", records, detections)
    write_yolo_segmentation(out_dir / "yolo" / "segment", records, detections)
    write_coco_detection(out_dir / "coco" / "instances.json", records, detections)
    write_jsonl(out_dir / "observations.jsonl", observations)
    write_jsonl(out_dir / "traps.jsonl", traps)
    write_jsonl(out_dir / "weather.jsonl", weather)
    meta = {
        "kind": "synthetic",
        "warning": "SYNTHETIC DATA FOR PIPELINE TESTS ONLY. Not field observations.",
        "n_images": len(records),
        "n_observations": len(observations),
        "n_traps": len(traps),
        "n_weather": len(weather),
        "seed": seed,
        "created_at": utc_now_iso(),
    }
    (out_dir / "SYNTHETIC.txt").write_text(
        "SYNTHETIC DATA — development/testing only. Do not report as real agricultural evidence.\n",
        encoding="utf-8",
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return records
