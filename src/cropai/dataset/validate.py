"""Automated dataset validation gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from cropai.config.loader import load_dataset_config
from cropai.dataset.leakage import (
    detect_group_leakage,
    detect_hash_leakage,
    detect_relpath_collision,
    detect_sample_id_collision,
)
from cropai.dataset.schema import ImageRecord, parse_iso
from cropai.domain.taxonomy import Taxonomy
from cropai.utils.paths import data_dir, repo_root


@dataclass
class ValidationReport:
    ok: bool
    n_records: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        if not self.ok:
            preview = "\n".join(self.errors[:20])
            raise ValueError(f"dataset validation failed ({len(self.errors)} errors)\n{preview}")


class DatasetValidator:
    def __init__(
        self,
        *,
        taxonomy: Taxonomy | None = None,
        config: dict[str, Any] | None = None,
        data_root: Path | None = None,
    ) -> None:
        self.taxonomy = taxonomy or Taxonomy()
        self.config = config or load_dataset_config()
        self.rules = dict(self.config.get("validation") or {})
        self.data_root = data_root or data_dir()

    def validate(self, records: list[ImageRecord]) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        errors.extend(detect_sample_id_collision(records))
        errors.extend(detect_relpath_collision(records))
        errors.extend(detect_group_leakage(records))
        errors.extend(detect_hash_leakage(records))

        min_w = int(self.rules.get("min_width", 64))
        min_h = int(self.rules.get("min_height", 64))
        max_w = int(self.rules.get("max_width", 8192))
        max_h = int(self.rules.get("max_height", 8192))
        suffixes = {s.lower() for s in self.rules.get("allowed_image_suffixes", [".png", ".jpg"])}
        lat_range = list(self.rules.get("lat_range", [-90, 90]))
        lon_range = list(self.rules.get("lon_range", [-180, 180]))
        known_crops = set(self.taxonomy.crop_ids())
        known_classes = set(self.taxonomy.classifier_classes())
        known_stages = set(self.taxonomy.crop_stages) | {"", "unknown"}
        known_regions = set(self.taxonomy.symptom_regions) | {""}

        missing_files = 0
        corrupted = 0
        resolution_problems = 0
        metadata_problems = 0
        geo_problems = 0
        temporal_problems = 0
        label_problems = 0
        bbox_problems = 0
        mask_problems = 0
        synthetic_unflagged = 0

        hashes: dict[str, list[str]] = defaultdict(list)
        class_counts: Counter[str] = Counter()
        crop_counts: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        field_real = 0
        synthetic = 0

        for rec in records:
            class_counts[rec.class_id] += 1
            crop_counts[rec.crop_id] += 1
            split_counts[rec.split or "unassigned"] += 1
            source_counts[rec.source] += 1
            if rec.is_synthetic:
                synthetic += 1
            if rec.is_field and not rec.is_synthetic:
                field_real += 1

            if rec.sha256:
                hashes[rec.sha256].append(rec.sample_id)

            if rec.crop_id not in known_crops:
                label_problems += 1
                errors.append(f"{rec.sample_id}: unknown crop_id {rec.crop_id}")
            if rec.class_id not in known_classes:
                label_problems += 1
                errors.append(f"{rec.sample_id}: unknown class_id {rec.class_id}")
            if rec.growth_stage not in known_stages:
                metadata_problems += 1
                warnings.append(f"{rec.sample_id}: unknown growth_stage {rec.growth_stage}")
            if rec.symptom_region not in known_regions:
                metadata_problems += 1
                warnings.append(f"{rec.sample_id}: unknown symptom_region {rec.symptom_region}")

            if self.rules.get("require_synthetic_flag") and rec.source.startswith("synthetic") and not rec.is_synthetic:
                synthetic_unflagged += 1
                errors.append(f"{rec.sample_id}: synthetic source without is_synthetic=true")

            suffix = Path(rec.relpath).suffix.lower()
            if suffix and suffix not in suffixes:
                errors.append(f"{rec.sample_id}: disallowed suffix {suffix}")

            image_path = self._resolve(rec.relpath)
            if not image_path.exists():
                missing_files += 1
                errors.append(f"{rec.sample_id}: missing file {rec.relpath}")
            else:
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                    with Image.open(image_path) as img:
                        width, height = img.size
                except (UnidentifiedImageError, OSError) as exc:
                    corrupted += 1
                    errors.append(f"{rec.sample_id}: corrupted image ({exc})")
                    width = rec.width
                    height = rec.height
                else:
                    if rec.width and rec.width != width:
                        metadata_problems += 1
                        warnings.append(f"{rec.sample_id}: width metadata {rec.width} != {width}")
                    if rec.height and rec.height != height:
                        metadata_problems += 1
                        warnings.append(f"{rec.sample_id}: height metadata {rec.height} != {height}")
                    if width < min_w or height < min_h or width > max_w or height > max_h:
                        resolution_problems += 1
                        errors.append(f"{rec.sample_id}: resolution {width}x{height} outside limits")

            if rec.lat is not None and not (lat_range[0] <= rec.lat <= lat_range[1]):
                geo_problems += 1
                errors.append(f"{rec.sample_id}: lat {rec.lat} out of range")
            if rec.lon is not None and not (lon_range[0] <= rec.lon <= lon_range[1]):
                geo_problems += 1
                errors.append(f"{rec.sample_id}: lon {rec.lon} out of range")
            if (rec.lat is None) ^ (rec.lon is None):
                geo_problems += 1
                errors.append(f"{rec.sample_id}: lat/lon must both be set or both empty")

            if rec.captured_at and parse_iso(rec.captured_at) is None:
                temporal_problems += 1
                errors.append(f"{rec.sample_id}: invalid captured_at {rec.captured_at}")

            if rec.annotation_path:
                ann_path = self._resolve(rec.annotation_path)
                if not ann_path.exists():
                    errors.append(f"{rec.sample_id}: missing annotation {rec.annotation_path}")
                else:
                    bbox_err, mask_err = self._check_annotation_file(ann_path, rec)
                    bbox_problems += bbox_err
                    mask_problems += mask_err
                    if bbox_err:
                        errors.append(f"{rec.sample_id}: invalid bounding boxes in {rec.annotation_path}")
                    if mask_err:
                        errors.append(f"{rec.sample_id}: invalid segmentation mask in {rec.annotation_path}")

        duplicate_hashes = {h: ids for h, ids in hashes.items() if len(ids) > 1}
        if duplicate_hashes:
            warnings.append(f"{len(duplicate_hashes)} duplicate content hashes (same image reused)")

        imbalance_ratio = float(self.rules.get("warn_imbalance_ratio", 10.0))
        if class_counts:
            most = class_counts.most_common(1)[0][1]
            least = min(class_counts.values())
            if least > 0 and most / least >= imbalance_ratio:
                warnings.append(
                    f"class imbalance {most}:{least} exceeds warn ratio {imbalance_ratio}"
                )

        if field_real == 0:
            warnings.append(
                "No real field images present. Public lab data is not a substitute for Maharashtra field images."
            )

        stats = {
            "n_records": len(records),
            "n_synthetic": synthetic,
            "n_real": len(records) - synthetic,
            "n_field_real": field_real,
            "missing_files": missing_files,
            "corrupted_images": corrupted,
            "resolution_problems": resolution_problems,
            "metadata_problems": metadata_problems,
            "geolocation_problems": geo_problems,
            "temporal_problems": temporal_problems,
            "label_problems": label_problems,
            "bbox_problems": bbox_problems,
            "mask_problems": mask_problems,
            "synthetic_unflagged": synthetic_unflagged,
            "duplicate_hash_groups": len(duplicate_hashes),
            "class_counts": dict(class_counts),
            "crop_counts": dict(crop_counts),
            "split_counts": dict(split_counts),
            "source_counts": dict(source_counts),
        }
        fail_on_leakage = bool(self.rules.get("fail_on_leakage", True))
        leakage_errors = [e for e in errors if "appears in splits" in e or "sha256" in e]
        if not fail_on_leakage:
            warnings.extend(leakage_errors)
            errors = [e for e in errors if e not in leakage_errors]

        return ValidationReport(
            ok=len(errors) == 0,
            n_records=len(records),
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    def _resolve(self, relpath: str) -> Path:
        path = Path(relpath)
        if path.is_absolute():
            return path
        candidate = self.data_root / path
        if candidate.exists():
            return candidate
        return repo_root() / path

    def _check_annotation_file(self, path: Path, rec: ImageRecord) -> tuple[int, int]:
        bbox_err = 0
        mask_err = 0
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return 0, 0
        width = rec.width or 1
        height = rec.height or 1
        min_box = float(self.rules.get("bbox_min_size_px", 2))
        min_mask = int(self.rules.get("mask_min_pixels", 4))
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 5:
                _, cx, cy, w, h = parts
                try:
                    bw = float(w) * width
                    bh = float(h) * height
                    fx, fy = float(cx), float(cy)
                except ValueError:
                    bbox_err += 1
                    continue
                if not (0 <= fx <= 1 and 0 <= fy <= 1):
                    bbox_err += 1
                if bw < min_box or bh < min_box:
                    bbox_err += 1
            elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                try:
                    coords = list(map(float, parts[1:]))
                except ValueError:
                    mask_err += 1
                    continue
                if any(c < -0.01 or c > 1.01 for c in coords):
                    mask_err += 1
                if len(coords) // 2 < 3:
                    mask_err += 1
                # crude pixel coverage proxy
                xs = coords[0::2]
                ys = coords[1::2]
                coverage = abs(max(xs) - min(xs)) * abs(max(ys) - min(ys)) * width * height
                if coverage < min_mask:
                    mask_err += 1
            else:
                bbox_err += 1
        return bbox_err, mask_err
