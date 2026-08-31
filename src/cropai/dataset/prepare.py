"""Dataset preparation entrypoint used by train_linux.sh / train_windows.bat."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cropai.config.loader import load_dataset_config
from cropai.dataset.adapters.base import SourceAdapter
from cropai.dataset.adapters.field_csv import FieldCSVAdapter
from cropai.dataset.adapters.ip102 import IP102Adapter
from cropai.dataset.adapters.plantdoc import PlantDocAdapter
from cropai.dataset.adapters.plantvillage import PlantVillageAdapter
from cropai.dataset.analysis import analyze_manifest, write_analysis
from cropai.dataset.manifest import Manifest, write_manifest
from cropai.dataset.schema import ImageRecord, write_csv
from cropai.dataset.split import split_records
from cropai.dataset.synthetic import generate_smoke_dataset
from cropai.dataset.validate import DatasetValidator
from cropai.domain.taxonomy import Taxonomy
from cropai.utils.logging import setup_logging, utc_now_iso
from cropai.utils.paths import data_dir, ensure_runtime_dirs, repo_root


def _source_root(cfg: dict[str, Any], source_id: str) -> Path:
    meta = (cfg.get("sources") or {}).get(source_id) or {}
    local = meta.get("local_dir")
    if local:
        path = Path(str(local))
        return path if path.is_absolute() else repo_root() / path
    return data_dir() / "raw" / source_id


def _ingest_public(cfg: dict[str, Any], version: str) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    adapters: list[tuple[str, SourceAdapter]] = [
        ("plantvillage", PlantVillageAdapter()),
        ("plantdoc", PlantDocAdapter()),
        ("ip102", IP102Adapter()),
    ]
    for source_id, adapter in adapters:
        root = _source_root(cfg, source_id)
        if not root.exists():
            continue
        records.extend(adapter.ingest(root, dataset_version=version))
    field_root = _source_root(cfg, "field_maharashtra")
    csv_path = field_root / "images.csv"
    if csv_path.exists():
        records.extend(
            FieldCSVAdapter().ingest(
                field_root,
                csv_path=csv_path,
                dataset_version=version,
                source="field_maharashtra",
            )
        )
    return records


def prepare_dataset(
    *,
    include_synthetic: bool = True,
    synthetic_only: bool = False,
    seed: int = 42,
    dataset_version: str | None = None,
) -> dict[str, Any]:
    log = setup_logging()
    ensure_runtime_dirs()
    cfg = load_dataset_config()
    taxonomy = Taxonomy()
    version = dataset_version or str(cfg.get("version") or "v0.1.0-phase1")
    split_cfg = dict(cfg.get("splits") or {})

    records: list[ImageRecord] = []
    if not synthetic_only:
        records.extend(_ingest_public(cfg, version))

    public_count = len(records)
    if include_synthetic or synthetic_only or public_count == 0:
        syn = generate_smoke_dataset(
            data_dir() / "synthetic" / "smoke",
            n_per_class=4,
            seed=seed,
            taxonomy=taxonomy,
            dataset_version=version,
        )
        records.extend(syn)
        if public_count == 0:
            log.warning(
                "No public/field images found under data/raw/. "
                "Prepared SYNTHETIC smoke dataset only. This is not field evidence."
            )

    # Classifier manifest: drop records whose class is outside taxonomy (e.g. raw IP102 ids).
    known = set(taxonomy.classifier_classes())
    known_crops = set(taxonomy.crop_ids())
    kept: list[ImageRecord] = []
    skipped = 0
    for rec in records:
        if rec.class_id in known and rec.crop_id in known_crops:
            rec.dataset_version = version
            kept.append(rec)
        else:
            skipped += 1
    records = kept

    records = split_records(
        records,
        train_ratio=float(split_cfg.get("train_ratio", 0.70)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        seed=int(split_cfg.get("seed", seed) if "seed" in split_cfg else seed),
    )

    report = DatasetValidator(taxonomy=taxonomy, config=cfg).validate(records)
    analysis = analyze_manifest(records, taxonomy=taxonomy)

    manifest = Manifest(
        version=version,
        created_at=utc_now_iso(),
        records=records,
        sources=sorted({r.source for r in records}),
        notes=(
            "Phase 1 manifest. Synthetic records are labeled is_synthetic=true. "
            f"Skipped {skipped} records whose class/crop is outside taxonomy."
        ),
        extra={
            "validation_ok": report.ok,
            "n_errors": len(report.errors),
            "n_warnings": len(report.warnings),
            "public_ingested_before_filter": public_count,
        },
    )
    sidecar = write_manifest(manifest)
    splits_dir = data_dir() / "splits" / version
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        subset = [r for r in records if r.split == split_name]
        write_csv(splits_dir / f"{split_name}.csv", subset)

    analysis_path = write_analysis(analysis, data_dir() / "manifests" / f"analysis_{version}.json")
    validation_path = data_dir() / "manifests" / f"validation_{version}.json"
    validation_path.write_text(
        json.dumps(
            {
                "ok": report.ok,
                "n_records": report.n_records,
                "errors": report.errors,
                "warnings": report.warnings,
                "stats": report.stats,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "version": version,
        "n_records": len(records),
        "n_synthetic": sum(1 for r in records if r.is_synthetic),
        "n_real": sum(1 for r in records if not r.is_synthetic),
        "n_field_real": sum(1 for r in records if r.is_field and not r.is_synthetic),
        "splits": {
            s: sum(1 for r in records if r.split == s) for s in ("train", "val", "test")
        },
        "validation_ok": report.ok,
        "n_errors": len(report.errors),
        "n_warnings": len(report.warnings),
        "manifest": str(sidecar),
        "analysis": str(analysis_path),
        "validation": str(validation_path),
        "skipped_out_of_taxonomy": skipped,
    }
    log.info("prepare_dataset complete: %s", json.dumps({k: summary[k] for k in ("version", "n_records", "n_synthetic", "n_field_real", "validation_ok")}))
    return summary
