# tests.md — full pytest report

Date: 2026-09-01  
Command: `PYTHONPATH=src .venv/bin/python -m pytest -v --tb=short`  
Root: `/home/user/CROP-AI`  
Config: `pyproject.toml` (`testpaths = tests`)

## Summary

| | |
| --- | --- |
| Result | **69 passed, 0 failed, 0 skipped, 0 errors** |
| Duration | **8.59 s** |
| pytest | 9.1.1 |
| Python | 3.11.2 |
| Platform | linux |
| GPU / CUDA / PyTorch | **not used** (dummy / heuristic / grid fallback) |

This run covers **every generated test file** under `tests/`. It is **not** a field, GPU, PlantVillage, or 5–60 minute endurance benchmark.

## Totals by file

| File | Collected | Result |
| --- | --- | --- |
| `tests/advisory/test_advisory.py` | 6 | passed |
| `tests/config/test_loader.py` | 1 | passed |
| `tests/config/test_taxonomy.py` | 9 | passed |
| `tests/dataset/test_annotations.py` | 3 | passed |
| `tests/dataset/test_hardware_and_store.py` | 2 | passed |
| `tests/dataset/test_prepare_and_analysis.py` | 2 | passed |
| `tests/dataset/test_synthetic_and_splits.py` | 5 | passed |
| `tests/geo/test_hotspots.py` | 10 | passed |
| `tests/risk/test_risk_engine.py` | 12 | passed |
| `tests/runtime/test_runtime.py` | 6 | passed |
| `tests/validate/test_phase6.py` | 3 | passed |
| `tests/vision/test_perception.py` | 10 | passed |
| **Total** | **69** | **69 passed** |

## Case list (all PASSED)

### Advisory (`tests/advisory/test_advisory.py`)

- `test_placeholder_ipm_blocks_chemicals_and_refers`
- `test_low_confidence_suppresses_diagnosis`
- `test_severe_triggers_laboratory_referral`
- `test_missing_ipm_pack_escalates`
- `test_multilingual_labels_do_not_change_actions`
- `test_never_emits_dosage`

### Config (`tests/config/`)

- `test_core_yaml_loads`
- `test_taxonomy_integrity`
- `test_counts_are_nonzero`
- `test_severity_bins_cover_0_100`
- `test_maharashtra_operational_crops_include_cropsap`
- `test_top10_indian_crops_are_present_and_operational`
- `test_tomato_potato_maize_grape_are_first_class`
- `test_list_was_not_narrowed`
- `test_no_local_field_images_are_claimed`
- `test_confidence_thresholds_are_conservative`

### Dataset (`tests/dataset/`)

- `test_yolo_boxes_are_normalized_and_parseable`
- `test_coco_contains_synthetic_flag`
- `test_invalid_yolo_bbox_detected`
- `test_hardware_report_does_not_invent_gpu`
- `test_observation_store_roundtrip`
- `test_prepare_synthetic_only_end_to_end`
- `test_analysis_does_not_claim_field_data`
- `test_synthetic_images_are_flagged`
- `test_split_does_not_leak_groups`
- `test_validator_accepts_synthetic_when_paths_resolvable`
- `test_validator_detects_missing_file`
- `test_csv_roundtrip`

### Geospatial / fusion (`tests/geo/test_hotspots.py`)

- `test_haversine_and_decay`
- `test_single_case_isolated`
- `test_nearby_cluster_not_isolated`
- `test_distant_cases_not_one_cluster`
- `test_old_versus_recent_declining`
- `test_recent_cluster_emerging_or_established`
- `test_low_confidence_downweighted_vs_validated`
- `test_fusion_missing_weather_and_trap`
- `test_fusion_conflict_referral`
- `test_benchmark_plan_does_not_invent_scores`

### Risk (`tests/risk/test_risk_engine.py`)

- `test_wet_humid_disease_risk_exceeds_dry`
- `test_high_trap_pest_risk_exceeds_zero`
- `test_missing_weather_continues_reduced_confidence`
- `test_missing_trap_continues_reduced_confidence`
- `test_insufficient_inputs_referral`
- `test_output_has_required_horizons`
- `test_history_days_since_outbreak`
- `test_synthetic_series_roundtrip`
- `test_metrics_separable_and_empty`
- `test_train_dry_run`
- `test_benchmark_plan_does_not_invent_scores`
- `test_evaluate_without_labels_is_not_measured`

### Runtime (`tests/runtime/test_runtime.py`)

- `test_select_runtime_is_dummy_without_weights`
- `test_queue_rejects_when_full`
- `test_export_plan_dry_run_does_not_invent_success`
- `test_benchmark_plan_honest`
- `test_profile_dummy_does_not_quote_fps`
- `test_endurance_smoke_marks_hour_not_run`

### Phase 6 (`tests/validate/test_phase6.py`)

- `test_no_checklist_item_is_verified`
- `test_final_report_does_not_invent_metrics`
- `test_cli_nonzero_when_incomplete`

### Vision (`tests/vision/test_perception.py`)

- `test_invalid_image_rejected`
- `test_corrupt_image_rejected`
- `test_pipeline_dummy_on_synthetic`
- `test_low_confidence_forces_referral`
- `test_quality_ok_on_normal_image`
- `test_tiny_image_rejected`
- `test_metrics_toy`
- `test_severity_from_mask`
- `test_train_dry_run`
- `test_benchmark_plan_does_not_invent_scores`

## What this run does **not** prove

These were **not executed** (and must not be inferred from 69 passed):

- PlantVillage / PlantDoc / IP102 / field-image tests (`@pytest.mark.requires_data` — no files on disk)
- GPU / CUDA / PyTorch / Ultralytics / TensorRT tests
- LightGBM / XGBoost fit on real outbreak labels
- 5 / 15 / 30 / 60 minute endurance
- `cropai validate-final` product success (harness tests assert it stays incomplete; CLI exits 3)

Synthetic fixtures are labeled synthetic. Passing CI is not field robustness.

## How to reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
export PYTHONPATH=src
python -m pytest -v --tb=short
```
