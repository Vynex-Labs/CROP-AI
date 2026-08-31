# test.md

Last updated: 2026-08-31

Latest local run: **66 passed** (`.venv/bin/python -m pytest -q`).

## How to run

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src pytest
```

## Inventory

| Layer | Tests | Status |
| --- | --- | --- |
| Unit — taxonomy / config | `tests/config/test_taxonomy.py`, `test_loader.py` | IMPLEMENTED |
| Dataset — synthetic, splits, leakage | `tests/dataset/test_synthetic_and_splits.py` | IMPLEMENTED |
| Dataset — YOLO/COCO/invalid bbox | `tests/dataset/test_annotations.py` | IMPLEMENTED |
| Dataset — prepare + analysis honesty | `tests/dataset/test_prepare_and_analysis.py` | IMPLEMENTED |
| Hardware report / observation store | `tests/dataset/test_hardware_and_store.py` | IMPLEMENTED |
| Model unit tests | `tests/vision/test_perception.py` (dummy, no weights) | IMPLEMENTED |
| Integration (vision pipeline) | dummy infer + reject | IMPLEMENTED (untrained) |
| Risk engine | `tests/risk/test_risk_engine.py` | IMPLEMENTED (untrained) |
| Geospatial hotspots | `tests/geo/test_hotspots.py` | IMPLEMENTED (synthetic scenarios) |
| Advisory / referral | `tests/advisory/test_advisory.py` | IMPLEMENTED |
| Runtime / export / endurance | `tests/runtime/test_runtime.py` | IMPLEMENTED (dummy) |
| Field robustness | — | NOT STARTED (needs field images) |
| Missing-data (risk) | weather/trap absence → reduced confidence, continues | IMPLEMENTED (synthetic) |
| Failure handling | invalid image missing-file validator + vision reject | PARTIAL |
| Regression | pytest on each change | IN PROGRESS |
| Performance / long-run | endurance smoke; 5–60 min NOT RUN | HARNESS ONLY |
| Final validation | — | NOT STARTED (Phase 6) |

## Dataset validation (product code, not only tests)

`cropai.dataset.validate.DatasetValidator` implements the Phase 1 minimum checks from MASTER_PROMPT §5.

## Known gaps

- No tests against PlantVillage/PlantDoc on disk (`@pytest.mark.requires_data`) because those files are not in the clone.
- No GPU tests.
- No advisory-engine tests beyond YAML presence (Phase 4).
- Synthetic data must never be used as a substitute for field robustness tests.
