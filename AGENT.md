# AGENT.md

Primary handoff / state document. If this file disagrees with `MASTER_PROMPT.md`, the master prompt wins.

## Current phase

**Phase 2 complete (code). Stopped at Phase 2.5 verification gate. Models NOT TRAINED / NOT SELECTED.**

Next action: user chooses PROCEED to Phase 3 (risk infrastructure) and/or runs GPU training. Do not treat YOLO11 / EfficientNetV2-S as winners.

## Completed work

- Inspected empty repository, sandbox hardware (no GPU), software (Python 3.11, no PyTorch)
- Saved exact `MASTER_PROMPT.md`
- Data-driven taxonomy (`configs/crop_config.yaml`) — 17 crops: top-10 India + tomato/potato/maize/grape + CROPSAP/onion/pepper (not narrowed)
- Dataset registry, adapters, validation, grouped splits, versioning
- Synthetic smoke dataset (labeled)
- Field collection protocol
- IPM placeholder knowledge with no invented doses
- `train_linux.sh` / `train_windows.bat` + `scripts/check_env.py`
- Required documentation files
- Phase 1 unit tests
- Phase 2 perception pipeline, dummy backends, train/eval/benchmark harness
- pytest vision tests

## Pending work

- User download of PlantVillage / PlantDoc / IP102
- Maharashtra field image collection
- CROPSAP / weather / soil authorised dumps
- User GPU training of detector / classifier / segmenter
- Measured Phase 2.5 model comparison
- All later phases

## Architecture

Working hypothesis only (MASTER_PROMPT §2). See README.md. No weights.

## Decisions

See `phase1.md` and `phase2.md`. No model selected.

## Commands

```bash
PYTHONPATH=src python -m cropai hardware
PYTHONPATH=src python -m cropai taxonomy
PYTHONPATH=src python -m cropai prepare-dataset
PYTHONPATH=src python -m cropai infer path.jpg --crop rice
PYTHONPATH=src python -m cropai.vision.train --dry-run
PYTHONPATH=src pytest
./train_linux.sh train --dry-run
```

## Known issues

- Sandbox has no NVIDIA GPU / CUDA / PyTorch
- `data/raw/` empty — prepare uses synthetic smoke only
- Operational crops cotton, sugarcane, tur, jowar, onion lack adequate public images
- IPM YAML is `status: placeholder` — chemical advice disabled by policy
- `src/cropai/vision/` exists; backends are dummy until checkpoints exist

## Requirement traceability

Status key: `[ ]` NOT STARTED · `[~]` IN PROGRESS · `[x]` IMPLEMENTED · `[T]` TESTED · `[V]` VERIFIED · `[P]` PENDING USER EXECUTION

| Requirement | Implementation | Test | Status |
| --- | --- | --- | --- |
| Save MASTER_PROMPT.md | `MASTER_PROMPT.md` | file present | [x] |
| Docs set (README, AGENT, phases, speed, test, techstack, precision, dataset, model_selection) | repo root | existence | [x] |
| Parse SIH26131 / Maharashtra domain | `configs/crop_config.yaml` + `docs/SIH26131.md` | `tests/config/` | [T] |
| Target crops / diseases / pests / healthy | 17 crops, 74 classes, 28 pests | taxonomy integrity + top10/hort tests | [T] |
| Severity labels | severity bins | `test_severity_bins_cover_0_100` | [T] |
| Symptom regions / crop stages | crop_config | taxonomy | [T] |
| Public dataset registry + licenses | `configs/dataset.yaml` `dataset.md` | loader test | [T] |
| Field dataset identified | protocol only, 0 images | analysis gate | [T] |
| Weather / soil / trap sources identified | dataset.yaml + synthetic writers | synthetic files | [x] |
| Geospatial structure (H3 res, observation fields) | crop_config + ObservationRecord | schema tests | [x] |
| Custom field dataset design | `data/field_protocol/` | n/a | [x] |
| Dataset-generation tooling | `prepare.py` `synthetic.py` adapters | prepare tests | [T] |
| Annotation formats YOLO/COCO/CSV | yolo.py coco.py schema.py | annotation tests | [T] |
| Train/val/isolated test + no leakage | split.py leakage.py | split tests | [T] |
| Provenance / licensing / synthetic flag / versioning | manifest.py SYNTHETIC.txt | prepare tests | [T] |
| Automated dataset validation (min checks) | validate.py | validator tests | [T] |
| Transfer-learning strategy documented | README / phase1 / dataset.md | n/a | [x] |
| Do not fabricate field data | analysis.gate.fabricated_field_data=False | test_analysis_does_not_claim_field_data | [T] |
| train_linux.sh / train_windows.bat | root scripts | manual smoke | [x] |
| CUDA detect + warn, no silent CPU train | hardware.py, train scripts | hardware test | [T] |
| PyTorch+CUDA training | `src/cropai/vision/train.py` | dry-run | [P] |
| YOLO11 detector | `train_detector.py` + dummy | vision tests | [x] code / [P] train |
| EfficientNetV2-S classifier | `train_classifier.py` + dummy | vision tests | [x] code / [P] train |
| YOLO11-seg | `train_detector.py --seg` + dummy | vision tests | [x] code / [P] train |
| LightGBM risk | — | — | [ ] |
| H3 + KDE/DBSCAN | — | — | [ ] |
| Advisory engine (structured, no gen override) | IPM YAML stubs + rules | — | [~] |
| Expert referral thresholds | crop_config confidence_thresholds | taxonomy test | [x] |
| Follow-up monitoring store | ObservationStore.follow_ups | store test | [T] |
| Multilingual presentation layer | languages listed en/hi/mr | — | [ ] |
| Offline diagnosis | PerceptionPipeline dummy path (no network) | infer test | [~] |
| Failure handling policies | preprocess reject + pipeline catch | vision tests | [T] |
| Official dashboard | — | — | [ ] |
| Farmer / extension UI | — | — | [ ] |
| ONNX / TensorRT / FP16 path | `vision/export.py` + configs | — | [~] code / [P] run |
| Reproducible run metadata | `vision/run_meta.py` | dry-run | [x] |
| Benchmarks (acc, F1, mAP, FPS, …) | — | — | [ ] |
| No invented pesticide dosage | knowledge/ipm + engine rules | — | [x] |

Phase 6 success checklist (MASTER_PROMPT §31): all items **NOT VERIFIED**.
