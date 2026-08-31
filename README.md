# CROP-AI — SIH26131

**Early detection and management of crop diseases and pest infestations**

Government of Maharashtra · Smart India Hackathon 2026 · Software

CROP-AI is a production-oriented, offline-capable farm-health system: image-based disease and pest detection, severity estimation, weather/trap risk forecasting, geospatial hotspots, structured IPM advisories, expert referral, follow-up monitoring, and official dashboards.

This is **not** a single-image classifier demo. Authoritative specification: [`MASTER_PROMPT.md`](MASTER_PROMPT.md). Live agent state: [`AGENT.md`](AGENT.md).

## Current status

| Item | State |
| --- | --- |
| Phase | **4 complete (code) → stopped at Phase 4.5. Models NOT TRAINED** |
| Perception models | Code + dummy backends. Candidates not selected. |
| Risk models | Heuristic fallback only. LightGBM **not selected**. |
| Geospatial / fusion | DBSCAN+KDE+decay + weighted fusion. **Unvalidated.** |
| Advisory | Structured IPM stubs; chemicals **blocked**. |
| Trained weights | NONE |
| Maharashtra field images in repo | **0** (protocol only) |
| Public datasets in repo | **not downloaded** |
| Smoke dataset | SYNTHETIC, labeled `is_synthetic=true` |
| GPU in this sandbox | **none** (user RTX 4050 is the intended trainer) |

Do not treat synthetic smoke images or untrained architecture names as evidence of detection accuracy.

## Architecture (working hypothesis)

```text
Farmer image → YOLO11 detect → EfficientNetV2-S classify → YOLO11-seg lesions
        ↘ pest boxes
Field observation + trap + weather → LightGBM risk → H3 + KDE/DBSCAN hotspots
        → structured IPM / expert referral / follow-up → dashboard
```

Candidates only. Verification gates may replace them. See [`model_selection.md`](model_selection.md).

## Setup

Python 3.10+ recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export PYTHONPATH=src
python -m cropai hardware
python -m cropai taxonomy
python -m cropai prepare-dataset
python -m pytest
```

`train_linux.sh` / `train_windows.bat` use `.venv` automatically when present.

Or:

```bash
./train_linux.sh dataset           # Linux / WSL
train_windows.bat dataset          # cmd.exe or PowerShell
```

Heavy training extras (`requirements-train.txt`) are for the CUDA machine, not this sandbox.

## Dataset

Domain taxonomy is data-driven: [`configs/crop_config.yaml`](configs/crop_config.yaml).

- **Top 10 Indian crops** (union, not a cut): rice, wheat, sugarcane, cotton, maize, soybean, chickpea, groundnut, mustard, pearl millet (bajra).
- **Required horticulture (kept, not a v1-only subset):** tomato, potato, maize, grape.
- Maharashtra CROPSAP: rice, cotton, soybean, sugarcane, maize, tur, jowar, gram. Also onion and chilli/pepper (not dropped).
- Public transfer-learning sources registered (not vendored): PlantVillage (CC0, lab), PlantDoc (field-scraped), FieldPlant (Cameroon field), IP102 (pests).
- Custom field collection protocol: [`data/field_protocol/COLLECTION_PROTOCOL.md`](data/field_protocol/COLLECTION_PROTOCOL.md).
- Download notes: [`scripts/download_public_datasets.md`](scripts/download_public_datasets.md).
- Provenance / licenses / counts: [`dataset.md`](dataset.md).

**Field data has not been fabricated.** Public lab accuracy will never be reported as deployment readiness.

## Training

User machine (NVIDIA RTX 4050 + CUDA):

```bash
pip install -r requirements-train.txt   # install the CUDA build of PyTorch yourself
./train_linux.sh all                    # or train_windows.bat all
```

If CUDA is missing, scripts print a warning and **refuse** heavy training unless `--dry-run` / `--smoke` or `CROP_AI_ALLOW_CPU_TRAIN=1`.

## Inference / deployment

```bash
python -m cropai infer path/to/leaf.jpg --crop rice
```

Without checkpoints this uses **dummy backends**: low confidence, expert referral, `disease_class=unknown`. That is intentional.

Training (user RTX 4050):

```bash
pip install -r requirements-train.txt
./train_linux.sh train --task classifier
python -m cropai.vision.benchmark
```

Export helpers: `src/cropai/vision/export.py` (needs torch). TensorRT is Phase 5.

Risk forecast (uncalibrated heuristic until outbreak labels exist):

```bash
python -m cropai forecast-risk --crop rice --humidity 90 --temperature 26 --rainfall-7d 20 --synthetic
python -m cropai.risk.train --dry-run
python -m cropai benchmark-risk
./train_linux.sh train-risk --dry-run
```

Do not treat those scores as outbreak probabilities.

Hotspots / fusion / advisory:

```bash
python -m cropai hotspots --observations data/synthetic/smoke/observations.jsonl --as-of 2026-08-31
python -m cropai fuse-risk --crop rice --weather-risk 0.7 --trap-risk 0.2 --vision-untrained --synthetic
python -m cropai advise --crop rice --disease rice_blast --confidence 0.9 --lang hi
python -m cropai benchmark-geo
```

Advisory never invents pesticide dose. Placeholder IPM → monitoring + extension referral.

## Documentation map

| File | Role |
| --- | --- |
| `MASTER_PROMPT.md` | Authoritative contract (do not silently edit) |
| `AGENT.md` | Phase, decisions, requirement traceability |
| `AGENT_REVIEW.md` | Gate reviews |
| `phase1.md` … `phase6.md` | Phase logs |
| `dataset.md` `techstack.md` `test.md` `speed.md` `precision.md` `model_selection.md` | Living engineering logs |

## Advisory safety

Treatment text comes only from structured IPM records under `knowledge/ipm/`. The system must not invent pesticide dose, interval, PHI, or legal use. Unverified records escalate to extension / laboratory consultation.

## License

Project code: Apache-2.0. Dataset licenses are **per source** — see `dataset.md`. Do not redistribute PlantDoc/IP102/CROPSAP dumps without checking terms.
