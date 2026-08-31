# phase2.md — Computer vision perception

Date: 2026-08-31  
Status: **IMPLEMENTED (code + dummy backends). NOT TRAINED. Metrics NOT MEASURED.**

## Objectives

MASTER_PROMPT §6: image → YOLO11 detect → EfficientNetV2-S classify → YOLO11-seg lesions → severity; required output fields; evaluation scripts; deployment measurements.

## Implementation

| Piece | Path | Status |
| --- | --- | --- |
| Output schema | `src/cropai/vision/schema.py` | [T] |
| Preprocess / quality reject | `preprocess.py` | [T] |
| Dummy backends (no weights) | `backends.py` | [T] |
| Torch / Ultralytics adapters | `torch_models.py` | [x] untested here (no torch) |
| Pipeline | `pipeline.py` | [T] |
| Severity (mask, box proxy) | `severity.py` | [T] |
| Metrics | `metrics.py` | [T] |
| Train classifier | `train_classifier.py` | [x] dry-run only |
| Train YOLO det/seg | `train_detector.py` | [x] dry-run only |
| CLI train | `python -m cropai.vision.train` | [T] dry-run |
| Evaluate | `python -m cropai.vision.evaluate` | [x] |
| Benchmark harness | `python -m cropai.vision.benchmark` | [T] |
| ONNX export helpers | `export.py` | [x] needs torch |
| Infer CLI | `python -m cropai infer IMAGE` | [T] |

Untrained dummy backends **force expert referral** and do not emit a confident disease name.

## Experiments

- pytest `tests/vision/` (invalid image, dummy pipeline, metrics, severity, dry-run, benchmark honesty)
- `./train_linux.sh train --dry-run` on this sandbox (no CUDA, no torch)
- `cropai infer` on a synthetic PNG → `expert_referral=true`, `disease_class=unknown`, backend `dummy+dummy+dummy`

No GPU training was run. No accuracy numbers were produced.

## Results

| Metric | Value | Status |
| --- | --- | --- |
| Detection mAP / P / R | n/a | NOT MEASURED |
| Classification acc / F1 | n/a | NOT MEASURED |
| Seg IoU / Dice | n/a | NOT MEASURED |
| Severity MAE / RMSE | n/a | NOT MEASURED |
| FPS / latency / VRAM | n/a | NOT MEASURED |

Dummy infer on synthetic (not a benchmark): referral rate 100% by design.

## Decisions

```
Decision: Torch-optional dummy backends so CI works without inventing diagnoses
Selected: dummy detector/classifier/segmenter until checkpoints exist
Alternatives: skip pipeline until GPU; fake high-confidence labels
Evidence: MASTER_PROMPT §27, §17
Reason: Untrained ≠ healthy. Referral is the safe default.
Date: 2026-08-31
```

```
Decision: Keep starting candidates (YOLO11n, EfficientNetV2-S, YOLO11n-seg) as candidates only
Selected: NOT SELECTED
Alternatives: listed in configs/training.yaml
Evidence: none measured
Date: 2026-08-31
```

## Failures

- PyTorch / Ultralytics / CUDA absent in sandbox → cannot train or export
- No public/field images → cannot measure field robustness
- Ultralytics YOLO11 weight download not attempted

## Verification (Phase 2.5)

See AGENT_REVIEW.md. Comparison harness exists. **No model is selected.**

## Confidence

Agent Confidence: **68/100** (code). Model-quality confidence: **0/100** (unmeasured).

Recommendation: **PROCEED** to Phase 3 risk-engine *infrastructure* in parallel with user-side training on RTX 4050. Do not treat candidates as winners.

## Next steps

Stop. Ask the user whether to proceed to Phase 3, or wait until they run:

```bash
pip install -r requirements-train.txt   # CUDA PyTorch build
./train_linux.sh dataset
./train_linux.sh train                  # needs CUDA
# or: python -m cropai.vision.train --task classifier
#     python -m cropai.vision.benchmark
```
