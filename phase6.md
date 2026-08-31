# phase6.md — Final validation and benchmark

Date: 2026-08-31  
Status: **HARNESS ONLY. Architecture NOT frozen. Project NOT complete. All §31 items NOT VERIFIED.**

## Objectives

MASTER_PROMPT §10: freeze architecture, run the full pipeline, report vision / forecast / spatial / system benchmarks. §31 success checkboxes.

## Implementation

| Piece | Path | Status |
| --- | --- | --- |
| Freeze record | `configs/architecture_freeze.yaml` | `frozen: false` |
| Checklist | `src/cropai/validate/checklist.py` | [T] |
| Report | `src/cropai/validate/report.py` | [T] |
| E2E smoke wrapper | `e2e.py` | [T] not a field bench |
| CLI | `cropai validate-final` | [T] exit 3 if incomplete |

No models were changed (none were selected). Freeze is **refused** because gates 2.5–5.5 have no measured winners.

## Experiments

- pytest `tests/validate/` — no checklist item VERIFIED; all numeric metrics NOT MEASURED
- `python -m cropai validate-final` → `project_complete=false`, exit 3

Dummy e2e smoke is available via `--smoke-image` and is labeled **not** a Phase 6 benchmark.

## Results

Every required metric is **NOT MEASURED**.

| Block | Status |
| --- | --- |
| Detection mAP / P / R | NOT MEASURED |
| Classification acc / F1 / calibration | NOT MEASURED |
| Segmentation IoU / Dice | NOT MEASURED |
| Severity MAE / RMSE | NOT MEASURED |
| Forecast ROC/PR/Brier/FNR/lead time | NOT MEASURED |
| Hotspot P/R / stability | NOT MEASURED |
| E2E latency / FPS / GPU / VRAM | NOT MEASURED |
| Official dashboard | NOT IMPLEMENTED |
| §31 verified count | **0 / 24** |

## Decisions

```
Decision: Do not freeze candidate architectures
Selected: frozen=false
Evidence: no measured Phase 2.5–5.5 winners
Date: 2026-08-31
```

```
Decision: Do not mark the SIH system complete
Selected: project_complete=false
Evidence: MASTER_PROMPT §31; 0 VERIFIED checkboxes
Date: 2026-08-31
```

## Failures

- No field images, no trained weights, no CUDA in this environment
- No dashboard UI
- No 1-hour endurance
- IPM still placeholder

## Verification

This **is** the Phase 6 gate. It fails on purpose rather than fabricating pass.

## Confidence

Agent Confidence: **75/100** that the *harness is honest*.  
Confidence the product meets §31: **0/100**.

Recommendation: **HOLD** on claiming completion. Collect field data, train on RTX 4050, attach verified IPM, then re-run `cropai validate-final`.

## Next steps

There is no Phase 7. Further work is user-side training, data, and a real dashboard — not another silent phase skip.
