# phase5.md — Edge deployment and performance engineering

Date: 2026-08-31  
Status: **IMPLEMENTED (runtime harness). NO TensorRT/ONNX export. GPU metrics NOT MEASURED.**

## Objectives

MASTER_PROMPT §9: PyTorch → ONNX → TensorRT FP16; ORT/PyTorch fallback; load/warmup/queue/cache/recovery; measure stage latencies; long-run tests; INT8 only after measured calibration.

## Implementation

| Piece | Path | Status |
| --- | --- | --- |
| Runtime config | `configs/runtime.yaml` | [T] |
| Runtime selector | `src/cropai/runtime/select.py` | [T] |
| Model cache | `cache.py` | [T] |
| Bounded queue | `queue.py` | [T] |
| Timed e2e session | `session.py` | [T] dummy |
| Profiler | `profile.py` | [T] dummy |
| Endurance harness | `endurance.py` | [T] smoke |
| Export plan | `export_plan.py` + `vision/export.py` | [T] dry-run |
| Phase 5.5 harness | `benchmark.py` | [T] |
| CLI | `profile-pipeline` `export-models` `endurance` `benchmark-runtime` | [T] |

Sandbox runtime is **dummy**. Dummy FPS is **not reported** as model throughput.

## Experiments

- pytest `tests/runtime/` — dummy selector, queue bound, export dry-run, honest benchmark, endurance 0.6s marks 5/60 min NOT RUN
- `cropai export-models --dry-run`
- `cropai benchmark-runtime`
- Dummy profile on a 96×96 PNG (low-contrast) — e2e timer only, labeled dummy

No ONNX file written. No TensorRT engine. No 5/15/30/60 minute run.

## Results

| Metric | Value | Status |
| --- | --- | --- |
| YOLO / classify / segment latency | n/a | NOT MEASURED |
| FPS (model) | n/a | NOT MEASURED (dummy FPS omitted) |
| GPU util / VRAM / thermal | n/a | NOT MEASURED |
| FP16 vs FP32 vs INT8 | n/a | NOT MEASURED |
| 5 / 15 / 30 / 60 min endurance | n/a | NOT RUN |
| INT8 | — | **NOT ADOPTED** |

Dummy-path wall time on this sandbox is an implementation smoke, not a deployment benchmark.

## Decisions

```
Decision: Do not adopt INT8
Selected: NOT ADOPTED
Evidence: no calibration set, no measured speedup
Date: 2026-08-31
```

```
Decision: FP16 remains GPU policy target, not a measured result
Selected: policy only
Evidence: no CUDA / no engines
Date: 2026-08-31
```

```
Decision: Do not optimize dummy backends
Selected: no kernel/queue tuning beyond a bounded reject queue
Evidence: MASTER_PROMPT §26 — do not optimize without a real profile
Date: 2026-08-31
```

## Failures

- No checkpoints, torch, ORT, or TensorRT
- Low-contrast synthetic PNG is rejected by quality gates (expected)
- Long-run tests not executed (agent environment is lightweight)

## Verification (Phase 5.5)

See AGENT_REVIEW.md.

- Measured performance: dummy smoke only
- Bottleneck: UNKNOWN
- Optimization applied: none
- Remaining issues: no GPU pipeline

## Confidence

Agent Confidence: **70/100** (code). Deployment-performance confidence: **0/100**.

Recommendation: **PROCEED** to Phase 6 *final validation harness* that reports NOT VERIFIED honestly. Do not claim the system is complete.

## Next steps

Stop. Ask the user whether to proceed to Phase 6.

On the RTX 4050 after training:

```bash
pip install -r requirements-train.txt
python -m cropai export-models --execute
python -m cropai profile-pipeline path.jpg --repeats 20
python -m cropai endurance path.jpg --seconds 300
```
