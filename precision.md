# precision.md

No trained model exists. Numbers below are **targets / policy**, not measurements.

| Precision | Status | Notes |
| --- | --- | --- |
| FP32 | NOT BENCHMARKED | Training baseline |
| FP16 | SELECTED as initial GPU deployment target | AMP during training when CUDA is present |
| INT8 | NOT ADOPTED | Only after calibration + measured accuracy/latency |

Selected precision: **FP16 (policy only)**. Accuracy, latency, memory, compatibility, and degradation: **PENDING BENCHMARK**.

Do not claim FP16 is “better” until Phase 5.5 numbers exist.
