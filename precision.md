# precision.md

No trained model exists. Numbers below are **policy**, not measurements.

| Precision | Status | Notes |
| --- | --- | --- |
| FP32 | NOT BENCHMARKED | Training baseline |
| FP16 | POLICY TARGET for GPU | AMP during training when CUDA is present. **Not measured** here. |
| INT8 | **NOT ADOPTED** | Requires reliable calibration, acceptable accuracy drop, and a measured speedup |

Selected precision: **FP16 (policy only)**. Accuracy, latency, memory, compatibility, and degradation: **PENDING BENCHMARK**.

Do not claim FP16 is faster or more accurate until Phase 5.5 numbers exist on the RTX 4050.
