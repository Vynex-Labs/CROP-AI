# speed.md

No trained inference stack is running. YOLO/classifier/TensorRT figures are **not measured**.

A dummy-path profiler exists (`cropai profile-pipeline`). Its wall times are **not** model FPS.

| Metric | Value | Status |
| --- | --- | --- |
| Inference FPS (YOLO / EfficientNet) | n/a | NOT MEASURED |
| YOLO latency | n/a | NOT MEASURED |
| Classifier latency | n/a | NOT MEASURED |
| Segmentation latency | n/a | NOT MEASURED |
| Risk latency | dummy-path only | NOT MEASURED as LightGBM |
| Fusion / advisory latency | dummy-path only | NOT a GPU benchmark |
| End-to-end latency (trained) | n/a | NOT MEASURED |
| CPU utilization | n/a | NOT MEASURED |
| GPU utilization | n/a | NOT MEASURED |
| RAM (runtime) | rss sampled on dummy smoke | NOT a production profile |
| VRAM | n/a | NOT MEASURED |
| Model size | n/a | no weights |
| Startup time | n/a | NOT MEASURED |
| Throughput | n/a | NOT MEASURED |
| 5 / 15 / 30 / 60 min | n/a | NOT RUN |

Sandbox hardware (honesty, not a production profile): 2 CPU, 3.8 GiB RAM, no GPU.

Do not quote dummy e2e milliseconds as SIH deployment performance.
