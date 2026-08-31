# techstack.md

Actual technologies in this repository / environment. Not an aspirational shopping list.

Last verified: 2026-08-31

## Agent sandbox (this clone)

| Component | Version / status |
| --- | --- |
| OS | Debian GNU/Linux 12 (bookworm), x86_64 |
| CPU | 2× Intel Xeon @ 2.60 GHz (KVM) |
| RAM | 3.8 GiB |
| GPU | **none** |
| nvidia-smi | not available |
| nvcc | not available |
| Python | 3.11.2 (`/usr/bin/python3`) |
| pip | 23.0.1 |
| Node | 22.22.3 (unused in Phase 1) |
| PyTorch | **not installed** |
| CUDA | **unavailable** |

## Phase 1 code (installed / declared)

| Package | Role | Declared | Installed in sandbox |
| --- | --- | --- | --- |
| PyYAML | configs | `>=6.0.1` | 6.0.3 in `.venv` |
| Pillow | image IO / synthetic PNG | `>=10.0.0` | 12.3.0 in `.venv` |
| pytest | tests | `>=7.4.0` | 9.1.1 in `.venv` |

Stdlib used: `dataclasses`, `hashlib`, `csv`, `json`, `pathlib`, `argparse`, `logging`, `unittest` via pytest.

## Intended training stack (NOT installed here)

To be installed by the user on the RTX 4050 machine. Exact versions must be recorded in `runs/*/meta.json` at train time — they are **not** claimed as currently used.

| Package | Planned role | Status |
| --- | --- | --- |
| PyTorch + torchvision | training | PENDING USER EXECUTION |
| CUDA / cuDNN | GPU | PENDING USER HARDWARE |
| Ultralytics YOLO11 | detect / segment | CODE present; package **not installed** |
| torchvision EfficientNetV2-S | classify | CODE present; package **not installed** |
| LightGBM / XGBoost | risk forecast candidates | CODE present; packages **not installed**; **not selected** |
| heuristic_unvalidated | risk runtime fallback | IMPLEMENTED (uncalibrated) |
| H3 | spatial index | CODE present; package **not installed**; grid fallback used |
| scikit-learn DBSCAN/KDE | geospatial | NOT USED — pure-Python DBSCAN + KDE |
| ONNX / ONNX Runtime / TensorRT | export | NOT IMPLEMENTED |
| FastAPI | API | NOT IMPLEMENTED |

See `requirements-train.txt` and `configs/training.yaml`.

## Storage (Phase 1)

| Store | Status |
| --- | --- |
| CSV manifests | IMPLEMENTED |
| JSONL observations / traps / weather | IMPLEMENTED (synthetic writer + ObservationStore) |
| SQLite | NOT IMPLEMENTED |
| Object store | NOT IMPLEMENTED |

## Frontend

NOT STARTED. Farmer / extension / official UIs are Phase 4–6.
