# model_selection.md

No model has been trained or benchmarked. Entries are **candidates**, not selections.

Date: 2026-08-31

## Disease classification

```
Task: disease classification
Candidate: EfficientNetV2-S (PyTorch, ImageNet pretrained)
Selected: NOT SELECTED — Phase 2.5 harness ready; metrics NOT MEASURED
Alternatives: ConvNeXt-Tiny, MobileNetV3-Large, ResNet-50 baseline
Accuracy: NOT MEASURED
Precision: NOT MEASURED
Recall: NOT MEASURED
Macro-F1: NOT MEASURED
Latency: NOT MEASURED
Memory: NOT MEASURED
Model size: NOT MEASURED
Field robustness: NOT MEASURED
Reason: Master-prompt starting candidate only. Must beat a baseline on field validation, not lab accuracy.
Date: 2026-08-31
```

## Object detection

```
Task: plant / leaf / pest detection
Candidate: YOLO11 (Ultralytics)
Selected: NOT SELECTED — Phase 2.5 harness ready; metrics NOT MEASURED
Alternatives: YOLO11s, YOLOv8n, other lightweight detector if measured better
Accuracy/mAP: NOT MEASURED
Precision: NOT MEASURED
Recall: NOT MEASURED
Macro-F1: n/a
Latency: NOT MEASURED
Memory: NOT MEASURED
Model size: NOT MEASURED
Field robustness: NOT MEASURED
Reason: Starting hypothesis. False negatives weighted more than leaderboard mAP.
Date: 2026-08-31
```

## Lesion segmentation

```
Task: symptom / lesion segmentation for affected-area %
Candidate: YOLO11-seg
Selected: NOT SELECTED — pending Phase 2.5 necessity test
Alternatives: skip segmentation and use detector box area; classical thresholding as weak baseline
IoU / Dice: NOT MEASURED
Reason: Segmentation is allowed only if it improves severity usefulness enough to justify cost.
Date: 2026-08-31
```

## Risk forecasting

```
Task: 1/3/7-day disease and pest risk
Candidate: LightGBM
Selected: NOT SELECTED — Phase 3.5 harness ready; 0 real outbreak labels
Alternatives: XGBoost, logistic baseline, heuristic_unvalidated. Transformer not justified.
Accuracy: NOT MEASURED
Reason: Dataset too small for ML forecasting (0 labels). Runtime fallback is heuristic_unvalidated (uncalibrated, not a validated Maharashtra ETL). Do not fabricate ML skill.
Date: 2026-08-31
```

## Geospatial

```
Task: hotspot detection
Candidate: H3 + KDE / DBSCAN + time decay
Selected: NOT SELECTED — pending Phase 4.5
Alternatives: H3 aggregation only. GNN not in scope unless measured benefit.
Date: 2026-08-31
```
