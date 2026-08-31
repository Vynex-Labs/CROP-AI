# phase4.md — Geospatial intelligence and decision system

Date: 2026-08-31  
Status: **IMPLEMENTED (code). Hotspot/fusion scores UNVALIDATED. No GNN. Chemicals blocked.**

## Objectives

MASTER_PROMPT §8: H3 indexing, hotspot states (isolated / emerging / established / declining), time decay, farm-level fusion, structured IPM advisory with expert referral. Never invent pesticide dosage.

## Implementation

| Piece | Path | Status |
| --- | --- | --- |
| Geo config | `configs/geo.yaml` | [T] |
| H3 or grid fallback | `src/cropai/geo/h3index.py` | [T] |
| Time decay | `decay.py` | [T] |
| Haversine DBSCAN | `dbscan.py` | [T] |
| Gaussian KDE | `kde.py` | [T] |
| Hotspot detector | `hotspots.py` | [T] qualitative |
| Weighted fusion | `src/cropai/fusion/` | [T] |
| Advisory engine | `src/cropai/advisory/` | [T] |
| i18n labels only | `advisory/i18n.py` | [T] |
| CLI | `hotspots` `fuse-risk` `advise` `benchmark-geo` | [T] |
| Phase 4.5 harness | `geo/benchmark.py` | [T] |

`h3` Python package is **not installed** here. Indexes are `gridN:i:j` until the user installs `h3`. They are **not** Uber H3 cells.

IPM YAML remains `status: placeholder` → monitoring + referral, **empty chemical_control**.

## Experiments

- pytest `tests/geo/` `tests/advisory/` — single/nearby/distant/old/recent, missing weather/trap fusion, conflict referral, placeholder chemicals blocked, low-confidence diagnosis suppressed, hi/mr labels
- `cropai advise --crop rice --disease rice_blast --confidence 0.9 --lang hi`
- `cropai fuse-risk --crop rice --weather-risk 0.7 --vision-untrained --synthetic`
- `cropai benchmark-geo`

No labeled Maharashtra hotspots. No dashboard UI.

## Results

| Metric | Value | Status |
| --- | --- | --- |
| Hotspot precision / recall | n/a | NOT MEASURED |
| False / missed hotspot rate | n/a | NOT MEASURED |
| Spatial / temporal stability | n/a | NOT MEASURED |
| Advisory consistency | placeholder → no chemicals, referral | TESTED |
| Expert-referral correctness | low conf / severe / missing IPM | TESTED (rules) |
| Labeled hotspots in repo | **0** | honest |

## Decisions

```
Decision: Do not introduce a GNN
Selected: NOT SELECTED (runtime: grid_or_h3 + DBSCAN + KDE + decay)
Alternatives: H3 aggregation only; GNN
Evidence: 0 labeled hotspots; MASTER_PROMPT §8
Date: 2026-08-31
```

```
Decision: Farm fusion is a weighted deterministic model, not LightGBM
Selected: NOT SELECTED (runtime fallback: weighted_deterministic, uncalibrated)
Evidence: no fitted fusion labels
Date: 2026-08-31
```

```
Decision: Placeholder IPM never emits chemical control
Selected: chemical_control_allowed=false until status=verified
Evidence: knowledge/ipm/*.yaml status placeholder; §28
Date: 2026-08-31
```

## Failures

- Uber H3 library absent → grid fallback
- No real observation coordinates / outbreak maps
- Hindi/Marathi labels only; IPM body stays English until verified translations exist
- Official dashboard UI not built (Phase 5–6)

## Verification (Phase 4.5)

See AGENT_REVIEW.md.

- Is H3+KDE/DBSCAN sufficient? **UNKNOWN**
- GNN introduced? **No**

## Confidence

Agent Confidence: **72/100** (code). Hotspot-skill confidence: **0/100**.

Recommendation: **PROCEED** to Phase 5 deployment *infrastructure* (ONNX path, runtime fallbacks) in parallel with field observations. Do not treat hotspot scores as surveillance truth.

## Next steps

Stop. Ask the user whether to proceed to Phase 5.
