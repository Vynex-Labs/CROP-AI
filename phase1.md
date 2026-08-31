# phase1.md — Dataset and agricultural domain definition

Date: 2026-08-31  
Status: IMPLEMENTED (tooling + taxonomy) / field images NOT AVAILABLE

## Objectives

MASTER_PROMPT §5 tasks 1–27: parse SIH26131, define crops/diseases/pests/healthy/severity/symptoms, register public and field sources, weather/soil/trap schemas, geospatial + crop-stage representation, custom field protocol, dataset tooling, annotation formats, train/val/isolated test, leakage prevention, provenance, licensing, synthetic vs real, versioning, automated validation.

## Implementation

| Area | Path |
| --- | --- |
| Domain taxonomy | `configs/crop_config.yaml` + `src/cropai/domain/taxonomy.py` |
| Dataset registry | `configs/dataset.yaml` |
| Class maps | `configs/mappings/plantvillage.yaml`, `plantdoc.yaml` |
| Records / CSV | `src/cropai/dataset/schema.py` |
| Adapters | `src/cropai/dataset/adapters/` |
| Synthetic smoke | `src/cropai/dataset/synthetic.py` (always `is_synthetic=true`) |
| Splits / leakage | `split.py`, `leakage.py` |
| Validator | `validate.py` |
| Prepare CLI | `python -m cropai prepare-dataset` |
| YOLO / COCO | `yolo.py`, `coco.py` |
| Field protocol | `data/field_protocol/` |
| IPM stubs | `knowledge/ipm/` (placeholder, no invented doses) |
| Train entrypoints | `train_linux.sh`, `train_windows.bat` |

## Experiments

None involving real images. Lightweight:

- taxonomy integrity
- synthetic generation
- grouped splits without leakage
- validator catches missing files and invalid YOLO boxes
- prepare-dataset on empty `data/raw/` produces synthetic-only manifest and warns

## Results

See pytest output and `data/manifests/` after prepare.

- pytest: 19 passed
- prepare-dataset: 76 synthetic records, 0 real, 0 field, validation_ok=true, 1 warning (no real field images)
- splits: train 57 / val 10 / test 9
- Real field image count: **0**. Public image count in clone: **0**.

## Decisions

```
Decision: Do not narrow. UNION of India's top-10 staples + CROPSAP + tomato/potato/maize/grape
Selected: 17 crops in crop_config.yaml crop_sets.top10_india + horticulture_required + cropsap_maharashtra + retained onion/pepper
Alternatives rejected: tomato-only demo; v1 cut to tomato/potato/maize/grape only
Evidence: user instruction 2026-08-31; MoA&FW area/production ranks; SIH26131 official statement in docs/SIH26131.md
Accuracy/Precision/Recall/F1/Latency/Memory: n/a (not a model decision)
Reliability: taxonomy is internally consistent (unit tested)
Field robustness: not applicable yet
Reason: User forbade narrowing. Tomato/potato/maize/grape stay first-class. Top-10 Indian crops are the national core.
Date: 2026-08-31
```

```
Decision: Do not vendor or fabricate field images
Selected: registry + adapters + protocol; synthetic smoke for CI only
Alternatives: download PlantVillage in sandbox; generate fake “field” JPEGs and call them real
Evidence: MASTER_PROMPT §27; sandbox has ~20 GiB and no GPU
Reason: Honesty and license/disk constraints.
Date: 2026-08-31
```

```
Decision: Severity = affected-area % bins + optional ordinal expert label
Selected: 5 bins in crop_config.yaml
Alternatives: Horsfall-Barratt 0–11; 0–9 ICAR scales only
Reason: Computer-vision measurable proxy; can map to ICAR scales later per crop when expert protocol exists.
Date: 2026-08-31
```

## Failures

- Cannot ingest PlantVillage/PlantDoc/IP102 in this environment (not present).
- CROPSAP dump not available.
- Weather/soil real series not available.
- Several operational crops have **no** public image coverage (tur, jowar, sugarcane, much of cotton/onion).

## Verification (Phase 1.5)

See `AGENT_REVIEW.md` and the gate block in this file.

Quantitative analysis is produced by `cropai.dataset.analysis.analyze_manifest`.

Gate conclusions (not fabricated):

- Dataset **quality of real images**: unknown / absent
- Class balance of real images: n/a
- Disease coverage in taxonomy: defined; in pixels: synthetic subset only
- Pest coverage: taxonomy yes; IP102 not downloaded; no trap counts
- Field-image realism: **no real field images**
- Annotation quality: synthetic boxes/polygons only
- Train/test separation: enforced on synthetic groups; untested on real data
- Licensing: documented per source
- Severity-label quality: unlabeled on public lab sets; synthetic proxy only
- Geographic diversity: none real
- Crop-stage diversity: metadata field exists; no real distribution
- Sufficient for transfer learning **start**: yes, once PlantVillage is downloaded by the user
- Additional field images required: **yes**

## Confidence

Agent Confidence: **72/100** (Phase 1 tooling). See AGENT_REVIEW.md.

Recommendation: **PROCEED** to Phase 2 infrastructure **with the field-data collection workstream in parallel**. Do not interpret this as deployment readiness.

## Next steps

Stop. Ask the user whether to proceed to Phase 2 or investigate an alternative dataset strategy (e.g. fewer crops with stronger public coverage, or hold for partner images).
