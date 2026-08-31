# AGENT_REVIEW.md

## Phase 5 / 5.5 — 2026-08-31

### Attempted work

- Runtime selector (TensorRT → ORT → PyTorch → dummy)
- Bounded queue, cache, warmup, timed e2e session
- Export dry-run, dummy profiler, endurance smoke
- Phase 5.5 honesty harness

### Completed work

- Dummy selected without weights
- Queue rejects when full (no unbounded growth)
- Export plan does not invent ONNX/TRT success
- Dummy profile omits FPS so it cannot be quoted as YOLO throughput
- Endurance 0.6s marks 5/15/30/60 min **NOT RUN**
- INT8 **not adopted**
- pytest **66 passed**

### Failed work

- Actual ONNX export (no torch, no checkpoints)
- TensorRT FP16 engine
- GPU util / VRAM / thermal
- 5+ minute endurance

### Benchmark results

Dummy-path smoke only. **Not** a deployment benchmark.

```
Measured performance: dummy smoke; YOLO/TRT/ORT NOT MEASURED
Bottleneck: UNKNOWN
Optimization applied: none
Remaining issues: no checkpoints, no CUDA, no TensorRT, INT8 not adopted, 1-hour test not run
```

### Agent Confidence: 70/100 (code) / 0/100 (deployed performance)

Why this confidence:

- Fallback order and INT8 policy match §9.
- Dummy timings cannot be mistaken for model FPS in the report object.
- Long-run keys stay NOT RUN unless actually run.

Remaining uncertainty:

- TensorRT builder API untested.
- ORT IO binding untested.
- Dummy e2e includes quality-reject on tiny PNGs.

Better alternatives considered:

- Quote dummy 200 ms as “real-time” — rejected (§27).
- Adopt INT8 from literature — rejected.
- Optimize dummy queue — rejected (§26).

Recommendation: **PROCEED** to Phase 6 final-validation *harness* (honest NOT VERIFIED). Do not claim SIH success.

---

## Phase 4 / 4.5 — 2026-08-31

### Attempted work

- H3 indexing (with grid fallback), DBSCAN, KDE, time decay, hotspot states
- Weighted farm-level fusion
- Structured IPM advisory + expert/lab referral
- Qualitative Phase 4.5 scenario tests

### Completed work

- Isolated / nearby-cluster / distant / old-vs-recent / low-confidence tests
- Missing weather/trap fusion continues with reduced confidence
- Conflicting fusion channels → referral
- Placeholder IPM: `chemical_control_allowed=false`, empty chemical list
- Low confidence suppresses disease name
- Hindi/Marathi **labels** only
- pytest **60 passed**

### Failed work

- Uber H3 package not installed
- Labeled hotspot precision/recall
- Dashboard UI
- Verified IPM packages (still placeholders)

### Benchmark results

NONE on real hotspots. Synthetic clusters are not surveillance skill.

### Model decisions

```
Selected spatial model: NOT SELECTED
Runtime: grid_or_h3 + DBSCAN + KDE + decay (unvalidated)
GNN: not introduced
Fusion: weighted_deterministic (uncalibrated, unfitted)
Why: 0 labeled hotspots. MASTER_PROMPT §8.
```

Answers required by §8 Phase 4.5:

- Is H3+KDE/DBSCAN sufficient? **UNKNOWN**
- Expert-referral correctness (rules): tested on stubs, not field cases

### Agent Confidence: 72/100 (code) / 0/100 (hotspot skill)

Why this confidence:

- Qualitative spatial tests match the required scenario list.
- Advisory cannot invent dosage.
- Grid fallback is labeled, not faked as H3.

Remaining uncertainty:

- Grid cells ≠ H3 hexes.
- No Maharashtra coordinate truth set.
- Fusion weights are guesses in config.

Better alternatives considered:

- Require h3 install in requirements.txt — skipped so CI stays tiny; documented fallback.
- GNN — rejected (§8).
- Emit chemical lists from NIPHM PDFs — rejected (not attached/verified).

Recommendation: **PROCEED** to Phase 5 deployment *infrastructure*. Do not freeze spatial methods.

---

## Phase 3 / 3.5 — 2026-08-31

### Attempted work

- Risk schema, temporal features, missing-data policy, heuristic fallback
- LightGBM/XGBoost adapters, identity calibration, train/eval/benchmark
- CLI `forecast-risk` / `benchmark-risk` / `evaluate-risk`

### Completed work

- 1/3/7-day disease and pest score fields
- Continue-on-missing weather/trap/soil with reduced confidence
- Insufficient-input expert referral
- Train refuses ML fit when real outbreak labels < 200
- pytest **44 passed** (includes risk)

### Failed work

- LightGBM / XGBoost training (packages absent; 0 real labels)
- Calibration evaluation
- Ingest of NASA POWER / Open-Meteo / CROPSAP / SoilGrids (not on disk)

### Benchmark results

NONE on real outbreaks. Synthetic heuristic scores are **not** forecast skill.

### Model decisions

```
Selected risk model: NOT SELECTED
Runtime fallback: heuristic_unvalidated (uncalibrated, unvalidated)
Why: 0 real outbreak labels. MASTER_PROMPT §7 forbids fabricating ML skill.
Rejected alternatives: none rejected by evidence (none measured)
Data limitations: no farm weather, no trap time series, no outbreak labels
```

Answers required by §7 Phase 3.5:

- Is LightGBM best? **UNKNOWN**
- Does data support ML forecasting? **NO**

### Agent Confidence: 70/100 (code) / 0/100 (forecast skill)

Why this confidence:

- Pipeline matches the Phase 3 contract without inventing ROC-AUC.
- Missing-data behaviour is tested.
- Heuristic is labeled unvalidated.

Remaining uncertainty:

- Heuristic weights are generic, not Maharashtra ETL.
- LightGBM API untested here.
- No lead-time evaluation possible without dated outbreaks.

Better alternatives considered:

- Constant 0.5 climatology only — weaker missing-data tests; kept as inner fallback when no features exist.
- Fake literature ROC-AUC — rejected (§27).
- Hold all of Phase 3 until CROPSAP arrives — rejected; user said Continue.

Recommendation: **PROCEED** to Phase 4 geospatial / fusion / advisory *infrastructure*. Do not freeze LightGBM.

---

## Phase 2 / 2.5 — 2026-08-31

### Attempted work

- Perception pipeline, dummy + torch/ultralytics adapters, train/eval/benchmark, infer CLI
- Lightweight tests without PyTorch

### Completed work

- Required vision output fields
- Invalid/corrupt image rejection
- Untrained dummy → expert referral (no fake diagnosis)
- Train dry-run; CUDA refuse for heavy jobs
- pytest **32 passed**

### Failed work

- Actual YOLO11 / EfficientNetV2-S / YOLO11-seg training (no torch, no CUDA, no public images)
- Measured accuracy / mAP / IoU / latency

### Benchmark results

NONE. Dummy infer is not a benchmark.

### Model decisions

**None selected.**

```
Selected models: none
Rejected alternatives: none rejected by evidence (none measured)
Evidence: harness only
Deployment tradeoffs: not measured
```

Answers required by §6 Phase 2.5:

- Is EfficientNetV2-S best? **UNKNOWN**
- Is YOLO11 best? **UNKNOWN**
- Is segmentation necessary? **UNKNOWN**

### Agent Confidence: 68/100 (code) / 0/100 (model skill)

Why this confidence:

- Pipeline matches the Phase 2 contract without inventing scores.
- Safety default is referral when untrained.
- Tests cover reject / dummy / metrics / dry-run.

Remaining uncertainty:

- No weights, no GPU, no field images.
- Ultralytics API / YOLO11 checkpoint names untested here.
- Classifier head-replace logic untested without torchvision.

Better alternatives considered:

- Wait to write vision code until the user trains — rejected; user said Continue.
- Fake 90%+ accuracy from literature — rejected (§27).

Recommendation: **PROCEED** to Phase 3 risk *infrastructure* in parallel with user RTX 4050 training. Do not freeze vision models.

---

## Phase 1 / 1.5 revision — 2026-08-31 (crop list)

User instruction: do **not** narrow; add tomato/potato/maize/grape; use top 10 Indian crops.

### Attempted work

- Stored official SIH26131 problem statement (`docs/SIH26131.md`)
- Replaced “narrow v1” option with a UNION crop policy
- Added groundnut, mustard (rapeseed), pearl millet (bajra)
- Promoted potato to operational + horticulture_required
- Kept onion, pepper, tur, jowar

### Completed work

- `crop_sets.top10_india` (10) + `horticulture_required` (tomato, potato, maize, grape) + CROPSAP + retained
- 17 crops, 74 disease/healthy classes, 28 pests, taxonomy integrity OK
- pytest **22 passed**

### Rejected alternatives

- Narrowing v1 to tomato/potato/maize/grape only — **rejected by user**

---

## Phase 1 / 1.5 — 2026-08-31 (original)

### Attempted work

- Repository, hardware, software, dataset inspection
- Persist master prompt
- Domain definition for SIH26131 (Maharashtra / CROPSAP-aligned)
- Dataset package, validation gate, training scripts, documentation
- Lightweight tests (no model training)
- `pytest`: 19 passed
- `prepare-dataset`: 76 synthetic records, 0 real, 0 field, validation_ok

### Completed work

- `MASTER_PROMPT.md` saved
- Taxonomy of 14 crops + diseases + pests + severity + stages
- Adapters registered for PlantVillage, PlantDoc, IP102, field CSV
- Synthetic smoke generator with difficulty tags, YOLO/COCO, observations/traps/weather
- Grouped train/val/test splits with leakage tests
- DatasetValidator covering the required minimum checks
- Field collection protocol
- IPM placeholders without dosages
- `train_linux.sh`, `train_windows.bat`
- Required living docs

### Failed work

- Ingesting real public datasets: **files not present** (not a code failure)
- Obtaining CROPSAP / IMD / Soil Health Card: **not available**
- GPU detection: CUDA unavailable in sandbox (reported, not hidden)

### Benchmark results

NONE. No model trained. No accuracy/FPS claims.

### Model decisions

NONE selected. Candidates recorded in `model_selection.md`.

### Rejected alternatives

- Tomato-only PlantVillage classifier demo — rejected as insufficient for SIH26131
- Fabricating Maharashtra field JPEGs — rejected (§27)
- Downloading tens of GB into the sandbox — rejected (disk, license, MASTER_PROMPT lightweight-execution rule)
- Immediate YOLO training from scratch — rejected (transfer learning + phase boundary)

### Unresolved problems

1. **Zero real images in the clone.** Transfer learning cannot actually run until the user downloads public sets.
2. **Zero Maharashtra field images.** Deployment metric is undefined.
3. Cotton, sugarcane, pigeon pea, sorghum, onion public coverage is weak or absent.
4. No pest-trap time series, no plot weather, no soil samples.
5. PlantDoc/IP102 licenses need a human check before redistribution.
6. IPM packages not attached — chemical advice must stay off.
7. RTX 4050 not visible in this environment; CUDA path untested here.

### Dataset verification gate (Phase 1.5)

| Question | Finding |
| --- | --- |
| Dataset quality | Real: unknown (absent). Synthetic: only for CI. |
| Class balance | Synthetic roughly even; real n/a |
| Disease coverage | Taxonomy defined; pixel coverage = synthetic subset |
| Pest coverage | Taxonomy defined; no IP102 files; no traps |
| Field-image realism | **No real field images** |
| Annotation quality | Synthetic boxes/polygons only |
| Train/test separation | Enforced on groups; tested on synthetic |
| Licensing | Documented; some “verify before redistribute” |
| Severity-label quality | Not present on PlantVillage; synthetic proxy only |
| Geographic diversity | None real |
| Crop-stage diversity | Field exists; no real distribution |
| Sufficient for transfer learning start? | **Yes, after user downloads PlantVillage (and ideally PlantDoc)** |
| Additional field images required? | **Yes, blocking for deployment / Phase 6** |
| public + field sufficient for Maharashtra? | **No** |

`analyze_manifest` sets:

- `public_plus_field_sufficient_for_maharashtra_deployment: False`
- `additional_field_images_required: True`
- `fabricated_field_data: False`

### Agent Confidence: 72/100

Why this confidence:

- Domain definition is explicit, data-driven, and internally consistent (tested).
- Tooling implements the Phase 1 checklist instead of a slide-only plan.
- Synthetic vs real is enforced in schema, validator, analysis, and files (`SYNTHETIC.txt`).
- Hardware/CUDA absence is reported rather than assumed.

Remaining uncertainty:

- Public dataset on-disk layout may differ from adapter assumptions (not integration-tested).
- PlantDoc/IP102 license and label noise.
- Whether partners can actually collect the field protocol volumes.
- SIH26131 full official PDF body was only partially retrieved (title, org = Government of Maharashtra, software PS confirmed).

Better alternatives considered:

- Narrow v1 to tomato+potato+maize+grape (strong PlantVillage) and treat other CROPSAP crops as risk/trap-only until images exist.
- Hold Phase 2 until a KVK/CROPSAP image dump is in hand.
- Add a rice-specific public set (Mendeley/Kaggle) as a first extra adapter — not done yet pending license review.

Recommendation: **PROCEED** to Phase 2 (perception *infrastructure* + user-side public-data training), while treating Maharashtra field collection as a parallel blocking workstream for any deployment claim.

Do **not** silently start Phase 2 until the user answers.

---

## Later phases

Phase 6 review: not written (phase not started).
