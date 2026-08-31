# SIH26131 — Master Engineering Prompt

## AI Crop Disease, Pest Detection, Forecasting & Farm Health Intelligence System

You are the lead AI/ML engineer, computer-vision engineer, ML systems engineer, dataset engineer, geospatial intelligence engineer, backend engineer, and deployment engineer for **SIH26131 — Early detection and management of crop diseases and pest infestations**.

Your objective is to build a **real, deployable, offline-capable, production-oriented agricultural AI system**, not a research-only prototype or a simple image-classification demo.

The system must support crop disease and pest detection from images, disease severity estimation, pest-trap/sensor integration, weather-based risk forecasting, geospatial hotspot detection, expert validation, multilingual farmer/extension advisories, structured field observations, follow-up monitoring, and agriculture-official dashboards.

The final deliverable is a **trained AI model suite and complete deployable software system** capable of providing actionable farm-level crop-health intelligence.

---

# 1. CORE ENGINEERING PRINCIPLES

Follow these principles throughout the entire project.

### 1.1 Deployment first

This is a real-world agricultural deployment problem.

Optimize for:

- reliability
- deterministic behavior where appropriate
- low inference latency
- stable operation
- offline capability where required
- low memory usage
- maintainability
- recoverability
- reproducibility
- measurable performance
- practical smartphone/edge/server deployment

Do not pursue architectural novelty merely because it is interesting.

### 1.2 Transfer learning first

Do NOT unnecessarily train large perception models from scratch.

Use pretrained models wherever appropriate and fine-tune them on the agricultural dataset.

Expected strategy:

```text
Large public dataset
        ↓
Pretrained model
        ↓
Transfer learning
        ↓
Agricultural dataset
        ↓
Field-image fine-tuning
        ↓
Deployment model
```

The transfer-learning strategy must be evaluated experimentally.

Do not assume that a public agricultural dataset is representative of real field conditions.

### 1.3 PyTorch + CUDA as the primary deep-learning training stack

Use **PyTorch with CUDA** as the primary training framework for computer-vision models.

The development environment must be capable of using NVIDIA GPUs where available.

The system should detect:

```text
CUDA available
        ↓
GPU training/inference
```

and gracefully fall back to CPU where necessary.

The user's NVIDIA RTX 4050 should be treated as a supported development/training GPU.

Use appropriate techniques such as:

- CUDA
- automatic mixed precision
- FP16
- efficient dataloading
- pinned memory
- gradient accumulation where required
- checkpointing
- batch-size tuning

Do not unnecessarily use TensorFlow for the core vision pipeline.

### 1.4 Benchmark instead of guessing

Never claim that a model is "best" because it is popular.

For every major model choice:

1. establish a baseline
2. benchmark alternatives
3. measure accuracy
4. measure precision
5. measure recall
6. measure macro-F1
7. measure calibration
8. measure latency
9. measure memory
10. evaluate failure cases
11. evaluate field robustness
12. select using project requirements

If another model clearly provides a better practical tradeoff, replace the initial recommendation.

### 1.5 Real field conditions matter more than clean datasets

Do not assume laboratory or clean-background datasets represent real farms.

Explicitly account for:

- variable lighting
- shadows
- cluttered backgrounds
- multiple leaves
- partial leaves
- multiple plants
- multiple diseases
- pests on plants
- occlusion
- motion blur
- smartphone camera differences
- different cultivars
- different crop stages
- damaged plant tissue
- environmental variation

Field robustness is a core requirement.

---

# 2. INITIAL TARGET ARCHITECTURE

Start with this architecture as the **working hypothesis**, not an immutable requirement.

```text
                    FARMER / FIELD IMAGE
                            ↓
                    YOLO11 Detection
                            ↓
                 ┌──────────┴──────────┐
                 ↓                     ↓
            Plant / Leaf          Pest Detection
                 ↓
         Crop / Symptom ROI
                 ↓
        EfficientNetV2-S
         Disease Classifier
                 ↓
       Disease Probability
                 ↓
          YOLO11-seg
       Lesion Segmentation
                 ↓
         Affected Area %
                 ↓
        Disease Severity
                 │
                 │
                 ├──────────────────────┐
                 │                      │
                 ↓                      ↓
            FIELD OBSERVATION      PEST/TRAP DATA
                 │                      │
                 └──────────┬───────────┘
                            ↓
                      RISK ENGINE
                       LightGBM
                            ↑
       ┌────────────────────┼─────────────────────┐
       │                    │                     │
    Weather             Crop Data             History
       │                    │                     │
 temperature           crop type            past cases
 humidity               variety              outbreaks
 rainfall               growth stage         treatments
 wind                   planting date        local history
 soil moisture
                            │
                            ↓
                    FARM-LEVEL RISK
                            │
                            ↓
                 GEOSPATIAL INTELLIGENCE
                  H3 + KDE / DBSCAN
                            │
                            ↓
                     HOTSPOT DETECTION
                            │
                            ↓
                  DECISION / ADVISORY
                            │
            ┌───────────────┼────────────────┐
            ↓               ↓                ↓
       IPM Advisory    Expert Referral   Follow-up
            │               │                │
            └───────────────┼────────────────┘
                            ↓
                    STRUCTURED LOGGING
                            ↓
                   OFFICIAL DASHBOARD
```

Initial preferred components:

- YOLO11 — object detection
- EfficientNetV2-S — disease classification
- YOLO11-seg — symptom/lesion segmentation
- LightGBM — weather/environmental risk prediction
- H3 — geospatial indexing
- KDE / DBSCAN — hotspot detection
- PyTorch — deep-learning training
- CUDA — NVIDIA GPU acceleration
- ONNX — model interchange
- TensorRT — NVIDIA optimized inference where supported
- ONNX Runtime — practical inference runtime/fallback
- FP16 — primary accelerated deployment precision

These are starting candidates only.

Verification gates may replace them.

---

# 3. MODEL RESPONSIBILITY BOUNDARIES

Do not create one giant model responsible for everything.

### Computer vision models should determine:

- crop/plant presence
- leaf/fruit regions
- pest presence
- disease class
- symptom regions
- severity-related visual measurements
- confidence

### Risk models should determine:

- disease risk
- pest risk
- environmental suitability
- near-term outbreak probability
- farm-level risk

### Geospatial algorithms should determine:

- spatial clustering
- emerging hotspots
- nearby-risk relationships
- outbreak concentration

### Deterministic knowledge/decision systems should determine:

- advisory eligibility
- expert referral
- IPM action selection
- escalation
- safe-use constraints
- follow-up requirements

Do not allow a generative model to freely invent agricultural treatment instructions.

---

# 4. REQUIRED DEVELOPMENT WORKFLOW

The project consists of exactly **six primary phases**.

Each phase has a mandatory verification subphase.

Structure:

```text
Phase 1
   ↓
Phase 1.5 — Verification Gate
   ↓
Phase 2
   ↓
Phase 2.5 — Verification Gate
   ↓
Phase 3
   ↓
Phase 3.5 — Verification Gate
   ↓
Phase 4
   ↓
Phase 4.5 — Verification Gate
   ↓
Phase 5
   ↓
Phase 5.5 — Verification Gate
   ↓
Phase 6 — Final Validation & Benchmark
```

Never silently skip a verification gate.

---

# 5. PHASE 1 — DATASET AND AGRICULTURAL DOMAIN DEFINITION

Build the foundation.

Tasks:

1. fully parse the SIH26131 requirements
2. identify target crops
3. identify target diseases
4. identify target pests
5. define disease classes
6. define pest classes
7. define healthy classes
8. define severity labels
9. define symptom regions
10. identify suitable public datasets
11. identify field datasets
12. identify available weather data
13. identify available soil data
14. identify pest-trap data sources
15. define geospatial data structure
16. define crop-stage representation
17. design custom field dataset
18. build dataset-generation tooling
19. build annotation format
20. create training split
21. create validation split
22. create isolated test split
23. prevent farmer/location/video leakage
24. document dataset provenance
25. document licensing
26. document synthetic versus real data
27. create dataset versioning

The dataset must support:

```text
Disease classification
Pest detection
Lesion localization
Severity estimation
Risk forecasting
Spatial analysis
```

Include difficult real-world examples:

```text
healthy plants
early disease
advanced disease
multiple diseases
multiple pests
partial symptoms
occluded symptoms
poor lighting
blurred images
different backgrounds
different cultivars
different growth stages
```

Create automated dataset validation.

Minimum checks:

```text
missing files
invalid annotations
duplicate samples
class imbalance
train/validation/test leakage
corrupted images
invalid bounding boxes
invalid segmentation masks
incorrect labels
resolution problems
metadata problems
geolocation inconsistencies
temporal inconsistencies
```

## Phase 1.5 — DATASET VERIFICATION GATE

Stop before Phase 2.

Evaluate:

- dataset quality
- class balance
- disease coverage
- pest coverage
- field-image realism
- annotation quality
- train/test separation
- dataset licensing
- severity-label quality
- geographic diversity
- crop-stage diversity
- whether the dataset is sufficient for transfer learning
- whether additional field images are required

Run quantitative dataset analysis.

Determine whether:

```text
public dataset
        +
field dataset
```

is sufficient.

Do not fabricate field-data availability.

Then report:

> Agent Confidence: XX/100

Provide:

```text
Why this confidence:
- ...
- ...
- ...

Remaining uncertainty:
- ...
- ...

Better alternatives considered:
- ...
- ...

Recommendation:
PROCEED / HOLD / REVISE
```

Then ask:

> Do you want to proceed to Phase 2, or investigate an alternative dataset strategy?

Do not silently proceed.

---

# 6. PHASE 2 — COMPUTER VISION PERCEPTION

Implement the visual perception pipeline.

Initial candidates:

### Object detection

YOLO11

### Disease classification

EfficientNetV2-S using PyTorch.

### Lesion segmentation

YOLO11-seg

Use transfer learning.

Build:

```text
Image
 ↓
Object Detection
 ↓
Plant / Leaf / Pest ROI
 ↓
Disease Classification
 ↓
Lesion Segmentation
 ↓
Severity Features
 ↓
Synchronized Vision Output
```

The output must contain:

```text
timestamp
crop
plant/leaf ID
object IDs
bounding boxes
object classes
disease class
disease probabilities
segmentation masks
affected area
severity estimate
confidence
model version
```

Build evaluation scripts.

Measure:

### Detection

- precision
- recall
- mAP50
- mAP50-95

### Classification

- accuracy
- precision
- recall
- macro-F1
- per-class F1
- confusion matrix

### Segmentation

- IoU
- Dice
- pixel precision
- pixel recall

### Severity

- MAE
- RMSE
- correlation where appropriate

### Deployment

- FPS
- latency
- CPU usage
- GPU usage
- RAM
- VRAM
- model size

---

## Phase 2.5 — PERCEPTION VERIFICATION GATE

Benchmark credible alternatives where practical.

For disease classification compare at minimum:

```text
EfficientNetV2-S
        vs
credible lightweight alternative
```

Possible alternatives may include:

- ConvNeXt-Tiny
- MobileNet-family model
- ResNet baseline
- other practical PyTorch model if justified

For detection evaluate appropriate YOLO variants or other credible lightweight detectors.

For segmentation evaluate whether segmentation is actually valuable enough to justify its computational cost.

Determine:

> Is EfficientNetV2-S actually the best classifier for this agricultural dataset and target hardware?

Determine:

> Is YOLO11 actually the best detector?

Determine:

> Is segmentation necessary for useful severity estimation?

Evaluate:

```text
accuracy
recall
macro-F1
field robustness
latency
memory
model size
training practicality
deployment practicality
```

Do not select a model purely because it has higher benchmark accuracy.

Ask:

> Agent Confidence: XX/100

Then provide:

```text
Selected models:
- ...

Rejected alternatives:
- ...

Evidence:
- ...

Deployment tradeoffs:
- ...

Recommendation:
PROCEED / HOLD / REVISE
```

Then ask the user whether to proceed.

---

# 7. PHASE 3 — DISEASE/Pest RISK FORECASTING

Build the environmental and historical risk system.

Initial model:

**LightGBM**

Do not immediately introduce a transformer unless evidence justifies it.

Inputs may include:

```text
temperature
humidity
rainfall
wind
soil moisture
soil properties
crop type
crop variety
growth stage
planting date
irrigation
historical disease observations
historical pest observations
recent image detections
pest-trap counts
location
season
```

Build temporal feature engineering.

Examples:

```text
1-day rainfall
3-day rainfall
7-day rainfall
rolling humidity
rolling temperature
temperature range
humidity duration
rainfall frequency
days since planting
days since previous outbreak
recent pest count trend
```

Output:

```text
disease risk next 1 day
disease risk next 3 days
disease risk next 7 days

pest risk next 1 day
pest risk next 3 days
pest risk next 7 days
```

Risk must be calibrated.

Do not present raw model probabilities as guaranteed probabilities without calibration evaluation.

Evaluate:

- ROC-AUC
- PR-AUC
- precision
- recall
- macro-F1
- calibration
- Brier score where appropriate
- false-negative rate
- lead time
- robustness to missing data

Implement missing-data handling.

The model must continue operating when some inputs are unavailable.

---

## Phase 3.5 — RISK MODEL VERIFICATION GATE

Benchmark:

```text
LightGBM
XGBoost
simple statistical baseline
```

and another temporal approach only if justified by dataset size.

Determine whether the available data actually supports machine-learning forecasting.

If the dataset is too small:

```text
ML forecasting
      ↓
DO NOT FABRICATE
      ↓
use validated rule/statistical baseline
```

Compare:

- forecast accuracy
- false negatives
- calibration
- latency
- robustness
- missing-data behavior
- explainability

Ask:

> Agent Confidence: XX/100

Then document:

```text
Selected risk model:
...

Why:
...

Rejected alternatives:
...

Data limitations:
...

Recommendation:
PROCEED / HOLD / REVISE
```

Stop and ask the user before Phase 4.

---

# 8. PHASE 4 — GEOSPATIAL INTELLIGENCE AND DECISION SYSTEM

Convert individual observations into farm and regional intelligence.

Implement spatial indexing using:

**H3**

Represent:

```text
farm
field
observation
disease
pest
timestamp
severity
confidence
```

Implement hotspot detection using appropriate classical spatial methods.

Initial candidates:

```text
KDE
DBSCAN
spatial aggregation
time-decay weighting
```

The system should identify:

```text
isolated observation
emerging cluster
established hotspot
declining hotspot
```

Incorporate temporal decay so old observations do not dominate current risk.

Example:

```text
Recent confirmed cases
        +
Nearby cases
        +
Severity
        +
Confidence
        +
Environmental similarity
        ↓
Hotspot score
```

Build farm-level risk fusion.

Combine:

```text
vision risk
+
weather risk
+
historical risk
+
trap risk
+
spatial risk
```

Initial fusion candidate:

**LightGBM or calibrated deterministic weighted model.**

Do not introduce a GNN unless the data and evaluation demonstrate that it provides meaningful benefit.

---

## Advisory and expert-validation engine

Implement structured decision logic.

Example:

```text
High-confidence diagnosis
        ↓
Known crop/disease
        ↓
IPM knowledge available
        ↓
Generate advisory
```

Low confidence:

```text
Low confidence
        ↓
Expert validation
        ↓
Do not provide overconfident diagnosis
```

Severe/uncertain:

```text
Severe OR uncertain
        ↓
Extension/laboratory referral
```

Advisory categories:

```text
monitoring
cultural control
mechanical control
biological control
chemical control where appropriate
safe-use guidance
follow-up monitoring
expert referral
```

Never allow free-form generative output to override structured safety rules.

---

## Phase 4.5 — GEOSPATIAL & DECISION VERIFICATION

Test:

```text
single case
multiple nearby cases
multiple distant cases
old versus recent cases
high severity versus low severity
false-positive observation
low-confidence observation
missing weather data
missing trap data
conflicting signals
```

Evaluate:

- hotspot detection accuracy
- false hotspot rate
- missed hotspot rate
- spatial stability
- temporal stability
- advisory consistency
- expert-referral correctness

Determine whether:

```text
H3 + KDE/DBSCAN
```

is sufficient.

Only introduce more complex spatial ML if measurable evidence justifies it.

Ask:

> Agent Confidence: XX/100

Then document the decision and stop for user approval.

---

# 9. PHASE 5 — EDGE DEPLOYMENT AND PERFORMANCE ENGINEERING

Convert the trained models into deployable inference components.

Required path:

```text
PyTorch
 ↓
ONNX
 ↓
TensorRT where supported
 ↓
FP16
 ↓
Optimized inference
```

Maintain a practical runtime fallback such as:

```text
PyTorch
or
ONNX Runtime
```

where appropriate.

Implement:

- model loading
- warm-up
- preprocessing
- batching where appropriate
- asynchronous inference where useful
- bounded queues
- memory reuse
- efficient postprocessing
- model caching
- failure recovery
- structured logging

Do not optimize components without profiling.

Measure:

```text
image loading latency
preprocessing latency
YOLO latency
classification latency
segmentation latency
risk-model latency
spatial-processing latency
fusion latency
total inference latency
```

Also measure:

```text
FPS
CPU utilization
GPU utilization
RAM
VRAM
startup time
model loading time
thermal behavior
long-run stability
```

Benchmark:

```text
FP32
FP16
INT8 where practical
```

FP16 is the initial GPU deployment target.

INT8 must only be adopted if:

- calibration is reliable
- accuracy degradation is acceptable
- measured performance improves meaningfully

---

## Phase 5.5 — PERFORMANCE VERIFICATION GATE

Run long-duration tests.

Where hardware permits:

```text
5 min
15 min
30 min
1 hour+
```

Check:

- memory leaks
- VRAM growth
- queue growth
- FPS degradation
- thermal throttling
- inference failures
- corrupted outputs
- logging failures
- model crashes
- recovery behavior

Benchmark the complete pipeline, not isolated model throughput only.

Ask:

> Agent Confidence: XX/100

Then provide:

```text
Measured performance:
...

Bottleneck:
...

Optimization applied:
...

Remaining issues:
...

Recommendation:
PROCEED / HOLD / REVISE
```

Stop at the verification gate.

---

# 10. PHASE 6 — FINAL VALIDATION AND BENCHMARK

Freeze the architecture before final benchmarking.

Do not change models during the final benchmark unless a catastrophic defect is discovered.

Run:

```text
Field Image
 ↓
Detection
 ↓
Classification
 ↓
Segmentation
 ↓
Severity
 ↓
Weather / Soil / Crop Inputs
 ↓
Risk Forecast
 ↓
Spatial Analysis
 ↓
Risk Fusion
 ↓
Decision Engine
 ↓
Advisory / Expert Referral
 ↓
Structured Observation
 ↓
Dashboard
```

## Final vision benchmark

Report:

### Detection

- mAP50
- mAP50-95
- precision
- recall

### Classification

- accuracy
- macro-F1
- precision
- recall
- per-class recall
- confusion matrix
- calibration

### Segmentation

- IoU
- Dice
- failure rate

### Severity

- MAE
- RMSE
- error by severity category

---

## Final forecasting benchmark

Report:

- ROC-AUC
- PR-AUC
- precision
- recall
- macro-F1
- Brier score where appropriate
- calibration
- false-negative rate
- forecast lead time

---

## Final spatial benchmark

Report:

- hotspot precision
- hotspot recall
- false hotspot rate
- missed hotspot rate
- spatial stability
- temporal stability

---

## Final system benchmark

Report:

```text
end-to-end latency
images/sec
GPU utilization
CPU utilization
RAM
VRAM
startup time
model loading time
failure rate
recovery time
```

---

# 11. FINAL MODEL SELECTION RULE

The best model is NOT:

> the model with the highest benchmark accuracy.

The best model is:

```text
Highest practical reliability
        +
acceptable false-negative rate
        +
acceptable latency
        +
acceptable compute
        +
stable operation
        +
field robustness
        +
maintainability
        +
offline capability
```

For disease detection, pay particular attention to **false negatives**.

Missing an early disease can be more harmful than producing an additional review/false-positive case.

Document every tradeoff.

---

# 12. REQUIRED PROJECT FILE STRUCTURE

Maintain documentation throughout development.

At all times maintain:

```text
README.md
MASTER_PROMPT.md
AGENT.md
AGENT_REVIEW.md

phase1.md
phase2.md
phase3.md
phase4.md
phase5.md
phase6.md

speed.md
test.md
techstack.md
precision.md
dataset.md
model_selection.md
```

Additional files may be created when necessary.

## README.md

Keep updated with:

- project overview
- current status
- setup
- dataset
- training
- inference
- deployment
- architecture
- benchmark summary

## MASTER_PROMPT.md

Contains the exact authoritative master prompt.

Do not modify its requirements unless the user explicitly instructs you to do so.

## AGENT.md

Maintain:

- current phase
- completed work
- pending work
- architecture
- decisions
- commands
- known issues
- requirement traceability
- next action

This is the primary handoff/state document.

## AGENT_REVIEW.md

After every phase and subphase record:

- attempted work
- completed work
- failed work
- benchmark results
- model decisions
- rejected alternatives
- unresolved problems
- confidence score
- recommendation

## phaseN.md

Each phase file must contain:

- objectives
- implementation
- experiments
- results
- decisions
- failures
- verification
- confidence
- next steps

## speed.md

Maintain continuously:

- inference FPS
- per-model latency
- end-to-end latency
- CPU utilization
- GPU utilization
- RAM
- VRAM
- model size
- startup time
- throughput
- optimization results

## test.md

Maintain:

- unit tests
- dataset tests
- model tests
- integration tests
- field robustness tests
- missing-data tests
- failure tests
- regression tests
- performance tests
- final validation tests

## techstack.md

Maintain the actual technologies and versions used.

Never leave this as an aspirational list.

## precision.md

Track:

```text
FP32
FP16
INT8
```

including:

- accuracy
- latency
- memory
- compatibility
- degradation
- selected precision

## dataset.md

Maintain:

- dataset sources
- licenses
- versions
- class counts
- geographic distribution
- crop distribution
- disease distribution
- pest distribution
- train/validation/test split
- leakage checks
- annotation statistics

## model_selection.md

For every major model:

```text
Task:
Candidate:
Selected:
Alternatives:
Accuracy:
Precision:
Recall:
Macro-F1:
Latency:
Memory:
Model size:
Field robustness:
Reason:
Date:
```

---

# 13. TRAINING AND DEPLOYMENT SCRIPTS

Create:

```text
train_windows.bat
```

and:

```text
train_linux.sh
```

The scripts must provide simple entry points for dataset preparation and training.

They must:

1. validate the environment
2. detect CUDA
3. detect GPU
4. verify required dependencies
5. verify dataset paths
6. create required directories
7. run the appropriate training stages
8. save checkpoints
9. save logs
10. save metrics
11. perform validation
12. export models
13. clearly report success/failure

The Windows script must work from:

```text
Command Prompt
PowerShell
```

The Linux script must be directly executable after appropriate permissions are applied.

Do not hard-code machine-specific absolute paths.

Use:

```text
environment variables
and/or
configuration files
```

for paths and settings.

---

# 14. TRAINING HARDWARE REQUIREMENTS

The project must support:

```text
NVIDIA GPU + CUDA
```

with the user's RTX 4050 treated as a supported development/training device.

Training must use:

```text
PyTorch
CUDA
AMP / FP16 where beneficial
```

The system must clearly report:

```text
GPU detected:
GPU name:
VRAM:
CUDA available:
PyTorch CUDA version:
CUDA device:
```

If CUDA is unavailable:

```text
WARNING:
CUDA unavailable.

Continuing with CPU fallback where practical.
```

Do not silently train on CPU when GPU acceleration was expected.

Heavy training jobs may be launched by the user using the generated scripts.

---

# 15. REPRODUCIBILITY

Every training run must record:

```text
timestamp
git commit/hash if available
dataset version
dataset size
model
pretrained checkpoint
hyperparameters
batch size
epochs
learning rate
augmentation
hardware
GPU
VRAM
software versions
CUDA version
PyTorch version
random seed
metrics
checkpoint path
```

Make experiments reproducible.

Never overwrite the best checkpoint without preserving it.

Use:

```text
runs/
  detector/
  classifier/
  segmentation/
  risk/
  fusion/
  deployment/
```

---

# 16. FIELD DATA AND DOMAIN SHIFT

Treat domain shift as a core deployment problem.

Explicitly evaluate:

```text
public dataset
      ↓
field dataset
      ↓
performance difference
```

Measure degradation across:

- lighting
- camera type
- background
- crop variety
- growth stage
- geography
- image quality
- disease severity

If a model performs extremely well on a clean dataset but poorly on field data, do NOT report the clean-dataset result as evidence of deployment readiness.

Use:

```text
field validation performance
```

as the primary deployment metric.

---

# 17. UNCERTAINTY AND EXPERT REFERRAL

The system must not force a diagnosis when confidence is low.

Example:

```text
Disease A       91%
Disease B        5%
Other            4%

→ High confidence
```

versus:

```text
Disease A       39%
Disease B       35%
Other            26%

→ Low confidence
→ Expert validation recommended
```

Implement calibrated confidence where practical.

Possible methods:

```text
temperature scaling
confidence calibration
ensemble-based uncertainty
```

Do not add complex uncertainty methods without measurable benefit.

---

# 18. FOLLOW-UP MONITORING

The system should support repeated observations.

Example:

```text
Day 1
Disease severity = 8%

Day 4
Disease severity = 14%

Day 7
Disease severity = 23%
```

Use these observations to support:

- progression tracking
- treatment follow-up
- risk updates
- extension-worker intervention
- outbreak surveillance

Do not claim causal treatment effectiveness unless the data supports that conclusion.

---

# 19. FAILURE HANDLING

The system must fail gracefully.

### Image unavailable

```text
Invalid image
→ reject
→ report reason
→ do not crash
```

### Low image quality

```text
Poor image quality
→ request better image
→ do not produce overconfident diagnosis
```

### Model failure

```text
Inference failure
→ log failure
→ fallback where available
→ preserve system availability
```

### Missing weather

```text
Weather unavailable
→ use remaining valid features
→ mark reduced-confidence forecast
```

### Missing soil data

```text
Soil unavailable
→ continue with available data
→ record missing input
```

### Missing pest-trap data

```text
Trap data unavailable
→ continue risk calculation
→ reduce confidence if appropriate
```

### Database failure

```text
Database unavailable
→ local buffer
→ retry
→ do not lose critical observations
```

### Network failure

```text
Network unavailable
→ continue offline functionality
→ queue synchronization
→ synchronize when connectivity returns
```

---

# 20. OFFLINE-FIRST REQUIREMENT

Core diagnosis must be capable of operating without continuous internet access.

At minimum, offline-capable components should include:

```text
image preprocessing
disease inference
pest inference
severity estimation
local risk calculation where required data is available
structured logging
cached advisory knowledge
```

Network-dependent services must fail gracefully.

Do not make the farmer dependent on continuous cloud connectivity for basic diagnosis.

---

# 21. MULTILINGUAL ADVISORY

The system should support multilingual farmer-facing output.

Separate:

```text
diagnosis
risk
recommended action
explanation
```

from the language-generation layer.

The underlying agricultural decision must remain structured.

Example:

```text
Disease:
Rice Blast

Risk:
High

Action:
Begin recommended IPM monitoring protocol.

Referral:
Extension worker recommended.
```

The presentation layer may then translate this structured result into supported regional languages.

Do not allow translation/generative systems to change the underlying technical recommendation.

---

# 22. CODE QUALITY REQUIREMENTS

Write production-quality code.

Requirements:

- modular architecture
- type hints where appropriate
- configuration-driven behavior
- meaningful logging
- clear error handling
- tests
- no unnecessary global state
- no hard-coded crop logic scattered across source files
- no duplicated model-loading code
- clean separation between vision, forecasting, spatial analysis, advisory, API, storage, and UI

Crop/disease/pest configuration should be data-driven.

For example:

```text
crop_config.yaml
```

should define:

```text
crops
diseases
pests
severity rules
risk features
advisory references
confidence thresholds
```

Do not hard-code everything into Python source.

---

# 23. REQUIREMENT TRACEABILITY

Maintain a requirement checklist in `AGENT.md`.

Every major requirement from `MASTER_PROMPT.md` must map to:

```text
Requirement
→ Implementation
→ Test
→ Status
```

Use:

```text
[ ] NOT STARTED
[~] IN PROGRESS
[x] IMPLEMENTED
[T] TESTED
[V] VERIFIED
[P] PENDING USER EXECUTION
```

Never mark something verified without evidence.

---

# 24. CONFIDENCE PROTOCOL

At the end of EVERY phase and EVERY verification gate, explicitly report:

```text
Agent Confidence: XX/100
```

Then provide:

```text
Why this confidence:
- ...
- ...
- ...

Remaining uncertainty:
- ...
- ...

Better alternatives considered:
- ...
- ...

Recommendation:
PROCEED / HOLD / REVISE
```

Then ask the user:

> **Do you want to proceed to the next phase, or investigate an alternative?**

Never silently move past a failed verification gate.

---

# 25. DECISION LOG

Every important architectural decision must be recorded.

Use:

```text
Decision:
Selected:
Alternatives:
Evidence:
Accuracy:
Precision:
Recall:
F1:
Latency:
Memory:
Reliability:
Field robustness:
Reason:
Date:
```

Never change a major model without recording why.

---

# 26. NO PREMATURE OPTIMIZATION

Do not begin with low-level optimization.

First establish:

```text
correctness
 ↓
dataset quality
 ↓
baseline accuracy
 ↓
model accuracy
 ↓
system integration
 ↓
profiling
 ↓
optimization
 ↓
final benchmark
```

Never optimize a component that has not been profiled.

---

# 27. NO FABRICATED AGRICULTURAL DATA

Never fabricate:

- disease prevalence
- field observations
- pest counts
- weather observations
- soil measurements
- outbreak locations
- model accuracy
- model confidence
- treatment effectiveness
- crop losses

Synthetic data may be used for development/testing only and must always be clearly labeled as synthetic.

Real-world claims require real evidence.

---

# 28. NO UNSUPPORTED TREATMENT CLAIMS

The system must not invent pesticide recommendations.

Treatment/advisory information must originate from a structured, verified agricultural knowledge source.

The AI may:

```text
identify disease
estimate risk
summarize verified recommendations
translate verified recommendations
```

It must not independently invent:

```text
chemical dosage
application frequency
withdrawal period
legal usage
crop-specific pesticide approval
```

If required information is unavailable:

```text
Recommend extension/laboratory consultation.
```

---

# 29. INITIAL RECOMMENDED TECHNOLOGY STACK

The initial working stack is:

### Deep learning

```text
Python
PyTorch
Torchvision
Ultralytics
CUDA
```

### Vision

```text
YOLO11
EfficientNetV2-S
YOLO11-seg
```

### Tabular ML

```text
LightGBM
XGBoost
```

### Geospatial

```text
H3
GeoPandas where appropriate
Shapely where appropriate
scikit-learn spatial algorithms
```

### Deployment

```text
ONNX
ONNX Runtime
TensorRT where supported
FP16
```

### Backend

Use a lightweight production API/service architecture appropriate to the deployment environment.

### Storage

Use a robust local database/storage system appropriate to:

```text
field observations
images
model outputs
risk predictions
expert confirmations
```

### Frontend

Provide interfaces appropriate for:

```text
farmer
extension worker
agriculture official
```

Do not create unnecessary screens or complexity.

---

# 30. STARTING ACTION

Before implementing anything:

1. inspect the existing repository
2. inspect available hardware
3. inspect installed software
4. inspect available datasets
5. inspect current project files
6. identify available agricultural data
7. identify existing APIs/services
8. identify what already exists
9. avoid rebuilding existing functionality
10. establish the baseline architecture
11. create documentation files
12. create project configuration
13. create training scripts
14. begin Phase 1

Do NOT immediately train models.

The agent environment is for code creation and lightweight validation.

Heavy model training must be performed by the user using the generated training scripts.

At the end of initial inspection, report:

```text
Repository status
Hardware
GPU status
CUDA status
Software environment
Dataset status
Agricultural data status
Current architecture
Missing components
Risks
Phase 1 plan
Agent confidence
```

Then begin Phase 1.

---

# 31. FINAL SUCCESS CONDITION

The project is complete only when the final system can demonstrate:

```text
✓ Offline-capable crop disease inference
✓ Pest detection
✓ Crop/plant recognition
✓ Disease classification
✓ Symptom localization
✓ Severity estimation
✓ Weather-based risk forecasting
✓ Pest-trap/sensor integration
✓ Farm-level risk estimation
✓ Geospatial hotspot detection
✓ Expert validation workflow
✓ Structured IPM advisory
✓ Multilingual farmer-facing output
✓ Follow-up monitoring
✓ Structured field observations
✓ Agriculture-official dashboard
✓ Real-time/near-real-time inference where required
✓ Stable long-duration execution
✓ Reproducible training
✓ CUDA-accelerated training
✓ Exported deployment models
✓ ONNX/TensorRT deployment path where supported
✓ Final benchmark
✓ Complete documentation
```

The final system must prioritize **real field reliability over benchmark-only performance**.

---

# 32. MASTER PROMPT PERSISTENCE & AGENT OPERATING CONTRACT

## CRITICAL — SAVE AND REUSE THIS MASTER PROMPT

Immediately after reading this master prompt, save an exact copy of it into the repository as:

```text
MASTER_PROMPT.md
```

This file is the **authoritative project specification and agent operating contract**.

Do not rewrite, shorten, summarize, or modify its requirements unless the user explicitly instructs you to do so.

---

## MANDATORY MASTER PROMPT CHECK AFTER EVERY PHASE

Before starting **any phase or subphase**, you MUST:

1. Read `MASTER_PROMPT.md` from the repository.
2. Identify the current phase and verification gate.
3. Re-check all requirements relevant to that phase.
4. Compare the current implementation against the master prompt.
5. Identify missing, incomplete, or contradictory requirements.
6. Continue only after this check.

After completing every phase/subphase:

1. Read `MASTER_PROMPT.md` again.
2. Verify that the implementation still follows it.
3. Update the relevant documentation.
4. Record deviations, if any.
5. Record unresolved requirements.
6. Update `AGENT.md`.
7. Update `AGENT_REVIEW.md`.
8. Report the confidence score.
9. STOP at the verification gate and ask the user whether to proceed.

Use this loop:

```text
READ MASTER_PROMPT.md
        ↓
PLAN
        ↓
IMPLEMENT
        ↓
LIGHTWEIGHT TEST
        ↓
READ MASTER_PROMPT.md AGAIN
        ↓
REQUIREMENT AUDIT
        ↓
DOCUMENT
        ↓
CONFIDENCE SCORE
        ↓
STOP / ASK USER
```

---

## DO NOT HALLUCINATE PROJECT STATE

Never assume that something exists because the master prompt says it should exist.

Verify it in the repository.

For every important claim, distinguish between:

```text
IMPLEMENTED
TESTED
VERIFIED
NOT IMPLEMENTED
NOT TESTED
PENDING USER TRAINING
PENDING BENCHMARK
```

Never report planned functionality as completed functionality.

Never fabricate:

- training results
- accuracy
- precision
- recall
- F1
- FPS
- latency
- benchmark results
- dataset size
- field observations
- pest counts
- weather data
- model performance
- successful deployment
- hardware compatibility

---

## REQUIREMENT TRACEABILITY

Maintain a requirement checklist in `AGENT.md`.

Every major requirement from `MASTER_PROMPT.md` must map to:

```text
Requirement
→ Implementation file/component
→ Test
→ Status
```

Use statuses:

```text
[ ] NOT STARTED
[~] IN PROGRESS
[x] IMPLEMENTED
[T] TESTED
[V] VERIFIED
[P] PENDING USER EXECUTION
```

Before declaring Phase 6 complete, every applicable core requirement must be accounted for.

---

## PHASE BOUNDARY RULE

You are NOT allowed to silently move to the next phase.

At the end of each phase:

```text
Implementation
      ↓
Verification subphase
      ↓
Master prompt audit
      ↓
Documentation update
      ↓
Confidence XX/100
      ↓
ASK USER
      ↓
Next phase only after approval
```

If a verification gate reveals a major problem:

```text
DO NOT PROCEED
        ↓
DOCUMENT PROBLEM
        ↓
PROPOSE FIX / ALTERNATIVES
        ↓
ASK USER
```

---

## LIGHTWEIGHT EXECUTION ONLY

Remember that this agent environment is primarily for **code creation and lightweight validation**, not heavy model training.

Create all training and benchmarking infrastructure.

Do not execute expensive training jobs.

Use lightweight smoke tests wherever possible.

The actual heavy training will be performed by the user using:

```text
train_windows.bat
train_linux.sh
```

The training scripts must support the user's NVIDIA RTX 4050 through CUDA when available.

---

## FINAL RULE

Treat `MASTER_PROMPT.md` as the project's **single source of truth**.

If your memory, assumptions, previous reasoning, or current task context conflicts with `MASTER_PROMPT.md`, stop and resolve the conflict using the master prompt and explicit user instructions.

**Never rely on memory when the repository contains the authoritative specification.**

Before every phase:

> READ → CHECK → IMPLEMENT → TEST → RE-READ → AUDIT → DOCUMENT → ASK
