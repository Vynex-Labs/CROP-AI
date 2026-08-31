# phase3.md — Disease / pest risk forecasting

Date: 2026-08-31  
Status: **IMPLEMENTED (code + unvalidated heuristic fallback). NOT TRAINED. Metrics NOT MEASURED.**

## Objectives

MASTER_PROMPT §7: weather / crop / history / trap features → 1/3/7-day disease and pest risk; calibration; missing-data handling; LightGBM as the starting candidate.

## Implementation

| Piece | Path | Status |
| --- | --- | --- |
| Config (data-driven coefficients) | `configs/risk.yaml` | [T] |
| Output schema (6 horizons) | `src/cropai/risk/schema.py` | [T] |
| Temporal features | `features.py` | [T] |
| Missing-data policy | `missing.py` | [T] |
| Heuristic fallback | `rules.py` | [T] |
| LightGBM / XGBoost adapters | `backends.py` | [x] untested (packages absent) |
| Identity calibration | `calibrate.py` | [T] |
| Engine | `engine.py` | [T] |
| Metrics (ROC/PR/Brier/FNR/ECE) | `metrics.py` | [T] toy only |
| Train / evaluate / benchmark | `train.py` `evaluate.py` `benchmark.py` | [T] dry-run |
| CLI | `cropai forecast-risk` | [T] |
| Synthetic panel | `tables.py` (labeled SYNTHETIC) | [T] |

Runtime default is **`heuristic_unvalidated`**, not LightGBM. Scores are **uncalibrated**. They are **not** outbreak probabilities.

## Experiments

- pytest `tests/risk/` — wet vs dry disease score, trap vs zero pest score, missing weather/trap continue, insufficient-input referral, benchmark honesty
- `python -m cropai.risk.train --dry-run` → `n_real_outbreak_labels=0`, `status=NOT_TRAINED`
- `cropai forecast-risk --crop rice --humidity 90 ... --synthetic` → heuristic scores, `calibrated=false`
- `./train_linux.sh train-risk --dry-run`

No LightGBM fit on real labels. No NASA POWER / CROPSAP ingest (not on disk).

## Results

| Metric | Value | Status |
| --- | --- | --- |
| ROC-AUC / PR-AUC | n/a | NOT MEASURED |
| Precision / recall / macro-F1 | n/a | NOT MEASURED |
| Brier / calibration / ECE | n/a | NOT MEASURED |
| False-negative rate | n/a | NOT MEASURED |
| Lead time | n/a | NOT MEASURED |
| Missing-data robustness | heuristic continues; confidence reduced | TESTED (synthetic) |
| Real outbreak labels in repo | **0** | honest |

## Decisions

```
Decision: If outbreak labels are too few, do not train/select LightGBM
Selected: NOT SELECTED
Runtime fallback: heuristic_unvalidated (uncalibrated, unvalidated)
Alternatives: LightGBM, XGBoost, logistic baseline, climatology 0.5
Evidence: n_real_outbreak_labels = 0; MASTER_PROMPT §7
Reason: ML forecasting → DO NOT FABRICATE → statistical/heuristic fallback
Date: 2026-08-31
```

```
Decision: Identity calibration until labeled forecasts exist
Selected: none (method: none)
Evidence: no paired (forecast, outbreak) set
Date: 2026-08-31
```

## Failures

- LightGBM / XGBoost / scikit-learn not installed here
- No IMD / NASA POWER / Open-Meteo / SoilGrids files in `data/raw/`
- No CROPSAP trap time series
- Heuristic is **not** a validated Maharashtra ETL rule set

## Verification (Phase 3.5)

See AGENT_REVIEW.md. Comparison harness exists. **No risk model is selected.**

- Is LightGBM best? **UNKNOWN**
- Does data support ML forecasting? **NO** (0 real labels)

## Confidence

Agent Confidence: **70/100** (code). Forecast-skill confidence: **0/100**.

Recommendation: **PROCEED** to Phase 4 geospatial / fusion / advisory *infrastructure* in parallel with collecting real weather, trap, and outbreak labels. Do not freeze LightGBM.

## Next steps

Stop. Ask the user whether to proceed to Phase 4.

On a data-authorized machine (after labels exist):

```bash
pip install -r requirements-train.txt
python -m cropai.risk.train --dry-run
python -m cropai benchmark-risk
# only then: python -m cropai.risk.train   # still refuses below min_real_labels
```
