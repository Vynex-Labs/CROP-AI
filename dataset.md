# dataset.md

Version: `v0.1.1-phase1`  
Date: 2026-08-31  
Honesty rule: counts below distinguish **defined taxonomy**, **registered sources**, and **files actually in this clone**.

## Sources and licenses

| Source | Kind | Task | License | In this repo? | Field realism |
| --- | --- | --- | --- | --- | --- |
| PlantVillage (Mohanty et al. 2016) | public lab | classification | CC0-1.0 | **not downloaded** | low |
| PlantDoc (Singh et al. 2020) | public scraped | cls + det | research — verify before redistribution | **not downloaded** | medium |
| FieldPlant (Moupojou et al. 2023) | public field (Cameroon) | det + cls | CC-BY-4.0 | **not downloaded** | high, **not India** |
| IP102 (Wu et al. 2019) | mixed pest photos | pest cls | research — verify | **not downloaded** | mixed |
| NASA POWER | gridded weather | risk features | NASA public | **not downloaded** | n/a |
| Open-Meteo | weather API | risk features | Open-Meteo terms | API only | n/a |
| SoilGrids | soil covariates | risk features | CC-BY-4.0 | **not downloaded** | n/a |
| CROPSAP Maharashtra | gov surveillance | traps + advisories | **not in repo** | **not available** | high if obtained |
| field_maharashtra | project field | all | pending consent | **0 images** | target |
| synthetic_smoke | generated PNG | pipeline tests | generated_in_repo | **generated on demand** | none |

Citations and URLs: `configs/dataset.yaml`.

## Taxonomy (defined, not image counts)

From `configs/crop_config.yaml` (validated by `Taxonomy.validate_integrity`):

- Crops (**17**, union — not narrowed):
  - Top 10 India: rice, wheat, sugarcane, cotton, maize, soybean, chickpea, groundnut, mustard, pearl_millet
  - Horticulture required: tomato, potato, maize, grape
  - CROPSAP / retained: pigeon_pea, sorghum, onion, pepper
- Disease/healthy classes: see YAML (healthy class per crop + named diseases)
- Pests: brown planthopper, stem borers, FAW, pink/american bollworm, whitefly, jassid, aphid, thrips, etc.
- Severity bins: healthy 0–1%, early 1–10%, moderate 10–25%, severe 25–50%, very_severe 50–100% (affected-area proxy)
- Symptom regions: leaf, stem, fruit, panicle, boll, cob, root, whole_plant, unknown
- Crop stages: seedling, vegetative, reproductive, ripening, harvest, unknown

## Image counts **in this clone**

Before `prepare-dataset`: **0 real images**.

After `python -m cropai prepare-dataset` on 2026-08-31 (synthetic only, because `data/raw/` is empty):

| Split | n | Real field | Lab public | Synthetic |
| --- | --- | --- | --- | --- |
| train | 57 | 0 | 0 | 57 |
| val | 10 | 0 | 0 | 10 |
| test | 9 | 0 | 0 | 9 |
| **total** | **76** | **0** | **0** | **76** |

19 synthetic classes, 11 crops represented, 0 pigeon_pea / sorghum / chickpea images even in smoke (those crops are operational-only and not in the smoke training-public loop except cotton/sugarcane/onion extras).

Exact sidecar: `data/manifests/latest.json`. Do not reuse these 76 numbers as field statistics.

Geographic distribution of real images: **none**. Synthetic lat/lon values are placeholders inside Maharashtra-ish bounds and are labeled `is_synthetic=true`.

## Splits

- Ratios: 70 / 15 / 15 train / val / isolated test
- Group keys: `farmer_id`, `location_id`, `video_id`, `capture_session_id`, `source_group`
- A group may not appear in two splits (`fail_on_leakage: true`)
- Hash-identical files may not cross splits

## Annotation formats

| Task | Format | Writer |
| --- | --- | --- |
| classification | CSV manifest (`IMAGE_FIELDS`) | `cropai.dataset.schema` |
| detection | YOLO txt + `dataset.yaml`, COCO JSON | `cropai.dataset.yolo`, `coco` |
| segmentation | YOLO-seg polygons, COCO polygons | same |
| observations | JSONL | `ObservationRecord` |
| traps | JSONL | `TrapRecord` |
| weather | JSONL | `WeatherRecord` |

JSON Schema: `configs/schemas/image_record.json`.

## Leakage checks

Automated in `DatasetValidator`:

- duplicate `sample_id` / `relpath`
- group leakage across splits
- sha256 leakage across splits
- missing / corrupted images
- resolution bounds
- unknown crop/class labels
- lat/lon range and pairing
- timestamp parse
- YOLO bbox / polygon sanity
- synthetic source without `is_synthetic=true`

## Severity labels

Public PlantVillage **does not** provide lesion area. Severity on PV would be unlabeled (`severity_bin=""`). Synthetic smoke computes a crude blob-area proxy for pipeline tests only. Real severity requires field masks or expert ordinal labels.

## What is NOT in the repo

- Maharashtra CROPSAP trap time series
- IMD / AWS station extracts
- Soil Health Card rows
- Expert-confirmed field photos
- Any prevalence, outbreak, or yield-loss statistics

Those absences are recorded, not filled with fake numbers.

## Versioning

`data/manifests/images_<version>.csv` + `.json` sidecar with `csv_sha256`, git commit, class/crop/split counts. `latest.json` points at the last prepare run.
