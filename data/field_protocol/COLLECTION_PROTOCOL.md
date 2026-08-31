# Maharashtra field dataset collection protocol

Status: **protocol only**. Zero field images are stored in this repository.

This protocol exists so partners (KVKs, SAUs, CROPSAP scouts, extension workers) can collect
a legally usable, leakage-safe, expert-reviewed dataset. It is not itself a dataset.

## 1. Purpose

Close the domain gap between:

```text
PlantVillage (lab, clean background)
        ↓
PlantDoc / FieldPlant (non-Maharashtra field)
        ↓
Maharashtra smartphone / scout images  ← required for deployment
```

## 2. Consent and licensing

- Obtain informed consent from the farmer before photographing a farm.
- Record whether the image may be used for model training, research, or internal use only.
- Do not publish GPS at house-level precision in public releases (snap to field centroid or H3 r8).
- Store original GPS only in the access-controlled observation store.

## 3. Capture rules (smartphone)

Take **at least 4 images per plant/issue**:

1. Canopy / whole-plant context (1–2 m)
2. Affected organ close-up (leaf / boll / panicle / fruit), in focus
3. Second close-up from a different angle
4. Healthy comparison leaf from the same plot when available

Also capture when possible:

- poor lighting and good lighting of the same symptom
- multiple cultivars and growth stages
- early *and* advanced symptoms
- mixed disease/pest when that is the real situation (label as multi-label, do not force one class)

Avoid:

- gloves or fingers covering the lesion
- flash hotspots that blow out lesion color
- screenshots of other apps
- laboratory cut-leaves on white paper (those belong in a lab split, not the field split)

## 4. Required metadata (CSV / app form)

| Field | Required | Notes |
| --- | --- | --- |
| sample_id | yes | UUID |
| farmer_id | yes | pseudonymous |
| location_id | yes | field, not village-only |
| capture_session_id | yes | one scouting stop |
| video_id | if from video | prevents frame leakage |
| captured_at | yes | ISO-8601 with timezone |
| lat, lon | yes if GPS on | WGS84 |
| crop_id | yes | from crop_config.yaml |
| variety | if known | |
| growth_stage | yes | seedling/vegetative/reproductive/ripening/harvest/unknown |
| class_id | after expert review | empty until reviewed |
| secondary_class_id | if mixed | |
| severity_bin | after review | healthy/early/moderate/severe/very_severe |
| affected_area_pct | if segmented | 0–100 |
| symptom_region | yes | leaf/stem/fruit/... |
| camera_make_model | auto | |
| consent | yes | |
| is_field | always true | |
| is_synthetic | always false | |

## 5. Expert label workflow

```text
Scout capture
    ↓
Extension worker pre-label (optional)
    ↓
Plant pathologist / entomologist confirmation
    ↓
Disagreement → laboratory / third expert
    ↓
Only confirmed labels enter train/val
    ↓
A held-out district stays in isolated test
```

Do not train on unconfirmed scout labels.

## 6. Leakage rules

- All images from one `capture_session_id`, `video_id`, `farmer_id`, or `location_id` stay in **one** split.
- Isolated test should use **different talukas/districts** than train when sample size allows.
- Do not put consecutive video frames in different splits.

## 7. Target volume (guidance, not a claim of existing data)

These are collection *targets*, not current inventory:

| Crop | Confirmed field images (target) | Notes |
| --- | --- | --- |
| Cotton | 2000+ | pink bollworm + sucking pests + blight |
| Soybean | 1500+ | |
| Rice | 1500+ | Konkan + irrigated |
| Sugarcane | 1500+ | |
| Tomato / grape / onion | 1000+ each | horticulture belt |
| Tur, jowar, gram, maize, wheat | 800+ each | |

Until these exist, models trained on public lab sets are **not deployment-ready**.

## 8. Pest traps

If pheromone / sticky / light trap counts are collected:

- trap_id, trap_type, pest_id, count, timestamp, lat/lon, crop_id
- Do not interpolate missing days as zeros without recording that they were missing.

## 9. Weather / soil

Prefer plot-adjacent AWS or IMD station IDs over guessed numbers.
Soil: Soil Health Card IDs if the farmer consents; otherwise SoilGrids as a covariate with `source=soilgrids`.
