# Downloading public datasets (user machine)

The agent sandbox does **not** download multi-gigabyte corpora. Run these on the training machine after reviewing each license.

Never commit raw images.

## PlantVillage (CC0 1.0) — lab classification

```bash
mkdir -p data/raw/plantvillage
git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset data/raw/plantvillage/src
# Expected layout after copy: data/raw/plantvillage/color/<ClassName>/*.jpg
```

Mendeley mirror: https://data.mendeley.com/datasets/tywbtsjrjv/1

## PlantDoc — in-the-wild (verify license before redistribution)

```bash
mkdir -p data/raw/plantdoc
git clone --depth 1 https://github.com/pratikkumar-patel/PlantDoc-Dataset data/raw/plantdoc/src
```

Detection labels (separate repo): https://github.com/pratikkumar-patel/PlantDoc-Object-Detection-Dataset

## FieldPlant (CC-BY-4.0) — Cameroon field images

See Moupojou et al., IEEE Access 2023. Place extracted images under `data/raw/fieldplant/` as ImageFolder or CSV.

## IP102 — pest recognition (verify license)

https://github.com/xpwu95/IP102 — place class folders under `data/raw/ip102/`.

## After download

```bash
python -m cropai prepare-dataset
```

The adapter will ingest whatever exists under `data/raw/` and keep synthetic smoke data labeled `is_synthetic=true`.
