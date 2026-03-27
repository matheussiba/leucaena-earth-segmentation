# leucaena-earth-segmentation

Binary semantic segmentation of **Leucaena leucocephala** (leucaena) tree canopy using aerial imagery and LiDAR data. Adapted from [felferrari/tree_fusion](https://github.com/felferrari/tree_fusion).

## Overview

This pipeline takes:
- **4-band aerial imagery** (B, G, R, NIR) at 25 cm resolution
- **LiDAR raster products** (CHM, intensity) from 4 pts/m² aerial survey
- **Polygon masks** (GeoJSON) created on [leucaena.earth](https://leucaena.earth)

And produces a trained **ResUNet** model for pixel-wise binary segmentation (leucaena vs. background).

## Pipeline

```
1. prep-data.py    Rasterize GeoJSON masks, normalise imagery, create patches
2. train.py        Train ResUNet with PyTorch (GPU recommended)
3. prediction.py   Sliding-window inference on full image
4. evaluation.py   Per-class precision, recall, F1, confusion matrix
```

## Project Structure

```
leucaena-earth-segmentation/
├── prep-data.py           # Data preparation (rasterize + patch)
├── train.py               # Model training
├── prediction.py          # Inference
├── evaluation.py          # Metrics
├── requirements.txt       # Python dependencies
├── conf/
│   ├── paths.py           # Data paths (edit for your machine)
│   ├── general.py         # Patch size, classes, hyperparameters
│   ├── default.py         # CLI defaults
│   ├── model_1.py         # Exp 1: Optical only
│   ├── model_2.py         # Exp 2: Early fusion (Optical + LiDAR)
│   └── model_3.py         # Exp 3: Late fusion (Optical | LiDAR)
├── models/
│   ├── resunet.py         # ResUNet architectures (+ fusion variants)
│   └── layers.py          # Residual block
├── utils/
│   ├── ops.py             # GeoTIFF I/O, GeoJSON rasterization, helpers
│   ├── dataloader.py      # PyTorch datasets
│   └── trainer.py         # Train/val loops, early stopping
├── data/                  # Your imagery + masks (not in git)
│   ├── optical.tif
│   ├── lidar.tif
│   └── masks.geojson
├── prepared/              # Generated numpy arrays (not in git)
└── experiments/           # Training outputs (not in git)
```

## Setup

```bash
# 1. Create conda environment (recommended for GDAL)
conda create -n leucaena python=3.11 gdal -c conda-forge
conda activate leucaena

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Edit paths
#    Open conf/paths.py and set paths to your data files
```

## Data Preparation

### Prerequisites

1. **Optical image**: 4-band GeoTIFF (B, G, R, NIR)
2. **LiDAR raster**: Pre-process your LAS/LAZ point cloud into raster bands (CHM, intensity). Tools: PDAL, LAStools, CloudCompare, or WhiteboxTools. The raster must match the optical image extent and resolution.
3. **Masks**: Export polygon masks from [leucaena.earth](https://map.leucaena.earth) as GeoJSON.

### Run

```bash
# With LiDAR
python prep-data.py --optical data/optical.tif --lidar data/lidar.tif --masks data/masks.geojson

# Without LiDAR (optical only)
python prep-data.py --optical data/optical.tif --masks data/masks.geojson --no-lidar
```

## Training

```bash
# Experiment 1: Optical only
python train.py -e 1

# Experiment 2: Early fusion (Optical + LiDAR)
python train.py -e 2

# Experiment 3: Late fusion
python train.py -e 3
```

## Prediction & Evaluation

```bash
python prediction.py -e 1
python evaluation.py -e 1
```

## Model Experiments

| Exp | Architecture | Input |
|-----|-------------|-------|
| 1 | ResUNet | Optical only (B,G,R,NIR) |
| 2 | ResUNet (early fusion) | Optical + LiDAR (concatenated) |
| 3 | ResUNet (late fusion) | Optical + LiDAR (separate encoders) |

## Credits

- Original tree segmentation pipeline: [Felipe Ferrari](https://github.com/felferrari/tree_fusion)
- Leucaena mapping platform: [leucaena.earth](https://leucaena.earth)
- Research: ESALQ/USP
