> **Part of the [PhD Leucaena Mapping Project](https://github.com/matheussiba/phd-leucaena-mapping)** — Mapping and Biomass Estimation of *Leucaena leucocephala* with Deep Learning (ESALQ/USP).
> See also: [leucaena-earth-platform](https://github.com/matheussiba/leucaena-earth-platform) (crowdmapping web app)

# leucaena-earth-segmentation

Binary semantic segmentation of **Leucaena leucocephala** (leucaena) tree canopy using aerial imagery and LiDAR data. Adapted from [felferrari/tree_fusion](https://github.com/felferrari/tree_fusion).

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [Key concepts for beginners](#2-key-concepts-for-beginners)
3. [How the pipeline works (big picture)](#3-how-the-pipeline-works-big-picture)
4. [Project structure](#4-project-structure)
5. [Setup](#5-setup)
6. [Step-by-step usage](#6-step-by-step-usage)
7. [Understanding the training output](#7-understanding-the-training-output)
8. [The three experiments](#8-the-three-experiments)
9. [Tuning the model](#9-tuning-the-model)
10. [Credits](#10-credits)

---

## 1. What is this project?

*Leucaena leucocephala* is an invasive tree species found across Brazil and many tropical regions.
This project trains a deep learning model to **automatically detect leucaena trees in aerial images**,
pixel by pixel — a task called **semantic segmentation**.

The model learns by looking at:
- **Aerial photos** (4 colour bands: Blue, Green, Red, and Near-Infrared)
- **LiDAR data** (a laser scan from the air that tells you how tall things are)
- **Human-drawn polygon masks** created on [leucaena.earth](https://leucaena.earth) that mark where leucaena trees are

After training, you point the model at a new image and it produces a map showing exactly which
pixels are leucaena and which are background.

---

## 2. Key concepts for beginners

> **Skip this section if you are already familiar with deep learning basics.**

### What is semantic segmentation?

Regular image classification asks *"what is in this image?"* (e.g., "a tree").
Semantic segmentation asks *"which pixels belong to each class?"* — every single pixel gets a label.
In our case: **0 = background**, **1 = leucaena**.

### What is a patch?

A full aerial image can be enormous (tens of thousands of pixels wide).
Neural networks typically work on small, fixed-size chunks called **patches** (here 256 × 256 pixels).
The pipeline slices the image into overlapping patches, trains on them, and then stitches predictions
back together to rebuild the full-resolution map.

### What is a ResUNet?

**U-Net** is a classic architecture for segmentation. It has:
- An **encoder** (the left side): compresses the image into a compact representation, learning
  "what features matter" — edges, colours, shapes.
- A **decoder** (the right side): gradually reconstructs the spatial map using those features.
- **Skip connections**: shortcuts from the encoder to the decoder so fine spatial details are not lost.

**ResUNet** adds **residual blocks** (from ResNet) to the encoder, which makes training
deeper networks easier by letting gradients flow more smoothly during backpropagation.

### What is fusion?

This project has both optical (colour + NIR) and LiDAR data. *Fusion* is how you combine them:

| Strategy | What happens |
|----------|-------------|
| **No fusion** (Exp 1) | LiDAR is ignored. Only optical bands are used. |
| **Early fusion** (Exp 2) | LiDAR bands are concatenated to optical bands *before* the first layer. The network sees all bands at once. |
| **Late fusion** (Exp 3) | Two separate encoders process optical and LiDAR independently. Their outputs are merged only near the end. |

### What is a loss function?

The loss measures how wrong the model's predictions are. During training, the optimizer tries to
minimise the loss. We use **CrossEntropyLoss** with class weights:
- `background` weight = 0.3
- `leucaena` weight = 0.7

The higher weight for leucaena tells the model "mistakes on the minority class cost more" — this
is important because leucaena pixels are much rarer than background pixels (class imbalance).

### What is early stopping?

Instead of running all 300 epochs, training stops automatically when the **validation loss**
stops improving for `patience = 15` consecutive epochs. The best model checkpoint is saved
and restored at the end. This prevents overfitting.

### What is overfitting?

When a model memorises the training data instead of learning generalizable patterns.
Signs: training loss keeps dropping but validation loss starts rising.
The validation set (data the model never trains on) is used to detect this.

---

## 3. How the pipeline works (big picture)

```
Your raw data
  ├─ optical.tif       (aerial photo, 4 bands, 25 cm resolution)
  ├─ lidar.tif         (LiDAR raster: CHM + intensity)
  └─ masks.geojson     (polygon annotations from leucaena.earth)
            │
            ▼
    [ 1. prep-data.py ]
    ┌──────────────────────────────────────────────────┐
    │ • Read and align optical + LiDAR rasters         │
    │ • Filter extreme pixel values (outlier removal)  │
    │ • Normalize pixel values to [0, 1]               │
    │ • Rasterize GeoJSON polygons → label raster      │
    │ • Slice everything into 256×256 patches          │
    │ • Split: 60% train / 20% val / 20% test          │
    │ • Save .npy arrays to prepared/                  │
    └──────────────────────────────────────────────────┘
            │
            ▼
    [ 2. train.py -e <1|2|3> ]
    ┌──────────────────────────────────────────────────┐
    │ • Load patches from prepared/                    │
    │ • Build ResUNet model (optical-only, or fusion)  │
    │ • Train for up to 300 epochs with early stopping │
    │ • Save best model to experiments/exp_N/models/   │
    │ • Save training log to experiments/exp_N/logs/   │
    └──────────────────────────────────────────────────┘
            │
            ▼
    [ 3. prediction.py -e <N> ]
    ┌──────────────────────────────────────────────────┐
    │ • Load the saved model                           │
    │ • Run sliding-window inference over full image   │
    │ • Average overlapping predictions (3 passes)     │
    │ • Save predicted class map (.npy + GeoTIFF)      │
    └──────────────────────────────────────────────────┘
            │
            ▼
    [ 4. evaluation.py -e <N> ]
    ┌──────────────────────────────────────────────────┐
    │ • Compare predictions vs. test labels            │
    │ • Print precision, recall, F1 per class          │
    │ • Print confusion matrix                         │
    └──────────────────────────────────────────────────┘
            │
            ▼
    experiments/exp_N/predicted/pred.tif   ← your output map
```

---

## 4. Project structure

```
leucaena-earth-segmentation/
│
├── prep-data.py        # Step 1 — data preparation
├── train.py            # Step 2 — model training
├── prediction.py       # Step 3 — run the model on the full image
├── evaluation.py       # Step 4 — measure how good the model is
├── requirements.txt    # Python packages needed (pip install -r requirements.txt)
│
├── conf/               # Configuration — edit these files to change behaviour
│   ├── paths.py        # Where your data files are on disk
│   ├── general.py      # Patch size, classes, learning rate, early stopping, etc.
│   ├── default.py      # Default values for command-line arguments
│   ├── model_1.py      # Experiment 1: optical-only ResUNet
│   ├── model_2.py      # Experiment 2: early fusion (optical + LiDAR concatenated)
│   └── model_3.py      # Experiment 3: late fusion (two separate encoders)
│
├── models/             # Neural network architecture code
│   ├── resunet.py      # ResUNet class and fusion variants
│   └── layers.py       # Residual block (the building block of ResUNet)
│
├── utils/              # Helper code used by the main scripts
│   ├── ops.py          # File I/O: read/write GeoTIFF, rasterize GeoJSON, etc.
│   ├── dataloader.py   # PyTorch Dataset classes (feed patches to the model)
│   └── trainer.py      # Training loop, validation loop, early stopping logic
│
├── data/               # YOUR data goes here (not tracked in git — add yours)
│   ├── optical.tif     # 4-band aerial image
│   ├── lidar.tif       # LiDAR raster
│   └── masks.geojson   # Polygon annotations
│
├── prepared/           # Auto-generated by prep-data.py (not in git)
│   ├── opt_img.npy         # Normalised optical image (H × W × 4)
│   ├── lidar_img.npy       # Normalised LiDAR image (H × W × 2)
│   ├── label_train.npy     # Label raster for training patches
│   ├── label_test.npy      # Label raster for test patches
│   ├── train_patches.npy   # Patch index arrays for training
│   └── val_patches.npy     # Patch index arrays for validation
│
└── experiments/        # Auto-generated by train.py (not in git)
    └── exp_1/          # One folder per experiment number
        ├── models/
        │   └── model.pt        # Best saved model weights
        ├── logs/
        │   ├── train_1.txt     # Full training log (printed output)
        │   └── eval_1.txt      # Evaluation results
        ├── predicted/
        │   ├── pred.npy        # Flat prediction array
        │   └── pred.tif        # Georeferenced output map (GeoTIFF)
        └── visual/             # Sample validation images saved during training
```

---

## 5. Setup

### Prerequisites

- Python 3.11 (recommended)
- [Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html) — easiest way to install GDAL
- A CUDA-capable GPU (strongly recommended; training on CPU is very slow)

### Step 1 — Create a conda environment

```bash
conda create -n leucaena python=3.11 gdal -c conda-forge
conda activate leucaena
```

> **Why conda for GDAL?** GDAL is a C library with complex system dependencies.
> Installing it through conda-forge handles all of that automatically.
> Installing via pip alone often fails.

### Step 2 — Install Python packages

```bash
pip install -r requirements.txt
```

### Step 3 — Put your data in the right place

Create a `data/` folder in the project root and place your files there:

```
data/
├── optical.tif      # 4-band GeoTIFF (Blue, Green, Red, NIR)
├── lidar.tif        # LiDAR raster (CHM, Intensity) — same CRS and resolution as optical
└── masks.geojson    # Polygon masks exported from leucaena.earth
```

> **Don't have LiDAR yet?** You can run Experiment 1 (optical only) and skip the LiDAR file entirely.

### Step 4 — Check your paths

Open `conf/paths.py` and verify the filenames match your actual files.
If you named them exactly as above, no changes are needed.

---

## 6. Step-by-step usage

### Step 1 — Prepare the data

```bash
# With LiDAR (runs experiments 2 and 3)
python prep-data.py --optical data/optical.tif --lidar data/lidar.tif --masks data/masks.geojson

# Without LiDAR (only experiment 1 will work)
python prep-data.py --optical data/optical.tif --masks data/masks.geojson --no-lidar
```

This creates the `prepared/` folder with all `.npy` arrays. You only need to run this once
(unless you change your masks or input imagery).

> **What is a .npy file?** It is a binary file saved by NumPy — very fast to load,
> much faster than reading a GeoTIFF on every training epoch.

### Step 2 — Train a model

```bash
python train.py -e 1     # Experiment 1: optical only
python train.py -e 2     # Experiment 2: early fusion
python train.py -e 3     # Experiment 3: late fusion
```

Extra options:

```bash
python train.py -e 1 -b 8      # batch size 8 (reduce if you run out of GPU memory)
python train.py -e 1 -a False  # disable data augmentation
python train.py -e 1 -c True   # continue a previously interrupted training run
```

Training output is written to `experiments/exp_<N>/logs/train_<N>.txt` (not the terminal).
To watch it live:

```bash
# Windows PowerShell
Get-Content experiments\exp_1\logs\train_1.txt -Wait

# Linux / macOS
tail -f experiments/exp_1/logs/train_1.txt
```

### Step 3 — Run prediction

```bash
python prediction.py -e 1
```

The model slides a 256 × 256 window across the full image (with 0%, 25%, and 50% overlap)
and averages the three prediction passes. This reduces border artefacts between patches.

Output: `experiments/exp_1/predicted/pred.tif` — a GeoTIFF you can open in QGIS.

### Step 4 — Evaluate results

```bash
python evaluation.py -e 1
```

Results are saved to `experiments/exp_1/logs/eval_1.txt`. They include:

```
            background: Acc=98.12%  F1=99.01%  Prec=98.50%  Rec=99.53%  Samples=1,204,312
              leucaena: Acc=92.45%  F1=87.23%  Prec=85.10%  Rec=89.45%  Samples=95,688

Confusion matrix:
  TN=1,188,100  FP=16,212
  FN=10,105     TP=85,583
```

> **How to read the confusion matrix:**
> - **TP** (True Positive): pixels the model correctly identified as leucaena
> - **TN** (True Negative): pixels the model correctly identified as background
> - **FP** (False Positive): background pixels wrongly called leucaena
> - **FN** (False Negative): leucaena pixels the model missed

---

## 7. Understanding the training output

During training, every epoch logs a line like:

```
Epoch 12
  train loss: 0.3241   val loss: 0.3589
```

- **Loss decreasing** → the model is learning
- **Train loss much lower than val loss** → possible overfitting (model memorising, not generalising)
- **Loss plateau for 15 epochs** → early stopping triggers, best checkpoint is restored

The `visual/` folder inside each experiment saves sample prediction images after each epoch
so you can visually see if the model is improving over time.

---

## 8. The three experiments

| Exp | Script flag | What the model sees | When to use |
|-----|-------------|---------------------|-------------|
| 1 | `-e 1` | Optical (B, G, R, NIR) only | Baseline; also use when you have no LiDAR |
| 2 | `-e 2` | Optical + LiDAR concatenated at the input | Simple, fast — good first fusion attempt |
| 3 | `-e 3` | Optical and LiDAR through two separate encoders, merged at the bottleneck | More expressive — often best, but heavier |

**Recommended order:** run all three, then compare F1 scores in the evaluation logs to
decide which fusion strategy works best for your data.

---

## 9. Tuning the model

All key numbers live in `conf/general.py`. The most important ones:

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `PATCH_SIZE` | 256 | Width and height of each training patch in pixels |
| `PATCH_OVERLAP` | 0.5 | Overlap fraction between adjacent patches during prep |
| `MAX_EPOCHS` | 300 | Maximum training epochs before forced stop |
| `LEARNING_RATE` | 1e-4 | How fast the optimizer adjusts weights (smaller = more stable but slower) |
| `LEARNING_RATE_SCHEDULER_GAMMA` | 0.995 | LR multiplied by this every epoch (LR decays slowly over time) |
| `CLASSES_WEIGHTS` | [0.3, 0.7] | Loss weight for [background, leucaena] — raise leucaena weight if it is very rare |
| `EARLY_STOP_PATIENCE` | 15 | Stop training if val loss does not improve for this many epochs |
| `TEST_SPLIT` | 0.2 | Fraction of patches held out for final evaluation |
| `VAL_SPLIT` | 0.2 | Fraction of training patches used for validation during training |

> **Tip for beginners:** start by changing only `CLASSES_WEIGHTS` and `LEARNING_RATE`.
> Those two have the largest effect on class-imbalanced problems like this one.

---

## 10. Credits

- Original tree segmentation pipeline: [Felipe Ferrari — tree_fusion](https://github.com/felferrari/tree_fusion)
- Aerial imagery and LiDAR data: donated by [IGC — Instituto Geográfico e Cartográfico de SP](http://www.igc.sp.gov.br/)
- Polygon annotations: collected via [leucaena.earth](https://leucaena.earth) crowdmapping platform
- Research institution: ESALQ/USP — Escola Superior de Agricultura "Luiz de Queiroz", Universidade de São Paulo
