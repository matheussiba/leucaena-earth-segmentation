# 01 — Migrate tree_fusion to leucaena binary segmentation with GeoJSON masks

- **Status:** `built`
- **Owner:** Matheus
- **Last update:** 2026-05-15
- **Upstream:** [felferrari/tree_fusion](https://github.com/felferrari/tree_fusion)

## Why

The starting point for this PhD pipeline was `tree_fusion`, provided by Prof.
Ferrari. That code targets a different problem: **multi-class species
segmentation** trained from **pre-rasterised label TIFFs**. For the leucaena
PhD project we need:

- **Binary** segmentation: leucaena (1) vs background (0).
- Labels coming from **GeoJSON polygons** drawn on
  [leucaena.earth](https://leucaena.earth), not from pre-baked rasters.
- A single command to ingest those polygons + a 4-band image and emit the
  arrays the trainer expects, without manual rasterisation in QGIS.

This plan documents the migration from `tree_fusion`'s data shape to the
current binary-with-GeoJSON pipeline that all later plans build upon.

## Starting point (tree_fusion)

- [`prep-data.py`](https://github.com/felferrari/tree_fusion/blob/main/prep-data.py)
  reads three TIFFs: `RGBNIR_allLiDAR_norm.tif`, `imgTrain_1.tif`,
  `imgTest_1.tif`.
- [`conf/general.py`](https://github.com/felferrari/tree_fusion/blob/main/conf/general.py)
  uses `N_CLASSES = 10 - len(REMOVED_CLASSES)`, optical band order
  `BLUE, RED, GREEN, NIR`, six LiDAR bands (`nx, ny, nz, curvature,
  intensity, chm`), patch `128`, overlap `0.9`.
- Class weights are per-species (`CLASSES_WEIGHTS = [B_W, T_W, ... 0]`).
- `prep-data.py` filters patches whose **non-zero, non-discarded** label
  fraction is above `MIN_TRAIN_CLASS`, then splits into train/val only.

## Decisions for the migration

- **Inputs become** a 4-band optical GeoTIFF + a GeoJSON of leucaena
  polygons + an optional LiDAR GeoTIFF. No pre-rasterised label files.
- **`utils/ops.py:rasterize_geojson`** added: opens GeoJSON via OGR,
  reprojects features to the optical raster CRS if needed, burns polygons
  to a `uint8` mask aligned to the optical grid.
- **Binary scheme**: `0 = background`, `1 = leucaena`, `255 = IGNORE_INDEX`
  for pixels that fall outside the test mask in held-out tiles.
- **Optical band order canonicalised to** `BLUE, GREEN, RED, NIR` (BGRN)
  in [`conf/general.py`](../conf/general.py) — this is what
  `prep-patches-from-tiles.py` in plan 03 also writes to disk.
- **Class weights** become `[0.3, 0.7]` (background, leucaena) in
  [`conf/general.py`](../conf/general.py).
- **Patch geometry** moved to `PATCH_SIZE = 256`, `PATCH_OVERLAP = 0.5`
  (larger context, less duplication than the upstream `128 / 0.9`).
- **Per-band min-max normalisation** added to `prep-data.py` so the scene
  feeds the network as `[0, 1]` float32 even when GeoTIFFs come from
  different sensors/calibrations.
- **Outlier clipping** kept (`filter_outliers` in
  [`utils/ops.py`](../utils/ops.py)) — useful with raw IGC imagery.
- **`--no-lidar`** flag in `prep-data.py` so experiment 1 works on
  optical-only input (LiDAR is replaced with a single zero band).
- **Train/test/val split** moves into `prep-data.py` itself; downstream
  code does not need separate hand-curated `imgTrain_X.tif` / `imgTest_X.tif`.

## Files touched

- [`prep-data.py`](../prep-data.py) — rewritten to consume GeoJSON masks
  and produce binary `train/val/test` patches.
- [`utils/ops.py`](../utils/ops.py) — new `rasterize_geojson`, kept
  `filter_outliers` and `save_geotiff` helpers.
- [`conf/general.py`](../conf/general.py) — binary task, `IGNORE_INDEX`,
  new band order, new patch geometry, new class weights, early-stopping
  defaults tuned for binary task.
- [`conf/paths.py`](../conf/paths.py) — local paths (`data/optical.tif`,
  `data/masks.geojson`, `data/lidar.tif`) instead of `D:/Ferrari/...`.
- [`conf/model_1.py`](../conf/model_1.py), [`conf/model_2.py`](../conf/model_2.py),
  [`conf/model_3.py`](../conf/model_3.py) — kept the optical / early-fusion /
  late-fusion variants used by the PhD experiments; models 4..9 from the
  upstream are dropped.
- [`utils/dataloader.py`](../utils/dataloader.py),
  [`utils/trainer.py`](../utils/trainer.py),
  [`evaluation.py`](../evaluation.py),
  [`prediction.py`](../prediction.py) — adjusted to the binary contract
  (`CrossEntropyLoss` with `ignore_index=IGNORE_INDEX`, `[0.3, 0.7]`
  weights, F1 for class 1).
- [`README.md`](../README.md) — beginner-friendly rewrite explaining
  segmentation, ResUNet, fusion and the four scripts.
- `.gitignore` — adds `data/`, `prepared/`, `experiments/`, raster/vector
  artifacts.

## How to run

```bash
python prep-data.py --optical data/optical.tif --masks data/masks.geojson --no-lidar
python train.py -e 1 -b 8
python prediction.py -e 1
python evaluation.py -e 1
```

(Same four commands documented in
[`README.md`](../README.md) §6.)

## Risks / mitigation (recorded at the time)

| Risk | Mitigation |
|------|------------|
| GeoJSON CRS differs from raster | OGR reprojection inside `rasterize_geojson` |
| Old multi-class config bleeds through | All references to `REMOVED_CLASSES` removed; only 2 classes left |
| LiDAR optional, but the model expects two inputs | Dummy single-band LiDAR raster when `--no-lidar`; experiment 1 ignores it via `lidar_bands=None` |
| Train/test mask leakage | Hold-out test patches sampled from the same scene; `IGNORE_INDEX` everywhere else so loss/metrics skip those pixels |
| Class imbalance | Higher weight on leucaena (`[0.3, 0.7]`) + min-target fraction `0.01` per training patch |

## Outcome

- Repository renamed to `leucaena-earth-segmentation` and made the canonical
  starting point for the PhD pipeline.
- Four scripts in the root (`prep-data → train → prediction → evaluation`)
  reproduce the binary segmentation flow with a single command each.
- This forms the foundation that plans 02 (Docker), 03 (tile-based patches)
  and 04 (predict-tiles, HDF5/Zarr, LiDAR in tile-based) extend.

## Out of scope (handled later)

- Reproducible container for the binary pipeline → plan 02.
- Scaling beyond one scene that fits in RAM → plan 03.
- Country-wide inference + storage migration + LiDAR in tile-based → plan 04.
