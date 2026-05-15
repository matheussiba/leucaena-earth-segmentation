# 04 — Tile-based pipeline part 2: predict-tiles, HDF5/Zarr, LiDAR

- **Status:** `not started`
- **Owner:** Matheus
- **Last update:** 2026-05-15

## Why

Plan 03 made training scale (tile-by-tile patch generation + per-patch
files). What is still missing to use the model on the whole Brazil:

1. **Inference at country scale** — current `prediction.py` reads the
   prepared full scene from `prepared/opt_img.npy`. To map ~293 000 km²
   that won't fit. We need a runner that walks a folder of tiles and
   produces one prediction GeoTIFF per tile.
2. **Patch storage** — once `prep-patches-from-tiles.py` runs on the full
   country, the number of patches will pass ~100k easily and a folder of
   `.npy` files becomes painful (filesystem listing, backup, transfer).
   Migrate to a chunked container (HDF5 or Zarr).
3. **LiDAR support in the tile-based path** — today LiDAR is bypassed
   (zero tensor) so only experiment 1 is reachable. Add a parallel
   `lidar.npy` per patch and the manifest columns to enable experiments
   2 and 3 on tile-based data.

## Decisions to make (open)

- HDF5 vs Zarr. Zarr is nicer with cloud / object storage; HDF5 is more
  common in PyTorch tutorials. Decide before implementation.
- Tile-level split (group patches from the same tile in the same split)
  vs patch-level. Tile-level is more honest; patch-level is simpler.
- Inference output: one GeoTIFF per tile, or per-tile + final VRT mosaic?
  VRT mosaic is essentially free and convenient for QGIS.

## Scope

### 3A. `predict-tiles.py`

CLI (proposed):

```text
--tiles-dir         folder with RGBN tiles
--tiles-glob        default "*.tif"
--model-path        experiments/exp_<N>/models/model.pt
--out-dir           default experiments/exp_<N>/predicted_tiles
--overlap           sliding-window overlap fraction (e.g. 0.25)
--band-order        RGBN|BGRN
--batch-size        default 128
--device            auto / cuda / cpu
```

Per-tile workflow:

1. Open the tile via GDAL; read in BGRN order.
2. Reflect-pad once by `PATCH_SIZE` (mirrors current `TreePredDataSet`).
3. Stream `PATCH_SIZE × PATCH_SIZE` windows; accumulate softmax averages.
4. Save `<tile>_pred.tif` (uint8 class map) and optionally
   `<tile>_prob.tif` (float32 class-1 probability).
5. After all tiles: `gdalbuildvrt predicted.vrt *_pred.tif` for a single
   mosaic layer in QGIS.

Reuse: ResUnet `get_model()`, `TreePredDataSet` patch averaging logic
factored out into `utils/inference.py`.

### 3B. HDF5 / Zarr migration

- New script `prep-patches-to-h5.py` (or `to_zarr.py`) that consumes the
  current manifest and writes one file per split:
  - `train.h5` with `opt` (N×H×W×4 uint8) and `lbl` (N×H×W uint8).
  - `val.h5`, `test.h5` similarly.
- New `PatchH5Dataset` in `utils/dataloader.py`; `train.py` gains
  `--patch-store {files,h5}` (default `files`).
- Keep the file-based path working — useful for debugging and small AOIs.

### 3C. LiDAR in tile-based

- `prep-patches-from-tiles.py` gains `--lidar-dir` and `--lidar-glob`. For
  each RGBN tile, it looks up a matching LiDAR tile (same stem or via a
  manifest mapping) and writes `lidar/<patch_id>.npy` (uint8 or float16).
- Manifest columns add: `lidar_tile_name`, `lidar_xoff`, `lidar_yoff`.
- `PatchFileDataset` reads the LiDAR patch instead of returning zeros.

## Files to create / change

- New: `predict-tiles.py`
- New: `prep-patches-to-h5.py` (or `to_zarr.py`)
- Maybe new: `utils/inference.py` (extracted from `prediction.py`)
- Edit: `utils/dataloader.py` (`PatchH5Dataset`)
- Edit: `train.py` (`--patch-store`)
- Edit: `prep-patches-from-tiles.py` (LiDAR options + columns)
- Docs: extend [`CHEATSHEET.md`](../CHEATSHEET.md) with a "Scale" section.

## Risks / unknowns

| Risk | Mitigation |
|------|------------|
| Tile boundary artefacts | High overlap (0.5) + softmax averaging in `predict-tiles.py` |
| LiDAR tile naming inconsistent with RGBN | Optional CSV mapping table |
| HDF5 file size | Use chunked storage + LZF/Blosc compression |
| Patches across split (leakage) | Add `--split-by tile` flag |
| Inference saturates disk | Write per-tile, parallelise per GPU only |

## How to evaluate before merging

- Run `predict-tiles.py` on the 17 fused tiles; visually inspect overlap
  with training masks in QGIS.
- Compare metrics from `prediction.py` (single VRT) vs `predict-tiles.py`
  (per tile) on the same AOI.
- Time `prep-patches-to-h5.py` on the local AOI; ensure training time
  stays within 1.2× of `--patch-store files`.

## Out of scope (-> future plan 05)

- Domain adaptation per region (different sensors / years).
- Active learning loops on `leucaena.earth`.
- Mosaicked country-wide GeoTIFF (likely never needed; VRT is enough).
