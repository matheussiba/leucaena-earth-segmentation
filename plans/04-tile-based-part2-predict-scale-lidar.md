# 04 — Tile-based pipeline part 2: predict-tiles, HDF5/Zarr, LiDAR

- **Status:** `partial` — 3C done (LAZ → patch); 3A and 3B still pending.
- **Owner:** Matheus
- **Last update:** 2026-05-16

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

---

## Outcome — 3C built (LiDAR no caminho tile-based)

**Implemented on branch `plan-04-predict-tiles-h5-lidar`** (2026-05-16).

### Problem we had

The tile-based path produced training patches only for **optical** (`opt/`)
and **labels** (`lbl/`). `PatchFileDataset` returned a *zero tensor* in
place of LiDAR, which kept `train.py -e 1` (optical-only) working but
left `-e 2` (early fusion) and `-e 3` (late fusion) effectively
disabled — the network would see all-zero LiDAR and learn nothing from
it.

We also had raw `.laz` point clouds in `D:\laz` (~300 IBGE tiles), not
the multi-band LiDAR GeoTIFFs the pipeline expects. So before we could
enable fusion experiments we needed a *LAZ → raster* preprocessing step.

### What was built

1. **`prep-lidar-rasters.py`** (new top-level script)
   - Walks `--laz-dir` for `.laz` / `.copc.laz` files.
   - For each LAZ:
     - Looks up the matching RGBN tile in `--tiles-dir` by stem
       (`A-B-C.copc.laz` -> `A-B-C.tif`, with `.copc` stripped).
     - Reads the RGBN geotransform → uses it as the **target grid**
       (extent, pixel size, CRS) for the output.
     - Runs a single tagged PDAL pipeline that reads the LAZ once and
       fans out into three writers (DSM = max Z of all clean returns,
       DTM = min Z of SMRF-classified ground returns, Intensity =
       mean raw intensity).
     - Computes `CHM = clip(DSM − DTM, 0, LIDAR_CHM_MAX_M)` in numpy,
       treating no-data as 0 m (open ground, the safe default for a
       vegetation feature).
     - Warps CHM (bilinear) and INTENSITY (average) to the *exact*
       RGBN grid via `gdal.Warp`.
     - Stacks both bands into a single 2-band float32 GeoTIFF at
       `<out-dir>/<stem>.tif` (band order `[CHM, INTENSITY]` matches
       `general.BAND_NAMES_LIDAR`).
   - Per-tile errors are caught and logged so a single bad LAZ does
     not abort the batch.
   - Records the result of every LAZ in `lidar_manifest.csv` with
     buckets `ok` / `skip-no-rgbn` / `skip-existing` / `error`,
     plus `n_points`, output dimensions, elapsed seconds and the
     error message.
   - Supports two *scales*:
     - `--max-tiles 2` — smoke run on 1–3 tiles before launching the
       full batch (cheap toolchain validation).
     - default — every LAZ that has a matching RGBN.
   - `--inspect-only` runs PDAL `filters.stats` and prints metadata
     without writing any raster (useful for first-time setup to
     verify CRS / point count / class histogram).
   - `--no-require-rgbn` produces standalone (native-bounds) rasters
     for the LAZ files that have no RGBN counterpart — useful for
     inspection in QGIS, **not** usable for training.

2. **`utils/lidar.py`** (new module)
   - `TileGrid.from_geotiff()` — reads a RGBN tile and packages
     extent + size + CRS into a single dataclass we can pass around.
   - `build_pdal_pipeline()` — composes the JSON pipeline above
     (reproject → drop class 7/18 noise → SMRF → 3 GDAL writers).
   - `compute_chm()` — numpy DSM/DTM subtraction with no-data
     handling and CHM cap.
   - `warp_to_grid()` — `gdal.Warp` wrapper that pins width / height /
     bounds / CRS, removing any chance of subtle sub-pixel drift
     between RGBN and LiDAR.
   - `stack_chm_intensity()` — writes the final 2-band float32
     GeoTIFF (DEFLATE, tiled, PREDICTOR=3) with `BAND_NAMES_LIDAR`
     metadata.
   - `process_laz_to_lidar_tif()` — orchestrates the three steps end
     to end and cleans up the per-tile tmpdir.
   - `inspect_laz()` — used by `--inspect-only` mode.

3. **`prep-patches-from-tiles.py`** (edited)
   - New flags `--lidar-dir` and `--lidar-glob`.
   - When `--lidar-dir` is set, for every kept patch the script also
     reads the same window from the matching LiDAR raster and writes
     `lidar/<patch_id>.npy` (float32, `(H, W, 2)`).
   - LiDAR matching is by tile stem (same convention as the RGBN ↔
     LAZ matching). Missing LiDAR is **non-fatal**: optical patches
     are still written and the manifest row carries `has_lidar=False`
     so the dataloader can fall back gracefully.
   - Two new manifest columns: `lidar_tile_name`, `has_lidar`. The
     manifest header is now derived from the dataclass field order
     to keep schemas stable across runs.

4. **`utils/dataloader.py`** (edited)
   - `PatchFileDataset` reads `lidar/<patch_id>.npy` when it exists
     (the manifest `has_lidar` column is the authoritative signal;
     filesystem presence is the fallback when the column is missing,
     to stay compatible with older manifests).
   - Real LiDAR is scaled to `[0, 1]` per band using the fixed
     constants `LIDAR_CHM_MAX_M` (CHM) and `LIDAR_INTENSITY_MAX`
     (everything else) added to `conf/general.py`. CHM is identified
     by its name in `BAND_NAMES_LIDAR`, not by its index, so the
     band order stays editable.
   - The RAM cache path was extended to keep the LiDAR array alongside
     `opt` / `lbl`, with a sensible 0-byte path when LiDAR is absent.
   - All augmentations (rotation, hflip, vflip) already operated on
     the LiDAR tensor in the placeholder code, so no changes there.

5. **`conf/paths.py`** (edited) — three new constants:
   - `PATH_LAZ_DIR`     -> env `LEUCAENA_LAZ_DIR`    (default `/data/laz`)
   - `PATH_LIDAR_DIR`   -> env `LEUCAENA_LIDAR_DIR`  (default `/data/lidar`)
   - `PATH_LIDAR_MANIFEST` -> `<PATH_LIDAR_DIR>/lidar_manifest.csv`

6. **`conf/general.py`** (edited) — three new constants:
   - `LIDAR_RASTER_RESOLUTION_M = 1.0`
   - `LIDAR_CHM_MAX_M = 50.0`
   - `LIDAR_INTENSITY_MAX = 32768.0`

7. **`Dockerfile`** (edited) — installs `pdal` + `python-pdal` from
   conda-forge in the same step as `gdal`. Adds roughly 500 MB to the
   image but keeps the LiDAR pipeline fully reproducible.

8. **`docker-compose.yml`** (edited) — two new bind mounts and two new
   env vars so the container sees `/data/laz` and `/data/lidar`.

9. **`.env.example` + `.env`** (edited) — added
   `LEUCAENA_LAZ_HOST_DIR`, `LEUCAENA_LIDAR_HOST_DIR`,
   `LEUCAENA_LAZ_DIR`, `LEUCAENA_LIDAR_DIR`.

10. **`CHEATSHEET.md`** (edited) — full *Pipeline LiDAR — do `.laz` ao
    patch* section with diagrams, exact commands for the **teste**
    and **final** scales, every CLI flag, and where each constant
    lives.

11. **`README.md`** (edited) — short paragraph linking to the new
    LiDAR rasterisation step.

### How to run (TL;DR)

```bash
# 1. Build the image once (now includes PDAL).
docker compose build

# 2. Smoke run: 2 LAZ, validates PDAL + GDAL warp + alignment.
docker compose run --rm segmentation \
    python prep-lidar-rasters.py --max-tiles 2

# 3. Full batch (every LAZ with a matching RGBN).
docker compose run --rm segmentation \
    python prep-lidar-rasters.py

# 4. Generate patches *with* LiDAR.
docker compose run --rm segmentation \
    python prep-patches-from-tiles.py --lidar-dir /data/lidar --band-order RGBN

# 5. Train early or late fusion.
docker compose run --rm segmentation \
    python train.py -e 2 -b 8 --patch-source file
```

### Decisions (and why)

- **PDAL via conda-forge, not pip.** PDAL is a heavy C++ library; the
  pip wheels are flaky on Windows / WSL. Conda-forge ships matching
  PROJ / GEOS / GDAL versions with the conda environment, which is
  what the rest of the image already uses.
- **DSM/DTM/Intensity in a single PDAL pipeline, fan-out via tags.**
  Three separate pipelines would re-read the LAZ three times; a
  tagged pipeline reads it once and ~3× the runtime is saved on big
  tiles.
- **Native PDAL resolution 1 m, then warp to RGBN.** Writing PDAL
  directly at 25 cm produces sparse, hole-ridden rasters for typical
  4 pts/m² aerial LiDAR. 1 m is dense enough to be meaningful; the
  warp upsamples (bilinear for CHM, average for Intensity) onto the
  optical grid.
- **CHM clipped at 50 m.** Treetop returns past 50 m are almost always
  spikes (mirror surfaces, birds, scanner noise). Capping protects
  the network from divide-by-50 producing values > 1.
- **Fixed scaling in the dataloader.** Per-tile min-max would mean
  the same canopy height looks different across tiles. A fixed
  divisor keeps "5 m of CHM" consistent everywhere.
- **`has_lidar` per-row in the manifest.** Mixed datasets (some tiles
  have LiDAR, some do not) are a realistic scenario in this PhD;
  encoding it explicitly avoids subtle bugs where `-e 2` ends up
  training on a mostly-zero LiDAR stream without telling anyone.

### Known limitations / next steps

- LAZ ↔ RGBN matching is only by stem. A `--mapping-csv` flag would
  cover the case where the two sources use different naming
  conventions (e.g. IBGE charts vs city lots). Not needed for the
  current PhD AOI; left as a TODO.
- The patch-level split in `prep-patches-from-tiles.py` does not
  honour tile boundaries: patches from the same tile can fall into
  train and test. A `--split-by tile` flag is still pending (also
  listed as a risk in this plan above).
- Sections **3A (predict-tiles.py)** and **3B (HDF5/Zarr migration)**
  remain unimplemented. The decisions section at the top of this
  plan still applies; nothing in 3C blocks either.
- Image size grew by ~500 MB because of PDAL. Acceptable for now;
  a slim image variant without PDAL would be the right move once
  LiDAR rasters are stable on disk and you only need training.
