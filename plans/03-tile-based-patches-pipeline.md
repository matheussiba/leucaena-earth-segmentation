# 02 — Tile-based patches pipeline

- **Status:** `built`
- **Owner:** Matheus
- **Last update:** 2026-05-15

## Why

`prep-data.py` loads one full scene into RAM and writes a single
`opt_img.npy`. That is fine for the 293 km² training AOI but does not scale
to nationwide data (~293 000 km², thousands of tiles, terabytes). The
sliding-window inside `prep-data.py` also forces a global min-max
normalisation that cannot be computed in a single pass over Brazil.

A complementary pipeline is added: iterate **one tile at a time**, write
per-patch `.npy` files plus a CSV manifest with deterministic
`train/val/test` splits. Same training code, different data source.

## Decisions

- Patches stored on disk as `uint8` (4× smaller than `float32`);
  normalisation `/255.0` is done by the dataloader at training time.
- Splits assigned at the **patch level**, with a fixed `--seed`. Patches
  from the same tile may end up in different splits — known limitation,
  see plan 04.
- LiDAR is **not** part of v1 of this pipeline; the dataloader returns a
  zero LiDAR tensor so the `((opt, lidar), label)` contract stays intact
  and experiment 1 works unchanged.
- Source band order is configurable (`--band-order RGBN` is the default to
  match the fusion script; it is rewritten to BGRN before saving, matching
  `conf/general.py`).
- `rasterize_geojson_for_tile` uses OGR `SetSpatialFilter` over the tile
  bbox so a Brazil-wide GeoJSON stays cheap per tile.

## Files touched

- New: [`prep-patches-from-tiles.py`](../prep-patches-from-tiles.py)
- New helper in [`utils/ops.py`](../utils/ops.py): `rasterize_geojson_for_tile`
- New class in [`utils/dataloader.py`](../utils/dataloader.py): `PatchFileDataset`
- [`train.py`](../train.py): `--patch-source {scene,file}` (default `scene`)
  and `--manifest`
- [`conf/paths.py`](../conf/paths.py): `PATH_TILES_DIR`, `PATH_PATCHES_DIR`,
  `PATH_PATCHES_MANIFEST`
- [`docker-compose.yml`](../docker-compose.yml): mount `D:/leucaena:/data`
- [`CHEATSHEET.md`](../CHEATSHEET.md) and [`README.md`](../README.md): docs

## Output layout

```
prepared/
  patches/
    opt/<patch_id>.npy   # uint8 (H, W, 4) BGRN
    lbl/<patch_id>.npy   # uint8 (H, W) 0/1
    manifest.csv         # one row per patch, with split column
    preparation.txt      # per-tile counts log
```

Manifest columns:
`patch_id, tile_name, row, col, xoff, yoff, win, leucaena_fraction, split`.

## How to run

Organise data once on the Windows host:

```
D:\leucaena\rgbir\*.tif
D:\leucaena\masks\leucaena_polygons.geojson
```

Inside the container:

```bash
python prep-patches-from-tiles.py \
  --tiles-dir /data/rgbir \
  --masks    /data/masks/leucaena_polygons.geojson \
  --band-order RGBN

python train.py -e 1 -b 8 --patch-source file
```

## Risks / mitigation

| Risk | Mitigation |
|------|------------|
| Huge GeoJSON | `SetSpatialFilter` by tile bbox, reproject only filtered features |
| Tile with no leucaena | OK; manifest empty for that tile (negative sampling = plan 04) |
| Many small `.npy` files | OK up to ~100k patches; migrate to HDF5/Zarr (plan 04) |
| Wrong band order | `--band-order` flag + assert `n_bands == 4` |
| Patch/split leakage | Document; tile-level split is plan 04 |
| `D:\leucaena` missing | Documented in CHEATSHEET; script aborts cleanly |

## Outcome

Implementation merged. Pending real-data run on the user's 17 fused RGBN
tiles + nationwide GeoJSON to validate end-to-end (just before the next
training run).

## Out of scope (-> plan 04)

- `predict-tiles.py` for tile-by-tile inference at country scale.
- HDF5/Zarr storage when patch count grows.
- LiDAR in the manifest.
- Tile-level (not patch-level) split.
- Negative sampling for tiles without leucaena.
