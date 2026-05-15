"""
Tile-based patch preparation (scalable replacement for prep-data.py).

Why this script exists
----------------------
``prep-data.py`` is fine for one small scene (or a VRT mosaic), but it loads
the whole image into RAM and saves a single ``opt_img.npy``. That does not
scale to thousands of tiles spread across Brazil.

This script processes a *folder of tiles* tile by tile, never materialising
the whole study area in RAM. For each tile it:

1. Reads the tile's bbox + CRS via GDAL.
2. Rasterises only the GeoJSON polygons whose bbox intersects the tile
   (``utils.ops.rasterize_geojson_for_tile``).
3. Slides a window over the label mask, keeps patches with at least
   ``--min-target-class`` leucaena pixels.
4. Reads only the chosen windows from the tile (cheap ``ReadAsArray``).
5. Writes each patch as ``opt/<patch_id>.npy`` (uint8) + ``lbl/<patch_id>.npy``
   (uint8) and appends a row to ``manifest.csv``.
6. After all tiles, performs a deterministic train/val/test split at the
   patch level and writes the final ``manifest.csv``.

Normalisation
-------------
Patches stay as ``uint8`` on disk. The dataloader (``PatchFileDataset`` in
``utils.dataloader``) divides by 255.0 at training time. This keeps patches
4x smaller than float32 and avoids a global min/max two-pass over Brazil.

Inputs
------
- ``--tiles-dir``: folder with single-tile multi-band GeoTIFFs (4 bands)
- ``--masks``: GeoJSON with leucaena polygons (anywhere in Brazil)

Outputs
-------
- ``<out-dir>/opt/<patch_id>.npy``  uint8  (H, W, 4)
- ``<out-dir>/lbl/<patch_id>.npy``  uint8  (H, W)
- ``<out-dir>/manifest.csv``         columns described below
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from osgeo import gdal, gdalconst
from skimage.util import view_as_windows

from conf import default, general, paths
from utils.ops import rasterize_geojson_for_tile


@dataclass
class PatchRecord:
    """One row of the manifest. ``split`` is filled after the global shuffle."""

    patch_id: str
    tile_name: str
    row: int
    col: int
    xoff: int
    yoff: int
    win: int
    leucaena_fraction: float
    split: str = ""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate train/val/test patches from a folder of tiles + a GeoJSON.",
    )
    p.add_argument(
        "--tiles-dir",
        default=paths.PATH_TILES_DIR,
        help="Folder containing single-tile GeoTIFFs (default: %(default)s)",
    )
    p.add_argument(
        "--tiles-glob",
        default="*.tif",
        help="Filename pattern under --tiles-dir (default: %(default)s)",
    )
    p.add_argument(
        "--masks",
        default=paths.PATH_MASKS,
        help="GeoJSON with leucaena polygons (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        default=paths.PATH_PATCHES_DIR,
        help="Destination for patches/manifest (default: %(default)s)",
    )
    p.add_argument(
        "--patch-size",
        type=int,
        default=general.PATCH_SIZE,
        help="Square patch side in pixels (default: %(default)s)",
    )
    p.add_argument(
        "--overlap",
        type=float,
        default=general.PATCH_OVERLAP,
        help="Sliding-window overlap fraction in [0, 1) (default: %(default)s)",
    )
    p.add_argument(
        "--min-target-class",
        type=float,
        default=default.MIN_TRAIN_CLASS,
        help="Minimum fraction of leucaena pixels to keep a patch (default: %(default)s)",
    )
    p.add_argument(
        "--test-split",
        type=float,
        default=general.TEST_SPLIT,
        help="Fraction of patches in test split (default: %(default)s)",
    )
    p.add_argument(
        "--val-split",
        type=float,
        default=general.VAL_SPLIT,
        help="Fraction of remaining train patches used for validation "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--band-order",
        choices=("RGBN", "BGRN"),
        default="RGBN",
        help="Order of the 4 source bands; will be rewritten to BGRN to match "
        "conf.general.BAND_NAMES_OPTICAL (default: %(default)s)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val/test shuffle (default: %(default)s)",
    )
    p.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="If set, process only the first N tiles (debug).",
    )
    return p.parse_args()


def _list_tiles(tiles_dir: str, pattern: str, max_tiles: int | None) -> list[str]:
    paths_ = sorted(glob.glob(os.path.join(tiles_dir, pattern)))
    if max_tiles is not None:
        paths_ = paths_[:max_tiles]
    return paths_


def _band_reorder_indices(band_order: str) -> list[int]:
    """Return the 0-based source-band indices needed to produce BGRN output."""
    if band_order == "BGRN":
        return [0, 1, 2, 3]
    if band_order == "RGBN":
        # RGBN means src[0]=R, src[1]=G, src[2]=B, src[3]=N
        # BGRN expected:    dst[0]=B, dst[1]=G, dst[2]=R, dst[3]=N
        return [2, 1, 0, 3]
    raise ValueError(f"Unsupported band order: {band_order}")


def _read_window_BGRN(
    ds: gdal.Dataset, xoff: int, yoff: int, win: int, src_order_idx: list[int]
) -> np.ndarray:
    """Read a (win, win, 4) uint8 window in BGRN order, regardless of source order."""
    out = np.empty((win, win, 4), dtype=np.uint8)
    for dst_b, src_b in enumerate(src_order_idx):
        band = ds.GetRasterBand(src_b + 1)
        arr = band.ReadAsArray(xoff, yoff, win, win)
        if arr is None:
            raise IOError(
                f"GDAL ReadAsArray returned None for band {src_b + 1} at "
                f"({xoff}, {yoff}, {win}, {win})"
            )
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        out[:, :, dst_b] = arr
    return out


def _process_tile(
    tile_path: str,
    geojson_path: str,
    out_dir: str,
    patch_size: int,
    overlap: float,
    min_target_class: float,
    band_order: str,
    log,
) -> list[PatchRecord]:
    tile_name = os.path.splitext(os.path.basename(tile_path))[0]
    log(f"\nTile: {tile_name}")

    ds = gdal.Open(tile_path, gdalconst.GA_ReadOnly)
    if ds is None:
        log(f"  [SKIP] cannot open tile.")
        return []

    width, height = ds.RasterXSize, ds.RasterYSize
    n_bands = ds.RasterCount
    if n_bands != 4:
        log(f"  [SKIP] expected 4 bands, got {n_bands}.")
        ds = None
        return []

    label, n_features = rasterize_geojson_for_tile(geojson_path, tile_path)
    log(f"  shape={width}x{height} | features used={n_features}")
    if n_features == 0:
        ds = None
        return []

    step = max(1, int((1 - overlap) * patch_size))
    label_windows = view_as_windows(label, (patch_size, patch_size), step)
    grid_rows, grid_cols = label_windows.shape[:2]
    flat = label_windows.reshape(-1, patch_size, patch_size)
    fraction = np.mean(flat == 1, axis=(1, 2))
    keep_mask = fraction >= min_target_class
    n_total = flat.shape[0]
    n_keep = int(keep_mask.sum())
    log(
        f"  windows={n_total:,} | step={step} | keep "
        f"(leucaena>={min_target_class:.2%}) {n_keep:,}"
    )
    if n_keep == 0:
        ds = None
        return []

    src_order_idx = _band_reorder_indices(band_order)

    opt_dir = os.path.join(out_dir, "opt")
    lbl_dir = os.path.join(out_dir, "lbl")
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    records: list[PatchRecord] = []
    grid_idx_flat = np.arange(grid_rows * grid_cols).reshape(grid_rows, grid_cols)
    rows_kept, cols_kept = np.unravel_index(
        np.flatnonzero(keep_mask), (grid_rows, grid_cols)
    )
    for r, c in zip(rows_kept, cols_kept):
        xoff = int(c * step)
        yoff = int(r * step)
        if xoff + patch_size > width or yoff + patch_size > height:
            continue
        patch_id = f"{tile_name}__r{yoff:06d}_c{xoff:06d}"

        opt_patch = _read_window_BGRN(ds, xoff, yoff, patch_size, src_order_idx)
        lbl_patch = label[yoff : yoff + patch_size, xoff : xoff + patch_size].astype(
            np.uint8
        )

        np.save(os.path.join(opt_dir, f"{patch_id}.npy"), opt_patch)
        np.save(os.path.join(lbl_dir, f"{patch_id}.npy"), lbl_patch)

        records.append(
            PatchRecord(
                patch_id=patch_id,
                tile_name=tile_name,
                row=int(r),
                col=int(c),
                xoff=xoff,
                yoff=yoff,
                win=patch_size,
                leucaena_fraction=float(fraction[grid_idx_flat[r, c]]),
            )
        )

    ds = None
    log(f"  wrote {len(records):,} patches")
    return records


def _assign_splits(
    records: list[PatchRecord],
    test_split: float,
    val_split: float,
    seed: int,
) -> None:
    if not records:
        return
    n_total = len(records)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_total)
    n_test = int(round(test_split * n_total))
    n_val = int(round(val_split * (n_total - n_test)))
    test_idx = set(order[:n_test].tolist())
    val_idx = set(order[n_test : n_test + n_val].tolist())
    for i, rec in enumerate(records):
        if i in test_idx:
            rec.split = "test"
        elif i in val_idx:
            rec.split = "val"
        else:
            rec.split = "train"


def _write_manifest(records: Iterable[PatchRecord], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    records = list(records)
    if not records:
        # Still write a header-only file so downstream tools detect the empty run.
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "patch_id",
                    "tile_name",
                    "row",
                    "col",
                    "xoff",
                    "yoff",
                    "win",
                    "leucaena_fraction",
                    "split",
                ]
            )
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def main() -> None:
    args = _parse_args()
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "preparation.txt")
    with open(log_path, "w") as log_f:

        def log(msg: str) -> None:
            print(msg, flush=True)
            log_f.write(msg + "\n")
            log_f.flush()

        log(f"tiles_dir         : {args.tiles_dir}")
        log(f"tiles_glob        : {args.tiles_glob}")
        log(f"masks             : {args.masks}")
        log(f"out_dir           : {out_dir}")
        log(f"patch_size        : {args.patch_size}")
        log(f"overlap           : {args.overlap}")
        log(f"min_target_class  : {args.min_target_class}")
        log(f"band_order (src)  : {args.band_order} -> stored as BGRN")
        log(f"splits            : test={args.test_split}  val={args.val_split}")
        log(f"seed              : {args.seed}")

        if not os.path.isdir(args.tiles_dir):
            log(f"[ABORT] tiles dir not found: {args.tiles_dir}")
            sys.exit(2)
        if not os.path.isfile(args.masks):
            log(f"[ABORT] masks file not found: {args.masks}")
            sys.exit(2)

        tile_paths = _list_tiles(args.tiles_dir, args.tiles_glob, args.max_tiles)
        log(f"Found {len(tile_paths)} tile(s)")
        if not tile_paths:
            log("[ABORT] no tiles matched.")
            sys.exit(2)

        t0 = time.time()
        all_records: list[PatchRecord] = []
        for i, tile_path in enumerate(tile_paths, 1):
            log(f"\n[{i}/{len(tile_paths)}] {tile_path}")
            try:
                records = _process_tile(
                    tile_path=tile_path,
                    geojson_path=str(args.masks),
                    out_dir=out_dir,
                    patch_size=args.patch_size,
                    overlap=args.overlap,
                    min_target_class=args.min_target_class,
                    band_order=args.band_order,
                    log=log,
                )
            except Exception as exc:  # noqa: BLE001 - keep processing other tiles
                log(f"  [ERROR] {type(exc).__name__}: {exc}")
                records = []
            all_records.extend(records)

        log("\n----- splits -----")
        _assign_splits(all_records, args.test_split, args.val_split, args.seed)
        n_train = sum(1 for r in all_records if r.split == "train")
        n_val = sum(1 for r in all_records if r.split == "val")
        n_test = sum(1 for r in all_records if r.split == "test")
        log(f"total : {len(all_records):,}")
        log(f"train : {n_train:,}")
        log(f"val   : {n_val:,}")
        log(f"test  : {n_test:,}")

        manifest_path = os.path.join(out_dir, "manifest.csv")
        _write_manifest(all_records, manifest_path)
        log(f"\nmanifest -> {manifest_path}")
        log(f"elapsed : {(time.time() - t0):.1f} s")


if __name__ == "__main__":
    main()
