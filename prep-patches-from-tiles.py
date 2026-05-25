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
7. Exports ``patch_footprints.geojson`` so the patch grid can be inspected
   directly in QGIS.

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
- ``<out-dir>/opt/<patch_id>.npy``    uint8    (H, W, 4)
- ``<out-dir>/lbl/<patch_id>.npy``    uint8    (H, W)
- ``<out-dir>/lidar/<patch_id>.npy``  float32  (H, W, k)  *(only when --lidar-dir is set)*
- ``<out-dir>/manifest.csv``           columns described below
- ``<out-dir>/patch_footprints.geojson`` one polygon per patch for QGIS inspection

LiDAR mode (optional, also enables refined labelling)
-----------------------------------------------------
When ``--lidar-dir`` is set, the script also extracts the same windows from
the matching LiDAR raster (produced by ``prep-lidar-rasters.py``). The match
is by stem: ``<tile_name>.tif`` in ``--tiles-dir`` pairs with
``<tile_name>.tif`` in ``--lidar-dir``.

Setting ``--lidar-dir`` ALSO activates the **refined labelling rule**
(professor's suggestion to reduce class confusion). Each pixel becomes:

- ``IGNORE_INDEX (255)`` when **outside every polygon** (loss + metrics skip it);
- ``1`` when **inside a polygon** AND ``CHM >= LEUCAENA_CHM_MIN_M`` AND
  ``NDVI >= LEUCAENA_NDVI_MIN`` (confirmed leucaena);
- ``0`` when **inside a polygon** but the CHM/NDVI checks fail (clearing /
  bare ground / tower / antenna — actively taught as NOT-leucaena).

Tiles without a matching LiDAR raster are **skipped entirely** when
``--lidar-dir`` is set (we never train on un-refinable polygons in this
mode). Without ``--lidar-dir`` the script keeps the legacy behaviour
(polygon interior = 1, everything else = 0) for backwards compatibility.

NDVI is computed directly from the RGBN tile:
``NDVI = (NIR - RED) / (NIR + RED)``.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import numpy as np
from osgeo import gdal, gdalconst
from skimage.util import view_as_windows

from conf import default, general, paths
from utils.ops import rasterize_geojson_for_tile


@dataclass
class PatchRecord:
    """One row of the manifest. ``split`` is filled after the global shuffle.

    LiDAR-specific fields (``lidar_tile_name``, ``has_lidar``) stay at their
    defaults when ``--lidar-dir`` is not used, which keeps backwards
    compatibility with manifests produced before LiDAR support.

    Refined-label fields:
    - ``polygon_fraction``: share of pixels with ``label != IGNORE_INDEX``
      (i.e. inside any annotated polygon). 1.0 in the legacy path.
    - ``leucaena_fraction``: share of pixels with ``label == 1`` (confirmed
      leucaena after CHM + NDVI refinement when applicable).
    """

    patch_id: str
    tile_name: str
    row: int
    col: int
    xoff: int
    yoff: int
    win: int
    leucaena_fraction: float
    polygon_fraction: float = 1.0
    split: str = ""
    lidar_tile_name: str = ""
    has_lidar: bool = False


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
    p.add_argument(
        "--lidar-dir",
        default=None,
        help="Folder with LiDAR GeoTIFFs aligned to the RGBN tiles "
        "(produced by prep-lidar-rasters.py). Same stem as the RGBN tile. "
        "If set, lidar/<patch_id>.npy is also written. "
        "Default (None) keeps the optical-only behaviour.",
    )
    p.add_argument(
        "--lidar-glob",
        default="*.tif",
        help="Filename pattern under --lidar-dir (default: %(default)s)",
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


def _read_window_lidar(
    ds: gdal.Dataset, xoff: int, yoff: int, win: int
) -> np.ndarray:
    """Read a (win, win, n_bands) float32 window from a LiDAR raster.

    No-data pixels become 0 so downstream training has well-defined inputs.
    """
    n_bands = ds.RasterCount
    out = np.empty((win, win, n_bands), dtype=np.float32)
    for b in range(n_bands):
        band = ds.GetRasterBand(b + 1)
        arr = band.ReadAsArray(xoff, yoff, win, win)
        if arr is None:
            raise IOError(
                f"GDAL ReadAsArray returned None for LiDAR band {b + 1} at "
                f"({xoff}, {yoff}, {win}, {win})"
            )
        arr = arr.astype(np.float32, copy=False)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            arr[np.isclose(arr, float(nodata))] = 0.0
        arr[~np.isfinite(arr)] = 0.0
        out[:, :, b] = arr
    return out


def _find_lidar_tile(tile_name: str, lidar_dir: Optional[str]) -> Optional[str]:
    """Look up ``<lidar_dir>/<tile_name>.tif`` (or ``.tiff``). Returns None if missing."""
    if not lidar_dir:
        return None
    for ext in (".tif", ".tiff"):
        candidate = os.path.join(lidar_dir, tile_name + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def _compute_ndvi_from_tile(ds: gdal.Dataset, band_order: str) -> np.ndarray:
    """Read RED and NIR from the full tile and return NDVI as float32 H x W.

    NDVI = (NIR - RED) / (NIR + RED), with the denominator clipped at a tiny
    epsilon so we never divide by zero. Non-finite results (e.g. when both
    bands are 0) become 0.0 so they fail the ``NDVI >= threshold`` check
    naturally.
    """
    if band_order == "RGBN":
        red_idx, nir_idx = 1, 4  # GDAL bands are 1-indexed: R=1, G=2, B=3, N=4
    elif band_order == "BGRN":
        red_idx, nir_idx = 3, 4
    else:
        raise ValueError(f"Unsupported band order: {band_order}")

    red = ds.GetRasterBand(red_idx).ReadAsArray().astype(np.float32)
    nir = ds.GetRasterBand(nir_idx).ReadAsArray().astype(np.float32)
    denom = nir + red
    denom = np.where(denom == 0.0, 1e-6, denom)
    ndvi = (nir - red) / denom
    ndvi[~np.isfinite(ndvi)] = 0.0
    return ndvi


def _read_full_lidar_band(ds: gdal.Dataset, band_idx_1based: int) -> np.ndarray:
    """Read one full LiDAR band as float32 H x W, replacing no-data with 0.0.

    Used by the label refinement step to fetch the CHM (band 1) for the whole
    tile at once. We never trust the in-LAZ CHM beyond ``LIDAR_CHM_MAX_M``
    because of treetop spikes or building artefacts (clipped at read time).
    """
    band = ds.GetRasterBand(band_idx_1based)
    arr = band.ReadAsArray()
    if arr is None:
        raise IOError(f"GDAL ReadAsArray returned None for LiDAR band {band_idx_1based}")
    arr = arr.astype(np.float32, copy=False)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr[np.isclose(arr, float(nodata))] = 0.0
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _refine_label_with_chm_ndvi(
    polygon_mask: np.ndarray,
    chm: np.ndarray,
    ndvi: np.ndarray,
    chm_min_m: float,
    ndvi_min: float,
    ignore_value: int,
) -> np.ndarray:
    """Build the refined per-pixel label from polygon + CHM + NDVI.

    Pixels outside every polygon become ``ignore_value`` (skipped by the loss
    and metrics). Pixels inside a polygon become ``1`` only when both
    ``CHM >= chm_min_m`` AND ``NDVI >= ndvi_min`` hold; otherwise they
    become ``0`` (background — actively taught as NOT-leucaena).

    Returns a ``uint8`` array with values in ``{0, 1, ignore_value}``.
    """
    in_poly = polygon_mask == 1
    is_tall = chm >= float(chm_min_m)
    is_veg = ndvi >= float(ndvi_min)
    refined = np.full(polygon_mask.shape, ignore_value, dtype=np.uint8)
    refined[in_poly] = 0
    refined[in_poly & is_tall & is_veg] = 1
    return refined


def _process_tile(
    tile_path: str,
    geojson_path: str,
    out_dir: str,
    patch_size: int,
    overlap: float,
    min_target_class: float,
    band_order: str,
    lidar_dir: Optional[str],
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

    polygon_mask, n_features = rasterize_geojson_for_tile(geojson_path, tile_path)
    log(f"  shape={width}x{height} | features used={n_features}")
    if n_features == 0:
        ds = None
        return []

    # --- LiDAR raster lookup (optional) ---------------------------------
    # When --lidar-dir is set, refined labelling (CHM + NDVI inside polygons,
    # IGNORE outside) is activated. Tiles WITHOUT a usable LiDAR raster are
    # skipped in this mode so the training set only contains refinable pixels.
    lidar_path = _find_lidar_tile(tile_name, lidar_dir)
    lidar_ds: Optional[gdal.Dataset] = None
    if lidar_dir:
        if lidar_path is None:
            log(f"  [SKIP] no LiDAR tile for {tile_name} (lidar_dir set: refined-label mode requires CHM).")
            ds = None
            return []
        lidar_ds = gdal.Open(lidar_path, gdalconst.GA_ReadOnly)
        if lidar_ds is None:
            log(f"  [SKIP] cannot open LiDAR tile: {lidar_path}.")
            ds = None
            return []
        if (lidar_ds.RasterXSize, lidar_ds.RasterYSize) != (width, height):
            log(
                f"  [SKIP] LiDAR shape {lidar_ds.RasterXSize}x{lidar_ds.RasterYSize} "
                f"differs from RGBN {width}x{height}. "
                "Re-run prep-lidar-rasters.py with --tiles-dir set."
            )
            lidar_ds = None
            ds = None
            return []
        log(f"  lidar    : {os.path.basename(lidar_path)} ({lidar_ds.RasterCount} bands)")

    # --- Label refinement (only when LiDAR is available) -----------------
    # Build the final ``label`` raster. Without LiDAR we keep the legacy
    # behaviour (polygon interior = 1, everything else = 0). With LiDAR we
    # apply the professor's rule: outside polygons -> IGNORE; inside ->
    # CHM>=threshold AND NDVI>=threshold => 1, else 0.
    if lidar_ds is not None:
        ndvi = _compute_ndvi_from_tile(ds, band_order)
        chm = _read_full_lidar_band(lidar_ds, band_idx_1based=1)  # band 1 is CHM
        label = _refine_label_with_chm_ndvi(
            polygon_mask=polygon_mask,
            chm=chm,
            ndvi=ndvi,
            chm_min_m=general.LEUCAENA_CHM_MIN_M,
            ndvi_min=general.LEUCAENA_NDVI_MIN,
            ignore_value=general.IGNORE_INDEX,
        )
        n_poly = int((polygon_mask == 1).sum())
        n_leu = int((label == 1).sum())
        n_ign = int((label == general.IGNORE_INDEX).sum())
        log(
            f"  refine   : in_poly={n_poly:,} px | leucaena(1)={n_leu:,} px | "
            f"bg-in-poly(0)={n_poly - n_leu:,} px | ignore(255)={n_ign:,} px "
            f"| CHM>={general.LEUCAENA_CHM_MIN_M}m NDVI>={general.LEUCAENA_NDVI_MIN}"
        )
    else:
        label = polygon_mask

    step = max(1, int((1 - overlap) * patch_size))
    label_windows = view_as_windows(label, (patch_size, patch_size), step)
    grid_rows, grid_cols = label_windows.shape[:2]
    flat = label_windows.reshape(-1, patch_size, patch_size)
    fraction = np.mean(flat == 1, axis=(1, 2))
    # Share of pixels that are NOT IGNORE (i.e. inside an annotated polygon).
    # In the legacy path (no LiDAR) the label has no IGNORE and this is 1.0
    # for every window, which keeps the manifest backwards-compatible.
    poly_fraction = np.mean(flat != general.IGNORE_INDEX, axis=(1, 2))
    keep_mask = fraction >= min_target_class
    n_total = flat.shape[0]
    n_keep = int(keep_mask.sum())
    log(
        f"  windows={n_total:,} | step={step} | keep "
        f"(leucaena>={min_target_class:.2%}) {n_keep:,}"
    )
    if n_keep == 0:
        ds = None
        if lidar_ds is not None:
            lidar_ds = None
        return []

    src_order_idx = _band_reorder_indices(band_order)

    opt_dir = os.path.join(out_dir, "opt")
    lbl_dir = os.path.join(out_dir, "lbl")
    lidar_out_dir = os.path.join(out_dir, "lidar")
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    if lidar_ds is not None:
        os.makedirs(lidar_out_dir, exist_ok=True)

    records: list[PatchRecord] = []
    grid_idx_flat = np.arange(grid_rows * grid_cols).reshape(grid_rows, grid_cols)
    rows_kept, cols_kept = np.unravel_index(
        np.flatnonzero(keep_mask), (grid_rows, grid_cols)
    )
    n_lidar_written = 0
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

        has_lidar = False
        if lidar_ds is not None:
            lidar_patch = _read_window_lidar(lidar_ds, xoff, yoff, patch_size)
            np.save(os.path.join(lidar_out_dir, f"{patch_id}.npy"), lidar_patch)
            has_lidar = True
            n_lidar_written += 1

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
                polygon_fraction=float(poly_fraction[grid_idx_flat[r, c]]),
                lidar_tile_name=os.path.basename(lidar_path) if has_lidar and lidar_path else "",
                has_lidar=has_lidar,
            )
        )

    ds = None
    if lidar_ds is not None:
        lidar_ds = None
    log(f"  wrote {len(records):,} patches (lidar={n_lidar_written:,})")
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
    # The dataclass field order is the canonical manifest schema; use it for
    # both the empty-run header and the populated case so a reader can rely
    # on the columns being present even when no patches were kept.
    fieldnames = list(PatchRecord.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def _pixel_to_xy(gt, px: int, py: int) -> list[float]:
    """Convert pixel coordinates into map coordinates using a GDAL geotransform."""
    x = gt[0] + px * gt[1] + py * gt[2]
    y = gt[3] + px * gt[4] + py * gt[5]
    return [float(x), float(y)]


def _write_patch_footprints(
    records: Iterable[PatchRecord],
    tiles_dir: str,
    out_path: str,
) -> tuple[int, str, dict[str, int]]:
    """Write one GeoJSON square per patch for visual inspection in QGIS."""
    records = list(records)
    tile_cache: dict[str, tuple[tuple, str | None]] = {}
    first_epsg: str | None = None
    counts: Counter[str] = Counter()
    features = []

    for rec in records:
        if rec.tile_name not in tile_cache:
            tile_path = os.path.join(tiles_dir, f"{rec.tile_name}.tif")
            ds = gdal.Open(tile_path, gdalconst.GA_ReadOnly)
            if ds is None:
                raise FileNotFoundError(tile_path)
            gt = ds.GetGeoTransform()
            srs = ds.GetSpatialRef()
            epsg = None
            if srs is not None:
                srs.AutoIdentifyEPSG()
                epsg = srs.GetAuthorityCode(None) or srs.GetAuthorityCode("PROJCS")
            ds = None
            tile_cache[rec.tile_name] = (gt, epsg)
            if first_epsg is None:
                first_epsg = epsg

        gt, _epsg = tile_cache[rec.tile_name]
        p1 = _pixel_to_xy(gt, rec.xoff, rec.yoff)
        p2 = _pixel_to_xy(gt, rec.xoff + rec.win, rec.yoff)
        p3 = _pixel_to_xy(gt, rec.xoff + rec.win, rec.yoff + rec.win)
        p4 = _pixel_to_xy(gt, rec.xoff, rec.yoff + rec.win)
        counts[rec.split] += 1

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[p1, p2, p3, p4, p1]],
                },
                "properties": {
                    "patch_id": rec.patch_id,
                    "tile_name": rec.tile_name,
                    "split": rec.split,
                    "row": rec.row,
                    "col": rec.col,
                    "xoff": rec.xoff,
                    "yoff": rec.yoff,
                    "win_px": rec.win,
                    "size_m": round(rec.win * abs(gt[1]), 3),
                    "leucaena_fraction": rec.leucaena_fraction,
                    "leucaena_pct": round(rec.leucaena_fraction * 100.0, 3),
                    "polygon_fraction": rec.polygon_fraction,
                    "polygon_pct": round(rec.polygon_fraction * 100.0, 3),
                    "has_lidar": rec.has_lidar,
                },
            }
        )

    crs_name = f"EPSG:{first_epsg}" if first_epsg else "unknown"
    geojson = {
        "type": "FeatureCollection",
        "name": "patch_footprints",
        "crs": {"type": "name", "properties": {"name": crs_name}},
        "features": features,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)
    return len(features), crs_name, dict(counts)


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
        log(f"lidar_dir         : {args.lidar_dir or '(disabled)'}")
        if args.lidar_dir:
            log(
                f"label_refinement  : in_poly + CHM>={general.LEUCAENA_CHM_MIN_M}m + "
                f"NDVI>={general.LEUCAENA_NDVI_MIN} (outside polygons = IGNORE {general.IGNORE_INDEX})"
            )
        else:
            log("label_refinement  : (disabled — polygon interior = 1, outside = 0)")

        if not os.path.isdir(args.tiles_dir):
            log(f"[ABORT] tiles dir not found: {args.tiles_dir}")
            sys.exit(2)
        if not os.path.isfile(args.masks):
            log(f"[ABORT] masks file not found: {args.masks}")
            sys.exit(2)
        if args.lidar_dir and not os.path.isdir(args.lidar_dir):
            log(
                f"[WARN] lidar_dir not found: {args.lidar_dir} "
                "-> continuing without LiDAR (manifest rows will have has_lidar=False)."
            )

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
                    lidar_dir=args.lidar_dir,
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

        footprints_path = os.path.join(out_dir, "patch_footprints.geojson")
        try:
            n_features, crs_name, split_counts = _write_patch_footprints(
                all_records,
                tiles_dir=args.tiles_dir,
                out_path=footprints_path,
            )
            log(f"footprints -> {footprints_path}")
            log(f"footprints features={n_features:,} crs={crs_name} splits={split_counts}")
        except Exception as exc:  # noqa: BLE001 - patches/manifest are still valid
            log(f"[WARN] could not write patch footprints GeoJSON: {type(exc).__name__}: {exc}")

        log(f"elapsed : {(time.time() - t0):.1f} s")


if __name__ == "__main__":
    main()
