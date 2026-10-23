"""
Single-scene, two-class, polygon-grouped patch preparation (no spatial leak).

This is the defensible alternative to ``prep-patches-from-tiles.py`` for the
reviewed single-scene experiment. It differs in three deliberate ways:

1. TWO annotation sources instead of one
   --------------------------------------
   - ``--leucaena``      polygons of *pure leucaena*  -> candidate class 1
   - ``--non-leucaena``  polygons of *everything that is NOT leucaena*
                         (bare ground, roofs, towers AND other tree species)
                         -> class 0 (hard negatives, kept even when tall).

   Every pixel OUTSIDE both polygon sets is ``IGNORE_INDEX`` (255): the loss
   and the metrics skip it, so we never claim to have mapped the whole scene
   — only the reviewed polygons.

2. Canopy-height gate on leucaena only
   -----------------------------------
   Inside a leucaena polygon a pixel becomes class 1 only where
   ``CHM >= --chm-min-m`` (default 3 m) — and ``NDVI >= --ndvi-min`` when that
   flag is set. Leucaena pixels below the height gate become IGNORE (we do not
   teach them as background, because they may still be young leucaena). The
   non-leucaena polygons are NOT height-gated: tall other-species trees are
   exactly the hard negatives we want.

3. Polygon-grouped train/val/test split (prevents the leak)
   --------------------------------------------------------
   Each polygon is assigned WHOLE to exactly one split (stratified per class,
   default 70/15/15). We then rasterise one mask per split and extract patches
   split-by-split: inside a split's patches, pixels belonging to *other*
   splits' polygons are forced to IGNORE. Because supervision for every split
   lives only on its own polygons, no annotated pixel is ever shared between
   train/val/test — unlike the random patch-level split in
   ``prep-patches-from-tiles.py`` where 60%-overlapping windows could land in
   different splits.

Outputs (compatible with PatchFileDataset / train.py --patch-source file /
eval-patches.py, exactly like prep-patches-from-tiles.py):

- ``<out-dir>/opt/<patch_id>.npy``    uint8    (H, W, 4)  BGRN
- ``<out-dir>/lbl/<patch_id>.npy``    uint8    (H, W)     {0, 1, 255}
- ``<out-dir>/lidar/<patch_id>.npy``  float32  (H, W, 2)  [CHM, INTENSITY]
- ``<out-dir>/manifest.csv``          one row per patch, with ``split`` column
- ``<out-dir>/patch_footprints.geojson``  patch squares (QGIS)
- ``<out-dir>/split_assignment.json``     which polygon went to which split
- ``<out-dir>/preparation.txt``           full run log

Example (inside the Docker image)
---------------------------------
python prep-scene-polygon-split.py \
  --optical       /data/rgbir/SF-23-Y-A-IV-4-SO-A.tif \
  --lidar         /data/lidar/SF-23-Y-A-IV-4-SO-A.tif \
  --leucaena      /data/masks/leucaena-polygons.geojson \
  --non-leucaena  /data/masks/non-leucaena-polygons.geojson \
  --out-dir       /data/patches \
  --chm-min-m 3.0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
from osgeo import gdal, gdalconst
from skimage.util import view_as_windows

from conf import general
from utils.ops import rasterize_geojson

SPLIT_CODES = {"train": 1, "val": 2, "test": 3}


# --------------------------------------------------------------------------- #
# Window readers (same byte contract as prep-patches-from-tiles.py)
# --------------------------------------------------------------------------- #
def _band_reorder_indices(band_order: str) -> list[int]:
    if band_order == "BGRN":
        return [0, 1, 2, 3]
    if band_order == "RGBN":
        return [2, 1, 0, 3]  # src R,G,B,N -> dst B,G,R,N
    raise ValueError(f"Unsupported band order: {band_order}")


def _read_window_BGRN(ds, xoff, yoff, win, src_order_idx) -> np.ndarray:
    out = np.empty((win, win, 4), dtype=np.uint8)
    for dst_b, src_b in enumerate(src_order_idx):
        arr = ds.GetRasterBand(src_b + 1).ReadAsArray(xoff, yoff, win, win)
        if arr is None:
            raise IOError(f"ReadAsArray None for band {src_b + 1} @ ({xoff},{yoff},{win})")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        out[:, :, dst_b] = arr
    return out


def _read_window_lidar(ds, xoff, yoff, win) -> np.ndarray:
    n = ds.RasterCount
    out = np.empty((win, win, n), dtype=np.float32)
    for b in range(n):
        band = ds.GetRasterBand(b + 1)
        arr = band.ReadAsArray(xoff, yoff, win, win)
        if arr is None:
            raise IOError(f"ReadAsArray None for LiDAR band {b + 1} @ ({xoff},{yoff},{win})")
        arr = arr.astype(np.float32, copy=False)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            arr[np.isclose(arr, float(nodata))] = 0.0
        arr[~np.isfinite(arr)] = 0.0
        out[:, :, b] = arr
    return out


def _read_full_band(ds, idx_1based: int) -> np.ndarray:
    band = ds.GetRasterBand(idx_1based)
    arr = band.ReadAsArray()
    if arr is None:
        raise IOError(f"ReadAsArray None for band {idx_1based}")
    arr = arr.astype(np.float32, copy=False)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr[np.isclose(arr, float(nodata))] = 0.0
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _compute_ndvi_full(ds, band_order: str) -> np.ndarray:
    if band_order == "RGBN":
        red_idx, nir_idx = 1, 4
    elif band_order == "BGRN":
        red_idx, nir_idx = 3, 4
    else:
        raise ValueError(band_order)
    red = ds.GetRasterBand(red_idx).ReadAsArray().astype(np.float32)
    nir = ds.GetRasterBand(nir_idx).ReadAsArray().astype(np.float32)
    denom = nir + red
    denom = np.where(denom == 0.0, 1e-6, denom)
    ndvi = (nir - red) / denom
    ndvi[~np.isfinite(ndvi)] = 0.0
    return ndvi


# --------------------------------------------------------------------------- #
# GeoJSON split helpers
# --------------------------------------------------------------------------- #
def _load_features(path: str) -> tuple[dict, list]:
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    return gj, gj.get("features", [])


def _assign_polygon_splits(n: int, fracs: tuple[float, float, float], rng) -> list[str]:
    """Return a per-polygon split label, stratified within one class."""
    order = rng.permutation(n)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    # Guarantee at least one polygon in val/test when the class is small enough.
    if n >= 3:
        n_train = min(n_train, n - 2)
        n_val = max(n_val, 1)
        if n_train + n_val > n - 1:
            n_val = n - 1 - n_train
    labels = ["test"] * n
    for pos, idx in enumerate(order):
        if pos < n_train:
            labels[idx] = "train"
        elif pos < n_train + n_val:
            labels[idx] = "val"
        else:
            labels[idx] = "test"
    return labels


def _write_split_geojson(template: dict, features: list, path: str) -> None:
    out = {
        "type": "FeatureCollection",
        "name": template.get("name", "split"),
        "crs": template.get("crs"),
        "features": features,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)


# --------------------------------------------------------------------------- #
# Manifest record
# --------------------------------------------------------------------------- #
@dataclass
class PatchRecord:
    patch_id: str
    scene: str
    split: str
    xoff: int
    yoff: int
    win: int
    leucaena_fraction: float
    polygon_fraction: float
    lidar_tile_name: str = ""
    has_lidar: bool = True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--optical", required=True, help="4-band RGBN GeoTIFF of the scene.")
    p.add_argument("--lidar", required=True, help="2-band [CHM, INTENSITY] GeoTIFF aligned to --optical (prep-lidar-scene.py).")
    p.add_argument("--leucaena", required=True, help="GeoJSON of pure-leucaena polygons.")
    p.add_argument("--non-leucaena", required=True, dest="non_leucaena", help="GeoJSON of not-leucaena polygons.")
    p.add_argument("--out-dir", required=True, help="Destination for opt/lbl/lidar/manifest.")
    p.add_argument("--patch-size", type=int, default=general.PATCH_SIZE)
    p.add_argument("--train-overlap", type=float, default=0.5, help="Sliding-window overlap for TRAIN patches.")
    p.add_argument("--eval-overlap", type=float, default=0.0, help="Overlap for VAL/TEST patches (0 = no double counting).")
    p.add_argument("--chm-min-m", type=float, default=3.0, help="Min canopy height (m) for a leucaena pixel to be class 1.")
    p.add_argument("--ndvi-min", type=float, default=None, help="Optional NDVI gate on leucaena pixels (default: off).")
    p.add_argument("--min-valid-frac", type=float, default=0.02, help="Min fraction of non-IGNORE pixels to keep a patch.")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--band-order", choices=("RGBN", "BGRN"), default="RGBN")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    opt_dir = os.path.join(out_dir, "opt")
    lbl_dir = os.path.join(out_dir, "lbl")
    lidar_out = os.path.join(out_dir, "lidar")
    for d in (opt_dir, lbl_dir, lidar_out):
        os.makedirs(d, exist_ok=True)

    log_path = os.path.join(out_dir, "preparation.txt")
    with open(log_path, "w", encoding="utf-8") as log_f:
        def log(msg: str) -> None:
            print(msg, flush=True)
            log_f.write(msg + "\n")
            log_f.flush()

        t0 = time.time()
        scene = os.path.splitext(os.path.basename(args.optical))[0]
        ps = int(args.patch_size)
        chm_min = float(args.chm_min_m)
        ndvi_min = args.ndvi_min
        fracs = (args.train_frac, args.val_frac, args.test_frac)

        log(f"scene           : {scene}")
        log(f"optical         : {args.optical}")
        log(f"lidar           : {args.lidar}")
        log(f"leucaena        : {args.leucaena}")
        log(f"non-leucaena    : {args.non_leucaena}")
        log(f"patch_size      : {ps}")
        log(f"overlap         : train={args.train_overlap} val/test={args.eval_overlap}")
        log(f"chm_min_m       : {chm_min}  ndvi_min: {ndvi_min if ndvi_min is not None else '(off)'}")
        log(f"min_valid_frac  : {args.min_valid_frac}")
        log(f"splits (poly)   : train={fracs[0]} val={fracs[1]} test={fracs[2]}  seed={args.seed}")

        # --- open rasters -------------------------------------------------- #
        ds = gdal.Open(args.optical, gdalconst.GA_ReadOnly)
        if ds is None or ds.RasterCount < 4:
            raise SystemExit(f"[ABORT] optical missing / <4 bands: {args.optical}")
        width, height = ds.RasterXSize, ds.RasterYSize
        lidar_ds = gdal.Open(args.lidar, gdalconst.GA_ReadOnly)
        if lidar_ds is None:
            raise SystemExit(f"[ABORT] cannot open lidar: {args.lidar}")
        if (lidar_ds.RasterXSize, lidar_ds.RasterYSize) != (width, height):
            raise SystemExit(
                f"[ABORT] lidar {lidar_ds.RasterXSize}x{lidar_ds.RasterYSize} != optical "
                f"{width}x{height}. Re-run prep-lidar-scene.py with this --optical."
            )
        log(f"grid            : {width}x{height} px")

        # --- rasterise full-scene class masks ------------------------------ #
        leu_mask = rasterize_geojson(args.leucaena, args.optical, burn_value=1, fill_value=0)
        non_mask = rasterize_geojson(args.non_leucaena, args.optical, burn_value=1, fill_value=0)
        chm = _read_full_band(lidar_ds, 1)
        ndvi = _compute_ndvi_full(ds, args.band_order) if ndvi_min is not None else None

        IGN = int(general.IGNORE_INDEX)
        label = np.full((height, width), IGN, dtype=np.uint8)
        # non-leucaena polygons -> class 0 (hard negatives, no height gate)
        label[non_mask == 1] = 0
        # leucaena polygons -> 1 where tall (and vegetated if ndvi gate on); else IGNORE
        leu_px = leu_mask == 1
        tall = chm >= chm_min
        gate = tall if ndvi is None else (tall & (ndvi >= float(ndvi_min)))
        label[leu_px] = IGN          # sub-threshold leucaena stays IGNORE (not taught as 0)
        label[leu_px & gate] = 1

        n_leu_poly_px = int(leu_px.sum())
        n_one = int((label == 1).sum())
        n_zero = int((label == 0).sum())
        n_ign = int((label == IGN).sum())
        log(
            f"label px        : leucaena(1)={n_one:,} | not-leucaena(0)={n_zero:,} | "
            f"ignore(255)={n_ign:,} | leucaena-poly-px={n_leu_poly_px:,} "
            f"(tall>={chm_min}m kept={n_one:,} / dropped={n_leu_poly_px - n_one:,})"
        )

        # --- polygon-grouped split ---------------------------------------- #
        rng = np.random.default_rng(args.seed)
        leu_gj, leu_feats = _load_features(args.leucaena)
        non_gj, non_feats = _load_features(args.non_leucaena)
        leu_splits = _assign_polygon_splits(len(leu_feats), fracs, rng)
        non_splits = _assign_polygon_splits(len(non_feats), fracs, rng)
        log(f"polygons        : leucaena={len(leu_feats)} non-leucaena={len(non_feats)}")
        for s in ("train", "val", "test"):
            log(
                f"  {s:<5}: leucaena={leu_splits.count(s)} non-leucaena={non_splits.count(s)}"
            )

        assignment = {
            "leucaena": [
                {"index": i, "fid": (leu_feats[i].get("properties") or {}).get("fid"), "split": leu_splits[i]}
                for i in range(len(leu_feats))
            ],
            "non_leucaena": [
                {"index": i, "fid": (non_feats[i].get("properties") or {}).get("fid"), "split": non_splits[i]}
                for i in range(len(non_feats))
            ],
            "seed": args.seed,
            "fracs": fracs,
        }
        with open(os.path.join(out_dir, "split_assignment.json"), "w", encoding="utf-8") as f:
            json.dump(assignment, f, indent=2)

        # one rasterised mask per split (union of both classes' polygons)
        split_masks: dict[str, np.ndarray] = {}
        tmp_paths: list[str] = []
        for s in ("train", "val", "test"):
            feats = [leu_feats[i] for i in range(len(leu_feats)) if leu_splits[i] == s]
            feats += [non_feats[i] for i in range(len(non_feats)) if non_splits[i] == s]
            tmp = os.path.join(out_dir, f"_split_{s}.geojson")
            _write_split_geojson(leu_gj, feats, tmp)
            tmp_paths.append(tmp)
            split_masks[s] = rasterize_geojson(tmp, args.optical, burn_value=1, fill_value=0)

        # --- extract patches split-by-split ------------------------------- #
        src_order = _band_reorder_indices(args.band_order)
        all_records: list[PatchRecord] = []
        lidar_name = os.path.basename(args.lidar)
        min_valid = float(args.min_valid_frac)

        for s in ("train", "val", "test"):
            overlap = args.train_overlap if s == "train" else args.eval_overlap
            step = max(1, int((1 - overlap) * ps))
            # keep only this split's polygons; everything else is IGNORE
            label_s = np.where(split_masks[s] == 1, label, np.uint8(IGN))
            win_view = view_as_windows(label_s, (ps, ps), step)
            gr, gc = win_view.shape[:2]
            flat = win_view.reshape(-1, ps, ps)
            poly_frac = np.mean(flat != IGN, axis=(1, 2))
            leu_frac = np.mean(flat == 1, axis=(1, 2))
            keep = np.flatnonzero(poly_frac >= min_valid)
            log(f"\n[{s}] windows={flat.shape[0]:,} step={step} keep(valid>={min_valid:.2%})={keep.size:,}")

            for k in keep:
                r, c = divmod(int(k), gc)
                xoff, yoff = c * step, r * step
                if xoff + ps > width or yoff + ps > height:
                    continue
                patch_id = f"{scene}__{s}__r{yoff:06d}_c{xoff:06d}"
                opt_patch = _read_window_BGRN(ds, xoff, yoff, ps, src_order)
                lbl_patch = label_s[yoff:yoff + ps, xoff:xoff + ps].astype(np.uint8)
                lidar_patch = _read_window_lidar(lidar_ds, xoff, yoff, ps)
                np.save(os.path.join(opt_dir, f"{patch_id}.npy"), opt_patch)
                np.save(os.path.join(lbl_dir, f"{patch_id}.npy"), lbl_patch)
                np.save(os.path.join(lidar_out, f"{patch_id}.npy"), lidar_patch)
                all_records.append(PatchRecord(
                    patch_id=patch_id, scene=scene, split=s, xoff=xoff, yoff=yoff, win=ps,
                    leucaena_fraction=float(leu_frac[k]), polygon_fraction=float(poly_frac[k]),
                    lidar_tile_name=lidar_name, has_lidar=True,
                ))

        # --- manifest + footprints ---------------------------------------- #
        manifest_path = os.path.join(out_dir, "manifest.csv")
        fieldnames = list(PatchRecord.__dataclass_fields__.keys())
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for rec in all_records:
                w.writerow(asdict(rec))

        counts = Counter(r.split for r in all_records)
        log("\n----- patches -----")
        log(f"total : {len(all_records):,}")
        for s in ("train", "val", "test"):
            log(f"{s:<5} : {counts.get(s, 0):,}")
        log(f"manifest -> {manifest_path}")

        _write_footprints(all_records, ds, os.path.join(out_dir, "patch_footprints.geojson"))

        ds = None
        lidar_ds = None
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        log(f"\nelapsed : {(time.time() - t0):.1f} s")


def _pixel_to_xy(gt, px, py):
    return [float(gt[0] + px * gt[1] + py * gt[2]), float(gt[3] + px * gt[4] + py * gt[5])]


def _write_footprints(records, ds, out_path) -> None:
    gt = ds.GetGeoTransform()
    srs = ds.GetSpatialRef()
    epsg = srs.GetAuthorityCode(None) if srs else None
    features = []
    for rec in records:
        p1 = _pixel_to_xy(gt, rec.xoff, rec.yoff)
        p2 = _pixel_to_xy(gt, rec.xoff + rec.win, rec.yoff)
        p3 = _pixel_to_xy(gt, rec.xoff + rec.win, rec.yoff + rec.win)
        p4 = _pixel_to_xy(gt, rec.xoff, rec.yoff + rec.win)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[p1, p2, p3, p4, p1]]},
            "properties": {
                "patch_id": rec.patch_id, "split": rec.split,
                "leucaena_pct": round(rec.leucaena_fraction * 100.0, 3),
                "polygon_pct": round(rec.polygon_fraction * 100.0, 3),
            },
        })
    gj = {
        "type": "FeatureCollection", "name": "patch_footprints",
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}" if epsg else "unknown"}},
        "features": features,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gj, f)


if __name__ == "__main__":
    main()
