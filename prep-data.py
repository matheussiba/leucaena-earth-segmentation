"""
Prepare data for leucaena binary segmentation (PhD / leucaena-earth pipeline).

Pipeline overview for analysts
------------------------------
1. Load optical + optional LiDAR rasters, same grid (H×W) and georeference.
2. Rasterize GeoJSON polygons from leucaena.earth into a pixel label mask.
3. Slide a fixed window over the train mask; keep patches with enough
   positive (leucaena) pixels; split indices into train / val / test.
4. Save full-scene numpy arrays plus patch index arrays for the PyTorch loader.

Inputs
------
  - Optical: 4-band GeoTIFF (B, G, R, NIR)
  - LiDAR:   multi-band GeoTIFF (CHM, intensity, …) — optional (--no-lidar)
  - Masks:   GeoJSON polygons (train-only, or train+test split via flags)

Outputs (prepared/)
-------------------
  - opt_img.npy, lidar_img.npy — float32, normalised per band to [0, 1]
  - label_train.npy, label_test.npy — uint8 0/1 (and IGNORE_INDEX where used)
  - train_patches.npy, val_patches.npy — arrays of shape (N, patch_h, patch_w)
    storing *linear indices* into the flattened image, not patch pixels themselves
  - test_label.tif — georeferenced test mask for evaluation / prediction overlay
  - map.data — pickle class map (binary: trivial here; kept for compatibility)
"""
import argparse
import pathlib
import os
import sys

import numpy as np
from osgeo import gdal, gdalconst
from skimage.util import view_as_windows
from matplotlib import pyplot as plt

from conf import paths, default, general
from utils.ops import (
    load_opt_image,
    rasterize_geojson,
    filter_outliers,
    save_dict,
)

parser = argparse.ArgumentParser(
    description='Prepare optical + LiDAR + GeoJSON masks for training'
)
parser.add_argument(
    '--optical', type=pathlib.Path, default=paths.PATH_OPTICAL,
    help='Path to 4-band optical GeoTIFF (B, G, R, NIR)'
)
parser.add_argument(
    '--lidar', type=pathlib.Path, default=paths.PATH_LIDAR,
    help='Path to multi-band LiDAR raster GeoTIFF (CHM, intensity, ...)'
)
parser.add_argument(
    '--masks', type=pathlib.Path, default=paths.PATH_MASKS,
    help='Path to GeoJSON with leucaena mask polygons'
)
parser.add_argument(
    '--train-masks', type=pathlib.Path, default=paths.PATH_TRAIN_MASKS,
    help='Optional: separate train masks GeoJSON (overrides --masks)'
)
parser.add_argument(
    '--test-masks', type=pathlib.Path, default=paths.PATH_TEST_MASKS,
    help='Optional: separate test masks GeoJSON (overrides --masks)'
)
parser.add_argument(
    '-m', '--min-target-class', type=float, default=default.MIN_TRAIN_CLASS,
    help='Minimum fraction of leucaena pixels to keep a patch [0-1]'
)
parser.add_argument(
    '--test-split', type=float, default=general.TEST_SPLIT,
    help='Fraction of patches to hold out for testing [0-1]'
)
parser.add_argument(
    '--val-split', type=float, default=general.VAL_SPLIT,
    help='Fraction of remaining patches for validation [0-1]'
)
parser.add_argument(
    '--no-lidar', action='store_true',
    help='Run without LiDAR data (optical only)'
)
parser.add_argument(
    '--seed', type=int, default=42,
    help='Random seed for reproducible splits'
)

args = parser.parse_args()

# All artefacts go under prepared/; paths.PREPARED_PATH must match conf.paths + dataloader.
os.makedirs(paths.PREPARED_PATH, exist_ok=True)

outfile = os.path.join(paths.PREPARED_PATH, 'preparation.txt')
with open(outfile, 'w') as log_f:
    def log(msg):
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    # ── 1. Load optical imagery ──────────────────────────────────────────
    log(f'Loading optical imagery: {args.optical}')
    opt_img = load_opt_image(str(args.optical)).astype(np.float32)
    h, w, n_bands = opt_img.shape
    log(f'  Shape: {h} x {w}, {n_bands} bands')
    assert n_bands >= general.N_OPTICAL_BANDS, (
        f'Expected at least {general.N_OPTICAL_BANDS} optical bands, got {n_bands}'
    )
    opt_img = opt_img[:, :, :general.N_OPTICAL_BANDS]

    log('  Filtering outliers...')
    opt_img = filter_outliers(opt_img)

    # Min–max per band → [0, 1]. Same idea as LiDAR below; keeps scales comparable across bands.
    for b in range(opt_img.shape[-1]):
        bmin, bmax = opt_img[:, :, b].min(), opt_img[:, :, b].max()
        if bmax > bmin:
            opt_img[:, :, b] = (opt_img[:, :, b] - bmin) / (bmax - bmin)
    log('  Normalised to [0, 1]')

    # ── 2. Load LiDAR raster ─────────────────────────────────────────────
    if not args.no_lidar and os.path.exists(str(args.lidar)):
        log(f'Loading LiDAR raster: {args.lidar}')
        lidar_img = load_opt_image(str(args.lidar)).astype(np.float32)
        log(f'  Shape: {lidar_img.shape[0]} x {lidar_img.shape[1]}, {lidar_img.shape[2]} bands')

        assert lidar_img.shape[:2] == (h, w), (
            f'LiDAR raster shape {lidar_img.shape[:2]} does not match optical {(h, w)}. '
            'Resample the LiDAR raster to the same resolution and extent as the optical image.'
        )

        log('  Filtering outliers...')
        lidar_img = filter_outliers(lidar_img)
        for b in range(lidar_img.shape[-1]):
            bmin, bmax = lidar_img[:, :, b].min(), lidar_img[:, :, b].max()
            if bmax > bmin:
                lidar_img[:, :, b] = (lidar_img[:, :, b] - bmin) / (bmax - bmin)
        log('  Normalised to [0, 1]')
    else:
        # Single dummy band so the dataloader always stacks (opt, lidar); model configs
        # that use lidar_bands=None skip LiDAR in the network (see conf/model_1.py).
        if args.no_lidar:
            log('LiDAR disabled (--no-lidar flag)')
        else:
            log(f'LiDAR file not found: {args.lidar} — continuing without LiDAR')
        lidar_img = np.zeros((h, w, 1), dtype=np.float32)

    # ── 3. Rasterize GeoJSON masks ───────────────────────────────────────
    have_separate = (args.train_masks is not None and args.test_masks is not None)

    if have_separate:
        log(f'Rasterizing TRAIN masks: {args.train_masks}')
        train_label = rasterize_geojson(str(args.train_masks), str(args.optical))
        log(f'Rasterizing TEST masks: {args.test_masks}')
        test_label = rasterize_geojson(str(args.test_masks), str(args.optical))
    else:
        log(f'Rasterizing masks: {args.masks}')
        full_label = rasterize_geojson(str(args.masks), str(args.optical))
        leucaena_pixels = (full_label == 1).sum()
        total_pixels = full_label.size
        log(f'  Leucaena pixels: {leucaena_pixels:,} / {total_pixels:,} ({100*leucaena_pixels/total_pixels:.2f}%)')
        train_label = full_label
        test_label = None

    # ── 4. Build patches ─────────────────────────────────────────────────
    patch_size = general.PATCH_SIZE
    train_step = max(1, int((1 - general.PATCH_OVERLAP) * patch_size))
    log(f'Patch size: {patch_size}, step: {train_step} (overlap {general.PATCH_OVERLAP})')

    # Flat row-major indices 0..H*W-1 reshaped to H×W; each sliding window picks
    # the same spatial window from labels and from this index grid.
    idx_matrix = np.arange(h * w, dtype=np.uint32).reshape((h, w))

    label_patches = view_as_windows(
        train_label, (patch_size, patch_size), train_step
    ).reshape((-1, patch_size, patch_size))

    idx_patches = view_as_windows(
        idx_matrix, (patch_size, patch_size), train_step
    ).reshape((-1, patch_size, patch_size))

    log(f'Total sliding windows: {label_patches.shape[0]:,}')

    # Positive-only sampling: drop windows that are mostly background (class imbalance).
    leucaena_fraction = np.mean(label_patches == 1, axis=(1, 2))
    keep = leucaena_fraction >= args.min_target_class
    idx_patches = idx_patches[keep]
    log(f'Patches after filtering (>={100*args.min_target_class:.1f}% leucaena): {idx_patches.shape[0]:,}')

    # ── 5. Train / test / val split ──────────────────────────────────────
    np.random.seed(args.seed)
    np.random.shuffle(idx_patches)
    n_total = idx_patches.shape[0]

    if have_separate:
        # Train GeoJSON drives patch locations; test GeoJSON is a full raster — no patch split for test.
        n_val = int(args.val_split * n_total)
        val_idx = idx_patches[:n_val]
        train_idx = idx_patches[n_val:]
        log(f'Using separate test masks. Train: {train_idx.shape[0]:,}, Val: {val_idx.shape[0]:,}')
    else:
        n_test = int(args.test_split * n_total)
        n_val = int(args.val_split * (n_total - n_test))
        test_idx = idx_patches[:n_test]
        val_idx = idx_patches[n_test:n_test + n_val]
        train_idx = idx_patches[n_test + n_val:]

        # Reconstruct a held-out label image: only pixels inside test patches get labels;
        # everything else stays IGNORE_INDEX so loss/metrics can mask them out.
        test_label = np.full((h, w), general.IGNORE_INDEX, dtype=np.uint8)
        for patch in test_idx:
            rows, cols = np.unravel_index(patch.flatten(), (h, w))
            test_label[rows, cols] = train_label[rows, cols]

        log(f'Train: {train_idx.shape[0]:,}, Val: {val_idx.shape[0]:,}, Test: {n_test:,}')

    # ── 6. Save prepared data ────────────────────────────────────────────
    log('Saving prepared arrays...')

    np.save(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_OPT}_img.npy'), opt_img)
    np.save(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LIDAR}_img.npy'), lidar_img)
    np.save(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LABEL}_train.npy'),
            train_label.astype(np.uint8))
    np.save(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LABEL}_test.npy'),
            test_label.astype(np.uint8))
    np.save(os.path.join(paths.PREPARED_PATH, 'train_patches.npy'), train_idx)
    np.save(os.path.join(paths.PREPARED_PATH, 'val_patches.npy'), val_idx)

    # Kept for parity with the original tree_fusion multi-class remap file.
    remap_dict = {0: 0, 1: 1}
    save_dict(remap_dict, os.path.join(paths.PREPARED_PATH, 'map.data'))

    # Save georeferenced test label as GeoTIFF
    ref_ds = gdal.Open(str(args.optical), gdalconst.GA_ReadOnly)
    geo_transform = ref_ds.GetGeoTransform()
    x_res = ref_ds.RasterXSize
    y_res = ref_ds.RasterYSize
    proj = ref_ds.GetProjection()
    crs = ref_ds.GetSpatialRef()
    ref_ds = None

    out_tif = os.path.join(paths.PREPARED_PATH, 'test_label.tif')
    target_ds = gdal.GetDriverByName('GTiff').Create(out_tif, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(proj)
    target_ds.SetSpatialRef(crs)
    target_ds.GetRasterBand(1).WriteArray(test_label.astype(np.uint8))
    target_ds = None

    log(f'Test label saved to {out_tif}')
    log('Preparation complete!')
