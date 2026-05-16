"""
predict-tiles.py — per-tile sliding-window inference for the leucaena pipeline.

Why this script exists
----------------------
``prediction.py`` (the legacy script) needs the whole AOI mosaiced into one
VRT / one ``.npy`` of shape ``(H, W, 4)`` and loads it into RAM. That works
for the small Piracicaba AOI (~4000×4000) but does **not** scale to a state-
or country-wide run: the optical mosaic alone would be hundreds of GB.

``predict-tiles.py`` walks one GeoTIFF tile at a time, runs the trained
ResUNet with multi-overlap sliding-window averaging (same maths as
``prediction.py`` and ``predict-tile-preview.py``), writes georeferenced
outputs to a local disk path, and finally calls ``gdalbuildvrt`` so that
QGIS / rasterio see one seamless mosaic from the per-tile rasters.

Output layout (``$LEUCAENA_PREDICTIONS_DIR/exp_<e>/`` by default)::

    <stem>_pred.tif         uint8   class map (0 = background, 1 = leucaena)
    <stem>_prob.tif         optional probability of class 1
    pred.vrt                virtual mosaic of every *_pred.tif
    prob.vrt                virtual mosaic of every *_prob.tif (if --save-prob)
    manifest.csv            per-tile log: status, lidar_status, elapsed_s, ...
    predict_<e>.txt         tee of the full stdout / stderr

Design defaults (documented in studies/predicao-em-escala.md)
-------------------------------------------------------------
- Output lives on **local disk** (``$LEUCAENA_PREDICTIONS_DIR``), NOT in the
  repo, because a country-wide run easily reaches tens of GB.
- ``--save-prob`` is **on** by default; the probability raster is what feeds
  scientific analysis (ROC curves, threshold sweeps).
- Probability dtype is **uint16** by default — 2 bytes/pixel, prob in
  ``[0, 1]`` quantised to 65 536 levels with a GeoTIFF ``scale_factor``. This
  is the closest portable equivalent of "float16" since GDAL / GeoTIFF have
  no native Float16. ``--prob-dtype float32`` keeps the old behaviour.
- Overlap default is ``[0, 0.25, 0.5]`` (averaged across) — same as
  ``general.PREDICTION_OVERLAPS``. ``--overlap 0`` gives a single fast pass
  for previews.
- Missing LiDAR tile = predict with zeros and log ``lidar_status=missing`` in
  the manifest. The script does not silently drop tiles; every decision is
  visible.

Examples
--------
::

    # default: every tile in PATH_TILES_DIR, overlaps [0, 0.25, 0.5],
    #          probability stored as uint16, output in /data/predictions/exp_1
    python predict-tiles.py -e 1

    # quick preview of 3 tiles, single overlap
    python predict-tiles.py -e 1 --max-tiles 3 --overlap 0

    # fusion experiment with real LiDAR; falls back to zeros if a tile lacks it
    python predict-tiles.py -e 3 --lidar-dir /data/lidar

    # keep raw float32 probabilities (4x bigger files)
    python predict-tiles.py -e 1 --prob-dtype float32
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib
import os
import sys
import time
import traceback
from typing import List, Optional

import numpy as np
import torch
from osgeo import gdal

from conf import general, paths
from utils.inference import (
    predict_tile_probability,
    read_lidar_as_array,
    read_tile_as_bgrn,
    write_class_geotiff,
    write_prob_geotiff,
)


class _Tee:
    """Mirror writes to multiple streams (used to tee stdout to a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def _parse_overlaps(args) -> List[float]:
    """``--overlap`` (single value) wins over ``--overlaps`` (comma list)."""
    if args.overlap is not None:
        return [float(args.overlap)]
    return [float(x.strip()) for x in args.overlaps.split(",") if x.strip() != ""]


def _find_lidar_path(stem: str, lidar_dir: Optional[str], suffix: str) -> Optional[str]:
    """Return ``<lidar_dir>/<stem><suffix>.tif`` (or plain ``<stem>.tif``) or ``None``."""
    if not lidar_dir or not os.path.isdir(lidar_dir):
        return None
    candidates = [f"{stem}{suffix}.tif", f"{stem}.tif"]
    for cand in candidates:
        full = os.path.join(lidar_dir, cand)
        if os.path.isfile(full):
            return full
    return None


MANIFEST_FIELDS = [
    "tile_name",
    "rgbn_path",
    "out_pred",
    "out_prob",
    "status",            # ok | skip-existing | error
    "lidar_status",      # present | missing | n/a
    "elapsed_s",
    "pred_pixels_total",
    "pred_pixels_leucaena",
    "frac_leucaena",
    "overlaps",
    "prob_dtype",
    "error_msg",
]


def _append_manifest(manifest_path: str, row: dict) -> None:
    """Append one row to the CSV manifest, creating the header on first call."""
    write_header = not os.path.isfile(manifest_path)
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Per-tile sliding-window inference (scalable replacement for "
            "prediction.py). See studies/predicao-em-escala.md for design notes."
        )
    )
    parser.add_argument("-e", "--experiment", type=int, required=True)
    parser.add_argument("--tiles-dir", type=str, default=paths.PATH_TILES_DIR)
    parser.add_argument("--tiles-glob", type=str, default="*.tif")
    parser.add_argument(
        "--lidar-dir",
        type=str,
        default=paths.PATH_LIDAR_DIR,
        help="Folder with 2-band LiDAR rasters from prep-lidar-rasters.py. "
             "Only used when the model is multimodal (model_2 / model_3).",
    )
    parser.add_argument(
        "--lidar-suffix",
        type=str,
        default="_lidar",
        help="Match the suffix used by prep-lidar-rasters.py.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override the checkpoint path (default: experiments/exp_<e>/models/model.pt).",
    )
    parser.add_argument("--experiments-path", type=str, default=str(paths.PATH_EXPERIMENTS))
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=f"Output folder (default: {paths.PATH_PREDICTIONS_DIR}/exp_<e>).",
    )
    parser.add_argument(
        "--band-order",
        choices=("RGBN", "BGRN"),
        default="RGBN",
        help="Source band order in the GeoTIFFs (must match training).",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=128)

    overlap_grp = parser.add_mutually_exclusive_group()
    overlap_grp.add_argument(
        "--overlaps",
        type=str,
        default=",".join(str(x) for x in general.PREDICTION_OVERLAPS),
        help="Comma-separated overlap values that are averaged. "
             "Default mirrors general.PREDICTION_OVERLAPS.",
    )
    overlap_grp.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Single overlap value (preview mode, e.g. 0). Overrides --overlaps.",
    )

    parser.add_argument("--save-prob", dest="save_prob", action="store_true", default=True)
    parser.add_argument("--no-save-prob", dest="save_prob", action="store_false")
    parser.add_argument(
        "--prob-dtype",
        choices=("float32", "uint16", "uint8"),
        default="uint16",
        help="Storage dtype for the probability raster. "
             "GeoTIFF has no native Float16; uint16 + scale_factor (default) "
             "halves disk size and is read transparently by QGIS/rasterio.",
    )
    parser.add_argument(
        "--no-build-vrt",
        dest="build_vrt",
        action="store_false",
        default=True,
        help="Skip building pred.vrt / prob.vrt at the end.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="Limit run to the first N tiles (debug). 0 = all tiles.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-run tiles even if their outputs already exist; also resets manifest.csv.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cuda | cpu (default: auto).",
    )

    args = parser.parse_args()

    overlaps = _parse_overlaps(args)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    exp_path = os.path.join(args.experiments_path, f"exp_{args.experiment}")
    models_path = os.path.join(exp_path, "models")

    out_dir = args.out_dir or os.path.join(paths.PATH_PREDICTIONS_DIR, f"exp_{args.experiment}")
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.csv")
    log_path = os.path.join(out_dir, f"predict_{args.experiment}.txt")

    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)

    print(f"== predict-tiles.py -- experiment={args.experiment} ==")
    print(f"Device          : {device}")
    print(f"Tiles dir       : {args.tiles_dir}")
    print(f"Tiles glob      : {args.tiles_glob}")
    print(f"LiDAR dir       : {args.lidar_dir} (consumed only if model is multimodal)")
    print(f"Out dir         : {out_dir}")
    print(f"Overlaps        : {overlaps}")
    print(f"Save prob       : {args.save_prob} (dtype={args.prob_dtype})")
    print(f"Build VRT       : {args.build_vrt}")
    print(f"Band order      : {args.band_order}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Max tiles       : {'all' if args.max_tiles == 0 else args.max_tiles}")
    print()

    model_m = importlib.import_module(f"conf.model_{args.experiment}")
    model, lidar_bands = model_m.get_model()
    model.to(device)
    model.eval()
    needs_lidar = lidar_bands is not None
    print(
        f"Model           : {type(model).__name__}  "
        f"lidar_bands={lidar_bands}  needs_lidar={needs_lidar}"
    )

    model_path = args.model_path or os.path.join(models_path, "model.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded weights  : {model_path}")
    print()

    pattern = os.path.join(args.tiles_dir, args.tiles_glob)
    tiles = sorted(glob.glob(pattern))
    if args.max_tiles > 0:
        tiles = tiles[: args.max_tiles]
    if not tiles:
        print(f"No tiles match {pattern}")
        sys.exit(1)
    print(f"Found {len(tiles)} tile(s).")

    if args.overwrite and os.path.isfile(manifest_path):
        os.remove(manifest_path)

    pred_paths_for_vrt: List[str] = []
    prob_paths_for_vrt: List[str] = []

    t_run = time.perf_counter()
    for i, tile_path in enumerate(tiles, 1):
        stem = os.path.splitext(os.path.basename(tile_path))[0]
        out_pred = os.path.join(out_dir, f"{stem}_pred.tif")
        out_prob = os.path.join(out_dir, f"{stem}_prob.tif") if args.save_prob else ""

        if (
            not args.overwrite
            and os.path.isfile(out_pred)
            and (not args.save_prob or os.path.isfile(out_prob))
        ):
            print(f"[{i:>4}/{len(tiles)}] {stem}  -> already done, skipping")
            pred_paths_for_vrt.append(out_pred)
            if out_prob:
                prob_paths_for_vrt.append(out_prob)
            _append_manifest(
                manifest_path,
                {
                    "tile_name": stem,
                    "rgbn_path": tile_path,
                    "out_pred": out_pred,
                    "out_prob": out_prob,
                    "status": "skip-existing",
                    "lidar_status": "",
                    "overlaps": ";".join(str(x) for x in overlaps),
                    "prob_dtype": args.prob_dtype if args.save_prob else "",
                },
            )
            continue

        t0 = time.perf_counter()
        try:
            print(f"[{i:>4}/{len(tiles)}] {stem}  reading...")
            opt_hwc, geo = read_tile_as_bgrn(tile_path, args.band_order)

            lidar_status = "n/a"
            lidar_arr = None
            if needs_lidar:
                lpath = _find_lidar_path(stem, args.lidar_dir, args.lidar_suffix)
                if lpath:
                    lidar_arr = read_lidar_as_array(lpath)
                    if lidar_arr.shape[:2] != opt_hwc.shape[:2]:
                        raise ValueError(
                            f"LiDAR shape {lidar_arr.shape[:2]} != optical "
                            f"{opt_hwc.shape[:2]} — did you run prep-lidar-rasters.py "
                            f"on the same grid?"
                        )
                    lidar_status = "present"
                else:
                    lidar_status = "missing"
                    print(
                        "          lidar=missing -- predicting with zeros, "
                        "decision logged to manifest."
                    )

            print("          inferring...")
            prob_hwc = predict_tile_probability(
                model=model,
                opt_bgrn_uint8=opt_hwc,
                lidar_arr_raw=lidar_arr,
                n_classes=general.N_CLASSES,
                patch_size=general.PATCH_SIZE,
                overlaps=overlaps,
                batch_size=args.batch_size,
                device=device,
                lidar_bands=lidar_bands,
            )
            pred_hw = np.argmax(prob_hwc, axis=-1).astype(np.uint8)

            write_class_geotiff(out_pred, geo, pred_hw)
            pred_paths_for_vrt.append(out_pred)

            if args.save_prob:
                write_prob_geotiff(out_prob, geo, prob_hwc[:, :, 1], args.prob_dtype)
                prob_paths_for_vrt.append(out_prob)

            total = int(pred_hw.size)
            leu = int(np.sum(pred_hw == 1))
            elapsed = time.perf_counter() - t0
            frac = leu / total if total > 0 else 0.0
            print(
                f"          done in {elapsed:.1f}s  "
                f"leucaena: {leu}/{total}  ({100*frac:.2f}%)  lidar={lidar_status}"
            )
            _append_manifest(
                manifest_path,
                {
                    "tile_name": stem,
                    "rgbn_path": tile_path,
                    "out_pred": out_pred,
                    "out_prob": out_prob,
                    "status": "ok",
                    "lidar_status": lidar_status,
                    "elapsed_s": f"{elapsed:.2f}",
                    "pred_pixels_total": total,
                    "pred_pixels_leucaena": leu,
                    "frac_leucaena": f"{frac:.6f}",
                    "overlaps": ";".join(str(x) for x in overlaps),
                    "prob_dtype": args.prob_dtype if args.save_prob else "",
                },
            )
        except Exception as ex:
            err_first = str(ex).splitlines()[0] if str(ex) else "unknown"
            err_full = traceback.format_exc()
            print(f"          ERROR: {err_first}\n{err_full}")
            _append_manifest(
                manifest_path,
                {
                    "tile_name": stem,
                    "rgbn_path": tile_path,
                    "out_pred": out_pred,
                    "out_prob": out_prob,
                    "status": "error",
                    "lidar_status": "",
                    "elapsed_s": f"{time.perf_counter() - t0:.2f}",
                    "overlaps": ";".join(str(x) for x in overlaps),
                    "prob_dtype": args.prob_dtype if args.save_prob else "",
                    "error_msg": err_first,
                },
            )

    total_min = (time.perf_counter() - t_run) / 60.0
    print(f"\nFinished {len(tiles)} tile(s) in {total_min:.1f} min")
    print(f"Manifest: {manifest_path}")

    if args.build_vrt and pred_paths_for_vrt:
        pred_vrt = os.path.join(out_dir, "pred.vrt")
        print(f"Building VRT mosaic: {pred_vrt} ({len(pred_paths_for_vrt)} tiles)")
        gdal.BuildVRT(pred_vrt, pred_paths_for_vrt)
        if args.save_prob and prob_paths_for_vrt:
            prob_vrt = os.path.join(out_dir, "prob.vrt")
            print(f"Building VRT mosaic: {prob_vrt} ({len(prob_paths_for_vrt)} tiles)")
            gdal.BuildVRT(prob_vrt, prob_paths_for_vrt)
    print("OK")


if __name__ == "__main__":
    main()
