"""
Rasterise a folder of LAZ point clouds into RGBN-aligned LiDAR GeoTIFFs.

Why this script exists
----------------------
The training pipeline (``prep-patches-from-tiles.py`` + ``PatchFileDataset``)
expects LiDAR as *multi-band raster tiles* aligned to the RGBN tiles. Most
LiDAR products on disk are ``.laz`` point clouds, not rasters. This script
performs the conversion in the same "one input tile -> one output tile"
spirit as ``prep-patches-from-tiles.py`` so the same scaling argument applies
(no RAM blow-up, per-tile errors do not abort the batch).

Inputs
------
- ``--laz-dir`` : folder with ``*.laz`` (or ``*.copc.laz``).
- ``--tiles-dir``: folder with matching RGBN GeoTIFFs. Same stem as the LAZ
  (``A-B-C.laz`` -> ``A-B-C.tif``); ``.copc`` is stripped before matching.
  When provided, every LiDAR tile is aligned pixel-for-pixel to its RGBN
  counterpart (this is what the training pipeline needs).

Outputs
-------
- ``<out-dir>/<stem>.tif`` : 2-band float32 GeoTIFF in the band order
  ``[CHM, INTENSITY]`` (matches ``conf.general.BAND_NAMES_LIDAR``).
- ``<out-dir>/lidar_manifest.csv`` : one row per LAZ with status
  (``ok`` / ``skip-no-rgbn`` / ``skip-existing`` / ``error``), n_points,
  elapsed time and error message.
- ``<out-dir>/preparation.txt`` : the full stdout log.

How to run
----------
**Smoke / "test" scale** (1 or 2 tiles, validates the toolchain)::

    docker compose run --rm segmentation \
        python prep-lidar-rasters.py --max-tiles 2

**Production scale** (every LAZ that has a matching RGBN)::

    docker compose run --rm segmentation \
        python prep-lidar-rasters.py

**Inspect mode** (no rasterisation, just print LAZ metadata)::

    docker compose run --rm segmentation \
        python prep-lidar-rasters.py --inspect-only --max-tiles 5
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

from conf import general, paths
from utils.lidar import (
    TileGrid,
    find_reference_tile,
    inspect_laz,
    normalise_stem,
    process_laz_to_lidar_tif,
)


@dataclass
class LidarRecord:
    """One row of ``lidar_manifest.csv``."""

    laz_name: str
    stem: str
    rgbn_tile: str = ""
    out_tif: str = ""
    status: str = ""  # ok / skip-no-rgbn / skip-existing / error
    n_points: int = 0
    out_width: int = 0
    out_height: int = 0
    aligned_to_rgbn: bool = False
    elapsed_s: float = 0.0
    error_msg: str = ""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rasterise LAZ point clouds into RGBN-aligned 2-band LiDAR GeoTIFFs.",
    )
    p.add_argument(
        "--laz-dir",
        default=paths.PATH_LAZ_DIR,
        help="Folder containing .laz / .copc.laz files (default: %(default)s)",
    )
    p.add_argument(
        "--laz-glob",
        default="*.laz",
        help="Filename pattern under --laz-dir (default: %(default)s)",
    )
    p.add_argument(
        "--tiles-dir",
        default=paths.PATH_TILES_DIR,
        help="Folder with matching RGBN GeoTIFFs. Required for training-grade "
        "output; set to empty string to allow standalone rasterisation "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        default=paths.PATH_LIDAR_DIR,
        help="Where to write LiDAR GeoTIFFs + manifest (default: %(default)s)",
    )
    p.add_argument(
        "--resolution",
        type=float,
        default=general.LIDAR_RASTER_RESOLUTION_M,
        help="Native PDAL rasterisation resolution in metres (default: %(default)s)",
    )
    p.add_argument(
        "--chm-max-m",
        type=float,
        default=general.LIDAR_CHM_MAX_M,
        help="Cap CHM values to this height in metres (default: %(default)s)",
    )
    p.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="If set, process only the first N LAZ (after sorting). Use 1-3 "
        "for the first smoke run before launching the full batch.",
    )
    p.add_argument(
        "--require-rgbn",
        action="store_true",
        default=True,
        help="Skip any LAZ that does not have a matching RGBN tile "
        "(default behaviour; use --no-require-rgbn to disable).",
    )
    p.add_argument(
        "--no-require-rgbn",
        dest="require_rgbn",
        action="store_false",
        help="Produce a standalone LiDAR GeoTIFF (LAZ native bounds) when no "
        "matching RGBN exists. Useful for inspection; NOT usable for training.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process LAZ even if the output GeoTIFF already exists.",
    )
    p.add_argument(
        "--inspect-only",
        action="store_true",
        help="Do not rasterise. Print PDAL metadata + RGBN match per LAZ and exit.",
    )
    return p.parse_args()


def _list_laz(laz_dir: str, pattern: str, max_tiles: Optional[int]) -> list[str]:
    files = sorted(glob.glob(os.path.join(laz_dir, pattern)))
    if max_tiles is not None:
        files = files[:max_tiles]
    return files


def _write_manifest(records: list[LidarRecord], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(LidarRecord.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def _format_status_counts(records: list[LidarRecord]) -> str:
    buckets: dict[str, int] = {}
    for r in records:
        buckets[r.status] = buckets.get(r.status, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(buckets.items()))


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

        log(f"laz_dir       : {args.laz_dir}")
        log(f"laz_glob      : {args.laz_glob}")
        log(f"tiles_dir     : {args.tiles_dir or '(none, standalone mode)'}")
        log(f"out_dir       : {out_dir}")
        log(f"resolution    : {args.resolution} m")
        log(f"chm_max_m     : {args.chm_max_m}")
        log(f"max_tiles     : {args.max_tiles}")
        log(f"require_rgbn  : {args.require_rgbn}")
        log(f"overwrite     : {args.overwrite}")
        log(f"inspect_only  : {args.inspect_only}")

        if not os.path.isdir(args.laz_dir):
            log(f"[ABORT] LAZ dir not found: {args.laz_dir}")
            sys.exit(2)
        if args.tiles_dir and not os.path.isdir(args.tiles_dir):
            log(f"[WARN] tiles dir not found: {args.tiles_dir}")

        laz_paths = _list_laz(args.laz_dir, args.laz_glob, args.max_tiles)
        log(f"\nFound {len(laz_paths)} LAZ file(s)")
        if not laz_paths:
            log("[ABORT] no LAZ matched.")
            sys.exit(2)

        records: list[LidarRecord] = []
        t0 = time.time()

        for i, laz_path in enumerate(laz_paths, 1):
            stem = normalise_stem(laz_path)
            log(f"\n[{i}/{len(laz_paths)}] {os.path.basename(laz_path)}  (stem={stem})")
            rec = LidarRecord(laz_name=os.path.basename(laz_path), stem=stem)

            # --- 1. Locate the matching RGBN tile (if any) ----------------
            rgbn_path = find_reference_tile(stem, args.tiles_dir) if args.tiles_dir else None
            rec.rgbn_tile = os.path.basename(rgbn_path) if rgbn_path else ""

            if rgbn_path is None:
                msg = "no matching RGBN tile"
                if args.require_rgbn:
                    log(f"  [SKIP] {msg}")
                    rec.status = "skip-no-rgbn"
                    rec.error_msg = msg
                    records.append(rec)
                    continue
                log(f"  [WARN] {msg} (continuing in standalone mode)")

            # --- 2. Inspect-only short-circuit ---------------------------
            if args.inspect_only:
                tic = time.time()
                try:
                    info = inspect_laz(laz_path)
                    rec.n_points = int(info["n_points"])
                    rec.status = "ok"
                    log(f"  points  : {rec.n_points:,}")
                    # Print a compact subset of PDAL's stats metadata.
                    log("  metadata: " + json.dumps(info["metadata"], default=str)[:500])
                except Exception as exc:  # noqa: BLE001
                    rec.status = "error"
                    rec.error_msg = f"{type(exc).__name__}: {exc}"
                    log(f"  [ERROR] {rec.error_msg}")
                rec.elapsed_s = round(time.time() - tic, 2)
                records.append(rec)
                continue

            # --- 3. Build target grid + output path ----------------------
            out_tif = os.path.join(out_dir, f"{stem}.tif")
            rec.out_tif = os.path.basename(out_tif)
            if os.path.isfile(out_tif) and not args.overwrite:
                log(f"  [SKIP] already exists: {out_tif} (use --overwrite to redo)")
                rec.status = "skip-existing"
                records.append(rec)
                continue

            ref_grid: Optional[TileGrid] = None
            if rgbn_path is not None:
                try:
                    ref_grid = TileGrid.from_geotiff(rgbn_path)
                    log(
                        f"  rgbn    : {os.path.basename(rgbn_path)} "
                        f"{ref_grid.width}x{ref_grid.height} px @ "
                        f"{abs(ref_grid.pixel_size_x):.3f} m (EPSG:{ref_grid.epsg_code})"
                    )
                except Exception as exc:  # noqa: BLE001
                    rec.status = "error"
                    rec.error_msg = f"read-rgbn: {type(exc).__name__}: {exc}"
                    log(f"  [ERROR] {rec.error_msg}")
                    records.append(rec)
                    continue

            # --- 4. Rasterise --------------------------------------------
            tic = time.time()
            try:
                stats = process_laz_to_lidar_tif(
                    laz_path=laz_path,
                    out_path=out_tif,
                    reference_grid=ref_grid,
                    resolution_m=args.resolution,
                    chm_max_m=args.chm_max_m,
                )
                rec.n_points = int(stats["n_points"])
                rec.out_width = int(stats["out_width"])
                rec.out_height = int(stats["out_height"])
                rec.aligned_to_rgbn = bool(stats["aligned_to_rgbn"])
                rec.status = "ok"
                log(
                    f"  written : {out_tif}  "
                    f"({rec.out_width}x{rec.out_height}, n_points={rec.n_points:,})"
                )
            except Exception as exc:  # noqa: BLE001 - keep batch alive
                rec.status = "error"
                rec.error_msg = f"{type(exc).__name__}: {exc}"
                log(f"  [ERROR] {rec.error_msg}")
            rec.elapsed_s = round(time.time() - tic, 2)
            log(f"  elapsed : {rec.elapsed_s:.1f} s")
            records.append(rec)

        # ------------- summary + manifest ---------------------------------
        log("\n----- summary -----")
        log(f"total    : {len(records)}")
        log(f"buckets  : {_format_status_counts(records)}")
        log(f"elapsed  : {(time.time() - t0):.1f} s")

        manifest_path = os.path.join(out_dir, os.path.basename(paths.PATH_LIDAR_MANIFEST))
        _write_manifest(records, manifest_path)
        log(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
