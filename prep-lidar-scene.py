"""
Build ONE aligned LiDAR raster (CHM + INTENSITY) for a single scene from
several LAZ tiles.

Why this script exists
----------------------
``prep-lidar-rasters.py`` pairs *one* LAZ with *one* RGBN tile by filename
stem. For the single-scene experiment the optical is a single large GeoTIFF
(e.g. ``SF-23-Y-A-IV-4-SO-A.tif``) but the LiDAR ships as several quadrant
LAZ files (``...-I.laz``, ``...-II.laz``, ...). This script rasterises every
LAZ onto the SAME optical reference grid and merges them into a single
2-band float32 GeoTIFF ``[CHM, INTENSITY]`` that lines up pixel-for-pixel
with the optical scene.

Merge rule: per pixel we take the element-wise MAX across all LAZ tiles for
both bands. Outside a tile's coverage the aligned raster is 0, so MAX simply
keeps the value from whichever tile actually covers that pixel; in the rare
overlap strips it keeps the taller canopy / stronger return (harmless here).

Requires PDAL (see CHEATSHEET: ``conda install -y -c conda-forge pdal
python-pdal``) — same dependency as ``prep-lidar-rasters.py``. Run it inside
the project's Docker image, which already has PDAL + GDAL.

Example
-------
python prep-lidar-scene.py \
  --optical /data/rgbir/SF-23-Y-A-IV-4-SO-A.tif \
  --laz-dir /data/laz \
  --out     /data/lidar/SF-23-Y-A-IV-4-SO-A.tif
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Optional

import numpy as np
from osgeo import gdal, gdalconst

from conf import general
from utils.lidar import TileGrid, process_laz_to_lidar_tif, normalise_stem


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--optical", required=True, help="Reference 4-band RGBN GeoTIFF (defines grid + CRS).")
    p.add_argument("--laz-dir", default=None, help="Folder with the scene's .laz/.las tiles.")
    p.add_argument("--laz-glob", default="*.la[sz]", help="Pattern under --laz-dir (default: %(default)s).")
    p.add_argument("--laz", nargs="*", default=None, help="Explicit LAZ paths (overrides --laz-dir).")
    p.add_argument("--out", required=True, help="Output 2-band [CHM, INTENSITY] GeoTIFF aligned to --optical.")
    p.add_argument("--resolution", type=float, default=general.LIDAR_RASTER_RESOLUTION_M,
                   help="PDAL native rasterisation resolution in metres (default: %(default)s).")
    p.add_argument("--chm-max-m", type=float, default=general.LIDAR_CHM_MAX_M,
                   help="Clip CHM above this height in metres (default: %(default)s).")
    p.add_argument("--keep-temp", action="store_true", help="Keep the per-LAZ aligned rasters next to --out.")
    return p.parse_args()


def _list_laz(args: argparse.Namespace) -> list[str]:
    if args.laz:
        return [os.path.abspath(p) for p in args.laz]
    if not args.laz_dir:
        raise SystemExit("[ABORT] pass --laz-dir or --laz.")
    found = sorted(glob.glob(os.path.join(args.laz_dir, args.laz_glob)))
    return [os.path.abspath(p) for p in found]


def _read_two_bands(path: str) -> tuple[np.ndarray, np.ndarray]:
    ds = gdal.Open(path, gdalconst.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)

    def _band(i: int) -> np.ndarray:
        b = ds.GetRasterBand(i)
        arr = b.ReadAsArray().astype(np.float32, copy=False)
        nodata = b.GetNoDataValue()
        if nodata is not None:
            arr[np.isclose(arr, float(nodata))] = 0.0
        arr[~np.isfinite(arr)] = 0.0
        return arr

    chm = _band(1)
    inten = _band(2) if ds.RasterCount >= 2 else np.zeros_like(chm)
    ds = None
    return chm, inten


def _write_scene_lidar(out_path: str, grid: TileGrid, chm: np.ndarray, inten: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(
        out_path, grid.width, grid.height, 2, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER", "PREDICTOR=3"],
    )
    out_ds.SetGeoTransform(grid.geo_transform)
    out_ds.SetProjection(grid.projection_wkt)
    b1 = out_ds.GetRasterBand(1)
    b1.SetDescription("CHM")
    b1.WriteArray(chm.astype(np.float32, copy=False))
    b2 = out_ds.GetRasterBand(2)
    b2.SetDescription("INTENSITY")
    b2.WriteArray(inten.astype(np.float32, copy=False))
    out_ds.FlushCache()
    out_ds = None


def main() -> None:
    args = _parse_args()
    log_path = os.path.splitext(args.out)[0] + ".prep.log"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_f:
        def log(msg: str) -> None:
            print(msg, flush=True)
            log_f.write(msg + "\n")
            log_f.flush()

        if not os.path.isfile(args.optical):
            raise SystemExit(f"[ABORT] optical not found: {args.optical}")
        laz_paths = _list_laz(args)
        if not laz_paths:
            raise SystemExit("[ABORT] no LAZ files matched.")

        grid = TileGrid.from_geotiff(args.optical)
        log(f"optical     : {args.optical}")
        log(f"grid        : {grid.width}x{grid.height} px | EPSG:{grid.epsg_code} | px={grid.pixel_size_x:.4f} m")
        log(f"laz tiles   : {len(laz_paths)}")
        log(f"resolution  : {args.resolution} m | chm_max: {args.chm_max_m} m")

        tmp_dir = os.path.join(os.path.dirname(args.out) or ".", "_lidar_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        chm_acc = np.zeros((grid.height, grid.width), dtype=np.float32)
        int_acc = np.zeros((grid.height, grid.width), dtype=np.float32)

        t0 = time.time()
        made: list[str] = []
        for i, laz in enumerate(laz_paths, 1):
            stem = normalise_stem(laz)
            tile_out = os.path.join(tmp_dir, f"{stem}.tif")
            log(f"\n[{i}/{len(laz_paths)}] {laz}")
            try:
                stats = process_laz_to_lidar_tif(
                    laz_path=laz,
                    out_path=tile_out,
                    reference_grid=grid,
                    resolution_m=args.resolution,
                    chm_max_m=args.chm_max_m,
                    tmpdir=tmp_dir,
                )
                log(f"  points={stats.get('n_points', 0):,} aligned={stats.get('aligned_to_rgbn')}")
            except Exception as exc:  # noqa: BLE001
                log(f"  [ERROR] {type(exc).__name__}: {exc}")
                continue
            chm, inten = _read_two_bands(tile_out)
            np.maximum(chm_acc, chm, out=chm_acc)
            np.maximum(int_acc, inten, out=int_acc)
            made.append(tile_out)

        if not made:
            raise SystemExit("[ABORT] no LAZ tile produced a raster.")

        _write_scene_lidar(args.out, grid, chm_acc, int_acc)
        log(f"\nmerged scene LiDAR -> {args.out}")
        log(f"CHM  : min={chm_acc.min():.2f} max={chm_acc.max():.2f} mean={chm_acc.mean():.3f} m")
        log(f"INT  : min={int_acc.min():.1f} max={int_acc.max():.1f} mean={int_acc.mean():.1f}")
        log(f"covered pixels (CHM>0): {(chm_acc > 0).sum():,} / {chm_acc.size:,}")

        if not args.keep_temp:
            for f in made:
                try:
                    os.remove(f)
                except OSError:
                    pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass
        log(f"elapsed : {(time.time() - t0):.1f} s")


if __name__ == "__main__":
    main()
