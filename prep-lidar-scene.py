"""
Build ONE aligned LiDAR raster (CHM + INTENSITY) for a single scene from
several LAZ tiles — memory-light version.

Why this script exists
----------------------
``prep-lidar-rasters.py`` pairs *one* LAZ with *one* RGBN tile by filename
stem. For the single-scene experiment the optical is a single large GeoTIFF
(e.g. ``SF-23-Y-A-IV-4-SO-A.tif``) but the LiDAR ships as several quadrant
LAZ files (``...-I.laz``, ``...-II.laz``, ...). This script turns every LAZ
into a small NATIVE-resolution CHM/INTENSITY raster, mosaics them, and does a
SINGLE streamed warp onto the optical grid. The final 2-band GeoTIFF lines up
pixel-for-pixel with the optical scene.

Memory strategy (important)
---------------------------
Warping every LAZ straight to the full 0.25 m / 526 Mpx grid materialised
~4 GB float32 arrays per tile and segfaulted the WSL VM on the 2nd tile.
Instead:
  1. PDAL rasterises each LAZ at NATIVE 1 m over its own (small) extent.
  2. CHM = DSM - DTM at 1 m (small).
  3. gdal.BuildVRT mosaics the per-tile 1 m rasters (cheap, just metadata).
  4. gdal.Warp streams the VRT onto the optical grid, writing to disk block
     by block (GDAL never holds the whole scene in RAM).
  5. The two warped bands are combined into one 2-band GeoTIFF block by block.
Peak RAM stays at a few hundred MB.

Requires PDAL + GDAL (present in the project's Docker image).

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
import json
import os
import sys
import tempfile
import time
from typing import Optional

import numpy as np
from osgeo import gdal, gdalconst

from conf import general
from utils.lidar import (
    TileGrid,
    build_single_writer_pipeline,
    compute_chm,
    normalise_stem,
    refine_dsm_to_percentile,
    run_pdal_pipeline,
)

NODATA = -9999.0


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
    p.add_argument("--laz-epsg", type=int, default=None,
                   help="EPSG of the LAZ when it has no embedded CRS "
                        "(default: same as --optical). IGC-SP LiDAR is EPSG:31983.")
    p.add_argument("--block-rows", type=int, default=2048, help="Row-block height for the final band combine.")
    p.add_argument("--keep-temp", action="store_true", help="Keep the per-LAZ native rasters and VRTs.")
    return p.parse_args()


def _list_laz(args: argparse.Namespace) -> list[str]:
    if args.laz:
        return [os.path.abspath(p) for p in args.laz]
    if not args.laz_dir:
        raise SystemExit("[ABORT] pass --laz-dir or --laz.")
    return [os.path.abspath(p) for p in sorted(glob.glob(os.path.join(args.laz_dir, args.laz_glob)))]


def _laz_bounds(laz: str, in_srs: Optional[int], buffer_m: float) -> tuple[float, float, float, float]:
    """Read (minx, maxx, miny, maxy) from the LAZ header via PDAL quickinfo.

    Header-only (no point read), so it is instant. A small buffer avoids
    clipping edge pixels during rasterisation. The three products (DSM / DTM /
    INTENSITY) MUST share these bounds so CHM = DSM - DTM is pixel-aligned.
    """
    import pdal  # noqa: PLC0415

    reader: dict = {"type": "readers.las", "filename": laz}
    if in_srs is not None:
        reader["default_srs"] = f"EPSG:{in_srs}"
    p = pdal.Pipeline(json.dumps({"pipeline": [reader]}))
    qi = p.quickinfo
    info = qi.get("readers.las") or next(iter(qi.values()))
    b = info["bounds"]
    return (b["minx"] - buffer_m, b["maxx"] + buffer_m, b["miny"] - buffer_m, b["maxy"] + buffer_m)


def _run_product(laz, out_path, product, res, target_epsg, in_srs, bounds,
                 dsm_stat: str = "p99", log=None) -> int:
    """Rasterise one product over the shared ``bounds`` at ``res`` metres."""
    def _go(stat: str) -> int:
        pipe = build_single_writer_pipeline(
            laz_path=laz, out_path=out_path, product=product,
            resolution_m=res, target_epsg=target_epsg, bounds=bounds,
            in_srs=in_srs, dsm_stat=stat,
        )
        n = run_pdal_pipeline(pipe)
        import gc
        gc.collect()
        if not os.path.isfile(out_path):
            raise RuntimeError(
                f"PDAL finished ({n:,} pts) but did not write {product} -> {out_path}"
            )
        return int(n)

    if product != "dsm":
        return _go(dsm_stat)

    try:
        return _go(dsm_stat)
    except Exception as exc:  # noqa: BLE001
        # PDAL < 2.10: sem percentile no writers.gdal.
        # 1) grava DSM com max (mesmo grid do DTM); 2) sobrescreve com pXX em Python.
        msg = str(exc).lower()
        want_pct = str(dsm_stat).lower().startswith("p")
        if want_pct and dsm_stat.lower() != "max" and (
            "percentile" in msg or "binmode" in msg or "p99" in msg
            or "output type" in msg or "invalid" in msg
        ):
            pct = float(str(dsm_stat).lstrip("pP"))
            if log:
                log(f"  [warn] PDAL sem {dsm_stat} ({type(exc).__name__}: {exc})")
                log(f"  [warn] DSM fallback: grid via max + p{pct:g} em Python "
                    f"(1º retorno, sem classes 7/18)")
            if os.path.isfile(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            n = _go("max")
            refine_dsm_to_percentile(
                laz_path=laz,
                dsm_path=out_path,
                percentile=pct,
                target_epsg=target_epsg,
                in_srs=in_srs,
                log=log,
            )
            return n
        raise


def _warp_to_grid(src_path: str, grid: TileGrid, out_path: str, resample: str) -> None:
    opts = gdal.WarpOptions(
        format="GTiff",
        width=grid.width, height=grid.height,
        outputBounds=(grid.xmin, grid.ymin, grid.xmax, grid.ymax),
        dstSRS=grid.projection_wkt,
        resampleAlg=resample,
        srcNodata=NODATA, dstNodata=NODATA,
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
        warpMemoryLimit=512,  # MB — keep GDAL's own warp buffer small
    )
    if gdal.Warp(out_path, src_path, options=opts) is None:
        raise RuntimeError(f"gdal.Warp failed: {src_path} -> {out_path}")


def _combine_blockwise(chm_path: str, int_path: str, out_path: str, grid: TileGrid,
                       log, block_rows: int) -> dict:
    """Stack two single-band warped rasters into one 2-band GeoTIFF, block by block."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(
        out_path, grid.width, grid.height, 2, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER", "PREDICTOR=3"],
    )
    out_ds.SetGeoTransform(grid.geo_transform)
    out_ds.SetProjection(grid.projection_wkt)
    ob1 = out_ds.GetRasterBand(1); ob1.SetDescription("CHM")
    ob2 = out_ds.GetRasterBand(2); ob2.SetDescription("INTENSITY")

    chm_ds = gdal.Open(chm_path, gdalconst.GA_ReadOnly)
    int_ds = gdal.Open(int_path, gdalconst.GA_ReadOnly)
    cb, ib = chm_ds.GetRasterBand(1), int_ds.GetRasterBand(1)

    w, h = grid.width, grid.height
    chm_min, chm_max, chm_sum, covered, int_max = np.inf, -np.inf, 0.0, 0, -np.inf
    for y0 in range(0, h, block_rows):
        ys = min(block_rows, h - y0)
        a1 = cb.ReadAsArray(0, y0, w, ys).astype(np.float32, copy=False)
        a2 = ib.ReadAsArray(0, y0, w, ys).astype(np.float32, copy=False)
        a1[np.isclose(a1, NODATA)] = 0.0; a1[~np.isfinite(a1)] = 0.0
        a2[np.isclose(a2, NODATA)] = 0.0; a2[~np.isfinite(a2)] = 0.0
        # extremos: CHM negativo = DTM>DSM (ruido / borda) → sem vegetacao
        a1[a1 < 0.0] = 0.0
        ob1.WriteArray(a1, 0, y0)
        ob2.WriteArray(a2, 0, y0)
        chm_min = min(chm_min, float(a1.min())); chm_max = max(chm_max, float(a1.max()))
        chm_sum += float(a1.sum()); covered += int((a1 > 0).sum())
        int_max = max(int_max, float(a2.max()))

    out_ds.FlushCache()
    out_ds = None; chm_ds = None; int_ds = None
    return {"chm_min": chm_min, "chm_max": chm_max, "chm_mean": chm_sum / float(w * h),
            "int_max": int_max, "covered": covered, "total": w * h}


def main() -> None:
    args = _parse_args()
    log_path = os.path.splitext(args.out)[0] + ".prep.log"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_f:
        def log(msg: str) -> None:
            print(msg, flush=True)
            log_f.write(msg + "\n"); log_f.flush()

        if not os.path.isfile(args.optical):
            raise SystemExit(f"[ABORT] optical not found: {args.optical}")
        laz_paths = _list_laz(args)
        if not laz_paths:
            raise SystemExit("[ABORT] no LAZ files matched.")

        grid = TileGrid.from_geotiff(args.optical)
        laz_epsg = args.laz_epsg if args.laz_epsg is not None else grid.epsg_code
        log(f"optical     : {args.optical}")
        log(f"grid        : {grid.width}x{grid.height} px | EPSG:{grid.epsg_code} | px={grid.pixel_size_x:.4f} m")
        log(f"laz tiles   : {len(laz_paths)} | laz_epsg: {laz_epsg}")
        log(f"resolution  : {args.resolution} m (native) | chm_max: {args.chm_max_m} m")
        log("dsm mode    : 1st return (ReturnNumber==1) + p99 (binmode); fallback max")
        log("dtm mode    : cell-wise MIN all returns (no SMRF — avoids WSL OOM / segfault)")

        tmp_dir = os.path.join(tempfile.gettempdir(), "leucaena_lidar_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        t0 = time.time()
        chm_paths: list[str] = []
        int_paths: list[str] = []
        for i, laz in enumerate(laz_paths, 1):
            stem = normalise_stem(laz)
            log(f"\n[{i}/{len(laz_paths)}] {laz}")
            dsm_p = os.path.join(tmp_dir, f"{stem}_dsm.tif")
            dtm_p = os.path.join(tmp_dir, f"{stem}_dtm.tif")
            int_p = os.path.join(tmp_dir, f"{stem}_int.tif")
            chm_p = os.path.join(tmp_dir, f"{stem}_chm.tif")
            try:
                bounds = _laz_bounds(laz, laz_epsg, buffer_m=args.resolution * 2.0)
                log(f"  bounds={bounds[0]:.1f},{bounds[1]:.1f},{bounds[2]:.1f},{bounds[3]:.1f}")
                n = _run_product(laz, dsm_p, "dsm", args.resolution, grid.epsg_code,
                                 laz_epsg, bounds, dsm_stat="p99", log=log)
                # Mensagem detalhada de p99 nativo vs Python vem do _run_product/refine.
                log(f"  dsm  ok ({n:,} pts) [1st return + p99]")
                _run_product(laz, dtm_p, "dtm", args.resolution, grid.epsg_code,
                             laz_epsg, bounds, log=log)
                log(f"  dtm  ok")
                _run_product(laz, int_p, "intensity", args.resolution, grid.epsg_code,
                             laz_epsg, bounds, log=log)
                log(f"  int  ok")
                compute_chm(dsm_p, dtm_p, chm_p, chm_max_m=args.chm_max_m)
                log(f"  chm  ok -> {os.path.basename(chm_p)}")
            except Exception as exc:  # noqa: BLE001
                log(f"  [ERROR] {type(exc).__name__}: {exc}")
                continue
            chm_paths.append(chm_p)
            int_paths.append(int_p)

        if not chm_paths:
            raise SystemExit("[ABORT] no LAZ tile produced a raster.")

        # --- mosaic (VRT) + single streamed warp to the optical grid ------- #
        chm_vrt = os.path.join(tmp_dir, "chm_mosaic.vrt")
        int_vrt = os.path.join(tmp_dir, "int_mosaic.vrt")
        gdal.BuildVRT(chm_vrt, chm_paths, options=gdal.BuildVRTOptions(srcNodata=NODATA, VRTNodata=NODATA))
        gdal.BuildVRT(int_vrt, int_paths, options=gdal.BuildVRTOptions(srcNodata=NODATA, VRTNodata=NODATA))
        log(f"\nmosaicked {len(chm_paths)} tile(s); warping to optical grid ...")

        chm_full = os.path.join(tmp_dir, "chm_full.tif")
        int_full = os.path.join(tmp_dir, "int_full.tif")
        # near (nao bilinear): nao espalha CHM para dentro de gaps / nao inventa altura
        _warp_to_grid(chm_vrt, grid, chm_full, resample="near")
        _warp_to_grid(int_vrt, grid, int_full, resample="near")

        log(f"combining bands block-wise -> {args.out}")
        m = _combine_blockwise(chm_full, int_full, args.out, grid, log, args.block_rows)
        log(f"\nscene LiDAR -> {args.out}")
        log(f"CHM  : min={m['chm_min']:.2f} max={m['chm_max']:.2f} mean={m['chm_mean']:.3f} m")
        log(f"INT  : max={m['int_max']:.1f}")
        log(f"covered pixels (CHM>0): {m['covered']:,} / {m['total']:,}")

        if not args.keep_temp:
            for p in chm_paths + int_paths + [chm_vrt, int_vrt, chm_full, int_full]:
                try:
                    os.remove(p)
                except OSError:
                    pass
            for stem in (normalise_stem(l) for l in laz_paths):
                for suf in ("_dsm.tif", "_dtm.tif"):
                    try:
                        os.remove(os.path.join(tmp_dir, stem + suf))
                    except OSError:
                        pass
        log(f"elapsed : {(time.time() - t0):.1f} s")


if __name__ == "__main__":
    main()
