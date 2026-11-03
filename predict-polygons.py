"""
Turn a trained model's class map into leucaena polygons (GeoJSON).

Two steps in one script
-----------------------
1. If ``--pred`` is omitted, run sliding-window inference on ``--optical``
   (same engine as ``predict-tiles.py``) and write ``*_pred.tif``.
2. Polygonize class 1 (optionally gated by CHM / probability / AOI), drop
   tiny blobs, simplify, write GeoJSON.

This is a **model map**, not reviewed ground truth. Metrics in
``eval-patches.py`` stay the honest number (inside annotated polygons only).

Example (inside the Docker image, after ``train.py -e 3``)
---------------------------------------------------------
# First time: infer + polygonize (overlap 0 = faster preview)
python predict-polygons.py -e 3 \
  --optical /data/rgbir/SF-23-Y-A-IV-4-SO-A.tif \
  --lidar   /data/lidar/SF-23-Y-A-IV-4-SO-A.tif \
  --aoi     /data/masks/261002-riodaspedras-aoi.geojson \
  --chm-min-m 3.0 --overlap 0 -b 8

# Already have a pred raster: skip inference
python predict-polygons.py \
  --pred /data/predictions/exp_3/SF-23-Y-A-IV-4-SO-A_pred.tif \
  --lidar /data/lidar/SF-23-Y-A-IV-4-SO-A.tif \
  --chm-min-m 3.0 --out /data/predictions/exp_3/leucaena-pred.geojson
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
from osgeo import gdal, gdalconst, ogr, osr

from conf import general, paths
from utils.inference import scale_lidar
from utils.ops import rasterize_geojson

NODATA = 0
gdal.SetConfigOption("GTIFF_SRS_SOURCE", "EPSG")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-e", "--experiment", type=int, default=3)
    p.add_argument("--optical", default=None, help="RGBN GeoTIFF (required if --pred is omitted).")
    p.add_argument("--lidar", default=None, help="2-band [CHM, INTENSITY] aligned to the optical.")
    p.add_argument("--pred", default=None, help="Existing class map (0/1). Skips inference.")
    p.add_argument("--prob", default=None, help="Optional probability raster (class 1).")
    p.add_argument("--aoi", default=None, help="GeoJSON AOI: pixels outside become 0 before polygonize.")
    p.add_argument("--out", default=None, help="Output GeoJSON (default: next to the pred raster).")
    p.add_argument("--out-dir", default=None, help="Folder for pred/prob if inference runs.")
    p.add_argument("--chm-min-m", type=float, default=3.0,
                   help="Keep predicted leucaena only where CHM >= this (0 = off). Needs --lidar.")
    p.add_argument("--prob-min", type=float, default=0.0,
                   help="Keep predicted leucaena only where P(class=1) >= this (0 = off).")
    p.add_argument("--sieve-px", type=int, default=64,
                   help="Remove connected blobs smaller than this many pixels before polygonize.")
    p.add_argument("--min-area-m2", type=float, default=50.0,
                   help="Drop polygons smaller than this area (m², in the raster CRS).")
    p.add_argument("--simplify-m", type=float, default=1.0,
                   help="Douglas-Peucker tolerance in metres (0 = no simplify).")
    p.add_argument("--to-epsg", type=int, default=4326,
                   help="Reproject output GeoJSON (default 4326). 0 = keep raster CRS.")
    p.add_argument("--band-order", choices=("RGBN", "BGRN"), default="RGBN")
    p.add_argument("-b", "--batch-size", type=int, default=8)
    p.add_argument("--overlap", type=float, default=0.0,
                   help="Single inference overlap (0 = fastest preview).")
    p.add_argument("--device", default="auto")
    p.add_argument("--model-path", default=None)
    return p.parse_args()


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _band_reorder_indices(band_order: str) -> list[int]:
    if band_order == "BGRN":
        return [0, 1, 2, 3]
    if band_order == "RGBN":
        return [2, 1, 0, 3]
    raise ValueError(band_order)


def _window_coords(h: int, w: int, patch: int, overlap: float) -> list[tuple[int, int]]:
    step = max(1, int(round(patch * (1.0 - float(overlap)))))
    ys = list(range(0, max(h - patch + 1, 1), step))
    xs = list(range(0, max(w - patch + 1, 1), step))
    if h >= patch and (h - patch) % step != 0:
        ys.append(h - patch)
    if w >= patch and (w - patch) % step != 0:
        xs.append(w - patch)
    return [(y, x) for y in sorted(set(ys)) for x in sorted(set(xs))]


def _read_opt_window(ds, xoff, yoff, win, src_order: list[int],
                     compute_ndvi: bool = False) -> np.ndarray:
    """Return (C, win, win) float32 in [0,1], BGRN (+ NDVI if requested)."""
    out = np.empty((4, win, win), dtype=np.float32)
    for dst_b, src_b in enumerate(src_order):
        arr = ds.GetRasterBand(src_b + 1).ReadAsArray(xoff, yoff, win, win)
        if arr is None:
            raise IOError(f"optical ReadAsArray failed @ ({xoff},{yoff})")
        out[dst_b] = arr.astype(np.float32) / 255.0
    if compute_ndvi:
        # BGRN: R=2, NIR=3 — same as PatchFileDataset
        red, nir = out[2], out[3]
        denom = nir + red
        denom = np.where(denom == 0.0, 1e-6, denom)
        ndvi = np.clip((nir - red) / denom, -1.0, 1.0).astype(np.float32)
        out = np.concatenate([out, ndvi[np.newaxis, :, :]], axis=0)
    return out


def _read_lidar_window(ds, xoff, yoff, win, lidar_bands) -> np.ndarray:
    """Return (C, win, win) float32 scaled like training."""
    n = ds.RasterCount
    raw = np.empty((win, win, n), dtype=np.float32)
    for b in range(n):
        band = ds.GetRasterBand(b + 1)
        arr = band.ReadAsArray(xoff, yoff, win, win)
        if arr is None:
            raise IOError(f"lidar ReadAsArray failed @ ({xoff},{yoff})")
        arr = arr.astype(np.float32, copy=False)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            arr[np.isclose(arr, float(nodata))] = 0.0
        arr[~np.isfinite(arr)] = 0.0
        raw[:, :, b] = arr
    scaled = scale_lidar(raw)
    if lidar_bands is not None:
        scaled = scaled[:, :, lidar_bands]
    return np.transpose(scaled, (2, 0, 1))


def _create_byte_tif(path: str, ref_ds, fill: int = 0) -> gdal.Dataset:
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        path, ref_ds.RasterXSize, ref_ds.RasterYSize, 1, gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    ds.SetGeoTransform(ref_ds.GetGeoTransform())
    ds.SetProjection(ref_ds.GetProjection())
    b = ds.GetRasterBand(1)
    b.Fill(fill)
    b.SetNoDataValue(255)
    return ds


def _create_uint16_prob_tif(path: str, ref_ds) -> gdal.Dataset:
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        path, ref_ds.RasterXSize, ref_ds.RasterYSize, 1, gdal.GDT_UInt16,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER", "PREDICTOR=2"],
    )
    ds.SetGeoTransform(ref_ds.GetGeoTransform())
    ds.SetProjection(ref_ds.GetProjection())
    b = ds.GetRasterBand(1)
    b.Fill(0)
    b.SetScale(1.0 / 65535.0)
    b.SetOffset(0.0)
    return ds


def _run_inference(args, out_dir: str) -> tuple[str, str]:
    """Memory-light inference: read/write one patch batch at a time (no full-scene RAM)."""
    if not args.optical or not os.path.isfile(args.optical):
        raise SystemExit("[ABORT] pass --optical (or --pred to skip inference).")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.optical))[0]
    out_pred = os.path.join(out_dir, f"{stem}_pred.tif")
    out_prob = os.path.join(out_dir, f"{stem}_prob.tif")

    device = _device(args.device)
    model_m = importlib.import_module(f"conf.model_{args.experiment}")
    model, lidar_bands = model_m.get_model()
    compute_ndvi = bool(getattr(model_m, "COMPUTE_NDVI", False))
    model.to(device)
    ckpt = args.model_path or os.path.join(
        paths.PATH_EXPERIMENTS, f"exp_{args.experiment}", "models", "model.pt"
    )
    if not os.path.isfile(ckpt):
        raise SystemExit(
            f"[ABORT] checkpoint not found: {ckpt}\n"
            f"Train first: python train.py -e {args.experiment} --patch-source file "
            f"--manifest /data/patches/manifest.csv --cache-patches"
        )
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {ckpt} on {device} | COMPUTE_NDVI={compute_ndvi}")

    opt_ds = gdal.Open(args.optical, gdalconst.GA_ReadOnly)
    if opt_ds is None or opt_ds.RasterCount < 4:
        raise SystemExit(f"[ABORT] cannot open 4-band optical: {args.optical}")
    H, W = opt_ds.RasterYSize, opt_ds.RasterXSize
    src_order = _band_reorder_indices(args.band_order)

    lidar_ds = None
    if lidar_bands is not None:
        if not args.lidar or not os.path.isfile(args.lidar):
            print("[WARN] no --lidar; predicting with zeros (late fusion will be weak).")
        else:
            lidar_ds = gdal.Open(args.lidar, gdalconst.GA_ReadOnly)
            if lidar_ds is None:
                raise SystemExit(f"[ABORT] cannot open lidar: {args.lidar}")
            if (lidar_ds.RasterXSize, lidar_ds.RasterYSize) != (W, H):
                raise SystemExit(
                    f"[ABORT] lidar {lidar_ds.RasterXSize}x{lidar_ds.RasterYSize} "
                    f"!= optical {W}x{H}"
                )

    patch = int(general.PATCH_SIZE)
    coords = _window_coords(H, W, patch, args.overlap)
    n_win = len(coords)
    print(f"Scene         : {W}x{H} px | windows={n_win:,} | overlap={args.overlap} | batch={args.batch_size}")
    print("Mode          : streaming (patch-by-patch, low RAM)")
    print(f"Writing       : {out_pred}")

    pred_ds = _create_byte_tif(out_pred, opt_ds, fill=0)
    pred_band = pred_ds.GetRasterBand(1)
    prob_ds = _create_uint16_prob_tif(out_prob, opt_ds)
    prob_band = prob_ds.GetRasterBand(1)

    # With overlap>0 the same pixel can be written multiple times; last write wins.
    # For overlap=0 (recommended for this scene) coverage is exact and unique.
    t0 = time.time()
    leu = 0
    bs = max(1, int(args.batch_size))
    lidar_c = len(lidar_bands) if lidar_bands is not None else 1

    for i0 in range(0, n_win, bs):
        batch = coords[i0 : i0 + bs]
        opt_list = []
        lidar_list = []
        for y, x in batch:
            opt_list.append(
                torch.from_numpy(
                    _read_opt_window(opt_ds, x, y, patch, src_order, compute_ndvi=compute_ndvi)
                )
            )
            if lidar_ds is not None:
                lidar_list.append(
                    torch.from_numpy(_read_lidar_window(lidar_ds, x, y, patch, lidar_bands))
                )
            else:
                lidar_list.append(torch.zeros((lidar_c, patch, patch), dtype=torch.float32))
        opt_t = torch.stack(opt_list, 0).to(device)
        lidar_t = torch.stack(lidar_list, 0).to(device)
        with torch.no_grad():
            # Models in this repo already apply Softmax in the classifier head.
            prob = model((opt_t, lidar_t)).cpu().numpy()
        for k, (y, x) in enumerate(batch):
            cls = np.argmax(prob[k], axis=0).astype(np.uint8)
            p1 = np.clip(prob[k, 1], 0.0, 1.0)
            pred_band.WriteArray(cls, x, y)
            prob_band.WriteArray((p1 * 65535.0).round().astype(np.uint16), x, y)
            leu += int((cls == 1).sum())

        done = min(i0 + bs, n_win)
        if done == n_win or done % max(bs * 20, 1) == 0 or i0 == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta = (n_win - done) / max(rate, 1e-6)
            print(
                f"  [{done:,}/{n_win:,}]  {100.0 * done / n_win:5.1f}%  "
                f"{rate:.1f} win/s  ETA {eta / 60.0:.1f} min",
                flush=True,
            )

    pred_ds.FlushCache()
    prob_ds.FlushCache()
    pred_ds = None
    prob_ds = None
    opt_ds = None
    if lidar_ds is not None:
        lidar_ds = None

    total_px = H * W
    print(
        f"  pred -> {out_pred}\n"
        f"  prob -> {out_prob}\n"
        f"  leucaena pixels (raw class map, pre CHM gate): {leu:,} / ~{total_px:,}  "
        f"({time.time() - t0:.0f}s)"
    )
    return out_pred, out_prob


def _block_mask(pred_path: str, mask_path: str, lidar_path: str | None, chm_min: float,
                prob_path: str | None, prob_min: float, aoi_path: str | None, log) -> None:
    """Write a binary uint8 mask (1 = keep as leucaena) aligned to pred_path."""
    pred_ds = gdal.Open(pred_path, gdalconst.GA_ReadOnly)
    if pred_ds is None:
        raise SystemExit(f"[ABORT] cannot open pred: {pred_path}")
    w, h = pred_ds.RasterXSize, pred_ds.RasterYSize
    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(mask_path, w, h, 1, gdal.GDT_Byte,
                        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"])
    out_ds.SetGeoTransform(pred_ds.GetGeoTransform())
    out_ds.SetProjection(pred_ds.GetProjection())
    out_b = out_ds.GetRasterBand(1)
    out_b.SetNoDataValue(NODATA)

    lidar_ds = gdal.Open(lidar_path, gdalconst.GA_ReadOnly) if lidar_path and os.path.isfile(lidar_path) else None
    if chm_min > 0 and lidar_ds is None:
        raise SystemExit("[ABORT] --chm-min-m > 0 requires a readable --lidar.")
    if lidar_ds is not None and (lidar_ds.RasterXSize, lidar_ds.RasterYSize) != (w, h):
        raise SystemExit("[ABORT] lidar size != pred size.")

    prob_ds = gdal.Open(prob_path, gdalconst.GA_ReadOnly) if prob_path and os.path.isfile(prob_path) else None
    aoi = None
    if aoi_path:
        aoi = rasterize_geojson(aoi_path, pred_path, burn_value=1, fill_value=0)
        log(f"  AOI pixels   : {int((aoi == 1).sum()):,}")

    block = 2048
    kept = 0
    for y0 in range(0, h, block):
        ys = min(block, h - y0)
        pred = pred_ds.GetRasterBand(1).ReadAsArray(0, y0, w, ys)
        keep = pred == 1
        if lidar_ds is not None and chm_min > 0:
            chm = lidar_ds.GetRasterBand(1).ReadAsArray(0, y0, w, ys).astype(np.float32)
            keep &= chm >= float(chm_min)
        if prob_ds is not None and prob_min > 0:
            pr = prob_ds.GetRasterBand(1).ReadAsArray(0, y0, w, ys).astype(np.float32)
            scale = prob_ds.GetRasterBand(1).GetScale() or 1.0
            keep &= (pr * float(scale)) >= float(prob_min)
        if aoi is not None:
            keep &= aoi[y0:y0 + ys, :] == 1
        out = keep.astype(np.uint8)
        out_b.WriteArray(out, 0, y0)
        kept += int(out.sum())

    out_ds.FlushCache()
    out_ds = None
    pred_ds = None
    if lidar_ds is not None:
        lidar_ds = None
    if prob_ds is not None:
        prob_ds = None
    log(f"  mask pixels  : {kept:,} (after CHM/prob/AOI gates)")


def _sieve(mask_path: str, sieve_px: int, log) -> None:
    if sieve_px <= 1:
        return
    ds = gdal.Open(mask_path, gdalconst.GA_Update)
    band = ds.GetRasterBand(1)
    gdal.SieveFilter(band, None, band, sieve_px, 8)
    ds.FlushCache()
    ds = None
    log(f"  sieve        : dropped blobs < {sieve_px} px")


def _polygonize(mask_path: str, out_geojson: str, min_area_m2: float, simplify_m: float,
                to_epsg: int, log) -> int:
    src = gdal.Open(mask_path, gdalconst.GA_ReadOnly)
    band = src.GetRasterBand(1)
    src_srs = osr.SpatialReference()
    wkt = src.GetProjection()
    if wkt:
        src_srs.ImportFromWkt(wkt)
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    else:
        src_srs.ImportFromEPSG(31983)

    mem = ogr.GetDriverByName("Memory").CreateDataSource("")
    mem_lyr = mem.CreateLayer("raw", src_srs, ogr.wkbPolygon)
    mem_lyr.CreateField(ogr.FieldDefn("value", ogr.OFTInteger))
    gdal.Polygonize(band, band, mem_lyr, 0, ["8CONNECTED=8"])
    src = None

    dst_srs = src_srs
    ct = None
    if to_epsg and to_epsg > 0:
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(int(to_epsg))
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not dst_srs.IsSame(src_srs):
            ct = osr.CoordinateTransformation(src_srs, dst_srs)

    if os.path.isfile(out_geojson):
        os.remove(out_geojson)
    drv = ogr.GetDriverByName("GeoJSON")
    out_ds = drv.CreateDataSource(out_geojson)
    layer = out_ds.CreateLayer("leucaena_pred", dst_srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("area_m2", ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("source", ogr.OFTString))

    kept = 0
    dropped = 0
    mem_lyr.ResetReading()
    for feat in mem_lyr:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        geom = geom.Clone()
        if simplify_m > 0:
            geom = geom.SimplifyPreserveTopology(float(simplify_m))
        area = geom.GetArea()  # m² in projected CRS (31983)
        if area < float(min_area_m2):
            dropped += 1
            continue
        if ct is not None:
            if geom.Transform(ct) != 0:
                dropped += 1
                continue
        out = ogr.Feature(layer.GetLayerDefn())
        out.SetGeometry(geom)
        out.SetField("area_m2", float(area))
        out.SetField("source", "model_pred")
        layer.CreateFeature(out)
        kept += 1

    out_ds = None
    mem = None
    log(f"  polygons     : kept={kept:,} dropped_small={dropped:,} -> {out_geojson}")
    return kept


def main() -> None:
    args = _parse_args()
    t0 = time.time()

    def log(msg: str) -> None:
        print(msg, flush=True)

    out_dir = args.out_dir or os.path.join(paths.PATH_PREDICTIONS_DIR, f"exp_{args.experiment}")
    pred_path = args.pred
    prob_path = args.prob
    if pred_path is None:
        pred_path, prob_path = _run_inference(args, out_dir)
    elif not os.path.isfile(pred_path):
        raise SystemExit(f"[ABORT] --pred not found: {pred_path}")

    out_geojson = args.out or os.path.join(os.path.dirname(pred_path) or out_dir, "leucaena-pred.geojson")
    mask_path = os.path.splitext(pred_path)[0] + "_leu_mask.tif"
    os.makedirs(os.path.dirname(out_geojson) or ".", exist_ok=True)

    log(f"pred          : {pred_path}")
    log(f"lidar / CHM   : {args.lidar or '(none)'}  chm_min={args.chm_min_m} m")
    log(f"AOI           : {args.aoi or '(full scene)'}")
    log(f"sieve / area  : {args.sieve_px} px / {args.min_area_m2} m²")

    _block_mask(pred_path, mask_path, args.lidar, args.chm_min_m, prob_path, args.prob_min, args.aoi, log)
    _sieve(mask_path, args.sieve_px, log)
    n = _polygonize(mask_path, out_geojson, args.min_area_m2, args.simplify_m, args.to_epsg, log)
    try:
        os.remove(mask_path)
    except OSError:
        pass

    sidecar = os.path.splitext(out_geojson)[0] + ".log.json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({
            "pred": pred_path,
            "out": out_geojson,
            "n_polygons": n,
            "chm_min_m": args.chm_min_m,
            "min_area_m2": args.min_area_m2,
            "sieve_px": args.sieve_px,
            "experiment": args.experiment,
            "elapsed_s": round(time.time() - t0, 1),
        }, f, indent=2)
    log(f"elapsed       : {time.time() - t0:.1f} s")
    log("NOTE: this GeoJSON is a model prediction, not reviewed labels.")


if __name__ == "__main__":
    main()
