"""
align_lidar_to_optical.py
=========================
Passo 2 do orquestrador (abordagem C). Pega os produtos de CHM gerados pelo R
(lidR: <base>_CHM.tif) e monta o raster LiDAR final usado no treino.

Dois mundos:
  • COM optico (--optical): mosaica os CHM e alinha (warp) a grade do RGBNIR,
    pixel a pixel. Gera 2 bandas [CHM, INTENSITY].
  • SEM optico: mantem o CHM na propria grade do LiDAR (nativo), tambem 2 bandas.

INTENSITY:
  • default: gerada via PDAL dos .laz (--laz-dir), alinhada a mesma grade.
  • --no-intensity: banda 2 preenchida com zeros (mantem lidar_channels=2).

Rodar DENTRO do Docker (tem GDAL + PDAL):
  python align_lidar_to_optical.py --chm-dir <produtos_R> --optical <rgbnir.tif> \
      --laz-dir <laz> --out <lidar/rasters/CENA.tif>
  python align_lidar_to_optical.py --chm-dir <produtos_R> --no-optical \
      --no-intensity --out <lidar/rasters/CENA.tif>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile

import numpy as np
from osgeo import gdal, gdalconst

# permite importar utils.* quando rodado de /workspace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.lidar import TileGrid, build_single_writer_pipeline, run_pdal_pipeline  # noqa: E402

NODATA = -9999.0


def _args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chm-dir", required=True, help="Pasta com <base>_CHM.tif (saida do R).")
    p.add_argument("--chm-glob", default="*_CHM.tif", help="Padrao dos CHM (default: %(default)s).")
    p.add_argument("--optical", default=None, help="RGBNIR de referencia (define grade/CRS).")
    p.add_argument("--no-optical", action="store_true",
                   help="Nao alinha; mantem o CHM na grade nativa do LiDAR.")
    p.add_argument("--out", required=True, help="GeoTIFF 2-band [CHM, INTENSITY] de saida.")
    p.add_argument("--laz-dir", default=None, help="Pasta com .laz (para gerar INTENSITY via PDAL).")
    p.add_argument("--laz-glob", default="*.la[sz]", help="Padrao dos LAZ (default: %(default)s).")
    p.add_argument("--laz-names", default=None,
                   help="Stems LAZ (virgula) a usar; filtra o pool sem pasta tmp.")
    p.add_argument("--no-intensity", action="store_true",
                   help="Nao gera intensidade; banda 2 = zeros.")
    p.add_argument("--laz-epsg", type=int, default=None,
                   help="EPSG dos LAZ sem CRS embutido (default: do optico).")
    p.add_argument("--resolution", type=float, default=1.0,
                   help="Resolucao nativa PDAL p/ intensidade (m).")
    return p.parse_args()


def _build_vrt(tifs, vrt_path):
    gdal.BuildVRT(vrt_path, tifs,
                  options=gdal.BuildVRTOptions(srcNodata=NODATA, VRTNodata=NODATA))
    return vrt_path


def _warp(src, grid: TileGrid, dst, resample):
    opts = gdal.WarpOptions(
        format="GTiff", width=grid.width, height=grid.height,
        outputBounds=(grid.xmin, grid.ymin, grid.xmax, grid.ymax),
        dstSRS=grid.projection_wkt, resampleAlg=resample,
        srcNodata=NODATA, dstNodata=NODATA,
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
        warpMemoryLimit=512,
    )
    if gdal.Warp(dst, src, options=opts) is None:
        raise RuntimeError(f"gdal.Warp falhou: {src} -> {dst}")


def _grid_from_tif(path) -> TileGrid:
    return TileGrid.from_geotiff(path)


def _laz_bounds(laz, in_srs, buffer_m):
    import pdal
    reader = {"type": "readers.las", "filename": laz}
    if in_srs is not None:
        reader["default_srs"] = f"EPSG:{in_srs}"
    pl = pdal.Pipeline(json.dumps({"pipeline": [reader]}))
    qi = pl.quickinfo
    info = qi.get("readers.las") or next(iter(qi.values()))
    b = info["bounds"]
    return (b["minx"] - buffer_m, b["maxx"] + buffer_m,
            b["miny"] - buffer_m, b["maxy"] + buffer_m)


def _intensity_mosaic(laz_paths, target_epsg, laz_epsg, res, tmp):
    """Gera intensidade nativa por LAZ e devolve um VRT mosaicado."""
    int_tifs = []
    for laz in laz_paths:
        stem = os.path.splitext(os.path.basename(laz))[0]
        out = os.path.join(tmp, f"{stem}_int.tif")
        try:
            bounds = _laz_bounds(laz, laz_epsg, buffer_m=res * 2.0)
            pipe = build_single_writer_pipeline(
                laz_path=laz, out_path=out, product="intensity",
                resolution_m=res, target_epsg=target_epsg, bounds=bounds, in_srs=laz_epsg,
            )
            run_pdal_pipeline(pipe)
            if os.path.isfile(out):
                int_tifs.append(out)
                print(f"  int ok: {stem}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] intensidade falhou p/ {stem}: {exc}")
    if not int_tifs:
        return None
    return _build_vrt(int_tifs, os.path.join(tmp, "int_mosaic.vrt"))


def _stack_2band(chm_path, int_path, out_path, grid: TileGrid):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(out_path, grid.width, grid.height, 2, gdal.GDT_Float32,
                    options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER", "PREDICTOR=3"])
    ds.SetGeoTransform(grid.geo_transform)
    ds.SetProjection(grid.projection_wkt)

    chm_ds = gdal.Open(chm_path, gdalconst.GA_ReadOnly)
    chm = chm_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    chm[np.isclose(chm, NODATA)] = 0.0
    chm[~np.isfinite(chm)] = 0.0

    if int_path is not None:
        int_ds = gdal.Open(int_path, gdalconst.GA_ReadOnly)
        inten = int_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        inten[np.isclose(inten, NODATA)] = 0.0
        inten[~np.isfinite(inten)] = 0.0
        int_ds = None
    else:
        inten = np.zeros_like(chm, dtype=np.float32)

    b1 = ds.GetRasterBand(1); b1.SetDescription("CHM"); b1.SetNoDataValue(NODATA); b1.WriteArray(chm)
    b2 = ds.GetRasterBand(2); b2.SetDescription("INTENSITY"); b2.SetNoDataValue(NODATA); b2.WriteArray(inten)
    ds.FlushCache()
    ds = None; chm_ds = None
    return {
        "chm_min": float(chm.min()), "chm_max": float(chm.max()),
        "chm_mean": float(chm.mean()), "covered": int((chm > 0).sum()),
        "total": int(chm.size), "int_max": float(inten.max()),
    }


def main():
    a = _args()
    chm_tifs = sorted(glob.glob(os.path.join(a.chm_dir, a.chm_glob)))
    if not chm_tifs:
        raise SystemExit(f"[ABORT] nenhum CHM em {a.chm_dir}/{a.chm_glob} "
                         f"(rode o passo R primeiro).")
    print(f"CHM products: {len(chm_tifs)} em {a.chm_dir}")

    tmp = tempfile.mkdtemp(prefix="lidar_align_")
    chm_vrt = _build_vrt(chm_tifs, os.path.join(tmp, "chm_mosaic.vrt"))

    use_optical = (not a.no_optical) and a.optical
    if use_optical:
        if not os.path.isfile(a.optical):
            raise SystemExit(f"[ABORT] optico nao encontrado: {a.optical}")
        grid = _grid_from_tif(a.optical)
        laz_epsg = a.laz_epsg if a.laz_epsg is not None else grid.epsg_code
        print(f"ALINHANDO ao optico: {grid.width}x{grid.height} EPSG:{grid.epsg_code}")
        chm_grid = os.path.join(tmp, "chm_grid.tif")
        _warp(chm_vrt, grid, chm_grid, "bilinear")
    else:
        print("SEM optico: mantendo CHM na grade nativa do LiDAR.")
        chm_grid = os.path.join(tmp, "chm_native.tif")
        gdal.Translate(chm_grid, chm_vrt,
                       options=gdal.TranslateOptions(
                           creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]))
        grid = _grid_from_tif(chm_grid)
        laz_epsg = a.laz_epsg if a.laz_epsg is not None else grid.epsg_code

    int_grid = None
    if not a.no_intensity and a.laz_dir:
        laz_paths = sorted(glob.glob(os.path.join(a.laz_dir, a.laz_glob)))
        if a.laz_names:
            want = {n.strip() for n in a.laz_names.split(",") if n.strip()}
            laz_paths = [p for p in laz_paths
                         if os.path.splitext(os.path.basename(p))[0] in want]
        if laz_paths:
            print(f"Gerando INTENSITY de {len(laz_paths)} LAZ (PDAL)...")
            int_vrt = _intensity_mosaic(laz_paths, grid.epsg_code, laz_epsg, a.resolution, tmp)
            if int_vrt is not None:
                int_grid = os.path.join(tmp, "int_grid.tif")
                _warp(int_vrt, grid, int_grid, "average")
        else:
            print(f"[WARN] nenhum LAZ em {a.laz_dir}; intensidade = zeros.")
    elif a.no_intensity:
        print("INTENSITY desativada (--no-intensity): banda 2 = zeros.")
    else:
        print("[WARN] sem --laz-dir; intensidade = zeros.")

    m = _stack_2band(chm_grid, int_grid, a.out, grid)
    print(f"\nLiDAR final -> {a.out}")
    print(f"CHM: min={m['chm_min']:.2f} max={m['chm_max']:.2f} mean={m['chm_mean']:.3f} m")
    print(f"INT: max={m['int_max']:.1f}")
    print(f"cobertura (CHM>0): {m['covered']:,}/{m['total']:,} "
          f"({100.0 * m['covered'] / max(m['total'], 1):.1f}%)")


if __name__ == "__main__":
    main()
