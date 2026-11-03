"""
aoi_resolve.py
==============
Resolve um AOI vetorial para os insumos de treino, sobre o POOL central de tiles.

Dado:
  - um pool  <root>/tiles/rgbnir/<tile>.tif  e  <root>/tiles/lidar/rasters/<tile>.tif
  - um AOI   <root>/aois/<aoi>.gpkg  (footprints com campos NOMENC_2K/5K/10K)
  - annotations global  <root>/annotations/<arquivo>.gpkg

Produz (em --out-dir):
  - aoi_rgbnir.vrt      mosaico virtual (sem copiar bytes) dos tiles RGBNIR do AOI
  - aoi_lidar.vrt       idem para os tiles LiDAR (so os que existem no pool)
  - aoi_annotations.gpkg  annotations recortado pela geometria do AOI
  - resolve.json        tiles resolvidos, campo NOMENC usado, caminhos

Como casa AOI -> tiles:
  1. auto-detecta qual coluna NOMENC_* casa com os nomes dos arquivos do pool
     (celulas podem ter varios nomes separados por ';');
  2. fallback: intersecao geometrica dos footprints do AOI com o bbox dos tiles.

Puro rasterio + geopandas (roda no host E no container). O caminho das fontes
do VRT usa --src-prefix (default = a pasta de tiles passada), para o mesmo .vrt
poder apontar p/ C:/... (host) ou /data/tiles (Docker), regenerando por ambiente.

Uso:
  python pipeline/aoi_resolve.py --root C:/00_PHD_MSB_RUN --aoi 261002-riodaspedras \
      --annotations C:/00_PHD_MSB_RUN/annotations/261030-classes.gpkg \
      --out-dir C:/00_PHD_MSB_RUN/datasets/261002-riodaspedras__lbl-v1/_resolve
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

import rasterio

NOMENC_FIELDS = ["NOMENC_2K", "NOMENC_5K", "NOMENC_10K"]

# rasterio dtype -> GDAL VRT dataType
_DT = {
    "uint8": "Byte", "int8": "Int8", "uint16": "UInt16", "int16": "Int16",
    "uint32": "UInt32", "int32": "Int32", "float32": "Float32",
    "float64": "Float64", "cint16": "CInt16", "cint32": "CInt32",
    "cfloat32": "CFloat32", "cfloat64": "CFloat64",
}


# ---------------------------------------------------------------------------
# Indice do pool (stem -> bounds/crs), cacheado em JSON
# ---------------------------------------------------------------------------
def build_tile_index(rgb_dir: Path, cache_path: Path, rebuild: bool = False) -> dict:
    if cache_path.exists() and not rebuild:
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    index: dict[str, dict] = {}
    for tif in sorted(rgb_dir.glob("*.tif")):
        try:
            with rasterio.open(tif) as ds:
                b = ds.bounds
                index[tif.stem] = {
                    "bounds": [b.left, b.bottom, b.right, b.top],
                    "crs": ds.crs.to_string() if ds.crs else None,
                }
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] falha indexando {tif.name}: {exc}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


# ---------------------------------------------------------------------------
# Resolucao AOI -> tiles
# ---------------------------------------------------------------------------
def _split_names(val) -> list[str]:
    if val is None:
        return []
    return [p.strip() for p in str(val).replace(",", ";").split(";") if p.strip()]


def resolve_tiles(aoi_gpkg: Path, index: dict, rgb_dir: Path,
                  nomenc_field: str | None = None):
    import geopandas as gpd

    gdf = gpd.read_file(aoi_gpkg)
    pool_stems = set(index.keys())
    pool_files = {p.stem for p in rgb_dir.glob("*.tif")} or pool_stems

    # 1) por NOME (campo NOMENC que melhor casa com os arquivos do pool)
    candidates = [nomenc_field] if nomenc_field else \
        [c for c in NOMENC_FIELDS if c in gdf.columns]
    best = None
    for col in candidates:
        if col not in gdf.columns:
            continue
        names = set()
        for v in gdf[col].tolist():
            names.update(_split_names(v))
        hits = names & pool_files
        if best is None or len(hits) > len(best[2]):
            best = (col, names, hits)
    if best and best[2]:
        col, names, hits = best
        tiles = sorted(hits)
        return tiles, f"nomenc:{col}"

    # 2) fallback geometrico (intersecao dos footprints do AOI x bbox dos tiles)
    from shapely.geometry import box
    aoi_union = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") \
        else gdf.geometry.unary_union
    tiles = []
    for stem, meta in index.items():
        bl, bb, br, bt = meta["bounds"]
        if box(bl, bb, br, bt).intersects(aoi_union):
            tiles.append(stem)
    return sorted(tiles), "geometric"


# ---------------------------------------------------------------------------
# VRT (mosaico virtual, puro XML)
# ---------------------------------------------------------------------------
def write_vrt(stems: list[str], src_dir: Path, out_vrt: Path,
              src_prefix: str | None = None) -> dict:
    """Monta um .vrt mosaicando <src_dir>/<stem>.tif. src_prefix e o prefixo
    escrito no <SourceFilename> (default = src_dir absoluto)."""
    if src_prefix is None:
        src_prefix = str(src_dir).replace("\\", "/")
    metas = []
    for stem in stems:
        f = src_dir / f"{stem}.tif"
        if not f.exists():
            continue
        with rasterio.open(f) as ds:
            t = ds.transform
            metas.append({
                "stem": stem, "w": ds.width, "h": ds.height,
                "count": ds.count, "dtype": ds.dtypes[0],
                "nodata": ds.nodata,
                "left": ds.bounds.left, "top": ds.bounds.top,
                "px": t.a, "py": -t.e,
                "crs": ds.crs.to_wkt() if ds.crs else "",
            })
    if not metas:
        return {"ok": False, "n": 0}

    px = metas[0]["px"]
    py = metas[0]["py"]
    count = metas[0]["count"]
    dtype = metas[0]["dtype"]
    nodata = metas[0]["nodata"]
    crs_wkt = metas[0]["crs"]
    xmin = min(m["left"] for m in metas)
    ymax = max(m["top"] for m in metas)
    xmax = max(m["left"] + m["w"] * m["px"] for m in metas)
    ymin = min(m["top"] - m["h"] * m["py"] for m in metas)
    W = int(round((xmax - xmin) / px))
    H = int(round((ymax - ymin) / py))
    gdal_dt = _DT.get(dtype, "Float32")

    lines = [f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">']
    if crs_wkt:
        lines.append(f"  <SRS>{escape(crs_wkt)}</SRS>")
    lines.append(
        f"  <GeoTransform>{xmin}, {px}, 0.0, {ymax}, 0.0, {-py}</GeoTransform>")
    for band in range(1, count + 1):
        lines.append(f'  <VRTRasterBand dataType="{gdal_dt}" band="{band}">')
        if nodata is not None:
            lines.append(f"    <NoDataValue>{nodata}</NoDataValue>")
        for m in metas:
            xoff = int(round((m["left"] - xmin) / px))
            yoff = int(round((ymax - m["top"]) / py))
            src = f'{src_prefix}/{m["stem"]}.tif'
            lines.append("    <SimpleSource>")
            lines.append(
                f'      <SourceFilename relativeToVRT="0">{escape(src)}</SourceFilename>')
            lines.append(f"      <SourceBand>{band}</SourceBand>")
            lines.append(
                f'      <SourceProperties RasterXSize="{m["w"]}" RasterYSize="{m["h"]}" '
                f'DataType="{_DT.get(m["dtype"], "Float32")}"/>')
            lines.append(
                f'      <SrcRect xOff="0" yOff="0" xSize="{m["w"]}" ySize="{m["h"]}"/>')
            lines.append(
                f'      <DstRect xOff="{xoff}" yOff="{yoff}" xSize="{m["w"]}" ySize="{m["h"]}"/>')
            if nodata is not None:
                lines.append(f"      <NODATA>{nodata}</NODATA>")
            lines.append("    </SimpleSource>")
        lines.append("  </VRTRasterBand>")
    lines.append("</VRTDataset>")

    out_vrt.parent.mkdir(parents=True, exist_ok=True)
    out_vrt.write_text("\n".join(lines), encoding="utf-8")
    return {"ok": True, "n": len(metas), "width": W, "height": H,
            "bands": count, "dtype": dtype}


# ---------------------------------------------------------------------------
# Recorte das anotacoes pelo AOI
# ---------------------------------------------------------------------------
def clip_annotations(annot_path: Path, aoi_gpkg: Path, out_path: Path) -> dict:
    import geopandas as gpd

    aoi = gpd.read_file(aoi_gpkg)
    ann = gpd.read_file(annot_path)
    if ann.crs != aoi.crs:
        aoi = aoi.to_crs(ann.crs)
    aoi_union = aoi.geometry.union_all() if hasattr(aoi.geometry, "union_all") \
        else aoi.geometry.unary_union
    clipped = gpd.clip(ann, aoi_union)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_file(out_path, driver="GPKG")
    return {"features_in": len(ann), "features_out": len(clipped)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help="PHD_RUN_ROOT (deriva tiles/aois/annotations).")
    p.add_argument("--aoi", default=None, help="Nome do AOI (=> aois/<aoi>.gpkg).")
    p.add_argument("--aoi-gpkg", default=None, help="Caminho explicito do gpkg do AOI.")
    p.add_argument("--tiles-dir", default=None, help="Pasta tiles/ (default <root>/tiles).")
    p.add_argument("--annotations", default=None, help="gpkg global de anotacoes.")
    p.add_argument("--out-dir", required=True, help="Saida (vrt + clip + resolve.json).")
    p.add_argument("--nomenc-field", default=None, help="Forcar coluna NOMENC_*.")
    p.add_argument("--src-prefix", default=None,
                   help="Prefixo escrito no SourceFilename do VRT (ex.: /data/tiles). "
                        "Default = pasta de tiles usada p/ ler.")
    p.add_argument("--rebuild-index", action="store_true")
    a = p.parse_args()

    root = Path(a.root) if a.root else None
    tiles_dir = Path(a.tiles_dir) if a.tiles_dir else (root / "tiles")
    rgb_dir = tiles_dir / "rgbnir"
    lid_dir = tiles_dir / "lidar" / "rasters"
    aoi_gpkg = Path(a.aoi_gpkg) if a.aoi_gpkg else (root / "aois" / f"{a.aoi}.gpkg")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not aoi_gpkg.exists():
        raise SystemExit(f"[ABORT] AOI gpkg nao encontrado: {aoi_gpkg}")

    print(f"Pool rgbnir: {rgb_dir}")
    index = build_tile_index(rgb_dir, tiles_dir / ".tile_index.json", a.rebuild_index)
    print(f"  {len(index)} tiles indexados")

    tiles, how = resolve_tiles(aoi_gpkg, index, rgb_dir, a.nomenc_field)
    print(f"AOI '{aoi_gpkg.stem}' -> {len(tiles)} tiles (via {how})")
    if not tiles:
        raise SystemExit("[ABORT] nenhum tile resolvido para o AOI.")

    rgb_prefix = a.src_prefix + "/rgbnir" if a.src_prefix else None
    lid_prefix = a.src_prefix + "/lidar/rasters" if a.src_prefix else None
    rgb_vrt = out_dir / "aoi_rgbnir.vrt"
    lid_vrt = out_dir / "aoi_lidar.vrt"
    rgb_info = write_vrt(tiles, rgb_dir, rgb_vrt, rgb_prefix)
    lid_info = write_vrt(tiles, lid_dir, lid_vrt, lid_prefix)
    print(f"  rgbnir VRT: {rgb_info}")
    print(f"  lidar  VRT: {lid_info} (tiles sem lidar sao ignorados)")

    annot_info = None
    if a.annotations:
        annot_path = Path(a.annotations)
        if annot_path.exists():
            annot_info = clip_annotations(annot_path, aoi_gpkg,
                                          out_dir / "aoi_annotations.gpkg")
            print(f"  annotations recortado: {annot_info}")
        else:
            print(f"  [warn] annotations nao encontrado: {annot_path}")

    resolve = {
        "aoi": aoi_gpkg.stem, "resolved_via": how, "n_tiles": len(tiles),
        "tiles": tiles, "rgbnir_vrt": str(rgb_vrt),
        "lidar_vrt": str(lid_vrt) if lid_info.get("ok") else None,
        "rgbnir": rgb_info, "lidar": lid_info, "annotations": annot_info,
    }
    (out_dir / "resolve.json").write_text(json.dumps(resolve, indent=2), encoding="utf-8")
    print(f"\nresolve.json -> {out_dir / 'resolve.json'}")


if __name__ == "__main__":
    main()
