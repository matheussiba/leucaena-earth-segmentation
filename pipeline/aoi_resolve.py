"""
aoi_resolve.py
==============
Resolve um AOI vetorial para os insumos de treino, sobre o POOL central de tiles.

Dado:
  - um pool  <root>/tiles/optical/rgbnir/<tile>.tif  e  <root>/tiles/lidar/rasters/<tile>.tif
  - um AOI   <root>/aois/<aoi>.gpkg  (footprints com campos NOMENC_2K/5K/10K)
  - annotations global  <root>/annotations/<arquivo>.gpkg

Produz (em --out-dir):
  - aoi_rgbnir.vrt/.tif   mosaico RGBNIR do AOI (tif se houver mask)
  - aoi_lidar.vrt/.tif    idem LiDAR
  - aoi_annotations.gpkg  annotations recortado (AOI ou mask)
  - resolve.json

Mask opcional (treino rapido / sem buraco LiDAR):
  aois/<aoi>-mask.gpkg  ou  --mask-gpkg PATH
  Depois dos VRTs, recorta optico + lidar + annotations ao poligono da mask.

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
from pathlib import Path
from xml.sax.saxutils import escape

import rasterio

NOMENC_FIELDS = ["NOMENC_2K", "NOMENC_5K", "NOMENC_10K"]

# Pool canonico: tiles/optical/rgbnir/<tile>.tif
RGBNIR_REL = "optical/rgbnir"


def find_rgbnir_dir(tiles_dir: Path) -> Path:
    """Pasta do pool RGBNIR 4 bandas: <tiles>/optical/rgbnir."""
    return Path(tiles_dir) / "optical" / "rgbnir"


def build_optical_pool(tiles_dir: Path) -> tuple[dict, dict[str, Path]]:
    """Indice + caminho por stem em tiles/optical/rgbnir."""
    tiles_dir = Path(tiles_dir)
    rgb_dir = find_rgbnir_dir(tiles_dir)
    index: dict[str, dict] = {}
    stem_paths: dict[str, Path] = {}
    if not rgb_dir.is_dir():
        return index, stem_paths
    for tif in sorted(rgb_dir.glob("*.tif")):
        try:
            with rasterio.open(tif) as ds:
                b = ds.bounds
                index[tif.stem] = {
                    "bounds": [b.left, b.bottom, b.right, b.top],
                    "crs": ds.crs.to_string() if ds.crs else None,
                    "rel": f"{RGBNIR_REL}/{tif.name}",
                }
            stem_paths[tif.stem] = tif
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] falha indexando {tif.name}: {exc}")
    return index, stem_paths


def rgbnir_rel_under_tiles(rgb_dir: Path | None = None, tiles_dir: Path | None = None) -> str:
    """Relativo a tiles/ p/ SourceFilename do VRT."""
    return RGBNIR_REL

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


def resolve_tiles(aoi_gpkg: Path, index: dict, rgb_dir: Path | None = None,
                  nomenc_field: str | None = None):
    """index deve conter so stems com arquivo real no disco (ver build_optical_pool)."""
    import geopandas as gpd

    gdf = gpd.read_file(aoi_gpkg)
    pool_files = set(index.keys())
    if rgb_dir is not None:
        try:
            pool_files |= {p.stem for p in Path(rgb_dir).glob("*.tif")}
        except OSError:
            pass

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
        col, _names, hits = best
        return sorted(hits), f"nomenc:{col}"

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
    """Monta um .vrt mosaicando <src_dir>/<stem>.tif.

    src_prefix: prefixo no SourceFilename (ex. /data/tiles/optical/rgbnir).
    Default = caminho absoluto de src_dir.
    """
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
                "stem": stem,
                "src": f"{src_prefix.rstrip('/')}/{stem}.tif",
                "w": ds.width, "h": ds.height,
                "count": ds.count, "dtype": ds.dtypes[0],
                "nodata": ds.nodata,
                "left": ds.bounds.left, "top": ds.bounds.top,
                "px": t.a, "py": -t.e,
                "crs": ds.crs.to_wkt() if ds.crs else "",
            })
    if not metas:
        return {"ok": False, "n": 0,
                "missing": [s for s in stems if not (src_dir / f"{s}.tif").exists()]}

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
            src = m["src"]
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
# Mask / recorte
# ---------------------------------------------------------------------------
_MASK_EXTS = (".gpkg", ".geojson", ".json", ".shp")


def find_aoi_mask(aois_dir: Path, aoi: str, explicit: Path | None = None) -> Path | None:
    """aois/<aoi>-mask.{gpkg,geojson,shp} (ou caminho explicito)."""
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_file() else None
    if not aoi:
        return None
    base = Path(aois_dir) / f"{aoi}-mask"
    for ext in _MASK_EXTS:
        cand = Path(str(base) + ext)
        if cand.is_file():
            return cand
    return None


def read_vector(path: Path):
    """Le vetor (GPKG/GeoJSON/SHP) com mensagem clara se o arquivo for invalido."""
    import geopandas as gpd

    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"[ABORT] vetor nao encontrado: {path}")
    size = path.stat().st_size
    if size < 64:
        raise SystemExit(
            f"[ABORT] arquivo muito pequeno/vazio ({size} bytes): {path}\n"
            f"        Exporte de novo no QGIS: botao direito na layer → Export → "
            f"Save Features As → Format=GeoPackage."
        )
    last_err: Exception | None = None
    # 1) pyogrio default  2) fiona  3) GPKG: prefix (Windows path quirks)
    attempts = [
        {},
        {"engine": "fiona"},
        {"engine": "pyogrio"},
    ]
    for kwargs in attempts:
        try:
            gdf = gpd.read_file(path, **kwargs)
            if gdf is not None and len(gdf) >= 0:
                return gdf
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    # prefixo DRIVER: (ajuda em alguns builds GDAL no Windows)
    for prefix in ("GPKG:", "GeoJSON:", "ESRI Shapefile:"):
        try:
            return gpd.read_file(f"{prefix}{path}")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise SystemExit(
        f"[ABORT] nao consegui ler a mask como vetor GIS: {path}\n"
        f"        Erro: {type(last_err).__name__}: {last_err}\n"
        f"        O arquivo existe mas NAO e um GeoPackage/GeoJSON/Shapefile valido "
        f"(extensao .gpkg com conteudo errado e comum).\n"
        f"        No QGIS: abra o poligono → Export → Save Features As → "
        f"Format = GeoPackage → salve como:\n"
        f"          aois/<aoi>-mask.gpkg\n"
        f"        Confira depois:  (Get-Item '...-mask.gpkg').Length"
    )


def clip_annotations(annot_path: Path, clip_gpkg: Path, out_path: Path) -> dict:
    import geopandas as gpd

    clip = read_vector(clip_gpkg)
    ann = read_vector(annot_path)
    if ann.crs != clip.crs:
        clip = clip.to_crs(ann.crs)
    clip_union = clip.geometry.union_all() if hasattr(clip.geometry, "union_all") \
        else clip.geometry.unary_union
    clipped = gpd.clip(ann, clip_union)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_file(out_path, driver="GPKG")
    return {"features_in": len(ann), "features_out": len(clipped),
            "clip": str(clip_gpkg)}


def crop_raster_to_mask(src_path: Path, mask_gpkg: Path, out_path: Path,
                        nodata_fill: float = 0.0) -> dict:
    """Recorta raster/VRT ao poligono da mask (crop=True); fora = nodata_fill."""
    from rasterio.mask import mask as rio_mask

    gdf = read_vector(mask_gpkg)
    if gdf.empty or gdf.geometry.is_empty.all():
        raise SystemExit(f"[ABORT] mask vazia: {mask_gpkg}")

    with rasterio.open(src_path) as src:
        if gdf.crs and src.crs and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        geoms = [g.__geo_interface__ for g in gdf.geometry
                 if g is not None and not g.is_empty]
        if not geoms:
            raise SystemExit(f"[ABORT] mask sem geometria valida: {mask_gpkg}")
        fill = src.nodata if src.nodata is not None else nodata_fill
        data, transform = rio_mask(
            src, geoms, crop=True, all_touched=True, filled=True, nodata=fill,
        )
        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform,
            "compress": "deflate",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        })
        if meta.get("nodata") is None:
            meta["nodata"] = fill
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data)
        return {
            "ok": True,
            "width": int(data.shape[2]),
            "height": int(data.shape[1]),
            "bands": int(data.shape[0]),
            "path": str(out_path).replace("\\", "/"),
        }


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
    p.add_argument("--require-lidar", action="store_true",
                   help="ABORT se aoi_lidar.vrt nao puder ser montado "
                        "(obrigatorio p/ latefusion).")
    p.add_argument("--mask-gpkg", default=None,
                   help="Poligono de recorte (default: aois/<aoi>-mask.gpkg se existir).")
    p.add_argument("--no-mask", action="store_true",
                   help="Ignora aois/<aoi>-mask.gpkg mesmo se existir.")
    p.add_argument("--force", action="store_true",
                   help="Refaz VRT/crop mesmo se aoi_rgbnir.tif/.vrt ja existir.")
    a = p.parse_args()

    root = Path(a.root) if a.root else None
    tiles_dir = Path(a.tiles_dir) if a.tiles_dir else (root / "tiles")
    aois_dir = (root / "aois") if root else Path(".")
    rgb_dir = find_rgbnir_dir(tiles_dir)
    lid_dir = tiles_dir / "lidar" / "rasters"
    aoi_gpkg = Path(a.aoi_gpkg) if a.aoi_gpkg else (root / "aois" / f"{a.aoi}.gpkg")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not aoi_gpkg.exists():
        raise SystemExit(f"[ABORT] AOI gpkg nao encontrado: {aoi_gpkg}")

    mask_gpkg = None
    if not a.no_mask:
        mask_gpkg = find_aoi_mask(
            aois_dir, a.aoi or aoi_gpkg.stem,
            Path(a.mask_gpkg) if a.mask_gpkg else None,
        )
        if mask_gpkg:
            print(f"Mask AOI: {mask_gpkg}")
        elif a.mask_gpkg:
            raise SystemExit(f"[ABORT] --mask-gpkg nao encontrado: {a.mask_gpkg}")

    # Cache: se saidas ja existem, nao recorta de novo (train/predict ficam rapidos)
    meta_path = out_dir / "resolve.json"
    rgb_tif = out_dir / "aoi_rgbnir.tif"
    lid_tif = out_dir / "aoi_lidar.tif"
    rgb_vrt_p = out_dir / "aoi_rgbnir.vrt"
    lid_vrt_p = out_dir / "aoi_lidar.vrt"
    ann_p = out_dir / "aoi_annotations.gpkg"
    if not a.force and meta_path.is_file():
        rgb_ok = rgb_tif.is_file() if mask_gpkg else (rgb_tif.is_file() or rgb_vrt_p.is_file())
        lid_ok = True
        if a.require_lidar:
            lid_ok = lid_tif.is_file() if mask_gpkg else (lid_tif.is_file() or lid_vrt_p.is_file())
        ann_ok = (not a.annotations) or ann_p.is_file()
        if rgb_ok and lid_ok and ann_ok:
            print(f"[skip] resolve ja pronto em {out_dir} (use --force p/ refazer)")
            print(f"resolve.json -> {meta_path}")
            return

    index, stem_paths = build_optical_pool(tiles_dir)
    # cache sempre reconstruido do disco
    (tiles_dir / ".tile_index.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rel"}
                    for k, v in index.items()}, indent=2),
        encoding="utf-8")
    print(f"Pool rgbnir: {rgb_dir}  ({len(stem_paths)} tiles)")
    if not stem_paths:
        raise SystemExit(
            f"[ABORT] nenhum .tif em {rgb_dir}. "
            "Coloque o RGBNIR 4 bandas em tiles/optical/rgbnir/."
        )

    tiles, how = resolve_tiles(aoi_gpkg, index, rgb_dir, a.nomenc_field)
    print(f"AOI '{aoi_gpkg.stem}' -> {len(tiles)} tiles (via {how})")
    if not tiles:
        import geopandas as gpd
        gdf = gpd.read_file(aoi_gpkg)
        wanted = set()
        for col in NOMENC_FIELDS:
            if col in gdf.columns:
                for v in gdf[col].tolist():
                    wanted.update(_split_names(v))
        missing_wanted = sorted(wanted - set(stem_paths))[:8]
        raise SystemExit(
            "[ABORT] nenhum tile resolvido para o AOI. "
            f"Pool tem {len(stem_paths)} RGBNIR em {rgb_dir}; "
            f"NOMENC sem arquivo (amostra): {missing_wanted}."
        )

    # Docker prefix so para VRT consumido no container SEM mask.
    # Com mask o crop roda no HOST → VRT precisa de paths do host (senao
    # /data/tiles/... nao existe e o rasterio quebra).
    docker_rgb_prefix = (f"{a.src_prefix.rstrip('/')}/{RGBNIR_REL}"
                         if a.src_prefix else None)
    docker_lid_prefix = (f"{a.src_prefix.rstrip('/')}/lidar/rasters"
                         if a.src_prefix else None)
    rgb_vrt = out_dir / "aoi_rgbnir.vrt"
    lid_vrt = out_dir / "aoi_lidar.vrt"
    for p in (rgb_vrt, lid_vrt, rgb_tif, lid_tif,
              out_dir / "aoi_rgbnir_full.vrt", out_dir / "aoi_lidar_full.vrt"):
        if p.exists():
            p.unlink()

    # Paths finais usados por labels/train/predict (podem virar .tif apos mask)
    rgb_final: Path
    lid_final: Path | None
    mask_info = None

    if mask_gpkg is not None:
        # VRT temporario com paths ABSOLUTOS do host p/ o crop
        host_rgb_vrt = out_dir / "_host_rgbnir.vrt"
        host_lid_vrt = out_dir / "_host_lidar.vrt"
        rgb_info = write_vrt(tiles, rgb_dir, host_rgb_vrt, src_prefix=None)
        lid_info = write_vrt(tiles, lid_dir, host_lid_vrt, src_prefix=None)
        print(f"  rgbnir VRT (host): {rgb_info}")
        print(f"  lidar  VRT (host): {lid_info} (tiles sem lidar sao ignorados)")
        if not rgb_info.get("ok"):
            raise SystemExit(
                f"[ABORT] falha ao montar VRT host rgbnir: {rgb_info}. "
                f"Verifique se os .tif do AOI existem em {rgb_dir}."
            )
        if not lid_info.get("ok"):
            msg = (
                f"sem LiDAR em {lid_dir} para tiles {tiles} "
                f"(missing={lid_info.get('missing')}). "
                f"Gere com: .\\run_lidar.cmd -aoi {aoi_gpkg.stem} -Engine python -Force"
            )
            if a.require_lidar:
                raise SystemExit(f"[ABORT] {msg}")
            print(f"[warn] {msg}")

        print(f"  recortando rasters pela mask {mask_gpkg.name} ...")
        rgb_tif = out_dir / "aoi_rgbnir.tif"
        mask_info = {"mask_gpkg": str(mask_gpkg).replace("\\", "/")}
        rgb_crop = crop_raster_to_mask(host_rgb_vrt, mask_gpkg, rgb_tif, nodata_fill=0.0)
        print(f"  rgbnir crop: {rgb_crop}")
        rgb_final = rgb_tif
        mask_info["rgbnir"] = rgb_crop
        lid_final = None
        if lid_info.get("ok") and host_lid_vrt.is_file():
            lid_tif = out_dir / "aoi_lidar.tif"
            lid_crop = crop_raster_to_mask(host_lid_vrt, mask_gpkg, lid_tif, nodata_fill=0.0)
            print(f"  lidar  crop: {lid_crop}")
            lid_final = lid_tif
            mask_info["lidar"] = lid_crop
        # guarda VRT full com prefixo docker (debug / uso sem mask)
        if docker_rgb_prefix:
            write_vrt(tiles, rgb_dir, out_dir / "aoi_rgbnir_full.vrt", docker_rgb_prefix)
        if lid_info.get("ok") and docker_lid_prefix:
            write_vrt(tiles, lid_dir, out_dir / "aoi_lidar_full.vrt", docker_lid_prefix)
        for tmp in (host_rgb_vrt, host_lid_vrt):
            if tmp.exists():
                tmp.unlink()
    else:
        rgb_info = write_vrt(tiles, rgb_dir, rgb_vrt, docker_rgb_prefix)
        lid_info = write_vrt(tiles, lid_dir, lid_vrt, docker_lid_prefix)
        print(f"  rgbnir VRT: {rgb_info}")
        print(f"  lidar  VRT: {lid_info} (tiles sem lidar sao ignorados)")
        if not rgb_info.get("ok"):
            raise SystemExit(
                f"[ABORT] falha ao montar aoi_rgbnir.vrt: {rgb_info}. "
                f"Verifique se os .tif do AOI existem em {rgb_dir}."
            )
        if not lid_info.get("ok"):
            msg = (
                f"sem LiDAR em {lid_dir} para tiles {tiles} "
                f"(missing={lid_info.get('missing')}). "
                f"Gere com: .\\run_lidar.cmd -aoi {aoi_gpkg.stem} -Engine python -Force"
            )
            if a.require_lidar:
                raise SystemExit(f"[ABORT] {msg}")
            print(f"[warn] {msg}")
        rgb_final = rgb_vrt
        lid_final = lid_vrt if lid_info.get("ok") else None

    clip_for_ann = mask_gpkg if mask_gpkg is not None else aoi_gpkg
    annot_info = None
    if a.annotations:
        annot_path = Path(a.annotations)
        if annot_path.exists():
            annot_info = clip_annotations(annot_path, clip_for_ann,
                                          out_dir / "aoi_annotations.gpkg")
            print(f"  annotations recortado ({'mask' if mask_gpkg else 'aoi'}): {annot_info}")
        else:
            print(f"  [warn] annotations nao encontrado: {annot_path}")

    resolve = {
        "aoi": aoi_gpkg.stem, "resolved_via": how, "n_tiles": len(tiles),
        "tiles": tiles,
        "tile_paths": {t: str(stem_paths[t]).replace("\\", "/") for t in tiles},
        "mask": mask_info,
        "rgbnir_vrt": str(rgb_final).replace("\\", "/"),
        "lidar_vrt": str(lid_final).replace("\\", "/") if lid_final else None,
        "rgbnir": rgb_info if mask_info is None else mask_info.get("rgbnir", rgb_info),
        "lidar": lid_info if mask_info is None else (mask_info.get("lidar") or lid_info),
        "annotations": annot_info,
    }
    (out_dir / "resolve.json").write_text(json.dumps(resolve, indent=2), encoding="utf-8")
    print(f"\nresolve.json -> {out_dir / 'resolve.json'}")
    if mask_gpkg:
        print(f"  (treino usara recorte mask: {rgb_final.name}"
              f"{', ' + Path(lid_final).name if lid_final else ''})")


if __name__ == "__main__":
    main()
