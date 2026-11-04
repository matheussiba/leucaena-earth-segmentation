"""
aoi_resolve.py
==============
Resolve um AOI vetorial para os insumos de treino, sobre o POOL central de tiles.

Dado:
  - um pool  <root>/tiles/optical/rgbnir/<tile>.tif  e  <root>/tiles/lidar/rasters/<tile>.tif
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

# Layout novo: tiles/optical/rgbnir ; legado: tiles/rgbnir
def rgbnir_candidate_dirs(tiles_dir: Path) -> list[Path]:
    tiles_dir = Path(tiles_dir)
    return [tiles_dir / "optical" / "rgbnir", tiles_dir / "rgbnir"]


def find_rgbnir_dir(tiles_dir: Path) -> Path:
    """Prefere tiles/optical/rgbnir; cai para tiles/rgbnir se so o legado tiver .tif."""
    for d in rgbnir_candidate_dirs(tiles_dir):
        if d.is_dir() and any(d.glob("*.tif")):
            return d
    return rgbnir_candidate_dirs(tiles_dir)[0]


def build_optical_pool(tiles_dir: Path) -> tuple[dict, dict[str, Path]]:
    """Indice + caminho real por stem. Legado primeiro, optical sobrescreve (prioridade)."""
    tiles_dir = Path(tiles_dir)
    index: dict[str, dict] = {}
    stem_paths: dict[str, Path] = {}
    for d in (tiles_dir / "rgbnir", tiles_dir / "optical" / "rgbnir"):
        if not d.is_dir():
            continue
        for tif in sorted(d.glob("*.tif")):
            try:
                with rasterio.open(tif) as ds:
                    b = ds.bounds
                    index[tif.stem] = {
                        "bounds": [b.left, b.bottom, b.right, b.top],
                        "crs": ds.crs.to_string() if ds.crs else None,
                        "rel": tif.resolve().relative_to(tiles_dir.resolve()).as_posix(),
                    }
                stem_paths[tif.stem] = tif
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] falha indexando {tif.name}: {exc}")
    return index, stem_paths


def rgbnir_rel_under_tiles(rgb_dir: Path, tiles_dir: Path) -> str:
    """Relativo a tiles/ p/ SourceFilename do VRT (ex.: optical/rgbnir)."""
    try:
        return Path(rgb_dir).resolve().relative_to(Path(tiles_dir).resolve()).as_posix()
    except ValueError:
        return "optical/rgbnir"

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
              src_prefix: str | None = None,
              stem_paths: dict[str, Path] | None = None,
              tiles_dir: Path | None = None) -> dict:
    """Monta um .vrt mosaicando <src_dir>/<stem>.tif (ou stem_paths).

    src_prefix: prefixo no SourceFilename (ex. /data/tiles). Com stem_paths +
    tiles_dir, cada fonte usa o relativo real (optical/rgbnir/... ou rgbnir/...).
    """
    metas = []
    for stem in stems:
        f = (stem_paths.get(stem) if stem_paths else None) or (src_dir / f"{stem}.tif")
        if f is None or not Path(f).exists():
            continue
        f = Path(f)
        with rasterio.open(f) as ds:
            t = ds.transform
            if src_prefix and tiles_dir is not None:
                try:
                    rel = f.resolve().relative_to(Path(tiles_dir).resolve()).as_posix()
                    src_name = f"{src_prefix.rstrip('/')}/{rel}"
                except ValueError:
                    src_name = f"{src_prefix.rstrip('/')}/{stem}.tif"
            elif src_prefix is not None:
                src_name = f"{src_prefix.rstrip('/')}/{stem}.tif"
            else:
                src_name = str(f).replace("\\", "/")
            metas.append({
                "stem": stem, "src": src_name,
                "w": ds.width, "h": ds.height,
                "count": ds.count, "dtype": ds.dtypes[0],
                "nodata": ds.nodata,
                "left": ds.bounds.left, "top": ds.bounds.top,
                "px": t.a, "py": -t.e,
                "crs": ds.crs.to_wkt() if ds.crs else "",
            })
    if not metas:
        return {"ok": False, "n": 0, "missing": [s for s in stems
                if not ((stem_paths or {}).get(s) or (src_dir / f"{s}.tif")).exists()]}

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
    lid_dir = tiles_dir / "lidar" / "rasters"
    aoi_gpkg = Path(a.aoi_gpkg) if a.aoi_gpkg else (root / "aois" / f"{a.aoi}.gpkg")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not aoi_gpkg.exists():
        raise SystemExit(f"[ABORT] AOI gpkg nao encontrado: {aoi_gpkg}")

    index, stem_paths = build_optical_pool(tiles_dir)
    # cache so para debug; sempre reconstruido do disco (optical + legado)
    (tiles_dir / ".tile_index.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rel"}
                    for k, v in index.items()}, indent=2),
        encoding="utf-8")
    print(f"Pool rgbnir: {len(stem_paths)} tiles "
          f"(optical/rgbnir + rgbnir legado)")
    if not stem_paths:
        raise SystemExit(
            "[ABORT] nenhum .tif em tiles/optical/rgbnir/ nem tiles/rgbnir/. "
            "Coloque o RGBNIR 4 bandas em tiles/optical/rgbnir/."
        )
    for stem, pth in sorted(stem_paths.items())[:3]:
        print(f"  ex.: {stem} -> {pth}")

    tiles, how = resolve_tiles(aoi_gpkg, index, None, a.nomenc_field)
    print(f"AOI '{aoi_gpkg.stem}' -> {len(tiles)} tiles (via {how})")
    if not tiles:
        # dica: NOMENC do AOI vs arquivos presentes
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
            f"Pool tem {len(stem_paths)} RGBNIR; "
            f"NOMENC sem arquivo no pool (amostra): {missing_wanted}. "
            "Coloque o .tif 4 bandas em tiles/optical/rgbnir/."
        )
    missing = [t for t in tiles if t not in stem_paths]
    if missing:
        raise SystemExit(
            f"[ABORT] tiles do AOI sem arquivo RGBNIR: {missing}. "
            f"Coloque em tiles/optical/rgbnir/ (ex.: {missing[0]}.tif)."
        )

    rgb_prefix = a.src_prefix  # /data/tiles ; write_vrt monta optical/rgbnir/...
    lid_prefix = a.src_prefix + "/lidar/rasters" if a.src_prefix else None
    rgb_vrt = out_dir / "aoi_rgbnir.vrt"
    lid_vrt = out_dir / "aoi_lidar.vrt"
    # remove VRT antigo (pode apontar p/ path errado)
    if rgb_vrt.exists():
        rgb_vrt.unlink()
    rgb_info = write_vrt(tiles, tiles_dir / "optical" / "rgbnir", rgb_vrt,
                         src_prefix=rgb_prefix, stem_paths=stem_paths,
                         tiles_dir=tiles_dir)
    lid_info = write_vrt(tiles, lid_dir, lid_vrt, lid_prefix)
    print(f"  rgbnir VRT: {rgb_info}")
    print(f"  lidar  VRT: {lid_info} (tiles sem lidar sao ignorados)")
    if not rgb_info.get("ok"):
        raise SystemExit(
            f"[ABORT] falha ao montar aoi_rgbnir.vrt: {rgb_info}. "
            f"Verifique se os .tif do AOI existem em tiles/optical/rgbnir/."
        )

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
        "tiles": tiles,
        "tile_paths": {t: str(stem_paths[t]).replace("\\", "/") for t in tiles},
        "rgbnir_vrt": str(rgb_vrt),
        "lidar_vrt": str(lid_vrt) if lid_info.get("ok") else None,
        "rgbnir": rgb_info, "lidar": lid_info, "annotations": annot_info,
    }
    (out_dir / "resolve.json").write_text(json.dumps(resolve, indent=2), encoding="utf-8")
    print(f"\nresolve.json -> {out_dir / 'resolve.json'}")


if __name__ == "__main__":
    main()
