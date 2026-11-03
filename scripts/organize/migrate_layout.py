"""
migrate_layout.py
=================
Migra a estrutura antiga (<root>/<aoi>-aoi/{opt,lidar,annotations}) para o
layout novo com POOL central de tiles:

    <root>/tiles/rgbnir/<tile>.tif
    <root>/tiles/lidar/rasters/<tile>.tif
    <root>/tiles/lidar/raw/*.laz
    <root>/tiles/lidar/products/*
    <root>/aois/<aoi>.gpkg            (footprints + NOMENC_10K = <tile>)
    <root>/annotations/               (anotacoes globais)
    <root>/datasets/  experiments/  reports/

Seguranca:
  - Move e sempre no MESMO volume (rename instantaneo), nunca copia GB.
  - Duplicatas (mesmo nome de tile em 2 AOIs) e lixo (_test_*) NAO sao apagados:
    vao para <root>/_migration_backup/ (quarentena).
  - Todo move e registrado em <root>/_migration_backup/manifest.json (undo).

Uso:
    python migrate_layout.py --root C:/00_PHD_MSB_RUN --dry-run
    python migrate_layout.py --root C:/00_PHD_MSB_RUN
    python migrate_layout.py --root C:/00_PHD_MSB_RUN --undo
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import rasterio
from shapely.geometry import box

# aoi legado -> id novo (sem sufixo -aoi)
LEGACY_AOIS = {
    "261002-riodaspedras-aoi": "261002-riodaspedras",
    "260515-piracicaba-aoi": "260515-piracicaba",
}
# anotacao global designada (sera copiada para annotations/)
GLOBAL_ANNOT = "261030-classes.gpkg"


def _manifest_path(root: Path) -> Path:
    return root / "_migration_backup" / "manifest.json"


def _load_manifest(root: Path) -> list[dict]:
    mp = _manifest_path(root)
    if mp.exists():
        return json.loads(mp.read_text(encoding="utf-8"))
    return []


def _save_manifest(root: Path, moves: list[dict]) -> None:
    mp = _manifest_path(root)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(moves, indent=2), encoding="utf-8")


def _move(src: Path, dst: Path, moves: list[dict], dry: bool, kind: str) -> None:
    print(f"  MOVE [{kind}] {src}  ->  {dst}")
    if dry:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    moves.append({"kind": kind, "src": str(src), "dst": str(dst)})


def _copy(src: Path, dst: Path, moves: list[dict], dry: bool) -> None:
    print(f"  COPY {src}  ->  {dst}")
    if dry:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    moves.append({"kind": "copy", "src": str(src), "dst": str(dst)})


def migrate(root: Path, dry: bool) -> None:
    pool_rgb = root / "tiles" / "rgbnir"
    pool_lid = root / "tiles" / "lidar" / "rasters"
    pool_raw = root / "tiles" / "lidar" / "raw"
    pool_prod = root / "tiles" / "lidar" / "products"
    annot_dir = root / "annotations"
    aois_dir = root / "aois"
    quarantine = root / "_migration_backup"
    for d in (pool_rgb, pool_lid, pool_raw, pool_prod, annot_dir, aois_dir,
              root / "datasets", root / "experiments", root / "reports"):
        if not dry:
            d.mkdir(parents=True, exist_ok=True)

    moves = _load_manifest(root)
    membership: dict[str, list[str]] = {}

    for legacy, aoi_id in LEGACY_AOIS.items():
        base = root / legacy
        if not base.is_dir():
            print(f"[skip] AOI legado ausente: {base}")
            continue
        print(f"\n===== {legacy} -> aoi '{aoi_id}' =====")
        stems: list[str] = []

        # --- rgbnir -> pool ---
        src_rgb = base / "opt" / "rgbnir"
        if src_rgb.is_dir():
            for tif in sorted(src_rgb.glob("*.tif")):
                stem = tif.stem
                stems.append(stem)
                target = pool_rgb / tif.name
                if target.exists():
                    q = quarantine / legacy / "opt" / "rgbnir" / tif.name
                    _move(tif, q, moves, dry, "rgb-dup")
                else:
                    _move(tif, target, moves, dry, "rgb")

        # --- lidar/rasters -> pool (lixo _test_* para quarentena) ---
        src_lid = base / "lidar" / "rasters"
        if src_lid.is_dir():
            for tif in sorted(src_lid.glob("*.tif")):
                if tif.name.startswith("_test"):
                    q = quarantine / legacy / "lidar" / "rasters" / tif.name
                    _move(tif, q, moves, dry, "lidar-junk")
                    continue
                target = pool_lid / tif.name
                if target.exists():
                    q = quarantine / legacy / "lidar" / "rasters" / tif.name
                    _move(tif, q, moves, dry, "lidar-dup")
                else:
                    _move(tif, target, moves, dry, "lidar")

        # --- lidar/raw -> pool ---
        src_raw = base / "lidar" / "raw"
        if src_raw.is_dir():
            for f in sorted(src_raw.iterdir()):
                if f.is_file():
                    target = pool_raw / f.name
                    if target.exists():
                        q = quarantine / legacy / "lidar" / "raw" / f.name
                        _move(f, q, moves, dry, "raw-dup")
                    else:
                        _move(f, target, moves, dry, "raw")

        # --- lidar/products -> pool ---
        src_prod = base / "lidar" / "products"
        if src_prod.is_dir():
            for f in sorted(src_prod.iterdir()):
                if f.is_file():
                    target = pool_prod / f.name
                    if not target.exists():
                        _move(f, target, moves, dry, "product")

        # --- annotations: copia a global designada ---
        gannot = base / "annotations" / GLOBAL_ANNOT
        if gannot.exists() and not (annot_dir / GLOBAL_ANNOT).exists():
            _copy(gannot, annot_dir / GLOBAL_ANNOT, moves, dry)

        membership[aoi_id] = sorted(set(stems))

    if not dry:
        _save_manifest(root, moves)
        (quarantine / "membership.json").write_text(
            json.dumps(membership, indent=2), encoding="utf-8")
    print("\nMembership (tiles por AOI):")
    for a, s in membership.items():
        print(f"  {a}: {len(s)} tiles")
    return membership


def build_aoi_gpkgs(root: Path, membership: dict[str, list[str]], dry: bool) -> None:
    import geopandas as gpd

    pool_rgb = root / "tiles" / "rgbnir"
    aois_dir = root / "aois"
    print("\n===== Gerando aois/<aoi>.gpkg (footprints + NOMENC_10K) =====")
    for aoi_id, stems in membership.items():
        geoms, names, crs = [], [], None
        for stem in stems:
            tif = pool_rgb / f"{stem}.tif"
            if not tif.exists():
                print(f"  [warn] sem raster no pool p/ {stem}")
                continue
            with rasterio.open(tif) as ds:
                b = ds.bounds
                crs = ds.crs
            geoms.append(box(b.left, b.bottom, b.right, b.top))
            names.append(stem)
        if not geoms:
            print(f"  [skip] {aoi_id}: nenhum footprint")
            continue
        gdf = gpd.GeoDataFrame({"NOMENC_10K": names}, geometry=geoms, crs=crs)
        out = aois_dir / f"{aoi_id}.gpkg"
        print(f"  {out}  ({len(gdf)} tiles, crs={crs})")
        if not dry:
            gdf.to_file(out, driver="GPKG")


def undo(root: Path) -> None:
    moves = _load_manifest(root)
    print(f"Revertendo {len(moves)} operacoes...")
    for m in reversed(moves):
        if m["kind"] == "copy":
            p = Path(m["dst"])
            if p.exists():
                p.unlink()
                print(f"  rm {p}")
            continue
        src, dst = Path(m["src"]), Path(m["dst"])
        if dst.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(src))
            print(f"  back {dst} -> {src}")
    print("Undo completo. (manifest mantido)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="C:/00_PHD_MSB_RUN")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--undo", action="store_true")
    a = p.parse_args()
    root = Path(a.root)

    if a.undo:
        undo(root)
        return

    t0 = time.time()
    membership = migrate(root, a.dry_run)
    build_aoi_gpkgs(root, membership, a.dry_run)
    print(f"\n{'[DRY-RUN] ' if a.dry_run else ''}Concluido em {time.time()-t0:.1f}s")
    if a.dry_run:
        print("Rode sem --dry-run para efetivar.")


if __name__ == "__main__":
    main()
