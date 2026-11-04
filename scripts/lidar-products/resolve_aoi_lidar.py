"""
resolve_aoi_lidar.py
====================
Dado PHD_RUN_ROOT + nome do AOI, resolve via NOMENC do gpkg:
  - optical_tiles : stems em tiles/optical/rgbnir que o AOI seleciona
  - laz_stems     : stems em tiles/lidar/raw que pertencem a esses tiles

Regra LAZ (sem pasta temporaria):
  1. qualquer NOMENC_2K/5K/10K do gpkg cujo .laz exista no pool;
  2. qualquer .laz cujo stem == tile optico OU comece com "<tile>-"
     (ex.: SF-23-Y-A-IV-4-SO-A-I sob o tile SF-23-Y-A-IV-4-SO-A).

Imprime JSON em stdout (consumido pelo run_lidar_pipeline.ps1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pipeline.aoi_resolve import (  # noqa: E402
    NOMENC_FIELDS, build_optical_pool, find_rgbnir_dir, resolve_tiles, _split_names,
)


def laz_for_tiles(aoi_gpkg: Path, optical: list[str], laz_dir: Path) -> list[str]:
    import geopandas as gpd

    laz_files = {p.stem: p for p in laz_dir.glob("*.laz")}
    laz_files.update({p.stem: p for p in laz_dir.glob("*.LAZ")})
    if not laz_files:
        return []

    stems: set[str] = set()
    gdf = gpd.read_file(aoi_gpkg)
    for col in NOMENC_FIELDS:
        if col not in gdf.columns:
            continue
        for v in gdf[col].tolist():
            for name in _split_names(v):
                if name in laz_files:
                    stems.add(name)

    opt = set(optical)
    for stem in laz_files:
        if stem in opt or any(stem.startswith(t + "-") for t in opt):
            stems.add(stem)

    return sorted(stems)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="PHD_RUN_ROOT")
    p.add_argument("--aoi", required=True, help="nome do AOI (aois/<aoi>.gpkg)")
    p.add_argument("--nomenc-field", default=None)
    a = p.parse_args()

    root = Path(a.root)
    aoi_gpkg = root / "aois" / f"{a.aoi}.gpkg"
    tiles_dir = root / "tiles"
    rgb_dir = find_rgbnir_dir(tiles_dir)
    laz_dir = tiles_dir / "lidar" / "raw"
    if not aoi_gpkg.exists():
        raise SystemExit(f"[ABORT] AOI nao encontrado: {aoi_gpkg}")

    index, stem_paths = build_optical_pool(tiles_dir)
    if not stem_paths:
        raise SystemExit(
            f"[ABORT] nenhum tile optico em {rgb_dir} "
            "(esperado tiles/optical/rgbnir/*.tif)."
        )

    optical, how = resolve_tiles(aoi_gpkg, index, rgb_dir, a.nomenc_field)
    if not optical:
        raise SystemExit(
            "[ABORT] AOI nao resolveu nenhum tile optico. "
            "Cheque NOMENC_* vs tiles/optical/rgbnir/*.tif."
        )
    laz = laz_for_tiles(aoi_gpkg, optical, laz_dir)

    out = {
        "aoi": a.aoi,
        "resolved_via": how,
        "optical_tiles": optical,
        "laz_stems": laz,
        "n_optical": len(optical),
        "n_laz": len(laz),
        "paths": {
            "rgbnir": str(rgb_dir).replace("\\", "/"),
            "laz": str(laz_dir).replace("\\", "/"),
            "products": str(root / "tiles" / "lidar" / "products").replace("\\", "/"),
            "rasters": str(root / "tiles" / "lidar" / "rasters").replace("\\", "/"),
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
