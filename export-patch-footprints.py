"""
Export prepared patch footprints to a GeoJSON layer for QGIS.

The training patches are stored as .npy arrays and are not georeferenced by
themselves. Their location is encoded in prepared/patches/manifest.csv:
tile_name + xoff + yoff + win. This script turns those rows into square
polygons in the tile CRS so you can inspect the training/validation/test
patches directly in QGIS.

Example (Docker):

    python export-patch-footprints.py

Output:

    prepared/patches/patch_footprints.geojson
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from osgeo import gdal

from conf import paths


def _pixel_to_xy(gt, px: int, py: int) -> list[float]:
    """Convert pixel coordinates to map coordinates using a GDAL geotransform."""
    x = gt[0] + px * gt[1] + py * gt[2]
    y = gt[3] + px * gt[4] + py * gt[5]
    return [float(x), float(y)]


def _tile_info(tile_path: Path) -> tuple[tuple, str | None]:
    ds = gdal.Open(str(tile_path))
    if ds is None:
        raise FileNotFoundError(tile_path)
    gt = ds.GetGeoTransform()
    srs = ds.GetSpatialRef()
    epsg = None
    if srs is not None:
        srs.AutoIdentifyEPSG()
        epsg = srs.GetAuthorityCode(None) or srs.GetAuthorityCode("PROJCS")
    ds = None
    return gt, epsg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a QGIS-readable GeoJSON with one polygon per training patch."
    )
    parser.add_argument(
        "--manifest",
        default=os.path.join(paths.PATH_PATCHES_DIR, "manifest.csv"),
        help="Patch manifest produced by prep-patches-from-tiles.py.",
    )
    parser.add_argument(
        "--tiles-dir",
        default=paths.PATH_TILES_DIR,
        help="Folder with the original RGBN tiles.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(paths.PATH_PATCHES_DIR, "patch_footprints.geojson"),
        help="Output GeoJSON path.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    tiles_dir = Path(args.tiles_dir)
    out_path = Path(args.out)

    features = []
    counts = Counter()
    tile_cache: dict[str, tuple[tuple, str | None]] = {}
    first_epsg = None

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile_name = row["tile_name"]
            if tile_name not in tile_cache:
                tile_cache[tile_name] = _tile_info(tiles_dir / f"{tile_name}.tif")
            gt, epsg = tile_cache[tile_name]
            if first_epsg is None:
                first_epsg = epsg

            xoff = int(row["xoff"])
            yoff = int(row["yoff"])
            win = int(row["win"])

            p1 = _pixel_to_xy(gt, xoff, yoff)
            p2 = _pixel_to_xy(gt, xoff + win, yoff)
            p3 = _pixel_to_xy(gt, xoff + win, yoff + win)
            p4 = _pixel_to_xy(gt, xoff, yoff + win)

            split = row.get("split", "")
            frac = float(row["leucaena_fraction"])
            counts[split] += 1

            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[p1, p2, p3, p4, p1]]},
                    "properties": {
                        "patch_id": row["patch_id"],
                        "tile_name": tile_name,
                        "split": split,
                        "row": int(row["row"]),
                        "col": int(row["col"]),
                        "xoff": xoff,
                        "yoff": yoff,
                        "win_px": win,
                        "size_m": round(win * abs(gt[1]), 3),
                        "leucaena_fraction": frac,
                        "leucaena_pct": round(frac * 100.0, 3),
                    },
                }
            )

    crs_name = f"EPSG:{first_epsg}" if first_epsg else "unknown"
    geojson = {
        "type": "FeatureCollection",
        "name": "patch_footprints",
        "crs": {"type": "name", "properties": {"name": crs_name}},
        "features": features,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    print(f"Wrote: {out_path}")
    print(f"Patches: {len(features):,}")
    print(f"CRS: {crs_name}")
    print(f"Splits: {dict(counts)}")


if __name__ == "__main__":
    main()
