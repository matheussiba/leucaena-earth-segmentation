"""
Step 1 of the data pipeline. Copy IGC-style aerial products (LAZ, RGB GeoTIFF,
IR GeoTIFF) from the raw archives (D:/laz, D:/rgb, D:/ir) into the dataset
tree used by the rest of the pipeline (``prep-rgbnir-from-rgb-ir.py``,
``prep-lidar-rasters.py``, ``prep-patches-from-tiles.py``).

Selection driver: tile IDs read from selected articulation shapefiles
(`*_selecao.shp`, exported from QGIS) — this script keeps the original
shapefile-driven interface that has been validated on Piracicaba. A GPKG
adapter (``--aoi xx.gpkg --layer ...``) is implemented in the sibling
``prep-rgbnir-from-rgb-ir.py`` and is a future addition here too.

Workflow (high level)
---------------------
1. In QGIS/ArcGIS, open the articulation shapefiles (Fehidro, Lote4, Voo22).
2. Select only the grid cells (tiles) you need.
3. Export the selection to a **new** shapefile named ``*_selecao.shp`` (same base
   name as the original, plus ``_selecao``).
4. Place **only** those ``*_selecao.shp`` files in ``ARTICULATION_FOLDER`` (see
   configuration block below).
5. Edit ``ARTICULATION_FOLDER``, ``DEST_ROOT``, ``SOURCE_LAZ`` / ``SOURCE_RGB`` /
   ``SOURCE_IR``, and the boolean switches, then run this script from a normal
   Python environment (not inside QGIS).

Output layout under ``DEST_ROOT``::

    laz/   # copied .laz files  -> downstream: prep-lidar-rasters.py
    rgb/   # copied .tif RGB    -> downstream: prep-rgbnir-from-rgb-ir.py
    ir/    # copied .tif IR     -> downstream: prep-rgbnir-from-rgb-ir.py

RGB/IR matching uses a **suffix cascade**: if the full tile ID from the
attribute does not match any filename, progressively shorter hyphen-separated
prefixes are tried until a match is found (useful when filenames are shorter
than the full nomenclature string). LAZ matching uses the full tile string only.

Dependencies: ``geopandas``, ``shapely`` (GeoPandas dependency), standard library.
Install with ``pip install -r requirements.txt`` at the repository root.

History
-------
Moved from ``leucaena-earth-utils/python/scripts/`` into this repository in
2026-05 so the full pre-processing chain (copy -> RGBNIR fuse -> LiDAR raster
-> patches -> train) lives in a single place.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from typing import Any

import geopandas as gpd

# =============================================================================
# USER CONFIGURATION — edit only this block
# =============================================================================

# Folder containing ONLY the *_selecao.shp files with selected tiles
ARTICULATION_FOLDER = r"C:\path\to\your\selecao_shapefiles"

# Destination root; subfolders laz/, rgb/, ir/ are created as needed
DEST_ROOT = r"D:\output\aerial_tiles_copy"

# Source drives/folders where full-resolution products are stored
SOURCE_LAZ = r"D:\laz"
SOURCE_RGB = r"E:\RGB"
SOURCE_IR = r"F:\IR"

# Copy switches
COPY_LAZ = True
COPY_RGB = True
COPY_IR = True

# If False, existing files in the destination are never replaced (resume-safe)
OVERWRITE_EXISTING_FILES = False

# =============================================================================
# Internal configuration — articulation layers and ID column names
# =============================================================================

SHAPEFILE_SPECS: dict[str, dict[str, str]] = {
    "Articulacao_Laser_Fehidro": {
        "filename": "Articulacao_Laser_Fehidro_selecao.shp",
        "id_column": "NOMENC_2K",
    },
    "Articulacao_Laser_Lote4": {
        "filename": "Articulacao_Laser_Lote4_selecao.shp",
        "id_column": "NOMENC_5K",
    },
    "Articulacao_Laser_Voo22": {
        "filename": "Articulacao_Laser_Voo22_selecao.shp",
        "id_column": "NOMENC_5K",
    },
}

DEST_LAZ = os.path.join(DEST_ROOT, "laz")
DEST_RGB = os.path.join(DEST_ROOT, "rgb")
DEST_IR = os.path.join(DEST_ROOT, "ir")


def _ensure_dest_dirs() -> None:
    if COPY_LAZ:
        os.makedirs(DEST_LAZ, exist_ok=True)
    if COPY_RGB:
        os.makedirs(DEST_RGB, exist_ok=True)
    if COPY_IR:
        os.makedirs(DEST_IR, exist_ok=True)


def _load_selected_shapefiles() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    print("Reading selected articulation shapefiles...")
    for layer_key, spec in SHAPEFILE_SPECS.items():
        path = os.path.join(ARTICULATION_FOLDER, spec["filename"])
        if os.path.exists(path):
            print(f"  [OK] Found: {spec['filename']}")
            loaded[layer_key] = {"gdf": gpd.read_file(path), "id_column": spec["id_column"]}
        else:
            print(f"  [SKIP] Missing (ignored): {spec['filename']}")
    return loaded


def _collect_tile_jobs(loaded: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build parallel job lists for RGB/IR (cascade) and LAZ (direct)."""
    jobs_rgb_ir: list[dict[str, Any]] = []
    jobs_laz: list[dict[str, Any]] = []

    for layer_key, bundle in loaded.items():
        gdf = bundle["gdf"]
        col = bundle["id_column"]
        print(f"Processing layer key: {layer_key}")
        if col not in gdf.columns:
            print(f"  [ERROR] Column not found: {col!r} in {layer_key}")
            continue
        for raw_id in gdf[col].dropna().unique().tolist():
            jobs_rgb_ir.append(
                {"tile_id": raw_id, "layer_key": layer_key, "id_column": col}
            )
            jobs_laz.append(
                {"search_term": raw_id, "layer_key": layer_key, "id_column": col}
            )

    return jobs_rgb_ir, jobs_laz


def _find_files_by_token(source_files: list[str], token: str) -> list[str]:
    """Return filenames whose basename matches ``token`` as a whole token before extension .tif/.laz."""
    pattern = re.compile(
        rf"(?<![a-zA-Z0-9]){re.escape(str(token))}(?![a-zA-Z0-9]).*\.(?:tif|laz)$",
        re.IGNORECASE,
    )
    return [f for f in source_files if pattern.search(f)]


def copy_rgb_or_ir_with_suffix_cascade(
    source_dir: str,
    dest_dir: str,
    jobs: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print(f"{label}: {source_dir} -> {dest_dir}")
    print("=" * 60)

    if not os.path.exists(source_dir):
        print(f"[ERROR] Source directory not found: {source_dir}")
        return {"expected": len(jobs), "copied": 0, "missing_by_layer": {}}

    source_names = os.listdir(source_dir)
    dest_names = set(os.listdir(dest_dir))

    copied = 0
    missing_by_layer: dict[str, list[str]] = {}

    for job in jobs:
        tile_id = job["tile_id"]
        layer_key = job["layer_key"]
        id_column = job["id_column"]

        parts = str(tile_id).split("-")
        matched_names: list[str] = []
        matched_token = ""

        while parts:
            token = "-".join(parts)
            matched_names = _find_files_by_token(source_names, token)
            if matched_names:
                matched_token = token
                break
            parts.pop()

        if matched_names:
            for fname in matched_names:
                if (fname not in dest_names) or OVERWRITE_EXISTING_FILES:
                    print(
                        f"  [COPY] {fname} | search_token={matched_token!r} | "
                        f"{id_column}={tile_id!r} | layer={layer_key}"
                    )
                    try:
                        shutil.copy2(
                            os.path.join(source_dir, fname),
                            os.path.join(dest_dir, fname),
                        )
                        copied += 1
                        dest_names.add(fname)
                    except OSError as exc:
                        print(f"  [ERROR] {fname}: {exc}")
        else:
            print(f"  [MISS] No file after cascade for tile_id={tile_id!r} | layer={layer_key}")
            missing_by_layer.setdefault(layer_key, []).append(str(tile_id))

    return {"expected": len(jobs), "copied": copied, "missing_by_layer": missing_by_layer}


def copy_laz_direct_match(
    source_dir: str,
    dest_dir: str,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print(f"LAZ: {source_dir} -> {dest_dir}")
    print("=" * 60)

    source_names = os.listdir(source_dir)
    dest_names = set(os.listdir(dest_dir))

    copied = 0
    missing_by_layer: dict[str, list[str]] = {}

    for job in jobs:
        term = job["search_term"]
        layer_key = job["layer_key"]
        hits = _find_files_by_token(source_names, term)
        if hits:
            for fname in hits:
                if (fname not in dest_names) or OVERWRITE_EXISTING_FILES:
                    print(f"  [COPY] {fname} | search_term={term!r}")
                    try:
                        shutil.copy2(
                            os.path.join(source_dir, fname),
                            os.path.join(dest_dir, fname),
                        )
                        copied += 1
                        dest_names.add(fname)
                    except OSError as exc:
                        print(f"  [ERROR] {fname}: {exc}")
        else:
            print(f"  [MISS] LAZ not found for search_term={term!r} | layer={layer_key}")
            missing_by_layer.setdefault(layer_key, []).append(str(term))

    return {"expected": len(jobs), "copied": copied, "missing_by_layer": missing_by_layer}


def print_run_summary(label: str, result: dict[str, Any]) -> None:
    print("\n" + "#" * 40)
    print(f"SUMMARY — {label}")
    print("#" * 40)
    print(f"Tile rows processed: {result['expected']}")
    print(f"Files copied this run: {result['copied']}")
    miss = result["missing_by_layer"]
    if miss:
        print("\nNot found (by articulation layer):")
        for layer_key, items in miss.items():
            uniq = sorted(set(items))
            print(f"  {layer_key} ({len(uniq)} unique):")
            for t in uniq:
                print(f"    - {t}")
    else:
        print("\nAll search terms matched at least one file (or copy was skipped by overwrite policy).")


def main() -> None:
    _ensure_dest_dirs()

    loaded = _load_selected_shapefiles()
    if not loaded:
        print("\n[ABORT] No articulation shapefiles were loaded. Check ARTICULATION_FOLDER and filenames.")
        return

    jobs_rgb_ir, jobs_laz = _collect_tile_jobs(loaded)
    if not jobs_rgb_ir:
        print("\n[ABORT] No tile IDs found in loaded shapefiles.")
        return

    t0 = time.time()

    if COPY_LAZ:
        t = time.time()
        print_run_summary("LAZ", copy_laz_direct_match(SOURCE_LAZ, DEST_LAZ, jobs_laz))
        print(f"Elapsed (LAZ): {time.time() - t:.2f} s\n")
    else:
        print("\n[INFO] LAZ copy disabled.\n")

    if COPY_RGB:
        t = time.time()
        print_run_summary(
            "RGB",
            copy_rgb_or_ir_with_suffix_cascade(SOURCE_RGB, DEST_RGB, jobs_rgb_ir, "RGB"),
        )
        print(f"Elapsed (RGB): {time.time() - t:.2f} s\n")
    else:
        print("\n[INFO] RGB copy disabled.\n")

    if COPY_IR:
        t = time.time()
        print_run_summary(
            "IR",
            copy_rgb_or_ir_with_suffix_cascade(SOURCE_IR, DEST_IR, jobs_rgb_ir, "IR"),
        )
        print(f"Elapsed (IR): {time.time() - t:.2f} s\n")
    else:
        print("\n[INFO] IR copy disabled.\n")

    print(f"Total elapsed: {time.time() - t0:.2f} s")
    print("\n" + "=" * 60)
    print("Finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
