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

import argparse
import os
import re
import shutil
import sys
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


def _simulate_copy(
    type_label: str,
    source_dir: str,
    dest_dir: str,
    tile_ids: list[str],
    use_cascade: bool = True,
) -> dict[str, Any]:
    """Check what would happen if we ran a copy without touching any file.

    Returns a dict with keys:
        type_label, source_dir, dest_dir,
        already_exist  (int)   -- files already in dest
        need_copy      (int)   -- files found in source but missing in dest
        missing_tiles  (list)  -- tile IDs for which no file was found in source
        source_missing (bool)  -- True when source_dir does not exist at all
    """
    if not os.path.exists(source_dir):
        return {
            "type_label": type_label,
            "source_dir": source_dir,
            "dest_dir": dest_dir,
            "already_exist": 0,
            "need_copy": 0,
            "missing_tiles": list(tile_ids),
            "source_missing": True,
        }

    source_names = os.listdir(source_dir)
    dest_names: set[str] = set(os.listdir(dest_dir)) if os.path.exists(dest_dir) else set()

    already_exist = 0
    need_copy = 0
    missing_tiles: list[str] = []

    for tid in tile_ids:
        parts = str(tid).split("-") if use_cascade else [str(tid)]
        matched: list[str] = []
        while parts:
            matched = _find_files_by_token(source_names, "-".join(parts))
            if matched:
                break
            if not use_cascade:
                break
            parts.pop()

        if matched:
            for fname in matched:
                if fname in dest_names:
                    already_exist += 1
                else:
                    need_copy += 1
        else:
            missing_tiles.append(str(tid))

    return {
        "type_label": type_label,
        "source_dir": source_dir,
        "dest_dir": dest_dir,
        "already_exist": already_exist,
        "need_copy": need_copy,
        "missing_tiles": missing_tiles,
        "source_missing": False,
    }


def _print_simulate_report(results: list[dict[str, Any]], enabled: list[bool]) -> None:
    """Print a human-readable dry-run report and a final verdict."""
    print()
    print("=" * 60)
    print("DRY-RUN REPORT")
    print("=" * 60)

    anything_to_do = False

    for r, active in zip(results, enabled):
        if not active:
            continue
        label = r["type_label"]
        print(f"\n  {label}")
        print(f"    source : {r['source_dir']}")
        print(f"    dest   : {r['dest_dir']}")
        if r["source_missing"]:
            print(f"    [ERROR] Source directory not found.")
            anything_to_do = True
            continue
        print(f"    already in dest : {r['already_exist']}")
        print(f"    need to copy    : {r['need_copy']}")
        print(f"    not in source   : {len(r['missing_tiles'])}")
        if r["missing_tiles"]:
            shown = r["missing_tiles"][:5]
            for m in shown:
                print(f"      MISS: {m}")
            if len(r["missing_tiles"]) > 5:
                print(f"      ... and {len(r['missing_tiles']) - 5} more")
        if r["need_copy"] > 0:
            anything_to_do = True

    print()
    print("-" * 60)
    if anything_to_do:
        total_copy = sum(r["need_copy"] for r in results)
        print(f"ACTION NEEDED: {total_copy} file(s) to copy.")
        print("Run the same command WITHOUT --dry-run to copy them.")
    else:
        print("OK: all expected files are already in the destination.")
        print("Run without --dry-run to confirm (nothing will be re-copied).")
    print("=" * 60)


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
    overwrite: bool = OVERWRITE_EXISTING_FILES,
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
                if (fname not in dest_names) or overwrite:
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
    overwrite: bool = OVERWRITE_EXISTING_FILES,
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
                if (fname not in dest_names) or overwrite:
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


# =============================================================================
# GeoPackage helper (used by the pipeline orchestrator)
# =============================================================================

#: Columns tried in order when --id-column is not specified.
_AUTO_ID_COLUMNS = ("NOMENC_10K", "NOMENC_5K", "NOMENC_2K")


def load_tile_ids_from_gpkg(
    gpkg_path: str,
    layer: str,
    id_column: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read tile IDs from a GeoPackage layer and return copy-job lists.

    Returns
    -------
    jobs_rgb_ir, jobs_laz
        Both lists use the same format expected by
        ``copy_rgb_or_ir_with_suffix_cascade`` and ``copy_laz_direct_match``.
    """
    gdf = gpd.read_file(gpkg_path, layer=layer)

    if id_column is None:
        for candidate in _AUTO_ID_COLUMNS:
            if candidate in gdf.columns:
                id_column = candidate
                break
        else:
            # Fall back to the first non-geometry object column
            for col in gdf.columns:
                if gdf[col].dtype == object and col.lower() != "geometry":
                    id_column = col
                    break
    if id_column is None:
        raise ValueError(
            f"Cannot detect tile-ID column in layer {layer!r}. "
            f"Available columns: {list(gdf.columns)}"
        )

    print(f"  [AOI] layer={layer!r}  id_column={id_column!r}  rows={len(gdf)}")
    tile_ids = (
        gdf[id_column]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", None)
        .dropna()
        .unique()
        .tolist()
    )
    print(f"  [AOI] unique tile IDs: {len(tile_ids)}")

    jobs_rgb_ir = [
        {"tile_id": tid, "layer_key": layer, "id_column": id_column}
        for tid in tile_ids
    ]
    jobs_laz = [
        {"search_term": tid, "layer_key": layer, "id_column": id_column}
        for tid in tile_ids
    ]
    return jobs_rgb_ir, jobs_laz


# =============================================================================
# Argparse (CLI mode — called by run_pipeline.py or directly)
# =============================================================================

def _parse_args() -> argparse.Namespace | None:
    """Return parsed args when the script is invoked with CLI flags, else None."""
    # When the script is run with no arguments it falls back to the legacy
    # USER CONFIGURATION block above, so existing users are unaffected.
    if len(sys.argv) == 1:
        return None

    p = argparse.ArgumentParser(
        description=(
            "Step 1 — Copy RGB, IR, and LAZ tiles that intersect an AOI "
            "defined in a GeoPackage layer."
        )
    )
    p.add_argument(
        "--aoi",
        required=True,
        help="Path to the GeoPackage (.gpkg) that contains the AOI layer.",
    )
    p.add_argument(
        "--layer",
        required=True,
        help="Layer name inside the GeoPackage, e.g. articulacao_laser_voo22_AOI_treino",
    )
    p.add_argument(
        "--id-column",
        default=None,
        help=(
            "Tile-ID column inside the layer "
            f"(auto-detected from {_AUTO_ID_COLUMNS} if not given)."
        ),
    )
    p.add_argument("--source-laz", default=SOURCE_LAZ, help="Source LAZ folder")
    p.add_argument("--source-rgb", default=SOURCE_RGB, help="Source RGB folder")
    p.add_argument("--source-ir",  default=SOURCE_IR,  help="Source IR folder")
    p.add_argument("--dest-laz",   default=DEST_LAZ,   help="Destination LAZ folder")
    p.add_argument("--dest-rgb",   default=DEST_RGB,   help="Destination RGB folder")
    p.add_argument("--dest-ir",    default=DEST_IR,    help="Destination IR folder")
    p.add_argument(
        "--no-laz", action="store_true", help="Skip copying LAZ files"
    )
    p.add_argument(
        "--no-rgb", action="store_true", help="Skip copying RGB files"
    )
    p.add_argument(
        "--no-ir", action="store_true", help="Skip copying IR files"
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-copy files that already exist at the destination",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be copied without touching the filesystem",
    )
    return p.parse_args()


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
    args = _parse_args()

    # ------------------------------------------------------------------
    # CLI / GPKG mode  (run_pipeline.py or direct invocation with flags)
    # ------------------------------------------------------------------
    if args is not None:
        print("Reading AOI from GeoPackage …")
        jobs_rgb_ir, jobs_laz = load_tile_ids_from_gpkg(
            args.aoi, args.layer, args.id_column
        )
        if not jobs_rgb_ir:
            print("\n[ABORT] No tile IDs found in the GeoPackage layer.")
            return

        src_laz  = args.source_laz
        src_rgb  = args.source_rgb
        src_ir   = args.source_ir
        dest_laz = args.dest_laz
        dest_rgb = args.dest_rgb
        dest_ir  = args.dest_ir
        do_laz   = not args.no_laz
        do_rgb   = not args.no_rgb
        do_ir    = not args.no_ir
        overwrite = args.overwrite
        dry_run   = args.dry_run

        if dry_run:
            tile_ids = [j["tile_id"] for j in jobs_rgb_ir]
            print(f"[DRY-RUN] AOI has {len(tile_ids)} tile(s). Simulating copy...")
            results = [
                _simulate_copy("LAZ", src_laz, dest_laz, tile_ids, use_cascade=False),
                _simulate_copy("RGB", src_rgb, dest_rgb, tile_ids, use_cascade=True),
                _simulate_copy("IR",  src_ir,  dest_ir,  tile_ids, use_cascade=True),
            ]
            _print_simulate_report(results, [do_laz, do_rgb, do_ir])
            return

        for d in (dest_laz, dest_rgb, dest_ir):
            os.makedirs(d, exist_ok=True)

        t0 = time.time()
        if do_laz:
            t = time.time()
            print_run_summary(
                "LAZ",
                copy_laz_direct_match(src_laz, dest_laz, jobs_laz, overwrite=overwrite),
            )
            print(f"Elapsed (LAZ): {time.time() - t:.2f} s\n")
        if do_rgb:
            t = time.time()
            print_run_summary(
                "RGB",
                copy_rgb_or_ir_with_suffix_cascade(
                    src_rgb, dest_rgb, jobs_rgb_ir, "RGB", overwrite=overwrite
                ),
            )
            print(f"Elapsed (RGB): {time.time() - t:.2f} s\n")
        if do_ir:
            t = time.time()
            print_run_summary(
                "IR",
                copy_rgb_or_ir_with_suffix_cascade(
                    src_ir, dest_ir, jobs_rgb_ir, "IR", overwrite=overwrite
                ),
            )
            print(f"Elapsed (IR): {time.time() - t:.2f} s\n")

        print(f"Total elapsed: {time.time() - t0:.2f} s")
        print("\n" + "=" * 60)
        print("Finished.")
        print("=" * 60)
        return

    # ------------------------------------------------------------------
    # Legacy shapefile mode  (no CLI args — reads USER CONFIGURATION block)
    # ------------------------------------------------------------------
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
