"""
Step 2 of the data pipeline. Build 4-band RGB+NIR GeoTIFFs per AOI tile by
fusing two co-registered single-tile rasters: an RGB raster (3 bands: R, G, B)
and an IR raster stored as a "false color" 3-band image whose **band 1 is the
actual near-infrared channel**.

Sits between ``prep-copy-tiles-from-aoi.py`` (which produces ``rgb/`` and
``ir/``) and ``prep-patches-from-tiles.py`` (which expects 4-band tiles).

Why this exists
---------------
Storing both rasters keeps the same area on disk twice (RGB plus a 3-band IR
where only one band is meaningful). This script writes a single 4-band GeoTIFF
per tile with band assignments::

    band 1 = Red       (from RGB band 1)
    band 2 = Green     (from RGB band 2)
    band 3 = Blue      (from RGB band 3)
    band 4 = NIR       (from IR  band 1)

AOI sources (multiple articulation layers)
------------------------------------------
The IGC dataset uses different articulation grids per acquisition: Fehidro,
Lote4 and Voo22. Each grid stores the tile identifier in a different column
(typically ``NOMENC_2K``, ``NOMENC_5K`` or ``NOMENC_10K``). This script accepts
**a list** of AOI layers, each with its own layer name and tile-id column,
unions and de-duplicates the tile ids before fusion.

Inputs / outputs
----------------
- ``AOI_GPKG_PATH``: GeoPackage containing the AOI layers (per spec ``gpkg_path``
  override is also supported).
- ``SOURCE_RGB_FOLDER`` and ``SOURCE_IR_FOLDER``: the existing one-tile-per-file
  folders. **They are never touched.**
- ``OUTPUT_FOLDER``: where ``<TILE_ID>.tif`` is written (created if missing).
- The script verifies that the two source rasters cover the same area at the
  same resolution and CRS; if not, the tile is skipped with a clear reason.

Dependencies (see repo root ``requirements.txt``)
-------------------------------------------------
    pip install rasterio geopandas

Then::

    python prep-rgbnir-from-rgb-ir.py

History
-------
Moved from ``leucaena-earth-utils/python/scripts/`` into this repository in
2026-05, alongside ``prep-copy-tiles-from-aoi.py``, so the full pre-processing
chain lives in a single place.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import glob
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

import geopandas as gpd
import rasterio
from rasterio.enums import ColorInterp
from rasterio.errors import RasterioIOError

# =============================================================================
# USER CONFIGURATION — edit only this block
# =============================================================================

# Default GeoPackage used unless an entry in AOI_LAYER_SPECS overrides it
AOI_GPKG_PATH = r"G:\My Drive\PHD\02-Tese\02-data\adote-uma-leucena\v1-LEUCENA MAPPING\gdb-leucena_v2.gpkg"

# Default suffix appended to each "base_layer" to build the actual layer name
# (e.g. "_AOI_treino" -> "articulacao_laser_voo22_AOI_treino").
# Change here to switch every entry at once (e.g. "_AOI_test"), or override
# per spec with the "suffix" key. To bypass entirely, set "layer" explicitly.
AOI_LAYER_SUFFIX = "_AOI_treino"

# One entry per articulation.
#   "base_layer"  : articulation name (suffix is appended automatically)
#   "id_column"   : tile-id column inside that layer
#   "enabled"     : optional bool, default True
#   "suffix"      : optional, overrides AOI_LAYER_SUFFIX for this entry only
#   "layer"       : optional, absolute layer name (skips base_layer + suffix)
#   "gpkg_path"   : optional, overrides AOI_GPKG_PATH for this entry only
AOI_LAYER_SPECS: list[dict] = [
    {"base_layer": "articulacao_laser_fehidro", "id_column": "NOMENC_2K",  "enabled": True},
    {"base_layer": "articulacao_laser_lote4",   "id_column": "NOMENC_5K",  "enabled": True},
    {"base_layer": "articulacao_laser_voo22",   "id_column": "NOMENC_10K", "enabled": True},
]

# Source folders with original tiles
SOURCE_RGB_FOLDER = r"D:\rgb"
SOURCE_IR_FOLDER = r"D:\ir"

# Output folder for fused 4-band rasters (created if missing)
OUTPUT_FOLDER = r"D:\rgbir"

# Which band of the IR raster carries the actual NIR signal (1-based)
IR_NIR_BAND_INDEX = 1

# If True, fall back to shorter prefixes of the tile id when no exact match
# is found (mirrors the cascade used in the IGC tile copy script).
USE_SUFFIX_CASCADE = True

# Skip output that already exists (resume-safe). Set to True to redo files.
OVERWRITE_EXISTING_FILES = False

# GeoTIFF write options
COMPRESS = "DEFLATE"    # DEFLATE / LZW / ZSTD / None
PREDICTOR = 2           # 2 for integer bands (uint8/uint16), 3 for float
TILED = True
BLOCKXSIZE = 512
BLOCKYSIZE = 512

# Performance knobs (Section 2)
# -----------------------------
# Number of tiles processed in parallel (separate processes). Each worker holds
# at least one full RGB+IR pair in memory, so size this for your RAM and disk.
# Rule of thumb on a 64 GB / 8c/16t i7 with SSD: 4-6 workers + 2-4 GDAL threads.
# Set to 1 to fully serialize (matches the previous behaviour).
MAX_WORKERS = 6

# Per-tile GDAL threads for (de)compression and IO. With MAX_WORKERS > 1 keep
# this low (2-4) so workers don't fight for cores. Use "ALL_CPUS" only when
# MAX_WORKERS == 1.
GDAL_NUM_THREADS = "4"

# GDAL raster cache size in MB. 1024 MB is a sane default for big tiles.
GDAL_CACHEMAX_MB = 1024

# =============================================================================
# Implementation
# =============================================================================

RASTER_EXTENSIONS = (".tif", ".tiff")

# Files in progress are written with this suffix and renamed on success.
# Any leftover ``*.tmp.tif`` is cleaned at startup (means the previous run died).
TMP_SUFFIX = ".tmp.tif"


# -----------------------------------------------------------------------------
# Keep Windows awake while the script is running
# -----------------------------------------------------------------------------
# https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002
_ES_AWAYMODE_REQUIRED = 0x00000040


@contextlib.contextmanager
def _keep_system_awake(keep_display_on: bool = True):
    """Prevent sleep / display-off while the block runs (Windows only).

    On non-Windows platforms this is a no-op so the script stays cross-platform.
    """
    if not sys.platform.startswith("win"):
        yield
        return

    flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
    if keep_display_on:
        flags |= _ES_DISPLAY_REQUIRED
    try:
        prev = ctypes.windll.kernel32.SetThreadExecutionState(flags)
        if prev == 0:
            print("[WARN] SetThreadExecutionState returned 0; sleep may still occur.", flush=True)
        else:
            print("[INFO] System sleep/display-off prevented while script runs.", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not request keep-awake: {exc}", flush=True)
        yield
        return

    try:
        yield
    finally:
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
            print("[INFO] Restored default sleep behavior.", flush=True)
        except Exception:  # noqa: BLE001
            pass


@dataclass
class TileJob:
    tile_id: str
    source_layer: str
    id_column: str


@dataclass
class TileSummary:
    tile_id: str
    source_layer: str
    rgb_name: str | None
    ir_name: str | None
    status: str
    detail: str = ""


def _list_rasters(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith(RASTER_EXTENSIONS)]


def _match_for_token(filenames: Iterable[str], token: str) -> list[str]:
    """Return files whose name contains ``token`` as a whole alphanumeric token."""
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return [f for f in filenames if pattern.search(f)]


def _find_tile_file(filenames: list[str], tile_id: str) -> str | None:
    """Find a single raster for ``tile_id`` (exact, then optional suffix cascade)."""
    hits = _match_for_token(filenames, tile_id)
    if hits:
        return sorted(hits)[0]
    if not USE_SUFFIX_CASCADE:
        return None
    parts = tile_id.split("-")
    while len(parts) > 1:
        parts.pop()
        token = "-".join(parts)
        hits = _match_for_token(filenames, token)
        if hits:
            return sorted(hits)[0]
    return None


def _approx_equal_transform(a, b, tol: float = 1e-6) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(tuple(a)[:6], tuple(b)[:6]))


def _fmt_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def _safe_remove(path: str) -> None:
    """Remove ``path`` if present, ignoring errors."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _fuse_one_tile(
    rgb_path: str, ir_path: str, out_path: str, log_prefix: str = ""
) -> tuple[str, str]:
    """Fuse one RGB + IR pair into a 4-band GeoTIFF. Returns (status, detail).

    No mosaicking is performed: bands are stacked block-by-block from the two
    co-registered single-tile rasters into a single 4-band output. Progress is
    reported per block so very large tiles do not look frozen.

    Output is written to ``<out_path>.tmp.tif`` and atomically renamed on
    success, so any leftover ``*.tmp.tif`` indicates a previous failed run.
    """
    out_dir = os.path.dirname(out_path) or "."
    base = os.path.splitext(os.path.basename(out_path))[0]
    tmp_path = os.path.join(out_dir, base + TMP_SUFFIX)

    try:
        os.makedirs(out_dir, exist_ok=True)
        _safe_remove(tmp_path)  # remove any stale tmp from a previous crash

        rgb_size = os.path.getsize(rgb_path)
        ir_size = os.path.getsize(ir_path)
        print(
            f"{log_prefix}  RGB src: {os.path.basename(rgb_path)} ({_fmt_size(rgb_size)})",
            flush=True,
        )
        print(
            f"{log_prefix}  IR  src: {os.path.basename(ir_path)} ({_fmt_size(ir_size)})",
            flush=True,
        )

        env_opts = {
            "GDAL_NUM_THREADS": str(GDAL_NUM_THREADS),
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        }
        # GDAL_CACHEMAX accepts an int (MB) directly in modern rasterio
        if GDAL_CACHEMAX_MB and GDAL_CACHEMAX_MB > 0:
            env_opts["GDAL_CACHEMAX"] = int(GDAL_CACHEMAX_MB)

        with rasterio.Env(**env_opts), rasterio.open(rgb_path) as rgb_src, rasterio.open(ir_path) as ir_src:
            if rgb_src.count < 3:
                return ("rgb_band_count", f"RGB has {rgb_src.count} bands (<3)")
            if ir_src.count < IR_NIR_BAND_INDEX:
                return ("ir_band_count", f"IR has {ir_src.count} bands; NIR index {IR_NIR_BAND_INDEX} unavailable")
            if (rgb_src.width, rgb_src.height) != (ir_src.width, ir_src.height):
                return ("shape_mismatch", f"RGB {rgb_src.width}x{rgb_src.height} vs IR {ir_src.width}x{ir_src.height}")
            if rgb_src.crs != ir_src.crs:
                return ("crs_mismatch", f"RGB={rgb_src.crs} IR={ir_src.crs}")
            if not _approx_equal_transform(rgb_src.transform, ir_src.transform):
                return ("transform_mismatch", "RGB and IR pixel grids differ")

            print(
                f"{log_prefix}  shape: {rgb_src.width}x{rgb_src.height}  "
                f"dtype RGB={rgb_src.dtypes[0]} IR={ir_src.dtypes[IR_NIR_BAND_INDEX-1]}  "
                f"CRS={rgb_src.crs}  "
                f"nodata RGB={rgb_src.nodata} IR={ir_src.nodata}",
                flush=True,
            )

            profile = rgb_src.profile.copy()
            # Strip inherited keys that would otherwise force GDAL to mark the
            # 4th band as alpha (EXTRASAMPLES=Alpha in the TIFF tag, which then
            # makes QGIS apply transparency/blending automatically).
            for k in ("photometric", "alpha", "Photometric", "ALPHA"):
                profile.pop(k, None)

            profile.update(
                count=4,
                driver="GTiff",
                tiled=TILED,
                blockxsize=BLOCKXSIZE,
                blockysize=BLOCKYSIZE,
                BIGTIFF="IF_SAFER",
                NUM_THREADS=GDAL_NUM_THREADS,
                # Explicitly tell GDAL: bands 1-3 are RGB and the 4th band is
                # an extra sample of UNSPECIFIED kind (NOT alpha). This is what
                # writes EXTRASAMPLES=UNSPECIFIED in the TIFF tag.
                PHOTOMETRIC="RGB",
                ALPHA="UNSPECIFIED",
            )
            if COMPRESS:
                profile.update(compress=COMPRESS)
                if PREDICTOR:
                    profile.update(predictor=PREDICTOR)

            print(f"{log_prefix}  writing -> {tmp_path} (will rename to {out_path})", flush=True)

            t_block = time.time()
            with rasterio.open(tmp_path, "w", **profile) as dst:
                # Belt-and-suspenders: force the per-band color interpretation
                # so any downstream tool (QGIS, gdalinfo) sees:
                #   band 1 = Red, band 2 = Green, band 3 = Blue, band 4 = Undefined
                # (NIR has no dedicated GCI; "Undefined" is the GDAL-correct
                # choice and prevents the alpha-channel treatment.)
                dst.colorinterp = (
                    ColorInterp.red,
                    ColorInterp.green,
                    ColorInterp.blue,
                    ColorInterp.undefined,
                )
                print(
                    f"{log_prefix}  colorinterp set: "
                    f"{tuple(ci.name for ci in dst.colorinterp)}",
                    flush=True,
                )

                windows = [w for _, w in dst.block_windows(1)]
                total = len(windows)
                step = max(1, total // 20)  # ~5% increments
                print(f"{log_prefix}  blocks to write: {total}", flush=True)

                for idx, window in enumerate(windows, 1):
                    # One read of all 3 RGB bands per window -> shape (3, h, w)
                    rgb_block = rgb_src.read(indexes=[1, 2, 3], window=window)
                    nir_block = ir_src.read(IR_NIR_BAND_INDEX, window=window)
                    # Use positional `indexes` to match rasterio's Cython API and
                    # avoid "TypeError: an integer is required" in some versions
                    dst.write(rgb_block, [1, 2, 3], window=window)
                    dst.write(nir_block, 4, window=window)

                    if idx % step == 0 or idx == total:
                        elapsed = time.time() - t_block
                        rate = idx / elapsed if elapsed > 0 else 0.0
                        eta = (total - idx) / rate if rate > 0 else 0.0
                        pct = idx / total * 100.0
                        print(
                            f"{log_prefix}    progress {pct:5.1f}%  "
                            f"block {idx}/{total}  "
                            f"elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s",
                            flush=True,
                        )

                dst.descriptions = ("Red", "Green", "Blue", "NIR")
                # Per-band tags so QGIS' Band Statistics / Layer Properties
                # clearly show which band is which.
                dst.update_tags(1, BAND_NAME="Red")
                dst.update_tags(2, BAND_NAME="Green")
                dst.update_tags(3, BAND_NAME="Blue")
                dst.update_tags(4, BAND_NAME="NIR")
                dst.update_tags(
                    PROCESSING_SCRIPT="build_4band_rgbir_geotiff_from_rgb_and_ir_false_color_per_aoi_tile.py",
                    SOURCE_RGB=os.path.basename(rgb_path),
                    SOURCE_IR=os.path.basename(ir_path),
                    IR_NIR_BAND_INDEX=str(IR_NIR_BAND_INDEX),
                    BAND_INTERPRETATION="band1=Red, band2=Green, band3=Blue, band4=NIR (Undefined; NOT alpha)",
                )

        # rasterio context closed cleanly -> promote tmp to final path
        if os.path.exists(out_path):
            _safe_remove(out_path)
        os.replace(tmp_path, out_path)

        out_size = os.path.getsize(out_path)
        print(
            f"{log_prefix}  done in {time.time() - t_block:.1f}s  "
            f"out: {_fmt_size(out_size)}",
            flush=True,
        )
        return ("ok", "")

    except KeyboardInterrupt:
        _safe_remove(tmp_path)
        print(f"{log_prefix}  [INTERRUPTED] removed partial {os.path.basename(tmp_path)}", flush=True)
        raise
    except RasterioIOError as exc:
        _safe_remove(tmp_path)
        print(f"{log_prefix}  [IO_ERROR] removed partial {os.path.basename(tmp_path)}", flush=True)
        return ("io_error", str(exc))
    except OSError as exc:
        _safe_remove(tmp_path)
        print(f"{log_prefix}  [OS_ERROR] removed partial {os.path.basename(tmp_path)}", flush=True)
        return ("os_error", str(exc))
    except Exception as exc:  # noqa: BLE001 - last-resort cleanup
        import traceback
        tb = traceback.format_exc()
        _safe_remove(tmp_path)
        print(f"{log_prefix}  [ERROR] removed partial {os.path.basename(tmp_path)}", flush=True)
        print(f"{log_prefix}  traceback:\n{tb}", flush=True)
        return ("error", f"{type(exc).__name__}: {exc}")


def _resolve_layer_name(spec: dict) -> str:
    """Resolve the final layer name from a spec entry.

    Priority:
      1. spec["layer"]                              (absolute name)
      2. spec["base_layer"] + spec["suffix"]        (per-spec suffix)
      3. spec["base_layer"] + AOI_LAYER_SUFFIX      (global default)
    """
    if spec.get("layer"):
        return str(spec["layer"])
    if "base_layer" not in spec:
        raise KeyError("AOI spec needs either 'layer' or 'base_layer'")
    suffix = spec.get("suffix", AOI_LAYER_SUFFIX) or ""
    return f"{spec['base_layer']}{suffix}"


def _load_aoi_tile_jobs() -> list[TileJob]:
    """Read every enabled AOI layer, union + de-duplicate tile ids."""
    jobs: list[TileJob] = []
    seen: set[str] = set()

    for spec in AOI_LAYER_SPECS:
        if not spec.get("enabled", True):
            continue
        layer = _resolve_layer_name(spec)
        col = spec["id_column"]
        gpkg = spec.get("gpkg_path", AOI_GPKG_PATH)
        print(f"Reading AOI layer: {gpkg} | layer={layer}", flush=True)

        try:
            gdf = gpd.read_file(gpkg, layer=layer)
        except Exception as exc:  # noqa: BLE001 - report and continue with other layers
            print(f"  [SKIP LAYER] failed to read: {exc}", flush=True)
            continue

        if col not in gdf.columns:
            print(
                f"  [SKIP LAYER] column {col!r} not found. "
                f"Columns available: {list(gdf.columns)}",
                flush=True,
            )
            continue

        cleaned = (
            gdf[col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", None)
            .dropna()
            .unique()
            .tolist()
        )
        added = 0
        for tid in cleaned:
            if tid not in seen:
                seen.add(tid)
                jobs.append(TileJob(tile_id=tid, source_layer=layer, id_column=col))
                added += 1
        print(f"  Tiles read: {len(cleaned)} | new (after dedup): {added}", flush=True)

    print(f"\nTotal unique tiles across all AOI layers: {len(jobs)}", flush=True)
    return jobs


def _process_tile_job(args: dict) -> dict:
    """Top-level worker entry-point used by ProcessPoolExecutor.

    Returns a dict so the parent process can rebuild a ``TileSummary``.
    """
    rgb_path = args["rgb_path"]
    ir_path = args["ir_path"]
    out_path = args["out_path"]
    log_prefix = args["log_prefix"]
    t0 = time.time()
    try:
        status, detail = _fuse_one_tile(rgb_path, ir_path, out_path, log_prefix=log_prefix)
    except KeyboardInterrupt:
        return {**args, "status": "interrupted", "detail": "", "elapsed_s": time.time() - t0}
    return {**args, "status": status, "detail": detail, "elapsed_s": time.time() - t0}


def _cleanup_stale_tmp(folder: str) -> int:
    """Remove leftover ``*.tmp.tif`` files (a previous run died mid-write)."""
    if not os.path.isdir(folder):
        return 0
    pattern = os.path.join(folder, f"*{TMP_SUFFIX}")
    removed = 0
    for p in glob.glob(pattern):
        try:
            os.remove(p)
            removed += 1
            print(f"[CLEANUP] removed stale partial: {os.path.basename(p)}", flush=True)
        except OSError as exc:
            print(f"[CLEANUP] could not remove {p}: {exc}", flush=True)
    return removed


# =============================================================================
# Argparse (CLI mode — called by run_pipeline.py or directly)
# =============================================================================

#: Columns tried in order when --id-column is not specified.
_AUTO_ID_COLUMNS_RGBNIR = ("NOMENC_10K", "NOMENC_5K", "NOMENC_2K")


def _detect_id_column(gpkg_path: str, layer: str) -> str:
    """Return the first matching auto-detect column, or raise."""
    import geopandas as gpd

    try:
        gdf = gpd.read_file(gpkg_path, layer=layer)
    except Exception as exc:
        raise RuntimeError(f"Cannot open {gpkg_path!r} layer={layer!r}: {exc}") from exc
    for candidate in _AUTO_ID_COLUMNS_RGBNIR:
        if candidate in gdf.columns:
            return candidate
    for col in gdf.columns:
        if gdf[col].dtype == object and col.lower() != "geometry":
            return col
    raise ValueError(
        f"Cannot detect tile-ID column in {layer!r}. "
        f"Available: {list(gdf.columns)}"
    )


def _parse_args_rgbnir() -> argparse.Namespace | None:
    """Return parsed args when the script is invoked with CLI flags, else None."""
    if len(sys.argv) == 1:
        return None

    p = argparse.ArgumentParser(
        description=(
            "Step 2 — Fuse RGB + IR tiles into 4-band RGBNIR GeoTIFFs. "
            "Tile IDs are read from a GeoPackage AOI layer."
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
        help="Layer name inside the GeoPackage.",
    )
    p.add_argument(
        "--id-column",
        default=None,
        help=(
            "Tile-ID column inside the layer "
            f"(auto-detected from {_AUTO_ID_COLUMNS_RGBNIR} if not given)."
        ),
    )
    p.add_argument(
        "--source-rgb",
        default=SOURCE_RGB_FOLDER,
        help="Folder with source RGB tiles (default: %(default)s)",
    )
    p.add_argument(
        "--source-ir",
        default=SOURCE_IR_FOLDER,
        help="Folder with source IR tiles (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        default=OUTPUT_FOLDER,
        help="Output folder for fused RGBNIR tiles (default: %(default)s)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Parallel worker processes (default: {MAX_WORKERS})",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process tiles that already exist at the destination.",
    )
    return p.parse_args()


def main() -> None:
    global AOI_GPKG_PATH, AOI_LAYER_SPECS, AOI_LAYER_SUFFIX
    global SOURCE_RGB_FOLDER, SOURCE_IR_FOLDER, OUTPUT_FOLDER
    global MAX_WORKERS, OVERWRITE_EXISTING_FILES

    # Patch module globals with CLI args when the script is invoked from the
    # pipeline (or directly with flags). When run with no flags, all globals
    # keep their default values and the script behaves exactly as before.
    _args = _parse_args_rgbnir()
    if _args is not None:
        AOI_GPKG_PATH = _args.aoi
        col = _args.id_column or _detect_id_column(_args.aoi, _args.layer)
        # Override AOI_LAYER_SPECS with a single explicit layer spec so that
        # _load_aoi_tile_jobs() reads exactly the layer the caller requested.
        AOI_LAYER_SPECS = [
            {
                "layer": _args.layer,
                "id_column": col,
                "enabled": True,
                "gpkg_path": _args.aoi,
            }
        ]
        SOURCE_RGB_FOLDER = _args.source_rgb
        SOURCE_IR_FOLDER = _args.source_ir
        OUTPUT_FOLDER = _args.out_dir
        if _args.workers is not None:
            MAX_WORKERS = _args.workers
        if _args.overwrite:
            OVERWRITE_EXISTING_FILES = True

    t_start = time.time()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    _cleanup_stale_tmp(OUTPUT_FOLDER)

    jobs = _load_aoi_tile_jobs()
    if not jobs:
        print("[ABORT] No tile ids in AOI.")
        return

    rgb_names = _list_rasters(SOURCE_RGB_FOLDER)
    ir_names = _list_rasters(SOURCE_IR_FOLDER)
    if not rgb_names:
        print(f"[ABORT] No rasters in {SOURCE_RGB_FOLDER}", flush=True)
        return
    if not ir_names:
        print(f"[ABORT] No rasters in {SOURCE_IR_FOLDER}", flush=True)
        return
    print(
        f"  RGB files available: {len(rgb_names)} | IR files available: {len(ir_names)}",
        flush=True,
    )

    summaries: list[TileSummary] = []
    written = 0
    skipped_existing = 0

    # ------------------------------------------------------------------
    # 1) Resolve file matches up-front -> ready-to-run job specs
    # ------------------------------------------------------------------
    ready_jobs: list[dict] = []
    for i, job in enumerate(jobs, 1):
        tile_id = job.tile_id
        prefix = f"[{i}/{len(jobs)}]"
        rgb_name = _find_tile_file(rgb_names, tile_id)
        ir_name = _find_tile_file(ir_names, tile_id)

        if not rgb_name and not ir_name:
            print(f"{prefix} {tile_id} SKIP missing_both", flush=True)
            summaries.append(TileSummary(tile_id, job.source_layer, None, None, "missing_both"))
            continue
        if not rgb_name:
            print(f"{prefix} {tile_id} SKIP missing_rgb (IR={ir_name})", flush=True)
            summaries.append(TileSummary(tile_id, job.source_layer, None, ir_name, "missing_rgb"))
            continue
        if not ir_name:
            print(f"{prefix} {tile_id} SKIP missing_ir (RGB={rgb_name})", flush=True)
            summaries.append(TileSummary(tile_id, job.source_layer, rgb_name, None, "missing_ir"))
            continue

        out_name = f"{tile_id}.tif"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        if os.path.exists(out_path) and not OVERWRITE_EXISTING_FILES:
            skipped_existing += 1
            print(f"{prefix} {tile_id} SKIP already exists -> {out_name}", flush=True)
            summaries.append(TileSummary(tile_id, job.source_layer, rgb_name, ir_name, "skipped_existing"))
            continue

        ready_jobs.append({
            "tile_id": tile_id,
            "source_layer": job.source_layer,
            "rgb_name": rgb_name,
            "ir_name": ir_name,
            "out_name": out_name,
            "rgb_path": os.path.join(SOURCE_RGB_FOLDER, rgb_name),
            "ir_path": os.path.join(SOURCE_IR_FOLDER, ir_name),
            "out_path": out_path,
            "log_prefix": prefix,
        })

    if not ready_jobs:
        print("\nNothing to do (all tiles missing or already exist).", flush=True)
    else:
        workers = max(1, int(MAX_WORKERS))
        mode = "serial" if workers == 1 else f"parallel ({workers} workers)"
        print(
            f"\nProcessing {len(ready_jobs)} tile(s) in {mode}. "
            f"GDAL threads/tile={GDAL_NUM_THREADS}, cache={GDAL_CACHEMAX_MB} MB.",
            flush=True,
        )

        # --------------------------------------------------------------
        # 2a) Serial path: identical to previous behaviour (logs per block)
        # --------------------------------------------------------------
        if workers == 1:
            for job_args in ready_jobs:
                t_tile = time.time()
                try:
                    status, detail = _fuse_one_tile(
                        job_args["rgb_path"],
                        job_args["ir_path"],
                        job_args["out_path"],
                        log_prefix=job_args["log_prefix"],
                    )
                except KeyboardInterrupt:
                    print(f"{job_args['log_prefix']} INTERRUPTED by user. Stopping.", flush=True)
                    summaries.append(TileSummary(
                        job_args["tile_id"], job_args["source_layer"],
                        job_args["rgb_name"], job_args["ir_name"], "interrupted",
                    ))
                    break

                summaries.append(TileSummary(
                    job_args["tile_id"], job_args["source_layer"],
                    job_args["rgb_name"], job_args["ir_name"], status, detail,
                ))
                if status == "ok":
                    written += 1
                    print(
                        f"{job_args['log_prefix']} OK   {job_args['tile_id']} -> {job_args['out_name']}  "
                        f"(tile total {time.time() - t_tile:.1f}s)",
                        flush=True,
                    )
                else:
                    print(
                        f"{job_args['log_prefix']} FAIL {job_args['tile_id']} {status}: {detail}",
                        flush=True,
                    )

        # --------------------------------------------------------------
        # 2b) Parallel path: ProcessPoolExecutor across tiles
        # --------------------------------------------------------------
        else:
            try:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_process_tile_job, j): j for j in ready_jobs}
                    completed = 0
                    for fut in as_completed(futures):
                        completed += 1
                        try:
                            result = fut.result()
                        except Exception as exc:  # noqa: BLE001
                            job_args = futures[fut]
                            print(
                                f"{job_args['log_prefix']} CRASH {job_args['tile_id']}: "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            summaries.append(TileSummary(
                                job_args["tile_id"], job_args["source_layer"],
                                job_args["rgb_name"], job_args["ir_name"], "crash", str(exc),
                            ))
                            continue

                        status = result["status"]
                        summaries.append(TileSummary(
                            result["tile_id"], result["source_layer"],
                            result["rgb_name"], result["ir_name"], status, result["detail"],
                        ))
                        if status == "ok":
                            written += 1
                            print(
                                f"[done {completed}/{len(ready_jobs)}] OK   "
                                f"{result['tile_id']} -> {result['out_name']}  "
                                f"({result['elapsed_s']:.1f}s)",
                                flush=True,
                            )
                        else:
                            print(
                                f"[done {completed}/{len(ready_jobs)}] FAIL "
                                f"{result['tile_id']} {status}: {result['detail']}",
                                flush=True,
                            )
            except KeyboardInterrupt:
                print("\nINTERRUPTED by user (Ctrl+C). Stopping pool.", flush=True)

    # ---- summary ----
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"  Unique tiles in AOI: {len(jobs)}")
    print(f"  Written:             {written}")
    print(f"  Skipped existing:    {skipped_existing}")

    by_status: dict[str, int] = {}
    for s in summaries:
        by_status[s.status] = by_status.get(s.status, 0) + 1
    print("\n  Counts by status:")
    for status, count in sorted(by_status.items()):
        print(f"    {status:18s} {count}")

    print("\n  Counts by source layer:")
    by_layer: dict[str, dict[str, int]] = {}
    for s in summaries:
        layer_map = by_layer.setdefault(s.source_layer, {})
        layer_map[s.status] = layer_map.get(s.status, 0) + 1
    for layer, status_map in by_layer.items():
        total = sum(status_map.values())
        print(f"    {layer} (n={total})")
        for status, count in sorted(status_map.items()):
            print(f"      {status:18s} {count}")

    print(f"\nElapsed: {time.time() - t_start:.2f} s")


if __name__ == "__main__":
    with _keep_system_awake(keep_display_on=True):
        main()
