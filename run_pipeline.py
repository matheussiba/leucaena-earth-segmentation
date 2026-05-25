"""
Leucaena Segmentation — End-to-End Pipeline Orchestrator
=========================================================

Single command that executes the full data-preparation and training workflow:

    Step 1   Tile copy         — copy RGB / IR / LAZ from D:\\ into dest\\ tree
    Step 2   RGBNIR fusion     — merge RGB + IR → 4-band RGBNIR GeoTIFFs
    Step 2b  Build overviews   — gdaladdo .ovr for QGIS performance (--build-ovr)
    Step 3   LAZ → CHM         — rasterise point clouds (Docker, PDAL)
    Step 4   Patch generation  — aligned optical + LiDAR patches (Docker, GDAL)
    Step 5   Training          — ResUNet training (Docker, CUDA)  [--train]

Typical usage
-------------
::

    python run_pipeline.py ^
      --aoi "G:\\My Drive\\...\\gdb-leucena_v2.gpkg" ^
      --layer articulacao_laser_voo22_AOI_treino ^
      --source D:\\ ^
      --dest "C:\\00_DATASETS_AI\\260515-piracicaba-aoi" ^
      --build-ovr ^
      --train

Quick dry-run (prints every command without touching any file)::

    python run_pipeline.py --aoi ... --layer ... --source D:\\ --dest ... --dry-run

Run only specific steps::

    python run_pipeline.py ... --steps 1,2   # copy + fuse only
    python run_pipeline.py ... --steps 3,4,5 # CHM + patches + train (Docker)

Logging
-------
Every line of stdout + stderr from every subprocess is captured and written
to a timestamped log file under  dest/models/logs/pipeline_<timestamp>.log.
A copy is also kept at  dest/models/logs/pipeline_latest.log.

If something goes wrong:
    1. Check the console for the [ERROR] lines.
    2. Open pipeline_latest.log for the full trace including subprocess output.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make sure the repo root is on sys.path so pipeline.* imports
# work when run_pipeline.py is called from outside the repo directory.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline.layout import DestLayout
from pipeline.log import setup_logging, update_latest
from pipeline.runners import run_cmd, docker_run
from pipeline.tile_index import TileIndex

log = logging.getLogger("pipeline")


# =============================================================================
# Argparse
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Leucaena segmentation end-to-end pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
Full pipeline (all steps + overviews):
  python run_pipeline.py \\
    --aoi "G:/...gpkg" --layer articulacao_laser_voo22_AOI_treino \\
    --source D:\\ --dest "C:\\00_DATASETS_AI\\260515-piracicaba-aoi" \\
    --build-ovr --train

Copy + fuse only:
  python run_pipeline.py ... --steps 1,2

Dry-run (print commands, no changes):
  python run_pipeline.py ... --dry-run --verbose

Resume from step 3 (skip copy + fuse already done):
  python run_pipeline.py ... --steps 3,4 --train
""",
    )

    # ---- Required ----------------------------------------------------------
    p.add_argument(
        "--aoi",
        required=True,
        metavar="PATH",
        help="GeoPackage (.gpkg) containing the AOI layer.",
    )
    p.add_argument(
        "--layer",
        required=True,
        metavar="LAYER",
        help="Layer name inside the GeoPackage, e.g. articulacao_laser_voo22_AOI_treino",
    )
    p.add_argument(
        "--dest",
        required=True,
        metavar="DIR",
        type=Path,
        help="Destination root, e.g. C:\\00_DATASETS_AI\\260515-piracicaba-aoi",
    )

    # ---- Optional / paths --------------------------------------------------
    p.add_argument(
        "--source",
        default=r"D:\\",
        metavar="DIR",
        type=Path,
        help="Source root with rgb/, ir/, laz/ sub-folders (default: D:\\\\)",
    )
    p.add_argument(
        "--id-column",
        default=None,
        metavar="COL",
        help=(
            "Tile-ID column inside the GeoPackage layer. "
            "Auto-detected from NOMENC_10K / NOMENC_5K / NOMENC_2K if omitted."
        ),
    )
    p.add_argument(
        "--annotations",
        default=None,
        metavar="PATH",
        type=Path,
        help=(
            "Path to the leucaena polygon GeoJSON / GeoPackage used for "
            "patch labelling (default: dest/annotations/leucaena.geojson). "
            "Must exist before running step 4."
        ),
    )
    p.add_argument(
        "--experiment",
        type=int,
        default=1,
        metavar="N",
        help="Experiment number passed to train.py -e (default: 1).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel workers for step 2 RGBNIR fusion (default: 4).",
    )

    # ---- Step selection ----------------------------------------------------
    p.add_argument(
        "--steps",
        default=None,
        metavar="1,2,3,4,5",
        help=(
            "Comma-separated steps to run (default: 1,2,3,4; or 1,2,3,4,5 "
            "when --train is given). Step 2b follows step 2 when --build-ovr "
            "is set."
        ),
    )
    p.add_argument(
        "--build-ovr",
        action="store_true",
        help="Build GeoTIFF overviews (.ovr) for QGIS after step 2.",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="Include step 5 (model training) in the pipeline.",
    )

    # ---- Behaviour flags ---------------------------------------------------
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run / overwrite outputs that already exist.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing anything.",
    )
    p.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force a fresh scan of --source (ignores cached tile_index.json).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging on the console (file always gets DEBUG).",
    )

    return p


# =============================================================================
# Pre-flight checks (logged, not fatal unless explicitly stated)
# =============================================================================

def _check_path(p: Path, label: str, *, required: bool = False) -> bool:
    """Log whether *p* exists. Return True if it exists."""
    if p.exists():
        log.debug("  [✓] %-35s %s", label, p)
        return True
    elif required:
        log.error("  [✗] MISSING (required): %-25s %s", label, p)
    else:
        log.warning("  [?] not yet / will be created: %-10s %s", label, p)
    return False


def _preflight_report(args: argparse.Namespace, layout: DestLayout, steps: set[int]) -> None:
    """Log a full preflight report: resolved paths, source dirs, dest dirs."""
    log.info("")
    log.info("  ┌─ PREFLIGHT CHECK ─────────────────────────────────────┐")
    log.info("  │ AOI GeoPackage")
    _check_path(Path(args.aoi), "AOI GeoPackage", required=True)
    log.info("  │")

    if 1 in steps:
        log.info("  │ Source directories (Step 1 input)")
        for sub in ("rgb", "ir", "laz"):
            _check_path(args.source / sub, f"source/{sub}", required=False)
        log.info("  │")

    log.info("  │ Destination directories")
    for label, path in [
        ("opt/raw/rgb",  layout.opt_raw_rgb),
        ("opt/raw/ir",   layout.opt_raw_ir),
        ("opt/rgbnir",   layout.opt_rgbnir),
        ("lidar/raw",    layout.lidar_raw),
        ("lidar/chm",    layout.lidar_chm),
        ("annotations",  layout.annotations),
        ("patches",      layout.patches),
        ("models/logs",  layout.models_logs),
    ]:
        _check_path(path, label)
    log.info("  │")

    if 4 in steps:
        ann = (
            Path(args.annotations) if args.annotations
            else _resolve_annotations(args, layout)
        )
        log.info("  │ Annotations file (Step 4 input)")
        _check_path(ann, "annotations file", required=False)
        log.info("  │")

    log.info("  └───────────────────────────────────────────────────────┘")
    log.info("")


# =============================================================================
# Pipeline steps
# =============================================================================

def step1_copy(args: argparse.Namespace, layout: DestLayout) -> int:
    """Step 1 — Copy RGB / IR / LAZ tiles from source to dest."""
    _step_header(1, "Tile Copy  (prep-copy-tiles-from-aoi.py)")

    src_laz = args.source / "laz"
    src_rgb = args.source / "rgb"
    src_ir  = args.source / "ir"

    log.info("  Source:")
    log.info("    LAZ : %s  (exists=%s)", src_laz, src_laz.exists())
    log.info("    RGB : %s  (exists=%s)", src_rgb, src_rgb.exists())
    log.info("    IR  : %s  (exists=%s)", src_ir,  src_ir.exists())
    log.info("  Destination:")
    log.info("    LAZ → %s", layout.lidar_raw)
    log.info("    RGB → %s", layout.opt_raw_rgb)
    log.info("    IR  → %s", layout.opt_raw_ir)
    log.info("  AOI: %s  layer=%s", args.aoi, args.layer)
    if args.id_column:
        log.info("  id-column: %s", args.id_column)

    cmd: list[str | Path] = [
        sys.executable,
        _REPO / "prep-copy-tiles-from-aoi.py",
        "--aoi",        str(args.aoi),
        "--layer",      args.layer,
        "--source-laz", str(src_laz),
        "--source-rgb", str(src_rgb),
        "--source-ir",  str(src_ir),
        "--dest-laz",   str(layout.lidar_raw),
        "--dest-rgb",   str(layout.opt_raw_rgb),
        "--dest-ir",    str(layout.opt_raw_ir),
    ]
    if args.id_column:
        cmd += ["--id-column", args.id_column]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.dry_run:
        cmd.append("--dry-run")

    log.debug("  Full command: %s", " ".join(str(c) for c in cmd))
    rc = run_cmd(cmd, cwd=_REPO, dry_run=False, label="Step 1 — tile copy")

    if rc == 0:
        _log_dir_count(layout.lidar_raw,   ".laz", "LAZ copied")
        _log_dir_count(layout.opt_raw_rgb, ".tif", "RGB copied")
        _log_dir_count(layout.opt_raw_ir,  ".tif", "IR  copied")
    return rc


def step2_rgbnir(args: argparse.Namespace, layout: DestLayout) -> int:
    """Step 2 — Fuse RGB + IR → 4-band RGBNIR GeoTIFFs."""
    _step_header(2, "RGBNIR Fusion  (prep-rgbnir-from-rgb-ir.py)")

    log.info("  Input RGB : %s  (%d .tif)",
             layout.opt_raw_rgb, _count(layout.opt_raw_rgb, ".tif"))
    log.info("  Input IR  : %s  (%d .tif)",
             layout.opt_raw_ir, _count(layout.opt_raw_ir, ".tif"))
    log.info("  Output    : %s", layout.opt_rgbnir)
    log.info("  Workers   : %d", args.workers)

    cmd: list[str | Path] = [
        sys.executable,
        _REPO / "prep-rgbnir-from-rgb-ir.py",
        "--aoi",        str(args.aoi),
        "--layer",      args.layer,
        "--source-rgb", str(layout.opt_raw_rgb),
        "--source-ir",  str(layout.opt_raw_ir),
        "--out-dir",    str(layout.opt_rgbnir),
        "--workers",    str(args.workers),
    ]
    if args.id_column:
        cmd += ["--id-column", args.id_column]
    if args.overwrite:
        cmd.append("--overwrite")

    log.debug("  Full command: %s", " ".join(str(c) for c in cmd))
    rc = run_cmd(cmd, cwd=_REPO, dry_run=args.dry_run, label="Step 2 — RGBNIR fusion")

    if rc == 0:
        _log_dir_count(layout.opt_rgbnir, ".tif", "RGBNIR tiles")
    return rc


def step2b_overviews(args: argparse.Namespace, layout: DestLayout) -> int:
    """Step 2b — Build .ovr GeoTIFF overviews (QGIS performance)."""
    _step_header("2b", "GeoTIFF Overviews  (gdaladdo)")

    tifs = sorted(layout.opt_rgbnir.glob("*.tif"))
    log.info("  Target dir : %s", layout.opt_rgbnir)
    log.info("  .tif files : %d", len(tifs))
    if not tifs:
        log.warning("  No .tif files found — skipping overview generation.")
        return 0

    # Check which already have an .ovr (skip them to avoid duplicate builds)
    missing_ovr = [t for t in tifs if not t.with_suffix(".tif.ovr").exists()
                   and not (t.parent / (t.name + ".ovr")).exists()]
    log.info("  Need .ovr  : %d  (already have: %d)", len(missing_ovr), len(tifs) - len(missing_ovr))
    if not missing_ovr and not args.overwrite:
        log.info("  All overviews already exist — nothing to do.")
        return 0

    # Build all overviews in a single Docker call using a bash loop.
    # "-ro" writes the .ovr as an external sidecar file alongside the .tif.
    bash_cmd = (
        "for f in /data/rgbnir/*.tif; do "
        "echo \"[OVR] $f\"; "
        "gdaladdo -ro "
        "--config COMPRESS_OVERVIEW DEFLATE "
        "--config PREDICTOR_OVERVIEW 2 "
        "--config BIGTIFF_OVERVIEW IF_SAFER "
        "--config GDAL_TIFF_OVR_BLOCKSIZE 512 "
        "--config GDAL_NUM_THREADS ALL_CPUS "
        "-r average "
        '"$f" '
        "2 4 8 16 32 64 || echo \"[OVR ERROR] $f\"; "
        "done; "
        "echo '[OVR] All done'"
    )

    log.info("  Docker image: %s", _docker_image())
    log.info("  Bash loop   : %s", bash_cmd[:80] + " …")
    log.debug("  Full bash cmd: %s", bash_cmd)

    rc = docker_run(
        ["bash", "-c", bash_cmd],
        repo_dir=_REPO,
        volumes={layout.opt_rgbnir: ("/data/rgbnir", "rw")},
        dry_run=args.dry_run,
        label="Step 2b — gdaladdo overviews",
        gpu=False,
    )

    if rc == 0:
        ovr_after = len(list(layout.opt_rgbnir.glob("*.tif.ovr"))
                        + list(layout.opt_rgbnir.glob("*.ovr")))
        log.info("  .ovr files  : %d", ovr_after)
    return rc


def step3_chm(args: argparse.Namespace, layout: DestLayout) -> int:
    """Step 3 — Rasterise LAZ point clouds into CHM GeoTIFFs (Docker/PDAL)."""
    _step_header(3, "LAZ → CHM  (prep-lidar-rasters.py, Docker/PDAL)")

    n_laz  = _count(layout.lidar_raw,  ".laz")
    n_rgbn = _count(layout.opt_rgbnir, ".tif")
    log.info("  Input LAZ   : %s  (%d .laz)", layout.lidar_raw,  n_laz)
    log.info("  Align RGBNIR: %s  (%d .tif)", layout.opt_rgbnir, n_rgbn)
    log.info("  Output CHM  : %s", layout.lidar_chm)

    if n_laz == 0:
        log.error(
            "  No .laz files found in %s — did step 1 run successfully?",
            layout.lidar_raw,
        )
        return 1

    _log_volume_check(layout.lidar_raw,  "/data/laz")
    _log_volume_check(layout.opt_rgbnir, "/data/rgbnir")
    _log_volume_check(layout.lidar_chm,  "/data/lidar")

    script_args: list[str] = [
        "python", "prep-lidar-rasters.py",
        "--laz-dir",   "/data/laz",
        "--tiles-dir", "/data/rgbnir",
        "--out-dir",   "/data/lidar",
    ]
    if args.overwrite:
        script_args.append("--overwrite")

    rc = docker_run(
        script_args,
        repo_dir=_REPO,
        volumes={
            layout.lidar_raw:  ("/data/laz",    "ro"),
            layout.opt_rgbnir: ("/data/rgbnir", "ro"),
            layout.lidar_chm:  ("/data/lidar",  "rw"),
        },
        dry_run=args.dry_run,
        label="Step 3 — LAZ → CHM",
        gpu=False,
    )

    if rc == 0:
        _log_dir_count(layout.lidar_chm, ".tif", "CHM tiles")
    return rc


def step4_patches(args: argparse.Namespace, layout: DestLayout) -> int:
    """Step 4 — Generate aligned optical + LiDAR patches (Docker/GDAL)."""
    _step_header(4, "Patch Generation  (prep-patches-from-tiles.py, Docker)")

    n_rgbn = _count(layout.opt_rgbnir, ".tif")
    n_chm  = _count(layout.lidar_chm,  ".tif")
    log.info("  RGBNIR tiles : %s  (%d)", layout.opt_rgbnir, n_rgbn)
    log.info("  CHM tiles    : %s  (%d)", layout.lidar_chm,  n_chm)

    ann = _resolve_annotations(args, layout)
    log.info("  Annotations  : %s  (exists=%s)", ann, ann.exists())
    log.info("  Output       : %s", layout.patches)

    if n_rgbn == 0:
        log.error(
            "  No RGBNIR tiles in %s — did step 2 run?",
            layout.opt_rgbnir,
        )
        return 1
    if not ann.exists():
        log.error(
            "  Annotations file not found: %s\n"
            "  Place your leucaena GeoJSON/GeoPackage there before "
            "running step 4, or pass --annotations <path>.",
            ann,
        )
        return 1

    ann_container = f"/data/masks/{ann.name}"

    _log_volume_check(layout.opt_rgbnir, "/data/rgbnir")
    _log_volume_check(layout.lidar_chm,  "/data/lidar")
    _log_volume_check(ann.parent,        "/data/masks")
    _log_volume_check(layout.patches,    "/data/patches")

    script_args: list[str] = [
        "python", "prep-patches-from-tiles.py",
        "--tiles-dir", "/data/rgbnir",
        "--lidar-dir", "/data/lidar",
        "--masks",     ann_container,
        "--out-dir",   "/data/patches",
    ]
    if args.overwrite:
        script_args.append("--overwrite")

    rc = docker_run(
        script_args,
        repo_dir=_REPO,
        volumes={
            layout.opt_rgbnir: ("/data/rgbnir", "ro"),
            layout.lidar_chm:  ("/data/lidar",  "ro"),
            ann.parent:        ("/data/masks",  "ro"),
            layout.patches:    ("/data/patches","rw"),
        },
        dry_run=args.dry_run,
        label="Step 4 — patch generation",
        gpu=False,
    )

    if rc == 0:
        manifest = layout.patches / "manifest.csv"
        if manifest.exists():
            try:
                n_patches = sum(1 for _ in manifest.open()) - 1  # minus header
                log.info("  Patches written: %d  (manifest: %s)", n_patches, manifest)
            except Exception:
                log.info("  Manifest: %s", manifest)
    return rc


def step5_train(args: argparse.Namespace, layout: DestLayout) -> int:
    """Step 5 — Train ResUNet model (Docker/CUDA)."""
    _step_header(5, f"Training  (train.py  -e {args.experiment}, Docker/GPU)")

    manifest = layout.patches / "manifest.csv"
    log.info("  Manifest    : %s  (exists=%s)", manifest, manifest.exists())
    log.info("  Experiment  : %d", args.experiment)
    log.info("  GPU         : yes (--gpus all)")
    log.info("  Exp outputs → repo/experiments/exp_%d/", args.experiment)

    if not manifest.exists() and not args.dry_run:
        log.error(
            "  Manifest not found: %s — did step 4 run?",
            manifest,
        )
        return 1

    _log_volume_check(layout.patches, "/data/patches")

    script_args: list[str] = [
        "python", "train.py",
        "--patch-source", "file",
        "--manifest",     "/data/patches/manifest.csv",
        "-e",             str(args.experiment),
    ]

    rc = docker_run(
        script_args,
        repo_dir=_REPO,
        volumes={layout.patches: ("/data/patches", "ro")},
        dry_run=args.dry_run,
        label=f"Step 5 — training (exp {args.experiment})",
        gpu=True,
    )
    return rc


# =============================================================================
# Logging helpers
# =============================================================================

def _step_header(num: int | str, title: str) -> None:
    log.info("")
    log.info("━" * 62)
    log.info("  STEP %s — %s", num, title)
    log.info("━" * 62)


def _count(d: Path, ext: str) -> int:
    """Count files with *ext* directly under *d*."""
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() == ext)


def _log_dir_count(d: Path, ext: str, label: str) -> None:
    n = _count(d, ext)
    log.info("  %-18s %d  (%s)", label + ":", n, d)


def _log_volume_check(host: Path, container: str) -> None:
    exists = "✓" if host.exists() else "✗ NOT FOUND"
    log.debug("  volume  %s → %s  [%s]", host, container, exists)


def _docker_image() -> str:
    from pipeline.runners import DOCKER_IMAGE
    return DOCKER_IMAGE


def _resolve_annotations(args: argparse.Namespace, layout: DestLayout) -> Path:
    if args.annotations:
        return Path(args.annotations)
    for candidate in (
        layout.annotations / "leucaena.geojson",
        layout.annotations / "polygons.geojson",
        layout.annotations / "leucaena.gpkg",
    ):
        if candidate.exists():
            return candidate
    return layout.annotations / "leucaena.geojson"  # default even if missing


# =============================================================================
# Utilities
# =============================================================================

def _parse_steps(raw: str | None, train: bool) -> list[int]:
    if raw is None:
        steps = [1, 2, 3, 4]
        if train:
            steps.append(5)
        return steps
    try:
        return sorted({int(s.strip()) for s in raw.split(",") if s.strip()})
    except ValueError:
        log.error("--steps must be comma-separated integers, e.g. 1,2,3")
        sys.exit(1)


def _verify_docker_image() -> bool:
    """Return True if the pre-built Docker image exists locally."""
    import subprocess as _sp
    from pipeline.runners import DOCKER_IMAGE
    r = _sp.run(
        ["docker", "image", "inspect", DOCKER_IMAGE],
        capture_output=True,
    )
    return r.returncode == 0


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    layout = DestLayout(root=args.dest.resolve())

    # ---- Logging: set up BEFORE any log.* calls ----------------------------
    setup_logging(layout.models_logs, verbose=args.verbose)

    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   Leucaena Segmentation Pipeline  —  %s    ║", run_ts)
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info("")
    log.info("  Repo        : %s", _REPO)
    log.info("  AOI         : %s", args.aoi)
    log.info("  Layer       : %s", args.layer)
    log.info("  Source root : %s", args.source)
    log.info("  Dest root   : %s", layout.root)
    log.info("  Overwrite   : %s", args.overwrite)
    if args.dry_run:
        log.info("  *** DRY-RUN MODE — no files will be written ***")

    # ---- Determine steps ---------------------------------------------------
    steps_list = _parse_steps(args.steps, args.train)
    steps_set  = set(steps_list)
    log.info("  Steps       : %s%s",
             steps_list,
             "  + 2b (overviews)" if args.build_ovr else "")
    log.info("")

    # ---- Pre-flight --------------------------------------------------------
    _preflight_report(args, layout, steps_set)

    # ---- Pre-flight: Docker check for steps 3-5 ----------------------------
    needs_docker = bool(steps_set & {3, 4, 5})
    if needs_docker and not args.dry_run:
        log.info("Checking Docker image …")
        if not _verify_docker_image():
            log.error(
                "Docker image 'leucaena-segmentation:cuda' not found.\n"
                "  Run:  docker compose build\n"
                "  in the repo root first, then retry."
            )
            sys.exit(1)
        log.info("  Docker image OK ✓")

    # ---- Create destination directories ------------------------------------
    if not args.dry_run:
        layout.ensure_all()
        log.debug("All destination directories created / verified.")

    # ---- Warm-up tile index (caches D:\\ listing) --------------------------
    if 1 in steps_set and not args.dry_run:
        log.info("Building / loading tile index for %s …", args.source)
        idx = TileIndex(
            source_root=args.source.resolve(),
            cache_dir=layout.pipeline_cache,
        )
        idx.load(rebuild=args.rebuild_index)

    # ====================================================================
    # Execute steps
    # ====================================================================

    t_pipeline_start = time.time()

    # Track per-step outcome: {step_label: (status, elapsed_s, note)}
    results: list[tuple[str, str, float, str]] = []

    def _run_step(
        label: str,
        fn,
        *fn_args,
        abort_on_fail: bool = True,
    ) -> bool:
        """Execute one pipeline step; return True on success."""
        t0 = time.time()
        try:
            rc = fn(*fn_args)
        except KeyboardInterrupt:
            elapsed = time.time() - t0
            log.warning("Step %s INTERRUPTED by user.", label)
            results.append((label, "interrupted", elapsed, "Ctrl+C"))
            _print_summary(results, layout)
            sys.exit(130)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            log.error(
                "Step %s raised an unhandled exception: %s: %s",
                label, type(exc).__name__, exc,
            )
            import traceback
            log.debug(traceback.format_exc())
            results.append((label, "exception", elapsed, str(exc)))
            if abort_on_fail:
                _print_summary(results, layout)
                sys.exit(1)
            return False

        elapsed = time.time() - t0
        if rc != 0:
            log.error(
                "Step %s FAILED with exit code %d  (%.1f s)",
                label, rc, elapsed,
            )
            log.error(
                "  → Check the log for details: %s",
                layout.models_logs / "pipeline_latest.log",
            )
            results.append((label, f"FAILED (exit {rc})", elapsed, ""))
            if abort_on_fail:
                _print_summary(results, layout)
                sys.exit(rc)
            return False

        log.info("Step %s ✓  (%.1f s)", label, elapsed)
        results.append((label, "OK", elapsed, ""))
        return True

    # ---- Steps ----------------------------------------------------------------

    if 1 in steps_set:
        _run_step("1", step1_copy, args, layout)

    if 2 in steps_set:
        ok = _run_step("2", step2_rgbnir, args, layout)
        if ok and args.build_ovr:
            _run_step("2b", step2b_overviews, args, layout, abort_on_fail=False)

    if 3 in steps_set:
        _run_step("3", step3_chm, args, layout)

    if 4 in steps_set:
        _run_step("4", step4_patches, args, layout)

    if 5 in steps_set:
        _run_step("5", step5_train, args, layout)

    # ---- Final summary --------------------------------------------------------
    _print_summary(results, layout)


def _print_summary(
    results: list[tuple[str, str, float, str]],
    layout: DestLayout,
) -> None:
    """Print a summary table and flush log to pipeline_latest.log."""
    total = time.time()  # not reliable here — used just for a divider
    all_ok = all(r[1] == "OK" for r in results)

    log.info("")
    log.info("━" * 62)
    log.info("  PIPELINE SUMMARY")
    log.info("━" * 62)
    log.info("  %-6s  %-22s  %s", "Step", "Status", "Duration")
    log.info("  %-6s  %-22s  %s", "------", "----------------------", "--------")
    for label, status, elapsed, note in results:
        note_str = f"  ({note})" if note else ""
        log.info(
            "  %-6s  %-22s  %.1f s%s",
            label, status, elapsed, note_str,
        )
    log.info("━" * 62)

    log_latest = layout.models_logs / "pipeline_latest.log"
    if all_ok:
        log.info("  ✓ All steps completed successfully.")
    else:
        failed = [r[0] for r in results if r[1] not in ("OK",)]
        log.warning("  ✗ Step(s) with errors: %s", failed)
        log.warning("  → Full log: %s", log_latest)

    log.info("  Log: %s", log_latest)
    log.info("━" * 62)

    update_latest(layout.models_logs)


if __name__ == "__main__":
    main()
