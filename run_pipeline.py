"""
Leucaena Segmentation — End-to-End Pipeline Orchestrator
=========================================================

Single command that executes the full data-preparation and training workflow:

    Step 1  Tile copy         — copy RGB / IR / LAZ from D:\\ into dest\\ tree
    Step 2  RGBNIR fusion     — merge RGB + IR → 4-band RGBNIR GeoTIFFs
    Step 2b Build overviews   — gdaladdo .ovr for QGIS performance (opt-in)
    Step 3  LAZ → CHM         — rasterise point clouds (Docker, PDAL)
    Step 4  Patch generation  — aligned optical + LiDAR patches (Docker, GDAL)
    Step 5  Training          — ResUNet training (Docker, CUDA)  [opt-in --train]

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

    python run_pipeline.py ... --steps 1,2   # only copy + fuse
    python run_pipeline.py ... --steps 3,4,5 # CHM + patches + train (Docker)

Architecture
------------
- Steps 1 + 2 run natively (no Docker) — only geopandas + rasterio needed.
- Steps 3, 4, 5 run inside the pre-built Docker image
  ``leucaena-segmentation:cuda`` so PDAL, GDAL-Python and PyTorch are
  available without a local conda install.
- All Docker calls use ``docker run`` with explicit ``-v`` bind-mounts so
  the orchestrator stays independent of the ``.env`` file / WSL paths.
- See ``pipeline/`` for modular helpers (layout, runners, tile_index, log).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make sure the repo root is on sys.path so pipeline.* imports work
# when run_pipeline.py is called from outside the repo directory.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline.layout import DestLayout
from pipeline.log import setup_logging
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
Full pipeline (all 5 steps + overviews):
  python run_pipeline.py --aoi "G:/...gpkg" --layer articulacao_laser_voo22_AOI_treino
    --source D:\\ --dest "C:\\00_DATASETS_AI\\260515-piracicaba-aoi" --build-ovr --train

Copy + fuse only:
  python run_pipeline.py ... --steps 1,2

Dry-run (print commands, no changes):
  python run_pipeline.py ... --dry-run
""",
    )

    # ---- Required ----
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
        help=(
            "Layer name inside the GeoPackage, "
            "e.g. articulacao_laser_voo22_AOI_treino"
        ),
    )
    p.add_argument(
        "--dest",
        required=True,
        metavar="DIR",
        type=Path,
        help="Destination root, e.g. C:\\00_DATASETS_AI\\260515-piracicaba-aoi",
    )

    # ---- Optional / paths ----
    p.add_argument(
        "--source",
        default=r"D:\\",
        metavar="DIR",
        type=Path,
        help=(
            "Source root with rgb/, ir/, laz/ sub-folders "
            "(default: D:\\\\)"
        ),
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

    # ---- Step selection ----
    p.add_argument(
        "--steps",
        default=None,
        metavar="1,2,3,4,5",
        help=(
            "Comma-separated list of steps to run. "
            "Default: 1,2,3,4 (or 1,2,3,4,5 when --train is given). "
            "Step 2b (overviews) always follows step 2 when --build-ovr is set."
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

    # ---- Behaviour flags ----
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
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return p


# =============================================================================
# Pipeline steps
# =============================================================================

def step1_copy(
    args: argparse.Namespace,
    layout: DestLayout,
) -> int:
    """Step 1 — Copy RGB / IR / LAZ tiles from source to dest."""
    log.info("=" * 60)
    log.info("STEP 1 — Tile Copy")
    log.info("=" * 60)

    cmd: list[str | Path] = [
        sys.executable,
        _REPO / "prep-copy-tiles-from-aoi.py",
        "--aoi", args.aoi,
        "--layer", args.layer,
        "--source-laz",  str(args.source / "laz"),
        "--source-rgb",  str(args.source / "rgb"),
        "--source-ir",   str(args.source / "ir"),
        "--dest-laz",    str(layout.lidar_raw),
        "--dest-rgb",    str(layout.opt_raw_rgb),
        "--dest-ir",     str(layout.opt_raw_ir),
    ]
    if args.id_column:
        cmd += ["--id-column", args.id_column]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.dry_run:
        cmd.append("--dry-run")

    return run_cmd(cmd, cwd=_REPO, dry_run=False, label="Step 1: tile copy")


def step2_rgbnir(
    args: argparse.Namespace,
    layout: DestLayout,
) -> int:
    """Step 2 — Fuse RGB + IR → 4-band RGBNIR GeoTIFFs."""
    log.info("=" * 60)
    log.info("STEP 2 — RGBNIR Fusion")
    log.info("=" * 60)

    cmd: list[str | Path] = [
        sys.executable,
        _REPO / "prep-rgbnir-from-rgb-ir.py",
        "--aoi", args.aoi,
        "--layer", args.layer,
        "--source-rgb", str(layout.opt_raw_rgb),
        "--source-ir",  str(layout.opt_raw_ir),
        "--out-dir",    str(layout.opt_rgbnir),
        "--workers",    str(args.workers),
    ]
    if args.id_column:
        cmd += ["--id-column", args.id_column]
    if args.overwrite:
        cmd.append("--overwrite")

    return run_cmd(cmd, cwd=_REPO, dry_run=args.dry_run, label="Step 2: RGBNIR fusion")


def step2b_overviews(
    args: argparse.Namespace,
    layout: DestLayout,
) -> int:
    """Step 2b — Build .ovr GeoTIFF overviews for QGIS performance."""
    log.info("=" * 60)
    log.info("STEP 2b — Build GeoTIFF Overviews")
    log.info("=" * 60)

    tifs = sorted(layout.opt_rgbnir.glob("*.tif"))
    if not tifs:
        log.warning("No .tif files found in %s; skipping overviews.", layout.opt_rgbnir)
        return 0

    log.info("Building overviews for %d tile(s) …", len(tifs))

    # Build all overviews in a single Docker invocation using a bash loop.
    # Mounting opt/rgbnir as /data/rgbnir inside the container.
    bash_cmd = (
        "for f in /data/rgbnir/*.tif; do "
        "gdaladdo -ro "
        "--config COMPRESS_OVERVIEW DEFLATE "
        "--config PREDICTOR_OVERVIEW 2 "
        "--config BIGTIFF_OVERVIEW IF_SAFER "
        "--config GDAL_TIFF_OVR_BLOCKSIZE 512 "
        "--config GDAL_NUM_THREADS ALL_CPUS "
        "-r average "
        '"$f" '
        "2 4 8 16 32 64; "
        "done"
    )

    return docker_run(
        ["bash", "-c", bash_cmd],
        repo_dir=_REPO,
        volumes={layout.opt_rgbnir: ("/data/rgbnir", "rw")},
        dry_run=args.dry_run,
        label="Step 2b: gdaladdo overviews",
        gpu=False,
    )


def step3_chm(
    args: argparse.Namespace,
    layout: DestLayout,
) -> int:
    """Step 3 — Rasterise LAZ point clouds into CHM GeoTIFFs (Docker/PDAL)."""
    log.info("=" * 60)
    log.info("STEP 3 — LAZ → CHM (Docker)")
    log.info("=" * 60)

    script_args: list[str] = [
        "python", "prep-lidar-rasters.py",
        "--laz-dir",   "/data/laz",
        "--tiles-dir", "/data/rgbnir",
        "--out-dir",   "/data/lidar",
    ]
    if args.overwrite:
        script_args.append("--overwrite")

    return docker_run(
        script_args,
        repo_dir=_REPO,
        volumes={
            layout.lidar_raw:  ("/data/laz",    "ro"),
            layout.opt_rgbnir: ("/data/rgbnir", "ro"),
            layout.lidar_chm:  ("/data/lidar",  "rw"),
        },
        dry_run=args.dry_run,
        label="Step 3: LAZ → CHM",
        gpu=False,
    )


def step4_patches(
    args: argparse.Namespace,
    layout: DestLayout,
) -> int:
    """Step 4 — Generate aligned optical + LiDAR patches (Docker/GDAL)."""
    log.info("=" * 60)
    log.info("STEP 4 — Patch Generation (Docker)")
    log.info("=" * 60)

    # Resolve the annotations file
    if args.annotations:
        annotations_path = Path(args.annotations)
    else:
        # Look for common filenames inside dest/annotations/
        candidates = [
            layout.annotations / "leucaena.geojson",
            layout.annotations / "polygons.geojson",
        ]
        annotations_path = next(
            (p for p in candidates if p.exists()),
            layout.annotations / "leucaena.geojson",  # default even if missing
        )

    # annotations dir is mounted as /data/masks; we need the basename
    annotations_basename = annotations_path.name
    container_masks_path = f"/data/masks/{annotations_basename}"

    if not annotations_path.exists():
        log.warning(
            "Annotations file not found: %s\n"
            "  Place your GeoJSON / GeoPackage there before running step 4, "
            "or pass --annotations <path>.",
            annotations_path,
        )

    script_args: list[str] = [
        "python", "prep-patches-from-tiles.py",
        "--tiles-dir", "/data/rgbnir",
        "--lidar-dir", "/data/lidar",
        "--masks",     container_masks_path,
        "--out-dir",   "/data/patches",
    ]
    if args.overwrite:
        script_args.append("--overwrite")

    return docker_run(
        script_args,
        repo_dir=_REPO,
        volumes={
            layout.opt_rgbnir:  ("/data/rgbnir", "ro"),
            layout.lidar_chm:   ("/data/lidar",  "ro"),
            annotations_path.parent: ("/data/masks", "ro"),
            layout.patches:     ("/data/patches", "rw"),
        },
        dry_run=args.dry_run,
        label="Step 4: patch generation",
        gpu=False,
    )


def step5_train(
    args: argparse.Namespace,
    layout: DestLayout,
) -> int:
    """Step 5 — Train ResUNet model (Docker/CUDA)."""
    log.info("=" * 60)
    log.info("STEP 5 — Training (Docker/GPU)")
    log.info("=" * 60)

    manifest_container = "/data/patches/manifest.csv"
    # Experiment outputs land in /workspace/experiments (repo-mounted volume)
    # so checkpoints and logs appear in the repository under experiments/exp_N/.

    script_args: list[str] = [
        "python", "train.py",
        "--patch-source", "file",
        "--manifest",     manifest_container,
        "-e",             str(args.experiment),
    ]

    return docker_run(
        script_args,
        repo_dir=_REPO,
        volumes={
            layout.patches: ("/data/patches", "ro"),
        },
        dry_run=args.dry_run,
        label=f"Step 5: training (experiment {args.experiment})",
        gpu=True,
    )


# =============================================================================
# Utilities
# =============================================================================

def _parse_steps(raw: str | None, train: bool) -> set[int]:
    """Convert --steps string to a set of integers."""
    if raw is None:
        steps = {1, 2, 3, 4}
        if train:
            steps.add(5)
        return steps
    try:
        return {int(s.strip()) for s in raw.split(",") if s.strip()}
    except ValueError:
        log.error("--steps must be a comma-separated list of integers, e.g. 1,2,3")
        sys.exit(1)


def _verify_docker_image() -> bool:
    """Return True if the Docker image exists locally."""
    from pipeline.runners import DOCKER_IMAGE
    import subprocess
    result = subprocess.run(
        ["docker", "image", "inspect", DOCKER_IMAGE],
        capture_output=True,
    )
    return result.returncode == 0


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ---- Destination layout ----
    layout = DestLayout(root=args.dest.resolve())

    # ---- Logging ----
    setup_logging(layout.models_logs, verbose=args.verbose)
    log.info("Leucaena Segmentation Pipeline")
    log.info("  repo   : %s", _REPO)
    log.info("  AOI    : %s  layer=%s", args.aoi, args.layer)
    log.info("  source : %s", args.source)
    log.info("  dest   : %s", layout.root)
    if args.dry_run:
        log.info("  mode   : DRY-RUN (no files will be written)")

    # ---- Determine which steps to run ----
    steps = _parse_steps(args.steps, args.train)
    log.info("  steps  : %s", sorted(steps))
    if args.build_ovr:
        log.info("  overviews: yes (after step 2)")

    # ---- Pre-flight checks ----
    needs_docker = bool(steps & {3, 4, 5})
    if needs_docker and not args.dry_run:
        if not _verify_docker_image():
            log.error(
                "Docker image 'leucaena-segmentation:cuda' not found. "
                "Run:  docker compose build  from the repo root first."
            )
            sys.exit(1)

    # ---- Create destination directories ----
    if not args.dry_run:
        layout.ensure_all()

    # ---- Warm up tile index (caches D:\ listing to avoid re-scan) ----
    if 1 in steps and not args.dry_run:
        idx = TileIndex(
            source_root=args.source.resolve(),
            cache_dir=layout.pipeline_cache,
        )
        idx.load(rebuild=args.rebuild_index)

    # ---- Execute steps ----
    t_start = time.time()
    failed: list[int] = []

    def _run(step_num: int, fn, *fn_args) -> bool:
        t = time.time()
        rc = fn(*fn_args)
        elapsed = time.time() - t
        if rc != 0:
            log.error("Step %d FAILED (exit %d) in %.1f s", step_num, rc, elapsed)
            failed.append(step_num)
            return False
        log.info("Step %d OK in %.1f s", step_num, elapsed)
        return True

    if 1 in steps:
        if not _run(1, step1_copy, args, layout):
            log.error("Aborting pipeline after step 1 failure.")
            sys.exit(1)

    if 2 in steps:
        if not _run(2, step2_rgbnir, args, layout):
            log.error("Aborting pipeline after step 2 failure.")
            sys.exit(1)
        if args.build_ovr:
            _run("2b", step2b_overviews, args, layout)

    if 3 in steps:
        _run(3, step3_chm, args, layout)

    if 4 in steps:
        _run(4, step4_patches, args, layout)

    if 5 in steps:
        _run(5, step5_train, args, layout)

    # ---- Summary ----
    elapsed = time.time() - t_start
    log.info("=" * 60)
    if failed:
        log.warning("Pipeline finished with failures in step(s): %s", failed)
    else:
        log.info("Pipeline completed successfully in %.1f s", elapsed)
    log.info("Outputs: %s", layout.root)
    log.info("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
