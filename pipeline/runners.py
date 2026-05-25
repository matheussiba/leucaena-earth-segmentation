"""Subprocess helpers for running pipeline steps natively or in Docker.

Two public helpers:

``run_cmd(cmd, ...)``
    Run any command (native Python, gdaladdo, …).  Logs the command,
    respects ``dry_run``, returns the exit code.

``docker_run(script_args, ...)``
    Build a ``docker run`` invocation against the pre-built image
    ``leucaena-segmentation:cuda``, mounting the repo + arbitrary extra
    volumes, then delegates to ``run_cmd``.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

# Image built by ``docker compose build`` in the repository root.
DOCKER_IMAGE = "leucaena-segmentation:cuda"

# Standard env vars the container expects (set by docker-compose, replicated
# here so ``docker run`` calls don't need an .env file).
_CONTAINER_ENV: dict[str, str] = {
    "PROJ_LIB": "/opt/conda/share/proj",
    "GDAL_DATA": "/opt/conda/share/gdal",
    "GTIFF_SRS_SOURCE": "EPSG",
    "PYTHONUNBUFFERED": "1",
}


def run_cmd(
    cmd: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
    label: str = "",
) -> int:
    """Run *cmd*, log it, and return the exit code.

    Parameters
    ----------
    cmd:
        Command + arguments (strings or Paths).
    cwd:
        Working directory for the subprocess (default: inherit).
    dry_run:
        When *True*, only log the command without executing it.
    label:
        Short human-readable description used in log messages instead of
        the full command (useful for long Docker invocations).
    """
    display = label or " ".join(str(c) for c in cmd)
    if dry_run:
        log.info("[DRY-RUN] %s", display)
        return 0

    log.info("$ %s", display)
    result = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if result.returncode != 0:
        log.error("Exit %d: %s", result.returncode, display)
    return result.returncode


def docker_run(
    script_args: Sequence[str | Path],
    *,
    repo_dir: Path,
    volumes: dict[Path, tuple[str, str]] | None = None,
    extra_env: dict[str, str] | None = None,
    dry_run: bool = False,
    label: str = "",
    gpu: bool = False,
) -> int:
    """Execute *script_args* inside the pre-built Docker image.

    Parameters
    ----------
    script_args:
        Command to run inside the container, e.g.
        ``["python", "prep-lidar-rasters.py", "--laz-dir", "/data/laz"]``.
    repo_dir:
        Absolute path to the repository root on the host.  Mounted as
        ``/workspace`` (the container's working directory).
    volumes:
        Extra host→container mounts, keyed by host ``Path``.
        Value is ``(container_path_str, mode)`` where *mode* is ``"ro"``
        or ``"rw"``.
    extra_env:
        Extra ``-e KEY=VALUE`` pairs forwarded into the container.
    dry_run:
        Print the docker command without running it.
    label:
        Short description for log messages.
    gpu:
        Add ``--gpus all`` (requires NVIDIA Docker runtime).
    """
    cmd: list[str] = ["docker", "run", "--rm"]

    if gpu:
        cmd += ["--gpus", "all"]

    # Repo → /workspace
    cmd += ["-v", f"{repo_dir}:/workspace", "-w", "/workspace"]

    # Standard container env
    for k, v in _CONTAINER_ENV.items():
        cmd += ["-e", f"{k}={v}"]
    for k, v in (extra_env or {}).items():
        cmd += ["-e", f"{k}={v}"]

    # Extra volume mounts
    for host_path, (container_path, mode) in (volumes or {}).items():
        cmd += ["-v", f"{host_path}:{container_path}:{mode}"]

    cmd.append(DOCKER_IMAGE)
    cmd.extend(str(a) for a in script_args)

    return run_cmd(
        cmd,
        cwd=repo_dir,
        dry_run=dry_run,
        label=label or " ".join(str(a) for a in script_args),
    )
