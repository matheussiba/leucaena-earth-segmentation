"""Subprocess helpers for running pipeline steps natively or in Docker.

Two public helpers:

``run_cmd(cmd, ...)``
    Run any command and **stream every line of stdout + stderr** to the
    logger in real time.  Both the console handler and the log-file handler
    receive the output, so errors are always captured even when the process
    crashes hours into a run.

``docker_run(script_args, ...)``
    Build a ``docker run`` invocation against the pre-built image
    ``leucaena-segmentation:cuda``, mount the repo + arbitrary extra
    volumes, and delegate to ``run_cmd``.

Why threading?
    Piping both stdout and stderr through the same ``Popen`` object can
    deadlock when one pipe's OS buffer fills while we are reading the
    other.  Using two daemon threads — one per pipe — drains both buffers
    concurrently and avoids the deadlock.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
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


# ---------------------------------------------------------------------------
# Internal streaming helper
# ---------------------------------------------------------------------------

def _stream_pipe(pipe, logger: logging.Logger, level: int) -> None:
    """Read *pipe* line-by-line and emit each line to *logger*.

    Runs inside a daemon thread so it cannot block the main process.
    """
    try:
        for raw in pipe:
            line = raw.rstrip("\n").rstrip("\r")
            if line:
                logger.log(level, "    %s", line)
    except Exception as exc:  # noqa: BLE001
        logger.debug("_stream_pipe ended with: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_cmd(
    cmd: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
    label: str = "",
) -> int:
    """Run *cmd*, stream all output to the logger, and return the exit code.

    Every line written to stdout or stderr by the subprocess is forwarded to
    the ``pipeline.runners`` logger at INFO level.  Because the file handler
    in ``pipeline.log`` is set to DEBUG, **nothing is silently discarded** —
    even if the console handler is at INFO only.

    Parameters
    ----------
    cmd:
        Command + arguments (strings or Paths).
    cwd:
        Working directory for the subprocess (default: inherit).
    dry_run:
        When *True*, only log the command without executing it.
    label:
        Short human-readable description used in the opening log line.
    """
    display = label or " ".join(str(c) for c in cmd)

    if dry_run:
        log.info("[DRY-RUN] %s", display)
        return 0

    # Log the full command at DEBUG so it always appears in the log file
    # even when the console is at INFO only.
    log.info("► %s", display)
    full_cmd_str = " ".join(str(c) for c in cmd)
    if full_cmd_str != display:
        log.debug("  full cmd: %s", full_cmd_str)
    if cwd:
        log.debug("  cwd: %s", cwd)

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        log.error(
            "Command not found: %s\n"
            "  Make sure the executable is on PATH.\n"
            "  Full error: %s",
            cmd[0],
            exc,
        )
        return 127

    # Drain stdout and stderr in parallel daemon threads
    t_out = threading.Thread(
        target=_stream_pipe,
        args=(proc.stdout, log, logging.INFO),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_stream_pipe,
        args=(proc.stderr, log, logging.WARNING),
        daemon=True,
    )
    t_out.start()
    t_err.start()

    proc.wait()
    t_out.join(timeout=10)
    t_err.join(timeout=10)

    if proc.returncode != 0:
        log.error(
            "Process exited with code %d  <- %s",
            proc.returncode,
            display,
        )
    else:
        log.debug("Process exited with code 0  <- %s", display)

    return proc.returncode


def build_docker_cmd(
    script_args: Sequence[str | Path],
    *,
    repo_dir: Path,
    volumes: dict[Path, tuple[str, str]] | None = None,
    extra_env: dict[str, str] | None = None,
    gpu: bool = False,
) -> list[str]:
    """Build a ``docker run`` command list without executing it.

    Separated from ``docker_run`` so callers can log or inspect the full
    Docker invocation before it runs.
    """
    cmd: list[str] = ["docker", "run", "--rm"]

    if gpu:
        cmd += ["--gpus", "all"]

    # Repo → /workspace (container working dir)
    cmd += ["-v", f"{repo_dir}:/workspace", "-w", "/workspace"]

    # Standard container env vars
    for k, v in _CONTAINER_ENV.items():
        cmd += ["-e", f"{k}={v}"]
    for k, v in (extra_env or {}).items():
        cmd += ["-e", f"{k}={v}"]

    # Extra volume mounts
    for host_path, (container_path, mode) in (volumes or {}).items():
        cmd += ["-v", f"{host_path}:{container_path}:{mode}"]

    cmd.append(DOCKER_IMAGE)
    cmd.extend(str(a) for a in script_args)

    return cmd


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
    cmd = build_docker_cmd(
        script_args,
        repo_dir=repo_dir,
        volumes=volumes,
        extra_env=extra_env,
        gpu=gpu,
    )

    # Log volume mounts at DEBUG for easy debugging
    if volumes:
        log.debug("  Docker volume mounts:")
        for host_path, (container_path, mode) in volumes.items():
            exists = "✓" if Path(host_path).exists() else "✗ NOT FOUND"
            log.debug("    %s  →  %s  [%s]  %s", host_path, container_path, mode, exists)

    return run_cmd(
        cmd,
        cwd=repo_dir,
        dry_run=dry_run,
        label=label or " ".join(str(a) for a in script_args),
    )
