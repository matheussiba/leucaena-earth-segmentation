"""Logging configuration: console (INFO) + persistent log file (DEBUG).

The file always receives DEBUG-level messages, which includes every line of
subprocess output captured by ``pipeline.runners.run_cmd``.  The console
only shows INFO+ by default (use ``--verbose`` for DEBUG on the console too).

Log files are timestamped so parallel runs and retries don't overwrite each
other:  ``pipeline_20260525_214200.log``.  A ``pipeline_latest.log``
symlink / copy always points to the most recent run.

Usage::

    from pipeline.log import setup_logging
    log = setup_logging(log_dir, verbose=args.verbose)
    log.info("Pipeline started")
"""
from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path


_FMT_FILE    = "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)s: %(message)s"
_FMT_CONSOLE = "%(asctime)s [%(levelname)-7s] %(message)s"
_DATEFMT     = "%H:%M:%S"


def setup_logging(
    log_dir: Path,
    *,
    verbose: bool = False,
) -> logging.Logger:
    """Attach a StreamHandler (stdout) and a timestamped FileHandler.

    Returns the root ``"pipeline"`` logger.  All child loggers
    (``pipeline.runners``, ``pipeline.tile_index`` …) inherit from it.

    Parameters
    ----------
    log_dir:
        Directory where log files are written.
        Created automatically if it does not exist.
    verbose:
        When *True*, set the console handler to DEBUG (same as the file).
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file    = log_dir / f"pipeline_{timestamp}.log"
    latest_file = log_dir / "pipeline_latest.log"

    # ---- Handlers ----------------------------------------

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(_FMT_CONSOLE, datefmt=_DATEFMT)
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)          # always capture everything
    file_handler.setFormatter(
        logging.Formatter(_FMT_FILE, datefmt=_DATEFMT)
    )

    # ---- Root pipeline logger ----------------------------

    root = logging.getLogger("pipeline")
    root.setLevel(logging.DEBUG)
    # Avoid duplicate handlers if setup_logging() is called more than once
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # ---- Keep pipeline_latest.log up to date ------------

    # Copy (not symlink) for Windows compatibility: symlinks need elevated
    # privileges on Windows unless Developer Mode is enabled.
    # We copy at the end of the run via update_latest(); here we just note
    # where to copy it.
    root._pipeline_log_file    = log_file     # type: ignore[attr-defined]
    root._pipeline_latest_file = latest_file  # type: ignore[attr-defined]

    root.info("Log file: %s", log_file)
    return root


def update_latest(log_dir: Path) -> None:
    """Copy the most recent ``pipeline_*.log`` to ``pipeline_latest.log``.

    Safe to call even when the file is still open for writing — we just copy
    its current state.
    """
    candidates = sorted(log_dir.glob("pipeline_2*.log"))
    if not candidates:
        return
    latest_src  = candidates[-1]
    latest_dest = log_dir / "pipeline_latest.log"
    try:
        shutil.copy2(latest_src, latest_dest)
    except Exception:
        pass  # best-effort
