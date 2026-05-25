"""Logging configuration: console + persistent log file.

Usage::

    from pipeline.log import setup_logging
    log = setup_logging(log_dir, verbose=args.verbose)
    log.info("Pipeline started")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_FMT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(log_dir: Path, *, verbose: bool = False) -> logging.Logger:
    """Attach a StreamHandler (stdout) and a FileHandler under *log_dir*.

    Returns the root ``"pipeline"`` logger.  All child loggers
    (``pipeline.steps``, ``pipeline.runners`` …) inherit its level.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8"),
    ]
    for h in handlers:
        h.setFormatter(formatter)
        h.setLevel(level)

    root = logging.getLogger("pipeline")
    root.setLevel(level)
    for h in handlers:
        root.addHandler(h)

    return root
