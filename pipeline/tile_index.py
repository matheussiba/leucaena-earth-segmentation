"""Cached directory listing of source files under the ``--source`` tree.

The source tree (typically ``D:\\``) can contain thousands of files.
Listing it once and caching to JSON means subsequent ``--dry-run`` or
resume invocations are fast without hitting the disk every time.

Usage::

    from pipeline.tile_index import TileIndex
    idx = TileIndex(source_root=Path("D:/"), cache_dir=layout.pipeline_cache)
    files = idx.load()          # {subdir: [filename, …]}
    idx.load(rebuild=True)      # force re-scan
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# Source sub-directories that the pipeline cares about.
_SUBDIRS = ("rgb", "ir", "laz")


class TileIndex:
    """Lazily scans and caches ``{subdir: [filename, …]}`` for ``source_root``.

    Parameters
    ----------
    source_root:
        Top of the raw source tree, e.g. ``Path("D:/")``.
    cache_dir:
        Directory where ``tile_index.json`` is persisted between runs
        (typically ``dest/.pipeline_cache``).
    """

    def __init__(self, source_root: Path, cache_dir: Path) -> None:
        self._root = source_root
        self._cache_file = cache_dir / "tile_index.json"
        self._data: dict[str, list[str]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, *, rebuild: bool = False) -> dict[str, list[str]]:
        """Return the file index, loading from cache when available.

        Parameters
        ----------
        rebuild:
            When *True* the on-disk cache is ignored and a fresh scan is run.
        """
        if self._data is not None and not rebuild:
            return self._data

        if not rebuild and self._cache_file.exists():
            log.info("Loading tile index from cache: %s", self._cache_file)
            try:
                with self._cache_file.open(encoding="utf-8") as fh:
                    self._data = json.load(fh)
                return self._data
            except Exception as exc:
                log.warning("Cache read failed (%s); rebuilding.", exc)

        self._data = self._scan()
        self._save()
        return self._data

    def invalidate(self) -> None:
        """Delete the on-disk cache (triggers a fresh scan on next ``load()``)."""
        if self._cache_file.exists():
            self._cache_file.unlink()
            log.info("Tile index cache removed: %s", self._cache_file)
        self._data = None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _scan(self) -> dict[str, list[str]]:
        log.info("Scanning source tree: %s …", self._root)
        data: dict[str, list[str]] = {}
        for sub in _SUBDIRS:
            d = self._root / sub
            if d.is_dir():
                names = sorted(os.listdir(d))
                data[sub] = names
                log.info("  %-4s  %d files", sub, len(names))
            else:
                data[sub] = []
                log.warning("  %-4s  directory not found: %s", sub, d)
        return data

    def _save(self) -> None:
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_file.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)
        log.info("Tile index cached → %s", self._cache_file)
