"""Canonical destination folder layout for the data pipeline.

All pipeline steps derive their output/input directories from a single
``DestLayout`` instance, so moving the dataset root only requires changing
``--dest``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DestLayout:
    """All paths under ``--dest`` expressed as ``pathlib.Path`` objects.

    Parameters
    ----------
    root:
        The top-level destination directory, e.g.
        ``C:\\00_DATASETS_AI\\260515-piracicaba-aoi``.
    """

    root: Path

    # ------------------------------------------------------------------
    # Optical imagery
    # ------------------------------------------------------------------

    @property
    def opt_raw_rgb(self) -> Path:
        """RGB tiles copied from D:\\rgb  (Step 1 output, Step 2 input)."""
        return self.root / "opt" / "raw" / "rgb"

    @property
    def opt_raw_ir(self) -> Path:
        """IR tiles copied from D:\\ir  (Step 1 output, Step 2 input)."""
        return self.root / "opt" / "raw" / "ir"

    @property
    def opt_rgbnir(self) -> Path:
        """4-band RGBNIR tiles  (Step 2 output, Steps 3 + 4 input)."""
        return self.root / "opt" / "rgbnir"

    # ------------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------------

    @property
    def lidar_raw(self) -> Path:
        """LAZ point clouds copied from D:\\laz  (Step 1 output, Step 3 input)."""
        return self.root / "lidar" / "raw"

    @property
    def lidar_chm(self) -> Path:
        """CHM + Intensity GeoTIFFs  (Step 3 output, Step 4 input)."""
        return self.root / "lidar" / "chm"

    # ------------------------------------------------------------------
    # ML data
    # ------------------------------------------------------------------

    @property
    def annotations(self) -> Path:
        """Leucaena polygon GeoJSON/GeoPackage (provided externally)."""
        return self.root / "annotations"

    @property
    def patches(self) -> Path:
        """Generated training patches  (Step 4 output, Step 5 input)."""
        return self.root / "patches"

    # ------------------------------------------------------------------
    # Models + experiment outputs
    # ------------------------------------------------------------------

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def models_logs(self) -> Path:
        return self.root / "models" / "logs"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def pipeline_cache(self) -> Path:
        """Hidden folder for tile-index JSON and other caches."""
        return self.root / ".pipeline_cache"

    def ensure_all(self) -> None:
        """Create all standard sub-directories (no-op if they already exist)."""
        for p in (
            self.opt_raw_rgb,
            self.opt_raw_ir,
            self.opt_rgbnir,
            self.lidar_raw,
            self.lidar_chm,
            self.annotations,
            self.patches,
            self.models_logs,
            self.pipeline_cache,
        ):
            p.mkdir(parents=True, exist_ok=True)
