"""
Sliding-window inference primitives used by ``predict-tiles.py``.

This module is the per-tile equivalent of the scene-level logic in
``prediction.py``: read one GeoTIFF tile, run the ResUNet over overlapping
``PATCH_SIZE`` windows, average the softmax probabilities, and write back
two georeferenced rasters (class map + optional probability).

Design notes
------------
- The forward contract of every model in this repo is ``model((opt, lidar))``;
  for optical-only experiments ``lidar`` is a zero tensor of shape
  ``(B, 1, PATCH_SIZE, PATCH_SIZE)``.
- LiDAR scaling MUST mirror ``utils.dataloader.PatchFileDataset._scale_lidar``
  exactly so the network sees the same input distribution it saw at training.
- Reflect-pad by ``PATCH_SIZE`` on each side so that every pixel of the real
  image is covered by the same number of overlapping windows regardless of
  position (no border under-sampling).
- The probability raster is written via ``write_prob_geotiff`` with a packed
  dtype option (uint16 with ``scale_factor`` = 1/65535) so the file is 2 bytes
  per pixel — half the size of float32 and the closest portable equivalent of
  Float16, which GDAL/GeoTIFF do not support natively.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from osgeo import gdal, gdalconst

from conf import general


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _band_reorder_indices(band_order: str) -> List[int]:
    """Map a source band order string to BGRN indices."""
    if band_order == "BGRN":
        return [0, 1, 2, 3]
    if band_order == "RGBN":
        return [2, 1, 0, 3]
    raise ValueError(f"Unsupported band order: {band_order}")


def read_tile_as_bgrn(tile_path: str, band_order: str) -> Tuple[np.ndarray, dict]:
    """
    Read a 4-band optical GeoTIFF.

    Returns a tuple ``(bgrn_hwc_uint8, geo)`` where ``bgrn_hwc_uint8`` is an
    ``(H, W, 4)`` ``uint8`` array reordered to ``BGRN`` (matches training
    convention) and ``geo`` carries the georeference info needed to write
    aligned outputs.
    """
    ds = gdal.Open(tile_path, gdalconst.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(tile_path)
    if ds.RasterCount != 4:
        raise ValueError(f"{tile_path}: expected 4 bands, got {ds.RasterCount}")
    bands_chw = ds.ReadAsArray().astype(np.uint8)
    geo = {
        "geotransform": ds.GetGeoTransform(),
        "projection": ds.GetProjection(),
        "x_size": ds.RasterXSize,
        "y_size": ds.RasterYSize,
        "spatial_ref": ds.GetSpatialRef(),
    }
    ds = None
    idx = _band_reorder_indices(band_order)
    bgrn = np.stack([bands_chw[i] for i in idx], axis=0)
    bgrn_hwc = np.transpose(bgrn, (1, 2, 0))
    return bgrn_hwc, geo


def read_lidar_as_array(lidar_path: str) -> np.ndarray:
    """
    Read every band of a LiDAR GeoTIFF and return ``(H, W, C)`` float32 in raw
    units (CHM in metres, INTENSITY in raw counts). Scaling to ``[0, 1]`` is
    done by :func:`scale_lidar`, not here, because the dtype consumed by the
    network is decided at training time and must match.
    """
    ds = gdal.Open(lidar_path, gdalconst.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(lidar_path)
    bands_chw = ds.ReadAsArray().astype(np.float32)
    ds = None
    if bands_chw.ndim == 2:
        bands_chw = bands_chw[np.newaxis, :, :]
    return np.transpose(bands_chw, (1, 2, 0))


def scale_lidar(lidar_arr: np.ndarray) -> np.ndarray:
    """
    Apply the same per-band normalisation used by ``PatchFileDataset``.

    Band order on disk is fixed to ``general.BAND_NAMES_LIDAR``
    (``[CHM, INTENSITY, ...]``). CHM is divided by ``LIDAR_CHM_MAX_M``; every
    other band by ``LIDAR_INTENSITY_MAX``. Values are clipped to ``[0, 1]``.
    """
    scaled = np.empty_like(lidar_arr, dtype=np.float32)
    for b in range(lidar_arr.shape[-1]):
        if b < len(general.BAND_NAMES_LIDAR) and general.BAND_NAMES_LIDAR[b].upper() == "CHM":
            denom = float(general.LIDAR_CHM_MAX_M)
        else:
            denom = float(general.LIDAR_INTENSITY_MAX)
        scaled[:, :, b] = np.clip(lidar_arr[:, :, b].astype(np.float32) / denom, 0.0, 1.0)
    return scaled


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


def _generate_window_coords(h: int, w: int, patch_size: int, overlap: float) -> List[Tuple[int, int]]:
    """
    Build a deterministic grid of (y, x) top-left coordinates for sliding
    windows of ``patch_size``. ``overlap`` in ``[0, 1)`` controls the step.
    The grid is "end-snapped": if the last regular step does not land flush
    with the right/bottom border, an extra coordinate is appended so the
    border pixels are always covered.
    """
    step = max(1, int(round(patch_size * (1.0 - overlap))))
    ys_list = list(range(0, h - patch_size + 1, step))
    if (h - patch_size) % step != 0:
        ys_list.append(h - patch_size)
    xs_list = list(range(0, w - patch_size + 1, step))
    if (w - patch_size) % step != 0:
        xs_list.append(w - patch_size)
    ys = sorted(set(ys_list))
    xs = sorted(set(xs_list))
    return [(y, x) for y in ys for x in xs]


def predict_tile_probability(
    *,
    model: torch.nn.Module,
    opt_bgrn_uint8: np.ndarray,
    lidar_arr_raw: Optional[np.ndarray],
    n_classes: int,
    patch_size: int,
    overlaps: Iterable[float],
    batch_size: int,
    device: str,
    lidar_bands: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Run multi-overlap sliding-window inference on a single tile.

    Parameters
    ----------
    model:
        A trained model whose forward signature is ``model((opt, lidar))``.
        Must be already moved to ``device`` and set to ``.eval()``.
    opt_bgrn_uint8:
        ``(H, W, 4)`` ``uint8`` optical raster in BGRN order.
    lidar_arr_raw:
        ``(H, W, C)`` ``float32`` LiDAR raster in raw units (CHM in metres,
        INTENSITY in raw counts), or ``None`` when the model does not need
        LiDAR (zero tensor will be passed). Must already match the optical
        grid pixel-for-pixel (``prep-lidar-rasters.py`` enforces that).
    n_classes:
        Number of output channels (2 in this project).
    patch_size, overlaps, batch_size, device:
        Self-explanatory. ``overlaps`` is iterable so the caller can pass a
        single overlap (preview mode) or the project default ``[0, 0.25, 0.5]``.
    lidar_bands:
        Same convention as in training: ``None`` (no LiDAR) or a list of
        integer indices into the disk band order ``BAND_NAMES_LIDAR``.

    Returns
    -------
    np.ndarray
        ``(H, W, n_classes)`` ``float32`` averaged probability map. ``argmax``
        on the last axis gives the class raster; ``[..., 1]`` gives the
        per-pixel probability of leucaena.
    """
    H, W, _ = opt_bgrn_uint8.shape
    pad = patch_size
    opt_float = opt_bgrn_uint8.astype(np.float32) / 255.0
    opt_pad = np.pad(opt_float, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    if lidar_arr_raw is not None:
        lidar_scaled = scale_lidar(lidar_arr_raw)
        if lidar_bands is not None:
            lidar_scaled = lidar_scaled[:, :, lidar_bands]
        lidar_pad = np.pad(lidar_scaled, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        lidar_c = lidar_pad.shape[-1]
    else:
        lidar_pad = None
        lidar_c = len(lidar_bands) if lidar_bands else 1

    hp, wp = opt_pad.shape[:2]
    overlaps_list = list(overlaps)
    if not overlaps_list:
        raise ValueError("overlaps must contain at least one value")

    overlap_avg = np.zeros((n_classes, hp, wp), dtype=np.float32)
    model.eval()

    for overlap in overlaps_list:
        coords = _generate_window_coords(hp, wp, patch_size, overlap)
        prob_sum = np.zeros((n_classes, hp, wp), dtype=np.float32)
        prob_count = np.zeros((hp, wp), dtype=np.float32)

        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i : i + batch_size]
            opt_batch = []
            lidar_batch = []
            for y, x in batch_coords:
                opt_chw = np.transpose(
                    opt_pad[y : y + patch_size, x : x + patch_size, :],
                    (2, 0, 1),
                )
                opt_batch.append(torch.from_numpy(opt_chw))
                if lidar_pad is not None:
                    lidar_chw = np.transpose(
                        lidar_pad[y : y + patch_size, x : x + patch_size, :],
                        (2, 0, 1),
                    )
                    lidar_batch.append(torch.from_numpy(lidar_chw))
            opt_t = torch.stack(opt_batch, dim=0).to(device)
            if lidar_pad is not None:
                lidar_t = torch.stack(lidar_batch, dim=0).to(device)
            else:
                lidar_t = torch.zeros(
                    (len(batch_coords), lidar_c, patch_size, patch_size),
                    dtype=torch.float32,
                    device=device,
                )

            with torch.no_grad():
                prob = model((opt_t, lidar_t))
            prob_np = prob.cpu().numpy()
            for k, (y, x) in enumerate(batch_coords):
                prob_sum[:, y : y + patch_size, x : x + patch_size] += prob_np[k]
                prob_count[y : y + patch_size, x : x + patch_size] += 1.0

        prob_count_safe = np.maximum(prob_count, 1e-6)
        overlap_avg += prob_sum / prob_count_safe[np.newaxis, ...]

    overlap_avg /= float(len(overlaps_list))
    final = overlap_avg[:, pad : pad + H, pad : pad + W]
    return np.transpose(final, (1, 2, 0))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _creation_options(gdt: int) -> List[str]:
    """COG-friendly creation options; predictor 3 only helps floating types."""
    if gdt == gdal.GDT_Float32:
        predictor = "3"
    else:
        predictor = "2"
    return [
        "COMPRESS=DEFLATE",
        f"PREDICTOR={predictor}",
        "TILED=YES",
        "BIGTIFF=IF_SAFER",
    ]


def write_class_geotiff(out_path: str, geo: dict, pred_hw: np.ndarray) -> None:
    """Write the argmax class raster as ``uint8`` (0/1, with ``255`` as nodata)."""
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        out_path,
        geo["x_size"],
        geo["y_size"],
        1,
        gdal.GDT_Byte,
        options=_creation_options(gdal.GDT_Byte),
    )
    ds.SetGeoTransform(geo["geotransform"])
    ds.SetProjection(geo["projection"])
    if geo.get("spatial_ref") is not None:
        ds.SetSpatialRef(geo["spatial_ref"])
    band = ds.GetRasterBand(1)
    band.WriteArray(pred_hw.astype(np.uint8))
    band.SetNoDataValue(255)
    band.SetDescription("class")
    ds.FlushCache()
    ds = None


def write_prob_geotiff(
    out_path: str,
    geo: dict,
    prob_class1_hw: np.ndarray,
    dtype: str,
) -> None:
    """
    Write the leucaena-class probability raster.

    ``dtype`` controls the storage and on-disk size:

    - ``float32``: 4 bytes per pixel, untouched probability.
    - ``uint16``  (default): 2 bytes per pixel; values stored as
      ``round(prob * 65535)`` with ``SetScale(1/65535)``. GeoTIFF has no
      native Float16, but ``uint16 + scale_factor`` is the standard
      packed-float trick used by ESA / CMIP datasets, halves the file size,
      and is read transparently by GDAL, rasterio, and QGIS — the user still
      sees a float in ``[0, 1]``.
    - ``uint8``: 1 byte per pixel; 256 levels — fine for visualisation but
      coarse for downstream statistics.
    """
    if dtype == "float32":
        arr = prob_class1_hw.astype(np.float32)
        gdt = gdal.GDT_Float32
        scale: Optional[float] = None
    elif dtype == "uint16":
        arr = np.clip(np.round(prob_class1_hw * 65535.0), 0, 65535).astype(np.uint16)
        gdt = gdal.GDT_UInt16
        scale = 1.0 / 65535.0
    elif dtype == "uint8":
        arr = np.clip(np.round(prob_class1_hw * 255.0), 0, 255).astype(np.uint8)
        gdt = gdal.GDT_Byte
        scale = 1.0 / 255.0
    else:
        raise ValueError(f"Unsupported prob dtype: {dtype}")

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        out_path,
        geo["x_size"],
        geo["y_size"],
        1,
        gdt,
        options=_creation_options(gdt),
    )
    ds.SetGeoTransform(geo["geotransform"])
    ds.SetProjection(geo["projection"])
    if geo.get("spatial_ref") is not None:
        ds.SetSpatialRef(geo["spatial_ref"])
    band = ds.GetRasterBand(1)
    band.WriteArray(arr)
    if scale is not None:
        band.SetScale(scale)
        band.SetOffset(0.0)
        # Plain metadata so downstream code can detect the packing without
        # relying solely on GDAL's scale/offset roundtrip.
        band.SetMetadataItem("LEUCAENA_PROB_PACKED", "true")
        band.SetMetadataItem("LEUCAENA_PROB_SCALE", f"{scale:.10g}")
    band.SetDescription("prob_class_leucaena")
    ds.FlushCache()
    ds = None
