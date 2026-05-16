"""
LiDAR helpers: LAZ -> aligned multi-band GeoTIFF (CHM + INTENSITY).

Pipeline summary
----------------
For each ``.laz`` we want to produce one ``.tif`` with two float32 bands
(CHM, INTENSITY) that is *exactly* aligned to the matching RGBN tile (same
extent, same pixel size, same CRS). The dataloader then crops the same
windows from RGBN and LiDAR, so they line up pixel-perfect.

We do that in three steps:

1. **Rasterise** the LAZ at a sensible native resolution (default 1 m)
   using PDAL. PDAL is the de-facto standard for point cloud processing
   and avoids hand-rolled binning. We produce three intermediate GeoTIFFs:

     - ``dsm.tif``     - max Z of all returns (top of canopy / buildings)
     - ``dtm.tif``     - min Z of ground returns after SMRF classification
     - ``intensity.tif`` - mean raw intensity of all returns

2. **Compute CHM** = DSM - DTM, clipped to [0, ``LIDAR_CHM_MAX_M``] to
   suppress spikes. Pixels where DSM or DTM are no-data become 0 m (open
   ground), which is the safest assumption for a vegetation index.

3. **Align** CHM + INTENSITY to the RGBN tile grid via ``gdal.Warp``
   (bilinear for CHM, average for INTENSITY). The resulting raster has
   the exact same width / height / geotransform as the reference RGBN.

The output is a single 2-band ``float32`` GeoTIFF in the order
``[CHM, INTENSITY]`` (matches ``conf.general.BAND_NAMES_LIDAR``).
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
from osgeo import gdal, gdalconst, osr

from conf import general


@dataclass
class TileGrid:
    """Reference grid (extent + size + CRS) read from a RGBN GeoTIFF."""

    width: int
    height: int
    geo_transform: tuple
    projection_wkt: str
    epsg_code: Optional[int]
    pixel_size_x: float
    pixel_size_y: float  # negative for north-up rasters
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @classmethod
    def from_geotiff(cls, path: str) -> "TileGrid":
        ds = gdal.Open(path, gdalconst.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(f"Cannot open reference tile: {path}")
        gt = ds.GetGeoTransform()
        w, h = ds.RasterXSize, ds.RasterYSize
        proj_wkt = ds.GetProjection()
        srs = osr.SpatialReference(wkt=proj_wkt) if proj_wkt else None
        epsg = None
        if srs is not None:
            code = srs.GetAuthorityCode(None)
            if code and str(code).isdigit():
                epsg = int(code)
        xmin = gt[0]
        ymax = gt[3]
        xmax = xmin + gt[1] * w
        ymin = ymax + gt[5] * h
        ds = None
        return cls(
            width=w,
            height=h,
            geo_transform=gt,
            projection_wkt=proj_wkt,
            epsg_code=epsg,
            pixel_size_x=gt[1],
            pixel_size_y=gt[5],
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )


def normalise_stem(filename: str) -> str:
    """Strip ``.copc`` and the file extension. ``A-B.copc.laz`` -> ``A-B``."""
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]  # drop .laz
    if base.endswith(".copc"):
        base = base[: -len(".copc")]
    return base


def find_reference_tile(stem: str, tiles_dir: str, exts=(".tif", ".tiff")) -> Optional[str]:
    """Look up ``<tiles_dir>/<stem>.<ext>`` for any of the given extensions."""
    if not tiles_dir or not os.path.isdir(tiles_dir):
        return None
    for ext in exts:
        candidate = os.path.join(tiles_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def build_pdal_pipeline(
    laz_path: str,
    dsm_path: str,
    dtm_path: str,
    intensity_path: str,
    resolution_m: float,
    target_epsg: Optional[int],
    bounds: Optional[tuple[float, float, float, float]],
) -> dict:
    """Build a PDAL pipeline (Python dict) that writes DSM, DTM and Intensity.

    The pipeline reads the LAZ once and fans out into three writers via
    PDAL tags. ``bounds`` is ``(xmin, xmax, ymin, ymax)`` in ``target_epsg``;
    when provided, the output rasters cover *exactly* that extent, which
    makes the later GDAL Warp step trivial.
    """
    stages: list = [
        {
            "type": "readers.las",
            "filename": laz_path,
            "tag": "src_raw",
        },
    ]
    # Reproject if the caller wants a specific CRS for the output rasters.
    last_tag = "src_raw"
    if target_epsg is not None:
        stages.append({
            "type": "filters.reprojection",
            "inputs": last_tag,
            "out_srs": f"EPSG:{target_epsg}",
            "tag": "src_proj",
        })
        last_tag = "src_proj"

    # Drop noise (LAS class 7) and high-noise (class 18) early; they wreck DSM.
    stages.append({
        "type": "filters.range",
        "inputs": last_tag,
        "limits": "Classification![7:7],Classification![18:18]",
        "tag": "src_clean",
    })

    # SMRF gives us a ground/non-ground classification suitable for CHM.
    stages.append({
        "type": "filters.smrf",
        "inputs": "src_clean",
        "tag": "src_smrf",
    })

    # Branch 1: ground-only for the DTM (min Z over class 2).
    stages.append({
        "type": "filters.range",
        "inputs": "src_smrf",
        "limits": "Classification[2:2]",
        "tag": "ground",
    })

    gdal_common = {
        "gdaldriver": "GTiff",
        "gdalopts": "COMPRESS=DEFLATE,TILED=YES,BIGTIFF=IF_SAFER",
        "data_type": "float32",
        "nodata": -9999.0,
        "resolution": float(resolution_m),
        # Bigger radius helps fill no-data pixels in sparse areas; tweakable.
        "radius": float(resolution_m) * 1.5,
    }
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        # PDAL bounds syntax: ([xmin,xmax],[ymin,ymax])
        gdal_common["bounds"] = f"([{xmin},{xmax}],[{ymin},{ymax}])"

    stages.append({
        **gdal_common,
        "type": "writers.gdal",
        "inputs": "ground",
        "filename": dtm_path,
        "output_type": "min",
    })

    stages.append({
        **gdal_common,
        "type": "writers.gdal",
        "inputs": "src_clean",
        "filename": dsm_path,
        "output_type": "max",
    })

    stages.append({
        **gdal_common,
        "type": "writers.gdal",
        "inputs": "src_clean",
        "filename": intensity_path,
        "output_type": "mean",
        "dimension": "Intensity",
    })

    return {"pipeline": stages}


def run_pdal_pipeline(pipeline: dict) -> int:
    """Execute a PDAL pipeline dict; returns the number of points processed."""
    try:
        import pdal  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise RuntimeError(
            "PDAL Python bindings not installed. In the Docker image they are "
            "pre-installed; for local conda: "
            "`conda install -y -c conda-forge pdal python-pdal`."
        ) from exc

    p = pdal.Pipeline(json.dumps(pipeline))
    p.validate()
    n_points = p.execute()
    return int(n_points)


def _read_band(path: str) -> tuple[np.ndarray, float]:
    """Return ``(array, nodata_value)`` for a single-band raster."""
    ds = gdal.Open(path, gdalconst.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    if nodata is None:
        nodata = float("nan")
    arr = band.ReadAsArray()
    ds = None
    return arr, float(nodata)


def compute_chm(
    dsm_path: str,
    dtm_path: str,
    out_path: str,
    chm_max_m: float = general.LIDAR_CHM_MAX_M,
) -> None:
    """Write ``CHM = clip(DSM - DTM, 0, chm_max_m)`` to a 1-band float32 TIF.

    DSM and DTM must already be on the same grid (same PDAL pipeline).
    No-data pixels in either input become 0 m in the CHM (we treat
    "no measurement" as "no vegetation" rather than NaN, which makes the
    subsequent network input well defined).
    """
    dsm, dsm_nodata = _read_band(dsm_path)
    dtm, dtm_nodata = _read_band(dtm_path)
    if dsm.shape != dtm.shape:
        raise ValueError(
            f"DSM and DTM shape mismatch: {dsm.shape} vs {dtm.shape}"
        )

    dsm_f = dsm.astype(np.float32, copy=False)
    dtm_f = dtm.astype(np.float32, copy=False)

    dsm_valid = ~np.isclose(dsm_f, dsm_nodata) & np.isfinite(dsm_f)
    dtm_valid = ~np.isclose(dtm_f, dtm_nodata) & np.isfinite(dtm_f)
    valid = dsm_valid & dtm_valid

    chm = np.zeros_like(dsm_f, dtype=np.float32)
    diff = dsm_f - dtm_f
    chm[valid] = np.clip(diff[valid], 0.0, float(chm_max_m))

    # Write using DSM's georef.
    src_ds = gdal.Open(dsm_path, gdalconst.GA_ReadOnly)
    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(
        out_path,
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    out_ds.SetGeoTransform(src_ds.GetGeoTransform())
    out_ds.SetProjection(src_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(-9999.0)
    out_band.WriteArray(chm)
    out_band.FlushCache()
    out_ds = None
    src_ds = None


def warp_to_grid(src_path: str, grid: TileGrid, out_path: str, resample_alg: str) -> None:
    """Resample ``src_path`` onto ``grid`` (exact width/height/extent/CRS)."""
    warp_opts = gdal.WarpOptions(
        format="GTiff",
        width=grid.width,
        height=grid.height,
        outputBounds=(grid.xmin, grid.ymin, grid.xmax, grid.ymax),
        dstSRS=grid.projection_wkt,
        resampleAlg=resample_alg,
        srcNodata=-9999.0,
        dstNodata=-9999.0,
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    out = gdal.Warp(out_path, src_path, options=warp_opts)
    if out is None:
        raise RuntimeError(f"gdal.Warp failed for {src_path} -> {out_path}")
    out = None


def stack_chm_intensity(
    chm_path: str, intensity_path: str, out_path: str, reference_grid: TileGrid
) -> None:
    """Combine CHM and INTENSITY into a single 2-band GeoTIFF.

    Output band order matches ``general.BAND_NAMES_LIDAR`` -> [CHM, INTENSITY].
    Bands are stored as float32; no-data is preserved as -9999.
    """
    chm, _ = _read_band(chm_path)
    inten, inten_nodata = _read_band(intensity_path)
    if chm.shape != inten.shape or chm.shape != (reference_grid.height, reference_grid.width):
        raise ValueError(
            f"Shape mismatch chm={chm.shape} inten={inten.shape} "
            f"grid=({reference_grid.height}, {reference_grid.width})"
        )

    # Replace NaNs/no-data in intensity with 0 (matches CHM convention).
    inten_f = inten.astype(np.float32, copy=False)
    inten_invalid = ~np.isfinite(inten_f) | np.isclose(inten_f, inten_nodata)
    inten_f[inten_invalid] = 0.0

    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(
        out_path,
        reference_grid.width,
        reference_grid.height,
        2,
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER", "PREDICTOR=3"],
    )
    out_ds.SetGeoTransform(reference_grid.geo_transform)
    out_ds.SetProjection(reference_grid.projection_wkt)
    band_chm = out_ds.GetRasterBand(1)
    band_chm.SetDescription("CHM")
    band_chm.SetNoDataValue(-9999.0)
    band_chm.WriteArray(chm.astype(np.float32, copy=False))
    band_inten = out_ds.GetRasterBand(2)
    band_inten.SetDescription("INTENSITY")
    band_inten.SetNoDataValue(-9999.0)
    band_inten.WriteArray(inten_f)
    out_ds.FlushCache()
    out_ds = None


def process_laz_to_lidar_tif(
    laz_path: str,
    out_path: str,
    reference_grid: Optional[TileGrid],
    resolution_m: float = general.LIDAR_RASTER_RESOLUTION_M,
    chm_max_m: float = general.LIDAR_CHM_MAX_M,
    tmpdir: Optional[str] = None,
) -> dict:
    """End-to-end LAZ -> 2-band LiDAR GeoTIFF.

    If ``reference_grid`` is provided, the output is aligned to it pixel
    for pixel. Otherwise the output uses the LAZ's native bounds at
    ``resolution_m`` (useful for inspection, not for training).

    Returns a small dict of stats (``n_points``, output shape, etc.) that
    the caller can persist into a manifest CSV.
    """
    stem = normalise_stem(laz_path)
    # tmp workspace per LAZ; auto-cleaned on success.
    workdir = tempfile.mkdtemp(prefix=f"lidar_{stem}_", dir=tmpdir)
    try:
        dsm_tmp = os.path.join(workdir, f"{stem}_dsm.tif")
        dtm_tmp = os.path.join(workdir, f"{stem}_dtm.tif")
        inten_tmp = os.path.join(workdir, f"{stem}_intensity.tif")
        chm_tmp = os.path.join(workdir, f"{stem}_chm.tif")

        # If we have a reference grid, expand the bounds by a small buffer so
        # PDAL doesn't drop edge pixels (interp radius rounding).
        bounds = None
        target_epsg = None
        if reference_grid is not None:
            buf = float(resolution_m) * 2.0
            bounds = (
                reference_grid.xmin - buf,
                reference_grid.xmax + buf,
                reference_grid.ymin - buf,
                reference_grid.ymax + buf,
            )
            target_epsg = reference_grid.epsg_code

        pipeline = build_pdal_pipeline(
            laz_path=laz_path,
            dsm_path=dsm_tmp,
            dtm_path=dtm_tmp,
            intensity_path=inten_tmp,
            resolution_m=resolution_m,
            target_epsg=target_epsg,
            bounds=bounds,
        )
        n_points = run_pdal_pipeline(pipeline)

        compute_chm(dsm_tmp, dtm_tmp, chm_tmp, chm_max_m=chm_max_m)

        if reference_grid is None:
            # Standalone mode: just stack at native resolution. Read both
            # rasters and write a 2-band TIF using the DSM as reference.
            ds = gdal.Open(dsm_tmp, gdalconst.GA_ReadOnly)
            native_grid = TileGrid(
                width=ds.RasterXSize,
                height=ds.RasterYSize,
                geo_transform=ds.GetGeoTransform(),
                projection_wkt=ds.GetProjection(),
                epsg_code=None,
                pixel_size_x=ds.GetGeoTransform()[1],
                pixel_size_y=ds.GetGeoTransform()[5],
                xmin=ds.GetGeoTransform()[0],
                ymin=ds.GetGeoTransform()[3] + ds.GetGeoTransform()[5] * ds.RasterYSize,
                xmax=ds.GetGeoTransform()[0] + ds.GetGeoTransform()[1] * ds.RasterXSize,
                ymax=ds.GetGeoTransform()[3],
            )
            ds = None
            stack_chm_intensity(chm_tmp, inten_tmp, out_path, native_grid)
            return {
                "n_points": n_points,
                "out_width": native_grid.width,
                "out_height": native_grid.height,
                "aligned_to_rgbn": False,
            }

        # Aligned mode: warp CHM and INTENSITY onto the RGBN grid.
        chm_warp = os.path.join(workdir, f"{stem}_chm_warp.tif")
        inten_warp = os.path.join(workdir, f"{stem}_intensity_warp.tif")
        warp_to_grid(chm_tmp, reference_grid, chm_warp, resample_alg="bilinear")
        warp_to_grid(inten_tmp, reference_grid, inten_warp, resample_alg="average")

        stack_chm_intensity(chm_warp, inten_warp, out_path, reference_grid)
        return {
            "n_points": n_points,
            "out_width": reference_grid.width,
            "out_height": reference_grid.height,
            "aligned_to_rgbn": True,
        }
    finally:
        # Best-effort cleanup; we don't fail the run on tmpdir leftovers.
        try:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


def inspect_laz(laz_path: str) -> dict:
    """Lightweight metadata read: bounds, point count, CRS, class histogram.

    Uses PDAL ``readers.las`` + ``filters.stats``; cheap because no rasterisation.
    Useful as a smoke test before launching a long batch.
    """
    try:
        import pdal  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PDAL not installed; see process_laz_to_lidar_tif().") from exc

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": laz_path},
            {"type": "filters.stats", "dimensions": "X,Y,Z,Intensity,Classification"},
        ]
    }
    p = pdal.Pipeline(json.dumps(pipeline))
    p.validate()
    n_points = p.execute()
    meta = json.loads(p.metadata) if isinstance(p.metadata, str) else p.metadata
    return {
        "n_points": int(n_points),
        "metadata": meta,
    }
