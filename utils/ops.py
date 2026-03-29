"""
Geospatial and I/O helpers for the leucaena segmentation pipeline.

GDAL/OGR is used for GeoTIFF read/write and GeoJSON rasterization so labels stay
aligned with the optical reference grid. NumPy arrays are HWC for multi-band images.
"""
import json
import numpy as np
import os
import pickle
from osgeo import ogr, gdal, gdalconst, gdal_array, osr


def save_json(dict_: dict, path_to_file: str):
    with open(path_to_file, 'w') as f:
        json.dump(dict_, f, indent=4)


def save_dict(dict_: dict, path_to_file: str):
    with open(path_to_file, 'wb') as f:
        pickle.dump(dict_, f)


def load_dict(path_to_file: str):
    with open(path_to_file, 'rb') as f:
        return pickle.load(f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_opt_image(path_to_file):
    """Load multi-band GeoTIFF as HWC numpy array."""
    img = gdal_array.LoadFile(path_to_file)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)
    return np.moveaxis(img, 0, -1)


def load_label_image(path_to_file):
    """Load single-band GeoTIFF label as 2D array."""
    return gdal_array.LoadFile(path_to_file)


def get_geo_info(path_to_file):
    """Extract georeference info from a GeoTIFF."""
    ds = gdal.Open(path_to_file, gdalconst.GA_ReadOnly)
    geo_transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    x_size = ds.RasterXSize
    y_size = ds.RasterYSize
    ds = None
    return geo_transform, projection, x_size, y_size


def rasterize_geojson(geojson_path, reference_tif_path, burn_value=1, fill_value=0):
    """Rasterize GeoJSON polygons to match a reference raster's grid.

    Burns polygon features as `burn_value` onto a background of `fill_value`,
    producing a label raster aligned to the reference image.

    Args:
        geojson_path: Path to GeoJSON file with polygon geometries.
        reference_tif_path: Path to the reference GeoTIFF (defines CRS, extent, resolution).
        burn_value: Pixel value inside polygons (default 1 = leucaena).
        fill_value: Background pixel value (default 0 = no leucaena).

    Returns:
        2D numpy array (H, W) with rasterized labels.
    """
    ref_ds = gdal.Open(reference_tif_path, gdalconst.GA_ReadOnly)
    geo_transform = ref_ds.GetGeoTransform()
    x_size = ref_ds.RasterXSize
    y_size = ref_ds.RasterYSize
    ref_srs = osr.SpatialReference()
    ref_srs.ImportFromWkt(ref_ds.GetProjection())

    target_ds = gdal.GetDriverByName('MEM').Create('', x_size, y_size, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(ref_ds.GetProjection())
    band = target_ds.GetRasterBand(1)
    band.Fill(fill_value)

    vec_ds = ogr.Open(geojson_path)
    if vec_ds is None:
        raise FileNotFoundError(f'Cannot open GeoJSON: {geojson_path}')
    layer = vec_ds.GetLayer()

    src_srs = layer.GetSpatialRef()
    if src_srs is not None and not src_srs.IsSame(ref_srs):
        print(f'  Reprojecting masks from {src_srs.GetAuthorityCode(None)} to {ref_srs.GetAuthorityCode(None)}')
        coord_transform = osr.CoordinateTransformation(src_srs, ref_srs)
        mem_driver = ogr.GetDriverByName('Memory')
        mem_ds = mem_driver.CreateDataSource('')
        mem_layer = mem_ds.CreateLayer('reprojected', ref_srs, ogr.wkbPolygon)
        for feat in layer:
            geom = feat.GetGeometryRef().Clone()
            geom.Transform(coord_transform)
            out_feat = ogr.Feature(mem_layer.GetLayerDefn())
            out_feat.SetGeometry(geom)
            mem_layer.CreateFeature(out_feat)
        layer = mem_layer

    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[burn_value])

    result = band.ReadAsArray()
    target_ds = None
    vec_ds = None
    ref_ds = None
    return result


def filter_outliers(img, bins=1000000, bth=0.001, uth=0.999, mask=[0]):
    """Clip per-band outliers based on cumulative histogram."""
    img[np.isnan(img)] = 0
    if len(mask) == 1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    for band in range(img.shape[-1]):
        hist = np.histogram(
            img[:mask.shape[0], :mask.shape[1]][mask != 2, band].ravel(),
            bins=bins
        )
        cum_hist = np.cumsum(hist[0]) / hist[0].sum()
        max_value = np.ceil(100 * hist[1][len(cum_hist[cum_hist < uth])]) / 100
        min_value = np.ceil(100 * hist[1][len(cum_hist[cum_hist < bth])]) / 100
        img[:, :, band][img[:, :, band] > max_value] = max_value
        img[:, :, band][img[:, :, band] < min_value] = min_value
    return img


def save_geotiff(base_image_path, dest_path, data, dtype):
    """Save array as GeoTIFF using georef from a base image."""
    base_data = gdal.Open(base_image_path, gdalconst.GA_ReadOnly)
    geo_transform = base_data.GetGeoTransform()
    x_res = base_data.RasterXSize
    y_res = base_data.RasterYSize
    crs = base_data.GetSpatialRef()
    proj = base_data.GetProjection()

    gdal_dtype = gdal.GDT_Byte if dtype == 'byte' else gdal.GDT_Float32
    np_dtype = np.uint8 if dtype == 'byte' else np.float32

    n_bands = 1 if len(data.shape) == 2 else data.shape[-1]
    target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, n_bands, gdal_dtype)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetSpatialRef(crs)
    target_ds.SetProjection(proj)

    data = data.astype(np_dtype)
    if len(data.shape) == 2:
        target_ds.GetRasterBand(1).WriteArray(data)
    else:
        for band_i in range(1, n_bands + 1):
            target_ds.GetRasterBand(band_i).WriteArray(data[:, :, band_i - 1])
    target_ds = None
