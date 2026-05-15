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

# IGC tiles often ship GeoKeys that GDAL reads as a broken "Engineering" CRS; prefer EPSG registry.
gdal.SetConfigOption('GTIFF_SRS_SOURCE', 'EPSG')

_POLYGON_TYPES = frozenset({
    ogr.wkbPolygon,
    ogr.wkbPolygon25D,
    ogr.wkbMultiPolygon,
    ogr.wkbMultiPolygon25D,
})


def _srs_from_epsg(epsg: int) -> osr.SpatialReference:
    srs = osr.SpatialReference()
    if srs.ImportFromEPSG(epsg) != 0:
        raise ValueError(f'Cannot import EPSG:{epsg}')
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _srs_from_dataset(ds: gdal.Dataset) -> osr.SpatialReference:
    """Resolve a raster CRS to a proper projected SRS (EPSG), not EngineeringCRS."""
    srs = ds.GetSpatialRef()
    if srs is None or not (srs.ExportToWkt() or "").strip():
        srs = osr.SpatialReference()
        if srs.ImportFromWkt(ds.GetProjection()) != 0:
            srs = None
    if srs is None or not (srs.ExportToWkt() or "").strip():
        raise ValueError('Tile has no CRS (missing projection / SRS).')

    code = srs.GetAuthorityCode(None)
    if code and code.isdigit():
        return _srs_from_epsg(int(code))

    name = (srs.GetName() or '').lower()
    if 'utm zone 23s' in name or '31983' in name:
        return _srs_from_epsg(31983)

    if srs.IsGeographic():
        return srs

    raise ValueError(
        f'Cannot resolve tile CRS (name={srs.GetName()!r}). '
        'Expected EPSG:31983 (SIRGAS 2000 / UTM 23S) or an authority code on the GeoTIFF.'
    )


def _srs_from_layer(layer: ogr.Layer) -> osr.SpatialReference:
    srs = layer.GetSpatialRef()
    wkt = (srs.ExportToWkt() if srs else "") or ""
    if not wkt.strip():
        return _srs_from_epsg(4326)
    code = srs.GetAuthorityCode(None)
    if code and code.isdigit():
        return _srs_from_epsg(int(code))
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _coord_transform(src: osr.SpatialReference, dst: osr.SpatialReference) -> osr.CoordinateTransformation:
    ct = osr.CoordinateTransformation(src, dst)
    if ct is None:
        raise RuntimeError(
            f'Cannot build CRS transform {src.GetAuthorityCode(None)} -> {dst.GetAuthorityCode(None)}'
        )
    return ct


def _tile_bbox_geometry(geo_transform, x_size: int, y_size: int) -> ogr.Geometry:
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * x_size
    y_min = y_max + geo_transform[5] * y_size
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(x_min, y_min)
    ring.AddPoint(x_max, y_min)
    ring.AddPoint(x_max, y_max)
    ring.AddPoint(x_min, y_max)
    ring.AddPoint(x_min, y_min)
    bbox_poly = ogr.Geometry(ogr.wkbPolygon)
    bbox_poly.AddGeometry(ring)
    return bbox_poly


def _reproject_layer_to_srs(
    layer: ogr.Layer, dst_srs: osr.SpatialReference
) -> tuple:
    """Return ``(vector DataSource/DataSet that must stay alive, burn_layer)``."""
    if hasattr(layer, 'GetDataSource'):
        owning_ds = layer.GetDataSource()
    elif hasattr(layer, 'GetDataset'):
        owning_ds = layer.GetDataset()
    else:
        owning_ds = None
    if owning_ds is None:
        raise RuntimeError('OGR Layer has no parent dataset (unexpected GDAL build).')
    src_srs = _srs_from_layer(layer)
    if src_srs.IsSame(dst_srs):
        return owning_ds, layer

    print(
        f'  Reprojecting masks {src_srs.GetAuthorityCode(None)} '
        f'-> {dst_srs.GetAuthorityCode(None)}'
    )
    ct = _coord_transform(src_srs, dst_srs)
    mem_driver = ogr.GetDriverByName('Memory')
    mem_ds = mem_driver.CreateDataSource('')
    mem_layer = mem_ds.CreateLayer('reprojected', dst_srs, ogr.wkbPolygon)
    layer.ResetReading()
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None or geom.GetGeometryType() not in _POLYGON_TYPES:
            continue
        geom = geom.Clone()
        if geom.Transform(ct) != 0:
            raise RuntimeError('Geometry reprojection failed (check PROJ/GDAL CRS setup).')
        out_feat = ogr.Feature(mem_layer.GetLayerDefn())
        out_feat.SetGeometry(geom)
        mem_layer.CreateFeature(out_feat)
    return mem_ds, mem_layer


def _rasterize_burn(target_ds: gdal.Dataset, burn_layer: ogr.Layer, burn_value: int) -> None:
    err = gdal.RasterizeLayer(
        target_ds, [1], burn_layer,
        burn_values=[int(burn_value)],
        options=['MERGE_ALG=REPLACE'],
    )
    if err not in (None, 0):
        raise RuntimeError(f'RasterizeLayer failed (code {err})')


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
    ref_srs = _srs_from_dataset(ref_ds)

    target_ds = gdal.GetDriverByName('MEM').Create('', x_size, y_size, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(ref_srs.ExportToWkt())
    band = target_ds.GetRasterBand(1)
    band.Fill(fill_value)

    vec_ds = ogr.Open(geojson_path)
    if vec_ds is None:
        raise FileNotFoundError(f'Cannot open GeoJSON: {geojson_path}')
    _mask_ds_alive, layer = _reproject_layer_to_srs(vec_ds.GetLayer(), ref_srs)

    _rasterize_burn(target_ds, layer, burn_value)

    result = band.ReadAsArray()
    target_ds = None
    vec_ds = None
    ref_ds = None
    return result


def rasterize_geojson_for_tile(geojson_path, tile_path, burn_value=1, fill_value=0):
    """Rasterize polygons whose bounding box intersects ``tile_path``'s extent.

    Same end-result as :func:`rasterize_geojson` but applies an OGR
    ``SetSpatialFilter`` on the tile's bbox (transformed to the GeoJSON CRS)
    before iterating features, so very large GeoJSONs (e.g. nationwide
    polygons) stay cheap to rasterize against a single tile.

    Args:
        geojson_path: Path to GeoJSON file with polygon geometries.
        tile_path: Path to the reference GeoTIFF (CRS, extent, resolution).
        burn_value: Pixel value inside polygons (default 1 = leucaena).
        fill_value: Background pixel value (default 0 = no leucaena).

    Returns:
        Tuple ``(label_array_HxW_uint8, n_features_used)``.
    """
    ref_ds = gdal.Open(tile_path, gdalconst.GA_ReadOnly)
    if ref_ds is None:
        raise FileNotFoundError(f'Cannot open tile: {tile_path}')
    geo_transform = ref_ds.GetGeoTransform()
    x_size = ref_ds.RasterXSize
    y_size = ref_ds.RasterYSize
    ref_srs = _srs_from_dataset(ref_ds)

    target_ds = gdal.GetDriverByName('MEM').Create('', x_size, y_size, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(ref_srs.ExportToWkt())
    band = target_ds.GetRasterBand(1)
    band.Fill(fill_value)

    vec_ds = ogr.Open(geojson_path)
    if vec_ds is None:
        ref_ds = None
        raise FileNotFoundError(f'Cannot open GeoJSON: {geojson_path}')
    layer = vec_ds.GetLayer()
    src_srs = _srs_from_layer(layer)

    bbox_poly = _tile_bbox_geometry(geo_transform, x_size, y_size)
    if not src_srs.IsSame(ref_srs):
        ct_bbox = _coord_transform(ref_srs, src_srs)
        bbox_poly_src = bbox_poly.Clone()
        if bbox_poly_src.Transform(ct_bbox) != 0:
            raise RuntimeError('Tile bbox reprojection for spatial filter failed.')
        layer.SetSpatialFilter(bbox_poly_src)
    else:
        layer.SetSpatialFilter(bbox_poly)

    n_features = layer.GetFeatureCount()
    if n_features == 0:
        result = band.ReadAsArray()
        target_ds = None
        vec_ds = None
        ref_ds = None
        return result, 0

    burn_keep_ds, burn_layer = _reproject_layer_to_srs(layer, ref_srs)
    _rasterize_burn(target_ds, burn_layer, burn_value)

    result = band.ReadAsArray()
    target_ds = None
    vec_ds = None
    ref_ds = None
    return result, n_features


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
