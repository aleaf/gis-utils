"""
Functions for working with rasters.
"""
import os
import collections
import warnings
import time
import numpy as np
import pandas as pd
from scipy import interpolate
try:
    from osgeo import gdal
except:
    gdal = False

try:
    import rasterio
    from rasterio import Affine
except:
    rasterio = False

from gisutils.projection import project, get_authority_crs
from gisutils.shapefile import shp2df


def get_transform(xul, yul, dx, dy=None, rotation=0.):
    """Returns an affine.Affine instance that can be
    used to locate raster grids in space. See
    https://www.perrygeo.com/python-affine-transforms.html
    https://rasterio.readthedocs.io/en/stable/topics/migrating-to-v1.html

    Parameters
    ----------
    xul : float
        x-coorindate of upper left corner of raster grid
    yul : float
        y-coorindate of upper left corner of raster grid
    dx : float
        cell spacing in the x-direction
    dy : float
        cell spacing in the y-direction
    rotation :
        rotation of the raster grid in degrees, clockwise
    Returns
    -------
    affine.Affine instance
    """
    if not rasterio:
        raise ImportError("This function requires rasterio.")
    if dy is None:
        dy = -dx
    return Affine(dx, 0., xul,
                  0., dy, yul) * \
           Affine.rotation(rotation)


def get_raster_crs(raster):
    """Get the coordinate reference system for a shapefile.

    Parameters
    ----------
    raster : str (filepath)
        Path to a raster

    Returns
    -------
    crs : pyproj.CRS instance

    """
    with rasterio.open(raster) as src:
        if src.crs is not None:
            crs = get_authority_crs(src.crs)
            return crs


def get_values_at_points(rasterfile, x=None, y=None, band=1,
                         points=None, points_crs=None,
                         out_of_bounds_errors='coerce',
                         method='nearest'):
    """Get raster values single point or list of points. Points in
    a different coordinate reference system (CRS) specified with a points_crs will be
    reprojected to the raster CRS prior to sampling.

    Parameters
    ----------
    rasterfile : str
        Filename of raster.
    x : 1D array
        X coordinate locations
    y : 1D array
        Y coordinate locations
    points : list of tuples or 2D numpy array (npoints, (row, col))
        Points at which to sample raster.
    points_crs : obj, optional
        Coordinate reference system for points or x, y. Only needed if
        different than the CRS for the raster, in which case the points will be
        reprojected to the raster CRS prior to getting the values.
        A Python int, dict, str, or pyproj.crs.CRS instance
        passed to the pyproj.crs.from_user_input
        See http://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input.
        Can be any of:
          - PROJ string
          - Dictionary of PROJ parameters
          - PROJ keyword arguments for parameters
          - JSON string with PROJ parameters
          - CRS WKT string
          - An authority string [i.e. 'epsg:4326']
          - An EPSG integer code [i.e. 4326]
          - A tuple of ("auth_name": "auth_code") [i.e ('epsg', '4326')]
          - An object with a `to_wkt` method.
          - A :class:`pyproj.crs.CRS` class
    out_of_bounds_errors : {‘raise’, ‘coerce’}, default 'raise'
        * If 'raise', then x, y locations outside of the raster will raise an exception.
        * If 'coerce', then x, y locations outside of the raster will be set to NaN.
    method : str 'nearest' or 'linear'
        If 'nearest', the rasterio.DatasetReader.index() method is used to
        return the raster values at the nearest cell centers. If 'linear',
        scipy.interpolate.interpn is used for bilinear interpolation of values
        between raster cell centers.

    Returns
    -------
    list of floats

    Notes
    -----
    requires rasterio
    """
    if not rasterio:
        raise ImportError("This function requires rasterio.")

    # read in sample points
    array_shape = None
    if x is not None and isinstance(x[0], tuple):
        x, y = np.array(x).transpose()
        warnings.warn(
            "new argument input for get_values_at_points is x, y, or points"
        )
    elif x is not None:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if len(x.shape) > 1:
            array_shape = x.shape
            x = x.ravel()
        if len(y.shape) > 1:
            array_shape = y.shape
            y = y.ravel()
    elif points is not None:
        if not isinstance(points, np.ndarray):
            x, y = np.array(points)
        else:
            x, y = points[:, 0], points[:, 1]
    else:
        print('Must supply x, y or list/array of points.')

    assert os.path.exists(rasterfile), "raster {} not found".format(rasterfile)
    t0 = time.time()

    print("reading data from {}...".format(rasterfile))
    with rasterio.open(rasterfile) as src:
        meta = src.meta
        nodata = meta['nodata']
        data = src.read(band)

    if points_crs is not None:
        points_crs = get_authority_crs(points_crs)
        raster_crs = get_authority_crs(src.crs)
        if points_crs != raster_crs:
            x, y = project((x, y), points_crs, raster_crs)

    if method == 'nearest':
        i, j = src.index(x, y)
        i = np.array(i, dtype=int)
        j = np.array(j, dtype=int)
        nrow, ncol = data.shape

        # mask row, col locations outside the raster
        within = (i >= 0) & (i < nrow) & (j >= 0) & (j < ncol)

        # get values at valid point locations
        results = np.ones(len(i), dtype=float) * np.nan
        results[within] = data[i[within], j[within]]
        if out_of_bounds_errors == 'raise' and np.any(np.isnan(results)):
            n_invalid = np.sum(np.isnan(results))
            raise ValueError("{} points outside of {} extent.".format(n_invalid, rasterfile))
    else:
        # map the points to interpolate to onto the raster coordinate system
        # (in case the raster is rotated)
        x_rx, y_ry = ~src.transform * (x, y)
        # coordinates of raster pixel centers in raster coordinate system
        # (e.g. i,j = 0, 0 = 0.5, 0.5)
        rx = np.arange(src.width) + 0.5
        ry = np.arange(src.height) + 0.5
        # pad the coordinates and the data, so that points within the outer pixels are still counted
        padded = np.pad(data.astype(float), pad_width=1, mode='edge')
        rx = np.array([0] + rx.tolist() + [rx[-1] + 0.5])
        ry = np.array([0] + ry.tolist() + [ry[-1] + 0.5])
        # exclude nodata points prior to interpolating
        padded[padded == nodata] = np.nan
        bounds_error = False
        if out_of_bounds_errors == 'raise':
            bounds_error = True
        results = interpolate.interpn((ry, rx), padded,
                                      (y_ry, x_rx), method=method,
                                       bounds_error=bounds_error, fill_value=nodata)
    # convert nodata values to np.nans
    results[results == nodata] = np.nan

    # reshape to input shape
    if array_shape is not None:
        results = np.reshape(results, array_shape)
    print("finished in {:.2f}s".format(time.time() - t0))
    return results


def write_raster(filename, array, xll=0., yll=0., xul=None, yul=None,
                 dx=1., dy=None, rotation=0., proj_str=None,
                 nodata=-9999, fieldname='value', verbose=False,
                 **kwargs):
    """
    Write a numpy array to Arc Ascii grid or shapefile with the model
    reference.

    Parameters
    ----------
    filename : str
        Path of output file. Export format is determined by
        file extention.
        '.asc'  Arc Ascii grid
        '.tif'  GeoTIFF (requries rasterio package)
    array : 2D numpy.ndarray
        Array to export
    xll : float
        x-coordinate of lower left corner of raster grid.
        Either xul, yul or xll, yll must be specified.
        Default = 0.
    yll : float
        y-coordinate of lower left corner of raster grid
        Default = 0.
    xul : float
        x-coordinate of upper left corner of raster grid.
        Either xul, yul or xll, yll must be specified.
    yul : float
        y-coordinate of upper left corner of raster grid
    dx : float
        cell spacing in the x-direction
    dy : float
        cell spacing in the y-direction
        (optional, assumed equal to dx by default)
    rotation :
        rotation of the raster grid in degrees, clockwise
    nodata : scalar
        Value to assign to np.nan entries (default -9999)
    fieldname : str
        Attribute field name for array values (shapefile export only).
        (default 'values')
    kwargs:
        keyword arguments to np.savetxt (ascii)
        rasterio.open (GeoTIFF)
        or flopy.export.shapefile_utils.write_grid_shapefile2

    Notes
    -----
    Rotated grids will be either be unrotated prior to export,
    using scipy.ndimage.rotate (Arc Ascii format) or rotation will be
    included in their transform property (GeoTiff format). In either case
    the pixels will be displayed in the (unrotated) projected geographic
    coordinate system, so the pixels will no longer align exactly with the
    model grid (as displayed from a shapefile, for example). A key difference
    between Arc Ascii and GeoTiff (besides disk usage) is that the
    unrotated Arc Ascii will have a different grid size, whereas the GeoTiff
    will have the same number of rows and pixels as the original.

    """
    if not rasterio:
        raise ImportError("This function requires rasterio.")
    t0 = time.time()
    a = array
    # third dimension is the number of bands
    if len(a.shape) == 2:
        a = np.reshape(a, (1, a.shape[0], a.shape[1]))
    count, height, width = a.shape

    if xul is not None and yul is not None:
        # default to decreasing y coordinates if upper left is specified
        if dy is None:  
            dy = -dx
        xll = _xul_to_xll(xul, height * dy, rotation)
        yll = _yul_to_yll(yul, height * dy, rotation)
    elif xll is not None and yll is not None:
        # default to increasing y coordinates if lower left is specified
        if dy is None:
            dy = dx
        xul = _xll_to_xul(xll, height * dy, rotation)
        yul = _yll_to_yul(yll, height * dy, rotation)
    if filename.lower().endswith(".tif"):
        trans = get_transform(xul=xul, yul=yul,
                              dx=dx, dy=-np.abs(dy), rotation=rotation)

        # third dimension is the number of bands
        if len(a.shape) == 2:
            a = np.reshape(a, (1, a.shape[0], a.shape[1]))

        if a.dtype == np.int64:
            a = a.astype(np.int32)
        meta = {'count': count,
                'width': width,
                'height': height,
                'nodata': nodata,
                'dtype': a.dtype,
                'driver': 'GTiff',
                'crs': proj_str,
                'transform': trans,
                'compress': 'lzw'
                }
        meta.update(kwargs)
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(a)
            if isinstance(a, np.ma.masked_array):
                dst.write_mask(~a.mask.transpose(1, 2, 0))
        print('wrote {}'.format(filename))

    elif filename.lower().endswith(".asc"):
        path, fname = os.path.split(filename)
        fname, ext = os.path.splitext(fname)
        for band in range(count):
            if count == 1:
                filename = os.path.join(path, fname + '.asc')
            else:
                filename =  os.path.join(path, fname + '_{}.asc'.format(band))
            write_arc_ascii(a[band], filename, xll=xll, yll=yll,
                            cellsize=dx,
                            nodata=nodata, **kwargs)
    if verbose:
        print("raster creation took {:.2f}s".format(time.time() - t0))


def zonal_stats(feature, raster, out_shape=None,
                stats=['mean']):
    try:
        from rasterstats import zonal_stats
    except:
        raise ImportError("This function requires rasterstats.")

    if not isinstance(feature, str):
        feature_name = 'feature'
    else:
        feature_name = feature
    t0 = time.time()
    print('computing {} {} for zones in {}...'.format(raster,
                                                      ', '.join(stats),
                                                      feature_name
                                                      ))
    print(stats)
    results = zonal_stats(feature, raster, stats=stats)
    print(out_shape)
    if out_shape is None:
        out_shape = (len(results),)
    #print(results)
    #means = [r['mean'] for r in results]
    #means = np.asarray(means)
    #means = np.reshape(means, out_shape).astype(float)
    #results = means

    #results = np.reshape(results, out_shape)
    #results = np.reshape(results, out_shape).astype(float)
    results_dict = {}
    for stat in stats:
        res = [r[stat] for r in results]
        res = np.asarray(res)
        res = np.reshape(res, out_shape).astype(float)
        results_dict[stat] = res
    print("finished in {:.2f}s".format(time.time() - t0))
    return results_dict


def read_arc_ascii(filename, shape=None):
    with open(filename) as src:
        meta = {}
        for i in range(6):
            k, v = next(src).strip().split()
            v = float(v) if '.' in v else int(v)
            meta[k.lower()] = v

        # make a gdal-style geotransform
        dx = meta['cellsize']
        dy = meta['cellsize']
        xul = meta['xllcorner']
        yul = meta['yllcorner'] + dy * meta['nrows']
        rx, ry = 0, 0
        meta['geotransform'] = dx, rx, xul, ry, -dy, yul

        if shape is not None:
            assert (meta['nrows'], meta['ncols']) == shape, \
                "Data in {} are {}x{}, expected {}x{}".format(filename,
                                                              meta['nrows'],
                                                              meta['ncols'],
                                                              *shape)
        arr = np.loadtxt(src)
    return arr, meta


def write_arc_ascii(array, filename, xll=0, yll=0, cellsize=1.,
                    nodata=-9999, **kwargs):
    """Write numpy array to Arc Ascii grid.

    Parameters
    ----------
    array : 2D numpy.ndarray
    filename : str (file path)
        Name of output arc ascii file
    xll : scalar
        X-coordinate of lower left corner of grid
    yll : scalar
        Y-coordinate of lower left corner of grid
    cellsize : scalar
        Grid spacing
    nodata : scalar
        Value indicating cells with no data.
    kwargs: keyword arguments to numpy.savetxt
    """
    array = array.copy()
    array[np.isnan(array)] = nodata

    filename = '.'.join(filename.split('.')[:-1]) + '.asc'  # enforce .asc ending
    nrow, ncol = array.shape
    txt = 'ncols  {:d}\n'.format(ncol)
    txt += 'nrows  {:d}\n'.format(nrow)
    txt += 'xllcorner  {:f}\n'.format(xll)
    txt += 'yllcorner  {:f}\n'.format(yll)
    txt += 'cellsize  {}\n'.format(cellsize)
    txt += 'NODATA_value  {:.0f}\n'.format(nodata)
    with open(filename, 'w') as output:
        output.write(txt)
    with open(filename, 'ab') as output:
        np.savetxt(output, array, **kwargs)
    print('wrote {}'.format(filename))


def _xul_to_xll(xul, height, rotation=0.):
    theta = rotation * np.pi / 180
    return xul - (np.sin(theta) * height)


def _xll_to_xul(xll, height, rotation=0.):
    theta = rotation * np.pi / 180
    return xll + (np.sin(theta) * height)


def _yul_to_yll(yul, height, rotation=0.):
    theta = rotation * np.pi / 180
    return yul - (np.cos(theta) * height)


def _yll_to_yul(yul, height, rotation=0.):
    theta = rotation * np.pi / 180
    return yul + (np.cos(theta) * height)

