"""
Functions for working with coordinate reference systems and projections.
"""
import warnings
import collections
from functools import partial
import numpy as np
from shapely.ops import transform
from shapely.geometry.base import BaseMultipartGeometry
import pyproj
from osgeo import osr


def __getattr__(name):
    if name == 'project':
        warnings.warn("The 'project' module was renamed to 'projection' "
                      "to avoid confusion with the project() function.",
                      DeprecationWarning)
        return project
    raise AttributeError('No module named ' + name)


def get_proj_str(prj):
    """Get the PROJ string from the well-known text in an ESRI projection file.

    Parameters
    ----------
    prj : string (filepath)
        ESRI Shapefile or projection file

    Returns
    -------
    proj_str : string (http://trac.osgeo.org/proj/)

    """
    prjfile = prj[:-4] + '.prj' # allows shp or prj to be argued
    try:
        with open(prjfile) as src:
            prjtext = src.read()
        srs = osr.SpatialReference()
        srs.ImportFromESRI([prjtext])
        proj_str = srs.ExportToProj4()
        return proj_str
    except:
        pass


def project(geom, projection1, projection2):
    """Reproject shapely geometry object(s) or scalar
    coodrinates to new coordinate system

    Parameters
    ----------
    geom: shapely geometry object, list of shapely geometry objects,
          list of (x, y) tuples, or (x, y) tuple.
    projection1: string
        Proj4 string specifying source projection
    projection2: string
        Proj4 string specifying destination projection
    """
    # pyproj 2 style
    # https://pyproj4.github.io/pyproj/dev/gotchas.html
    transformer = pyproj.Transformer.from_crs(projection1, projection2, always_xy=True)

    # check for x, y values instead of shapely objects
    if isinstance(geom, tuple):
        # tuple of scalar values
        if np.isscalar(geom[0]):
            return transformer.transform(*geom)
        elif isinstance(geom[0], collections.Iterable):
            return transformer.transform(*geom)

    # sequence of tuples or shapely objects
    if isinstance(geom, BaseMultipartGeometry):
        geom0 = geom
    elif isinstance(geom, collections.Iterable):
        geom = list(geom) # in case it's a generator
        geom0 = geom[0]
    else:
        geom0 = geom

    # sequence of tuples
    if isinstance(geom0, tuple):
        a = np.array(geom)
        x = a[:, 0]
        y = a[:, 1]
        return transformer.transform(x, y)

    project = partial(transformer.transform)

    # do the transformation!
    if isinstance(geom, collections.Iterable) and not isinstance(geom, BaseMultipartGeometry):
        return [transform(project, g) for g in geom]
    return transform(project, geom)


def get_authority_crs(crs):
    """Try to get the authority representation
    for a CRS, for more robust comparison with other CRS
    objects.

    Parameters
    ----------
    crs : obj
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

    Returns
    -------
    authority_crs : pyproj.crs.CRS instance
        CRS instance initiallized with the name
        and authority code (e.g. epsg: 5070) produced by
        pyproj.crs.CRS.to_authority()

    Notes
    -----
    pyproj.crs.CRS.to_authority() will return None if a matching
    authority name and code can't be found. In this case,
    the input crs instance will be returned.

    References
    ----------
    http://pyproj4.github.io/pyproj/stable/api/crs/crs.html

    """
    crs = pyproj.crs.CRS.from_user_input(crs)
    authority = crs.to_authority()
    if authority is not None:
        return pyproj.CRS.from_user_input(authority)
    return crs