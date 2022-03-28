import warnings
from . import _version
__version__ = _version.get_versions()['version']

from gisutils.projection import get_proj_str, get_authority_crs, project, get_shapefile_crs
from gisutils.raster import get_values_at_points, get_raster_crs, write_raster
from gisutils.shapefile import df2shp, shp2df
