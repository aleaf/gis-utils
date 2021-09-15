import warnings
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from gisutils.projection import get_proj_str, get_authority_crs, project
from gisutils.raster import get_values_at_points, get_raster_crs, write_raster
from gisutils.shapefile import df2shp, shp2df, get_shapefile_crs
