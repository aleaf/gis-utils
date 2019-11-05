
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .shapefile import df2shp, shp2df
from .project import project, get_proj_str
from .raster import get_values_at_points
