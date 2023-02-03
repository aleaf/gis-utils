import os
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import Point, MultiPolygon, box
from shapely.geometry.base import BaseMultipartGeometry
import rasterio
import pytest
from gisutils.projection import get_proj_str, project, get_authority_crs, project_raster, get_rasterio_crs
from gisutils.shapefile import df2shp
from gisutils.raster import write_raster, get_values_at_points


def test_import():
    print('from gisutils import project...')
    from gisutils import project
    assert callable(project)  # verify that the function is imported, not the module
    print('from gisutils.project import project...')
    from gisutils.projection import project
    assert callable(project)  # old module name should still work for time being
    print('from gisutils.project import get_proj_str...')
    from gisutils.projection import get_proj_str
    assert callable(get_proj_str)

def test_get_proj_str(test_output_path):
    proj_str = '+proj=tmerc +lat_0=0 +lon_0=-90 +k=0.9996 +x_0=520000 +y_0=-4480000 +datum=NAD83 +units=m +no_defs '
    p1 = pyproj.Proj(proj_str)
    f = os.path.join(test_output_path, 'junk.shp')
    df2shp(pd.DataFrame({'id': [0],
                         'geometry': [Point(0, 0)]
                         }),
           f, proj_str=proj_str)
    proj4_2 = get_proj_str(f.replace('shp', 'prj'))
    p2 = pyproj.Proj(proj4_2)
    assert p1 == p2


@pytest.mark.parametrize('crs', (True, False))
@pytest.mark.parametrize('input', [(554220.0, 391780.0, 3070, 3071),
                                   (177955.0, 939285.0, 'epsg:5070', 'epsg:4269'),
                                   (-91.87370, 34.93738, 'epsg:4269', 'epsg:5070'),
                                   (-94.16583369760917, 31.142591218327198, 4269, 4326)
                                   ]
)
def test_project_point(input, crs):
    x1, y1, proj_str_1, proj_str_2 = input
    if crs:
        proj_str_1 = get_authority_crs(proj_str_1)
        proj_str_2 = get_authority_crs(proj_str_2)
    point_1 = (x1, y1)

    # tuple
    point_2 = project(point_1, proj_str_1, proj_str_2)
    point_3 = project(point_2, proj_str_2, proj_str_1)
    assert isinstance(point_2, tuple)
    assert np.allclose(point_1, point_3)

    # list of tuples
    points_5070_list = [point_1] * 3
    point_2 = project(points_5070_list, proj_str_1, proj_str_2)
    x, y = point_2
    x2, y2 = project((x, y), proj_str_2, proj_str_1)
    assert len(x) == len(x2)
    assert np.allclose(np.array(points_5070_list).transpose(),
                       np.array([x2, y2]))

    # shapely Point
    point_2 = project(Point(point_1), proj_str_1, proj_str_2)
    point_3 = project(Point(point_2), proj_str_2, proj_str_1)
    assert isinstance(point_2, Point)
    assert np.allclose(point_1, (point_3.x, point_3.y))

    # list of Points
    point_2 = project([Point(point_1),
                          Point(point_1)], proj_str_1, proj_str_2)
    point_3 = project(point_2, proj_str_2, proj_str_1)
    assert isinstance(point_2, list)
    for p in point_3:
        assert np.allclose(list(p.coords)[0], point_1)


def test_project_polygon():
    nrow, ncol = 409, 614
    spacing = 500 * .3048 # meters
    height, width = nrow * spacing, ncol * spacing
    xul = 617822.3
    yul = 5177152.3
    yll = 5177152.3 - height

    bbox = box(xul, yll, xul + width, yll + height)
    bbox_4269 = project(bbox, 26715, 4269)
    assert np.allclose(bbox_4269.bounds, 
                       (-91.47365622892065, 46.156112832038794, 
                        -90.23414472574912, 46.739476860395655))
    
    
def test_project_multipolygon():

    p1 = box(0, 0, 1, 1)
    p2 = box(0, 1, 2, 1)
    geom = MultiPolygon([p1, p2])
    result = project(geom, 'epsg:3070', 'epsg:26916')
    assert isinstance(result, BaseMultipartGeometry)
    assert isinstance(result, MultiPolygon)


@pytest.mark.parametrize('input,expected_srs', (pytest.param(None, None, marks=pytest.mark.xfail),
                                                (5070, 'EPSG:5070'),
                                                ('epsg:26910', 'EPSG:26910'),
                                                ('epsg:4269', 'EPSG:4269'),
                                                 # an example of an uncommon CRS
                                                (('PROJCS["NAD_1983_California_Teale_Albers",'
                                                  'GEOGCS["GCS_North_American_1983",'
                                                  'DATUM["D_North_American_1983",'
                                                  'SPHEROID["GRS_1980",6378137.0,298.257222101]],'
                                                  'PRIMEM["Greenwich",0.0],'
                                                  'UNIT["Degree",0.0174532925199433]],'
                                                  'PROJECTION["Albers"],'
                                                  'PARAMETER["False_Easting",0.0],'
                                                  'PARAMETER["False_Northing",-4000000.0],'
                                                  'PARAMETER["Central_Meridian",-120.0],'
                                                  'PARAMETER["Standard_Parallel_1",34.0],'
                                                  'PARAMETER["Standard_Parallel_2",40.5],'
                                                  'PARAMETER["Latitude_Of_Origin",0.0],'
                                                  'UNIT["Meter",1.0]]'), 'EPSG:3310'),
                                                # CRS for Nation Hydrogeologic Grid
                                                # which has no epgs code
                                                # (Albers WGS 84)
                                                (('PROJCS["Albers NHG",'
                                                  'GEOGCS["GCS_WGS_1984",'
                                                  'DATUM["D_WGS_1984",'
                                                  'SPHEROID["WGS_1984",6378137,298.257223563,'
                                                  'AUTHORITY["EPSG","7030"]],'
                                                  'TOWGS84[0,0,0,0,0,0,0],'
                                                  'AUTHORITY["EPSG","6326"]],'
                                                  'PRIMEM["Greenwich",0,'
                                                  'AUTHORITY["EPSG","8901"]],'
                                                  'UNIT["degree",0.0174532925199433,'
                                                  'AUTHORITY["EPSG","9122"]],'
                                                  'AUTHORITY["EPSG","4326"]],'
                                                  'PROJECTION["Albers_Conic_Equal_Area"],'
                                                  'PARAMETER["standard_parallel_1",29.5],'
                                                  'PARAMETER["standard_parallel_2",45.5],'
                                                  'PARAMETER["latitude_of_origin",23],'
                                                  'PARAMETER["central_meridian",-96],'
                                                  'PARAMETER["false_easting",0],'
                                                  'PARAMETER["false_northing",0],'
                                                  'UNIT["Meter",1]]'), None)
                                      ))

def test_get_authority_crs(input, expected_srs):
    if expected_srs is None:
        expected_srs = input
    crs = get_authority_crs(input)
    assert crs.srs == expected_srs


@pytest.fixture
def geotiff_3070(test_output_path, rotation=0):
    filename = os.path.join(test_output_path, 'test_raster_3070.tif')

    array = np.array([[0, 1, 2],
                      [3, 4, 5],
                      [6, 7, 8]])
    dx = 5.
    xll, yll = 0., 0.
    write_raster(filename, array, xll=xll, yll=yll,
                 dx=dx, dy=None, rotation=rotation, crs='epsg:3070',
                 nodata=-9999)
    return filename


@pytest.fixture
def arc_ascii_3070(test_output_path, rotation=0):
    filename = os.path.join(test_output_path, 'test_arcascii_3070.asc')

    array = np.array([[0, 1, 2],
                      [3, 4, 5],
                      [6, 7, 8]])
    dx = 5.
    xll, yll = 0., 0.
    write_raster(filename, array, xll=xll, yll=yll,
                 dx=dx, dy=None, rotation=rotation, proj_str='epsg:3070',
                 nodata=-9999)
    return filename


def test_project_raster(test_output_path, geotiff_3070):
    filename = geotiff_3070
    filename2 = os.path.join(test_output_path, 'test_raster_4269.tif')
    project_raster(filename, filename2, 'epsg:4269',
                   resampling=0, resolution=None, num_threads=2,
                   driver='GTiff')
    filename3 = os.path.join(test_output_path, 'test_raster_4269_3070.tif')
    project_raster(filename2, filename3, 'epsg:3070',
                   resampling=0, resolution=None, num_threads=2,
                   driver='GTiff')
    with rasterio.open(filename) as src:
        array = src.read(1)
    with rasterio.open(filename2) as src2:
        array2 = src2.read(1)
    with rasterio.open(filename3) as src3:
        array3 = src3.read(1)

    # verify that get_values_at_points returns the same results
    # for the original and round-tripped 3070 rasters
    original_cell_xcenters = np.array([0.5, 1.5, 2.5] * 3)
    original_cell_ycenters = np.array([0.5] * 3 + [1.5] * 3 + [2.5] * 3)
    x, y = src.transform * (original_cell_xcenters, original_cell_ycenters)
    results = get_values_at_points(filename, x=x, y=y)
    expected = np.arange(0, 9)
    assert np.allclose(results, expected)
    results3 = get_values_at_points(filename3, x=x, y=y)
    assert np.allclose(results3, expected)


@pytest.mark.parametrize('input,expected', (pytest.param(None, None, marks=pytest.mark.xfail),
                                                (5070, 'EPSG:5070'),
                                                ('epsg:26910', 'EPSG:26910'),
                                                ('epsg:4269', 'EPSG:4269'),
                                                 # an example of an uncommon CRS
                                                (('PROJCS["NAD_1983_California_Teale_Albers",'
                                                  'GEOGCS["GCS_North_American_1983",'
                                                  'DATUM["D_North_American_1983",'
                                                  'SPHEROID["GRS_1980",6378137.0,298.257222101]],'
                                                  'PRIMEM["Greenwich",0.0],'
                                                  'UNIT["Degree",0.0174532925199433]],'
                                                  'PROJECTION["Albers"],'
                                                  'PARAMETER["False_Easting",0.0],'
                                                  'PARAMETER["False_Northing",-4000000.0],'
                                                  'PARAMETER["Central_Meridian",-120.0],'
                                                  'PARAMETER["Standard_Parallel_1",34.0],'
                                                  'PARAMETER["Standard_Parallel_2",40.5],'
                                                  'PARAMETER["Latitude_Of_Origin",0.0],'
                                                  'UNIT["Meter",1.0]]'), 'EPSG:3310'),
                                                # CRS for Nation Hydrogeologic Grid
                                                # which has no epgs code
                                                # (Albers WGS 84)
                                                # apparently in some form of ESRI wkt dialect
                                                (('PROJCS["Albers NHG",'
                                                  'GEOGCS["GCS_WGS_1984",'
                                                  'DATUM["D_WGS_1984",'
                                                  'SPHEROID["WGS_1984",6378137,298.257223563,'
                                                  'AUTHORITY["EPSG","7030"]],'
                                                  'TOWGS84[0,0,0,0,0,0,0],'
                                                  'AUTHORITY["EPSG","6326"]],'
                                                  'PRIMEM["Greenwich",0,'
                                                  'AUTHORITY["EPSG","8901"]],'
                                                  'UNIT["degree",0.0174532925199433,'
                                                  'AUTHORITY["EPSG","9122"]],'
                                                  'AUTHORITY["EPSG","4326"]],'
                                                  'PROJECTION["Albers_Conic_Equal_Area"],'
                                                  'PARAMETER["standard_parallel_1",29.5],'
                                                  'PARAMETER["standard_parallel_2",45.5],'
                                                  'PARAMETER["latitude_of_origin",23],'
                                                  'PARAMETER["central_meridian",-96],'
                                                  'PARAMETER["false_easting",0],'
                                                  'PARAMETER["false_northing",0],'
                                                  'UNIT["Meter",1]]'), None)
                                      ))
def test_get_rasterio_crs(input, expected):
    if expected is None:
        expected = input
    rasterio_crs = get_rasterio_crs(input)
    # convert both rasterio crs and expected to pyproj to compare
    # (small differences in the wkt output from rasterio will fail the Albers WGS 84 otherwise)
    pyproj.crs.CRS.from_user_input(expected) == pyproj.crs.CRS.from_user_input(rasterio_crs)
