import os
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import Point, MultiPolygon, box
from shapely.geometry.base import BaseMultipartGeometry
import pytest
from ..project import get_proj_str, project
from ..shapefile import df2shp


def test_get_proj_str(tmpdir):
    proj_str = '+proj=tmerc +lat_0=0 +lon_0=-90 +k=0.9996 +x_0=520000 +y_0=-4480000 +datum=NAD83 +units=m +no_defs '
    p1 = pyproj.Proj(proj_str)
    f = os.path.join(tmpdir, 'junk.shp')
    df2shp(pd.DataFrame({'id': [0],
                         'geometry': [Point(0, 0)]
                         }),
           f, proj_str=proj_str)
    proj4_2 = get_proj_str(f.replace('shp', 'prj'))
    p2 = pyproj.Proj(proj4_2)
    assert p1 == p2


@pytest.mark.parametrize('input', [(177955.0, 939285.0, 'epsg:5070', 'epsg:4269'),
                                   (-91.87370, 34.93738, 'epsg:4269', 'epsg:5070')]
)
def test_project_point(input):
    x1, y1, proj_str_1, proj_str_2 = input
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
    assert np.allclose(point_1, point_3)

    # list of Points
    point_2 = project([Point(point_1),
                          Point(point_1)], proj_str_1, proj_str_2)
    point_3 = project(point_2, proj_str_2, proj_str_1)
    assert isinstance(point_2, list)
    for p in point_3:
        assert np.allclose(list(p.coords)[0], point_1)


def test_project_multipolygon():

    p1 = box(0, 0, 1, 1)
    p2 = box(0, 1, 2, 1)
    geom = MultiPolygon([p1, p2])
    result = project(geom, 'epsg:3070', 'epsg:26916')
    assert isinstance(result, BaseMultipartGeometry)
    assert isinstance(result, MultiPolygon)
