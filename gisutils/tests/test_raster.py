import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import rasterio
from rasterio import Affine
from rasterio.crs import CRS
from shapely.geometry import box
import pytest
from gisutils import df2shp
from gisutils.projection import project, get_authority_crs
from gisutils.raster import (_xll_to_xul, _xul_to_xll, _yll_to_yul, _yul_to_yll,
                      write_raster, get_transform, read_arc_ascii,
                      get_values_at_points, zonal_stats, get_raster_crs)
from .test_projection import geotiff_3070, arc_ascii_3070


def geotiff(tmpdir, rotation=45.):
    filename = os.path.join(tmpdir, 'test_raster.tif')

    array = np.array([[0, 1],
                      [2, 3]])
    height, width = array.shape
    dx = 5.
    xll, yll = 0., 0.
    write_raster(filename, array, xll=xll, yll=yll,
                     dx=dx, dy=None, rotation=rotation, proj_str='epsg:3070',
                     nodata=-9999)
    xul = _xll_to_xul(xll, height*dx, rotation)
    yul = _yll_to_yul(yll, height*dx, rotation)
    transform = get_transform(xul=xul, yul=yul,
                              dx=dx, dy=-dx, rotation=rotation)
    return filename, transform


def arc_ascii(tmpdir):
    filename = os.path.join(tmpdir, 'test_raster.asc')

    array = np.array([[0, 1],
                      [2, 3]])
    height, width = array.shape
    dx = 5.
    xll, yll = 0., 0.
    rotation = 0.
    write_raster(filename, array, xll=xll, yll=yll,
                     dx=dx, dy=None, rotation=rotation,
                     nodata=-9999)
    xul = _xll_to_xul(xll, height*dx, rotation)
    yul = _yll_to_yul(yll, height*dx, rotation)
    transform = get_transform(xul=xul, yul=yul,
                              dx=dx, dy=-dx, rotation=rotation)
    return filename, transform


def test_get_transform():
    dx = 5.
    height = 2
    xll, yll = 0., 0.
    rotation = 30.
    xul = _xll_to_xul(xll, height*dx, rotation)
    yul = _yll_to_yul(yll, height*dx, rotation)
    transform = get_transform(xul=xul, yul=yul,
                              dx=dx, dy=-dx, rotation=rotation)
    transform2 = Affine(dx, 0., xul,
                        0., -dx, yul) * \
                 Affine.rotation(rotation)
    assert transform == transform2


@pytest.mark.parametrize('xll_height_rotation', [(0., 10., 30.)])
def test_xll_to_xul(xll_height_rotation):
    xll, height, rotation = xll_height_rotation
    expected = np.sin(rotation * np.pi/180) * height
    xul = _xll_to_xul(xll, height, rotation)
    assert np.allclose(xul, expected)
    xll2 = _xul_to_xll(xul, height, rotation)
    assert np.allclose(xll2, xll)


@pytest.mark.parametrize('yll_height_rotation', [(0., 10., 30.)])
def test_yll_to_yul(yll_height_rotation):
    yll, height, rotation = yll_height_rotation
    expected = np.cos(rotation * np.pi / 180) * height
    yul = _yll_to_yul(yll, height, rotation)
    assert np.allclose(yul, expected)
    yll2 = _yul_to_yll(yul, height, rotation)
    assert np.allclose(yll2, yll)


@pytest.mark.parametrize('method', ('nearest', 'linear'))
@pytest.mark.parametrize('x, y, rotation, expected', (([2.5, 7.5, -1], [2.5, 7.5, -1], 0., {'nearest': [2, 1, -9999],
                                                                                            'linear': [2, 1, -9999]
                                                                                            }
                                                       ),
                                                      ([np.sqrt(2)*2.5, 12, -1], [0, 1, -1], 45., {'nearest': [2, 1, -9999],
                                                                                        'linear': [2, 1, -9999]
                                                                                        }
                                                                                    )
                                                     ))
def test_get_values_at_points_geotiff(tmpdir, x, y, rotation, method, expected):
    filename, transform = geotiff(tmpdir, rotation=rotation)
    result = get_values_at_points(filename, x=x, y=y, method=method,
                                  out_of_bounds_errors='coerce')
    result[np.isnan(result)] = -9999
    assert np.allclose(result, expected[method])


def test_get_values_at_points_arc_ascii(tmpdir):
    filename, _ = arc_ascii(tmpdir)
    result = get_values_at_points(filename,
                                  x=[2.5, 7.5, -1],
                                  y=[2.5, 7.5, -1],
                                  out_of_bounds_errors='coerce')
    result[np.isnan(result)] = -9999
    expected = [2, 1, -9999]
    assert np.allclose(result, expected)


def test_get_values_at_points_in_a_different_crs(geotiff_3070):

    # get a dataset reader handle for the raster
    with rasterio.open(geotiff_3070) as src:
        pass
    # points that represent the cell centers of the raster in epsg:3070
    original_cell_xcenters = np.array([0.5, 1.5, 2.5] * 3)
    original_cell_ycenters = np.array([0.5] * 3 + [1.5] * 3 + [2.5] * 3)
    x, y = src.transform * (original_cell_xcenters, original_cell_ycenters)
    results = get_values_at_points(geotiff_3070, x=x, y=y)
    expected = np.arange(0, 9)
    assert np.allclose(results, expected)

    # reproject the points to epsg:4326
    x_4326, y_4326 = project((x, y), 'epsg:3070', 'epsg:4326')
    results2 = get_values_at_points(geotiff_3070, x=x_4326, y=y_4326, points_crs='epsg:4326')
    assert np.allclose(results2, expected)


def test_get_raster_crs(geotiff_3070):
    crs = get_raster_crs(geotiff_3070)
    expected = get_authority_crs(3070)
    assert crs == expected


def test_get_arc_ascii_crs(arc_ascii_3070):
    crs = get_raster_crs(arc_ascii_3070)
    assert crs is None


@pytest.fixture(scope='module')
def polygon_features():
    features = [box(-1, 0, 1, 1),  # polygon doesn't include any cell centers
                box(0, 0, 2.6, 2.6),  # 0, 1 should be included, because cell center is within polygon
                box(0.5, 0.5, 100, 100),
                box(0, 0, 2.5, 2.5),
                ]
    return features


@pytest.fixture(scope='module')
def shapefile_features(polygon_features, tmpdir):
    df = pd.DataFrame({'id': list(range(len(polygon_features))),
                       'geometry': polygon_features
                       }
                      )
    shapefile_name = '{}/zstats_features.shp'.format(tmpdir)
    df2shp(df, shapefile_name, epsg=3070)
    return shapefile_name


# ugly work-around for fixtures not being supported as test parameters yet
# https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(params=['polygon_features',
                        'shapefile_features'])
def features(request,
             polygon_features,
             shapefile_features):
    return {'polygon_features': polygon_features,
            'shapefile_features': shapefile_features}[request.param]


def test_zonal_stats(features, tmpdir):
    filename, transform = geotiff(tmpdir, rotation=0.)
    result = zonal_stats(features, filename, out_shape=None,
                stats=['mean'])
    result = result['mean']
    result[np.isnan(result)] = -9999
    assert np.array_equal(result, [-9999, 2., np.arange(4).mean(), 2.])
    result = zonal_stats(features, filename, out_shape=(2, 2),
                         stats=['mean'])
    assert result['mean'].shape == (2, 2)


def test_write_raster_geotiff(tmpdir):
    fname, tranform = geotiff(tmpdir)
    expected = {'driver': 'GTiff',
                'dtype': 'int32',
                'nodata': -9999.0,
                'width': 2,
                'height': 2,
                'count': 1,
                'crs': CRS.from_epsg(3070),
                'transform': tranform
                }
    with rasterio.open(fname) as src:
        meta = src.meta
        data = src.read(1)
    assert meta == expected
    assert data.dtype == np.int32
    assert data.sum() == 6


def test_write_raster_ascii(tmpdir):
    fname, transform = arc_ascii(tmpdir)
    data, metadata = read_arc_ascii(fname, shape=(2, 2))
    expected = {'ncols': 2,
                'nrows': 2,
                'xllcorner': 0.0,
                'yllcorner': 0.0,
                'cellsize': 5.0,
                'nodata_value': -9999,
                'geotransform': (transform.a, transform.b, transform.c,
                                 transform.d, transform.e, transform.f)}
    assert metadata == expected
    assert data.sum() == 6



