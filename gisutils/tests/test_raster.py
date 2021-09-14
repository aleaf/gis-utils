import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
import rasterio
from rasterio import Affine
from rasterio.crs import CRS
from shapely.geometry import box, Point
import pytest
from gisutils import df2shp, shp2df
from gisutils.projection import project, get_authority_crs, project_raster
from gisutils.raster import (_xll_to_xul, _xul_to_xll, _yll_to_yul, _yul_to_yll,
                      write_raster, get_transform, read_arc_ascii,
                      get_values_at_points, points_to_raster, zonal_stats,
                             get_raster_crs, clip_raster)
from gisutils.tests.test_projection import geotiff_3070, arc_ascii_3070
from gisutils.tests.test_shapefile import crs_test_params


def geotiff(test_output_path, rotation=45.):
    filename = os.path.join(test_output_path, 'test_raster.tif')

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


def nc_file(test_output_path):
    filename = os.path.join(test_output_path, 'test_netcdf.nc')

    array = np.array([[0, 1],
                      [2, 3]])
    height, width = array.shape
    dx = 5.
    xll, yll = 0., 0.
    x = xll + np.add.accumulate(np.ones(array.shape[1])*dx) - (dx/2)
    y = yll + np.add.accumulate(np.ones(array.shape[0])*dx) - (dx/2)
    y = y[::-1]
    
    da = xr.DataArray(array,
                  dims=['y', 'x'],
                  coords={'x': x, 'y': y},
                  name='values')
    da.to_netcdf(filename)
    return filename, None


def arc_ascii(test_output_path):
    filename = os.path.join(test_output_path, 'test_raster.asc')

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


@pytest.mark.parametrize('size_thresh', (0, 1e9))
@pytest.mark.parametrize('method', ('nearest', 'linear'))
@pytest.mark.parametrize('x, y, rotation, expected', 
                         (([2.5, 7.5, -1], [2.5, 7.5, -1], 0., {'nearest': [2, 1, -9999],
                                                                'linear': [2, 1, -9999]
                                                                }
                                                       ),
                          ([2.5], [2.5], 0., {'nearest': [2],
                                    'linear': [2]
                                    }
                            ),
                                                      ([np.sqrt(2)*2.5, 12, -1], [0, 1, -1], 45., {'nearest': [2, 1, -9999],
                                                                                        'linear': [2, 1, -9999]
                                                                                        }
                                                                                    )
                                                     ))
def test_get_values_at_points_geotiff(test_output_path, x, y, rotation, method, expected, size_thresh):
    filename, transform = geotiff(test_output_path, rotation=rotation)
    result = get_values_at_points(filename, x=x, y=y, method=method,
                                  out_of_bounds_errors='coerce',
                                  size_thresh=size_thresh)
    result[np.isnan(result)] = -9999
    if size_thresh == 0:
        method = 'nearest'
    assert np.allclose(result, expected[method])


def test_get_values_at_points_arc_ascii(test_output_path):
    filename, _ = arc_ascii(test_output_path)
    result = get_values_at_points(filename,
                                  x=[2.5, 7.5, -1],
                                  y=[2.5, 7.5, -1],
                                  out_of_bounds_errors='coerce')
    result[np.isnan(result)] = -9999
    expected = [2, 1, -9999]
    assert np.allclose(result, expected)


def test_get_values_at_points_netcdf(test_output_path):
    filename, _ = nc_file(test_output_path)
    result = get_values_at_points(filename,
                                  x=[2.5, 7.5, -1],
                                  y=[2.5, 7.5, -1],
                                  xarray_variable='values',
                                  out_of_bounds_errors='coerce')
    result[np.isnan(result)] = -9999
    expected = [2., 1., -9999.]
    assert np.allclose(result, expected)
    

def test_get_values_at_points_in_a_different_crs(geotiff_3070):

    # get a dataset reader handle for the raster
    with rasterio.open(geotiff_3070) as src:
        pass
    # points that represent the cell centers of the raster in epsg:3070
    original_cell_xcenters = np.array([0.5, 1.5, 2.5] * 3)
    original_cell_ycenters = np.array([0.5] * 3 + [1.5] * 3 + [2.5] * 3)
    x, y = src.transform * (original_cell_xcenters, original_cell_ycenters)
    print(src.transform)
    results = get_values_at_points(geotiff_3070, x=x, y=y)
    expected = np.arange(0, 9)
    assert np.allclose(results, expected)

    # reproject the points to epsg:4326
    print((x,y))
    x_4326, y_4326 = project((x, y), 'epsg:3070', 'epsg:4326')
    print((x_4326, y_4326))
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
def shapefile_features(polygon_features, test_output_path):
    df = pd.DataFrame({'id': list(range(len(polygon_features))),
                       'geometry': polygon_features
                       }
                      )
    shapefile_name = '{}/zstats_features.shp'.format(test_output_path)
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


def test_zonal_stats(features, test_output_path):
    filename, transform = geotiff(test_output_path, rotation=0.)
    result = zonal_stats(features, filename, out_shape=None,
                stats=['mean'])
    result = result['mean']
    result[np.isnan(result)] = -9999
    assert np.array_equal(result, [-9999, 2., np.arange(4).mean(), 2.])
    result = zonal_stats(features, filename, out_shape=(2, 2),
                         stats=['mean'])
    assert result['mean'].shape == (2, 2)


def test_write_raster_geotiff(test_output_path):
    fname, tranform = geotiff(test_output_path)
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


def test_write_raster_ascii(test_output_path):
    fname, transform = arc_ascii(test_output_path)
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


@pytest.mark.parametrize('crs', crs_test_params)
def test_write_raster_crs(crs, test_output_path):
    filename = os.path.join(test_output_path, 'test_raster.tif')

    array = np.array([[0, 1],
                      [2, 3]])
    height, width = array.shape
    dx = 5.
    xll, yll = 0., 0.
    write_raster(filename, array, xll=xll, yll=yll,
                 dx=dx, dy=None, rotation=0, crs=crs,
                 nodata=-9999)
    if crs is not None:
        with rasterio.open(filename) as src:
            written_crs = get_authority_crs(src.crs)
            assert written_crs == get_authority_crs(crs)


@pytest.fixture
def geotiff_4269(test_output_path, geotiff_3070):
    filename = geotiff_3070
    filename2 = os.path.join(test_output_path, 'test_raster_4269.tif')
    project_raster(filename, filename2, 'epsg:4269',
                   resampling=0, resolution=None, num_threads=2,
                   driver='GTiff')
    return filename2


@pytest.mark.parametrize('bounds', (box(5, 5, 10, 10),
                                   [box(5, 5, 10, 10)],
                                    {'type': 'Polygon',  # geojson
                                     'coordinates': (((10.0, 5.0),
                                                      (10.0, 10.0),
                                                      (5.0, 10.0),
                                                      (5.0, 5.0),
                                                      (10.0, 5.0)),)},
                                    'POLYGON ((10 5, 10 10, 5 10, 5 5, 10 5))'  # wkt
                                    ))
def test_clip_raster(geotiff_4269, bounds, test_output_path):
    # make some bounds to clip out the "4" (middle value) in geotiff_3070
    outraster = os.path.join(test_output_path, 'clipped_raster.tif')
    clip_raster(geotiff_4269, clip_features=bounds, outraster=outraster,
                clip_features_crs='epsg:3070',
                clip_kwargs={'all_touched': False}, resampling=0)
    assert os.path.exists(outraster)
    with rasterio.open(outraster) as src:
        result = src.read(1)
    result[result == src.nodata] = 0
    assert result.sum() == 4


@pytest.fixture
def point_data(test_output_path):
    df = pd.DataFrame({'x': [1, 3, 5, 5, 3, 2],
                       'y': [1, 1, 1, 3, 2, 4],
                       'values': np.random.randn(6),
                       })
    df['geometry'] = [Point(x, y) for x, y in zip(df.x, df.y)]
    df2shp(df, test_output_path / 'test_points.shp', crs=5070)


def test_points_to_raster(point_data, test_output_path):
    bottom_shapefiles = [test_output_path / 'test_points.shp']
    outfile = test_output_path / 'test_points_raster.tif'
    points_to_raster(bottom_shapefiles,
                             data_col='values',
                             output_resolution=0.1,
                             outfile=outfile)
    source_data = shp2df(str(bottom_shapefiles[0]))
    x = [g.x for g in source_data.geometry]
    y = [g.y for g in source_data.geometry]
    results = get_values_at_points(outfile, x, y)
    assert np.allclose(results, source_data['values'].values)