import os
import numpy as np
import pandas as pd
import shapely.wkt
import pyproj
import pytest
from gisutils.projection import project, get_authority_crs, get_shapefile_crs
from gisutils.shapefile import (df2shp, shp2df, shp_properties,
                         rename_fields_to_10_characters)


def test_shp_properties():
    df = pd.DataFrame({'reach': [1], 'value': [1.0], 'name': ['stuff']}, index=[0])
    df = df[['name', 'reach', 'value']].copy()
    assert [d.name for d in df.dtypes] == ['object', 'int64', 'float64']
    assert shp_properties(df) == {'name': 'str', 'reach': 'int', 'value': 'float'}


def test_shp_integer_dtypes(test_output_path):

    # verify that pandas is recasting numpy ints as python ints when converting to dict
    # (numpy ints invalid for shapefiles)
    d = pd.DataFrame(np.ones((3, 3)), dtype=int).astype(object).to_dict(orient='records')
    for i in range(3):
        assert isinstance(d[i][0], int)

    df = pd.DataFrame({'r': np.arange(100), 'c': np.arange(100)})
    f = '{}/ints.dbf'.format(test_output_path)
    df2shp(df, f)
    df2 = shp2df(f)
    assert np.all(df == df2)


def test_shp_boolean_dtypes(test_output_path):

    df = pd.DataFrame([False, True]).transpose()
    df.columns = ['true', 'false']
    f = '{}/bool.dbf'.format(test_output_path)
    df2shp(df, f)
    df2 = shp2df(f, true_values='True', false_values='False')
    assert np.all(df == df2)


def test_rename_fields_to_10_characters(test_output_path):
    columns = ['atthelimit'] + ['overthelimit', 'overtheli1',
                                'overthelimit2', 'overthelimit3', 'tomanycharacters']
    columns += ['{}{}'.format(s, i) for i, s in enumerate(['tomanycharacters'] * 11)]
    expected = ['atthelimit', 'overthelimit'[:10], 'overtheli1', 'overtheli0', 'overtheli2',
                'tomanycharacters'[:10]]
    expected += ['{}{}'.format(s, i)
                 for i, s in enumerate(['tomanycharacters'[:9]] * 10)]
    expected += ['tomanycharacters'[:8] + '10']
    result = rename_fields_to_10_characters(columns)
    assert set([len(s) for s in result]) == {10}
    assert result == expected
    f = '{}/fields.dbf'.format(test_output_path)
    df = pd.DataFrame(dict(zip(columns, [[1, 2]]* len(columns))))
    df2shp(df, f)
    df2 = shp2df(f)
    assert df2.columns.tolist() == expected


@pytest.fixture(scope='module')
def eel_river_polygon(test_output_path):
    polygon_wkt = ('POLYGON ((-2345010.181299999 2314860.9384, '
                   '-2292510.181299999 2314860.9384, -2292510.181299999 2281360.9384, '
                   '-2345010.181299999 2281360.9384, -2345010.181299999 2314860.9384))')
    polygon = shapely.wkt.loads(polygon_wkt)
    return polygon


@pytest.fixture(scope='module')
def eel_river_polygon_shapefile(test_output_path, eel_river_polygon):
    df = pd.DataFrame({'geometry': [eel_river_polygon],
                       'id': [0]})
    outfile = os.path.join(test_output_path, 'bbox.shp')

    # write out to 5070
    df2shp(df, outfile, epsg=5070)
    return outfile


def test_get_shapefile_crs(eel_river_polygon_shapefile):
    crs = get_shapefile_crs(eel_river_polygon_shapefile)
    expected = pyproj.crs.CRS.from_epsg(5070)
    assert crs == expected


crs_test_params = (None,
                   5070,
                   'epsg:26910',
                   'epsg:4269',
                   # an example of an uncommon CRS
                   ('PROJCS["NAD_1983_California_Teale_Albers",'
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
                    'UNIT["Meter",1.0]]')
                                      )


@pytest.mark.parametrize('dest_crs', crs_test_params)
def test_shp2df_df2shp_crs(dest_crs, test_output_path, eel_river_polygon,
                eel_river_polygon_shapefile, request):

    # read in to dest_crs
    df_dest_crs = shp2df(eel_river_polygon_shapefile, dest_crs=dest_crs)

    # reproject back to 5070
    if dest_crs is not None:
        geoms = project(df_dest_crs['geometry'], dest_crs, 5070)
    else:
        geoms = df_dest_crs['geometry']
    # verify that polygon is the same as original in 5070
    assert geoms[0].almost_equals(eel_river_polygon)

    # check that when writing the polygon back to a shapefile
    # a valid projection file is produced
    output_shapefile = os.path.join(test_output_path, 'results.shp')
    df2shp(df_dest_crs, output_shapefile, crs=dest_crs)
    written_crs = get_shapefile_crs(output_shapefile)
    if dest_crs is not None:
        assert written_crs == get_authority_crs(dest_crs)

