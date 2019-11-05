import numpy as np
import pandas as pd
from ..shapefile import (df2shp, shp2df, shp_properties,
                         rename_fields_to_10_characters)


def test_shp_properties():
    df = pd.DataFrame({'reach': [1], 'value': [1.0], 'name': ['stuff']}, index=[0])
    df = df[['name', 'reach', 'value']].copy()
    assert [d.name for d in df.dtypes] == ['object', 'int64', 'float64']
    assert shp_properties(df) == {'name': 'str', 'reach': 'int', 'value': 'float'}


def test_shp_integer_dtypes(tmpdir):

    # verify that pandas is recasting numpy ints as python ints when converting to dict
    # (numpy ints invalid for shapefiles)
    d = pd.DataFrame(np.ones((3, 3)), dtype=int).astype(object).to_dict(orient='records')
    for i in range(3):
        assert isinstance(d[i][0], int)

    df = pd.DataFrame({'r': np.arange(100), 'c': np.arange(100)})
    f = '{}/ints.dbf'.format(tmpdir)
    df2shp(df, f)
    df2 = shp2df(f)
    assert np.all(df == df2)


def test_shp_boolean_dtypes(tmpdir):

    df = pd.DataFrame([False, True]).transpose()
    df.columns = ['true', 'false']
    f = '{}/bool.dbf'.format(tmpdir)
    df2shp(df, f)
    df2 = shp2df(f, true_values='True', false_values='False')
    assert np.all(df == df2)


def test_rename_fields_to_10_characters(tmpdir):
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
    f = '{}/fields.dbf'.format(tmpdir)
    df = pd.DataFrame(dict(zip(columns, [[1, 2]]* len(columns))))
    df2shp(df, f)
    df2 = shp2df(f)
    assert df2.columns.tolist() == expected