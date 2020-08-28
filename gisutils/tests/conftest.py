import os
import shutil
import pytest


@pytest.fixture(scope="module", autouse=True)
def test_output_path():
    folder = 'gisutils/tests/output'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return folder
