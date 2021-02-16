import os
from pathlib import Path
import shutil
import pytest


@pytest.fixture(scope="session")
def project_root_path():
    filepath = os.path.split(os.path.abspath(__file__))[0]
    return Path(os.path.normpath(os.path.join(filepath, '../../')))


@pytest.fixture(scope="module", autouse=True)
def test_output_path():
    folder = 'gisutils/tests/output'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return Path(folder)
