from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture()
def test_data_mrc_dir():
    """Fixture to provide the MRC test data directory."""
    return Path(Path(__file__).parent.joinpath("testdata_mrc"))


@pytest.fixture()
def test_data_npy_dir():
    """Fixture to provide the NPY test data directory."""
    return Path(Path(__file__).parent.joinpath("testdata_npy"))


@pytest.fixture()
def test_corrupt_file():
    """Fixture to provide the path to a corrupt file for testing."""
    return Path(__file__).parent / "corrupt.mrc"


@pytest.fixture()
def test_data_single_mrc_dir():
    """Fixture to provide a single MRC file for testing."""
    return Path(Path(__file__).parent.joinpath("testdata_mrc", "mrc"))


@pytest.fixture()
def test_data_single_mrc_temp_dir():
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        test_data_single_mrc_dir = Path(
            Path(__file__).parent.joinpath("testdata_mrc", "mrc")
        )
        for file in test_data_single_mrc_dir.glob("*"):
            shutil.copy(file, temp_dir_path)
        yield temp_dir_path
