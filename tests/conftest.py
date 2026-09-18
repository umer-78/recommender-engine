from pathlib import Path

import pytest

from recs import load, temporal_split

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def dataset():
    return load(DATA)


@pytest.fixture(scope="session")
def split(dataset):
    return temporal_split(dataset, holdout=3, minimum_history=5)
