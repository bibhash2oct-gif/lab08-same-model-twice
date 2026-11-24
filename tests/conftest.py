"""Shared fixtures: the splits are built once and reused by every test."""

import pytest

from prepare import make_loaders, make_splits


@pytest.fixture(scope="session")
def splits():
    return make_splits()


@pytest.fixture
def loaders(splits):
    return make_loaders(splits, batch_size=64)
