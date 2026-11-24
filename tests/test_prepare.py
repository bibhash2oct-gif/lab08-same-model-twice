"""Tests for the split, the preparation, the Dataset and the loaders."""

import torch

from prepare import NUMERIC, make_loaders


def test_dataset_length_and_dtypes(splits):
    x, y = splits.train[0]
    assert len(splits.train) == 3600
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert x.shape == (splits.n_features,)
    assert y.shape == ()


def test_splits_add_up_and_do_not_overlap(splits):
    train = set(splits.train_rows.index)
    val = set(splits.val_rows.index)
    test = set(splits.test_rows.index)
    assert (len(train), len(val), len(test)) == (3600, 1200, 1200)
    assert len(train | val | test) == 6000
    assert not train & val
    assert not train & test
    assert not val & test


def test_scaler_fitted_on_training_rows_only(splits):
    n = len(NUMERIC)
    train_means = splits.train.x[:, :n].mean(dim=0)
    val_means = splits.val.x[:, :n].mean(dim=0)
    assert torch.allclose(train_means, torch.zeros(n), atol=1e-5)
    assert val_means.abs().max() > 1e-3


def test_training_loader_shuffles_but_is_repeatable(splits):
    first, _, _ = make_loaders(splits, seed=7)
    second, _, _ = make_loaders(splits, seed=7)
    _, y_first = next(iter(first))
    _, y_second = next(iter(second))
    assert torch.equal(y_first, y_second)
    assert not torch.equal(y_first, splits.train.y[:64])
