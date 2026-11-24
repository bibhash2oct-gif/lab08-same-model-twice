"""Tests for the Keras version and the comparison."""

import dataclasses

import keras
import numpy as np
import pytest
import torch
from torch import nn

from compare import COLUMNS, compare
from keras_model import (
    TOLERANCE,
    build_keras_net,
    copy_torch_into_keras,
    count_torch_parameters,
    largest_gap,
    predict_keras,
    predict_torch,
)
from model import BikeNet
from prepare import BikeDataset, make_loaders
from train import MAIN_RUN, best_epoch
from train_keras import keras_arrays, mae, train_keras

SHORT = {**MAIN_RUN, "epochs": 3}


def squared_error(model, x, y):
    return float(np.mean((predict_keras(model, x) - y) ** 2))


def rising_validation(splits):
    """The same rows, but every validation target is zero.

    As the network learns real counts, its validation loss can only get worse,
    so the best epoch comes before the last one.
    """
    zeros = np.zeros(len(splits.val), dtype=np.float32)
    return dataclasses.replace(splits, val=BikeDataset(splits.val.x.numpy(), zeros))


def test_keras_net_returns_one_value_per_row(splits):
    x = splits.val.x.numpy()[:10]
    net = build_keras_net(splits.n_features)
    assert net.predict(x, verbose=0).shape == (10, 1)
    assert predict_keras(net, x).shape == (10,)


@pytest.mark.parametrize("hidden, n_layers", [(1, 1), (8, 2), (64, 2), (100, 3)])
def test_parameter_count_matches_pytorch(hidden, n_layers):
    torch_net = BikeNet(n_features=8, hidden=hidden, n_layers=n_layers)
    keras_net = build_keras_net(n_features=8, hidden=hidden, n_layers=n_layers)
    assert keras_net.count_params() == count_torch_parameters(torch_net)


@pytest.mark.parametrize("hidden", [8, 64])
def test_weight_copy_gives_the_same_predictions(splits, hidden):
    # With hidden=8 the first two kernels are square (8 by 8), so a copy that
    # forgot to transpose would not fail on shape. Only the values can catch it.
    torch.manual_seed(0)
    torch_net = BikeNet(splits.n_features, hidden)
    keras_net = build_keras_net(splits.n_features, hidden)
    copy_torch_into_keras(torch_net, keras_net)
    x_val = splits.val.x.numpy()
    assert np.std(predict_torch(torch_net, x_val)) > 1e-3  # not a constant output
    assert largest_gap(torch_net, keras_net, x_val) < TOLERANCE


def test_a_copy_without_the_transpose_is_caught(splits):
    torch.manual_seed(0)
    torch_net = BikeNet(splits.n_features, hidden=8)
    keras_net = build_keras_net(splits.n_features, hidden=8)
    linears = [m for m in torch_net.modules() if isinstance(m, nn.Linear)]
    for linear, dense in zip(linears, keras_net.layers, strict=True):
        wrong = linear.weight.detach().numpy().reshape(dense.kernel.shape)
        dense.set_weights([wrong, linear.bias.detach().numpy()])
    assert largest_gap(torch_net, keras_net, splits.val.x.numpy()) > 100 * TOLERANCE


def test_both_frameworks_get_the_same_rows(splits):
    arrays = keras_arrays(splits)
    for name in ("train", "val", "test"):
        tensors = getattr(splits, name)
        x, y = arrays[name]
        assert x.dtype == np.float32 and y.dtype == np.float32
        assert np.array_equal(x, tensors.x.numpy())
        assert np.array_equal(y, tensors.y.numpy())
    _, val_loader, test_loader = make_loaders(splits)
    for name, loader in (("val", val_loader), ("test", test_loader)):
        x_batches = torch.cat([x for x, _ in loader]).numpy()
        assert np.array_equal(arrays[name][0], x_batches)


def test_keras_net_can_memorize_one_small_batch(splits):
    keras.utils.set_random_seed(0)
    x, y = keras_arrays(splits)["train"]
    x, y = x[:32], y[:32]
    net = build_keras_net(splits.n_features)
    net.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
    start = squared_error(net, x, y)
    net.fit(x, y, epochs=300, batch_size=32, verbose=0)
    assert squared_error(net, x, y) < 0.05 * start


def test_saved_and_reloaded_model_predicts_the_same(splits, tmp_path):
    keras.utils.set_random_seed(0)
    x, y = keras_arrays(splits)["train"]
    net = build_keras_net(splits.n_features)
    net.compile(optimizer="adam", loss="mse")
    net.fit(x[:256], y[:256], epochs=2, verbose=0)
    path = tmp_path / "bike_net.keras"
    net.save(path)
    loaded = keras.models.load_model(path)
    x_val = splits.val.x.numpy()
    assert np.array_equal(predict_keras(net, x_val), predict_keras(loaded, x_val))


def test_best_weights_are_restored(splits):
    rising = rising_validation(splits)
    model, history = train_keras(SHORT, rising, seed=0)
    best = best_epoch(history)
    assert best < len(history)
    x_val, y_val = keras_arrays(rising)["val"]
    score = squared_error(model, x_val, y_val)
    assert score == pytest.approx(history[best - 1]["val_loss"], rel=1e-4)
    assert score < history[-1]["val_loss"]


def test_checkpoint_holds_the_best_epoch(splits, tmp_path):
    rising = rising_validation(splits)
    path = tmp_path / "best.keras"
    _, history = train_keras(SHORT, rising, seed=0, checkpoint=path)
    saved = keras.models.load_model(path)
    x_val, y_val = keras_arrays(rising)["val"]
    best_loss = min(row["val_loss"] for row in history)
    assert squared_error(saved, x_val, y_val) == pytest.approx(best_loss, rel=1e-4)


def test_history_logs_every_epoch(splits):
    _, history = train_keras(SHORT, splits, seed=0)
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert {"train_loss", "val_loss", "val_mae"} <= set(history[0])


def test_same_seed_repeats_a_short_run(splits):
    _, first = train_keras(SHORT, splits, seed=5)
    _, again = train_keras(SHORT, splits, seed=5)
    _, other = train_keras(SHORT, splits, seed=6)
    losses = [row["val_loss"] for row in first]
    assert [row["val_loss"] for row in again] == pytest.approx(losses, rel=1e-5)
    assert [row["val_loss"] for row in other] != pytest.approx(losses, rel=1e-5)


def test_mae_flattens_keras_predictions():
    prediction = np.array([[1.0], [2.0], [4.0]])  # shape (3, 1), like Keras
    target = np.array([1.0, 2.0, 3.0])  # shape (3,)
    assert mae(prediction, target) == pytest.approx(1 / 3)
    # Without the flatten, numpy compares every prediction with every target.
    assert np.mean(np.abs(prediction - target)) != pytest.approx(1 / 3)


def test_comparison_trains_both_frameworks_on_every_seed():
    table, check = compare({**MAIN_RUN, "epochs": 2}, seeds=(1, 2))
    assert list(table.columns) == COLUMNS
    assert sorted(table["framework"]) == ["keras", "keras", "pytorch", "pytorch"]
    assert sorted(table["seed"]) == [1, 1, 2, 2]
    assert table["best_epoch"].between(1, 2).all()
    assert (table["test_mae"] > 0).all() and (table["seconds"] > 0).all()
    assert check["same_function"]
