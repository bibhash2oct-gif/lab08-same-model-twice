"""Tests for the epoch function and the training function."""

from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from model import BikeNet
from train import best_epoch, run_epoch, train


def copy_weights(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_one_step_changes_every_layer(splits):
    torch.manual_seed(0)
    model = BikeNet(splits.n_features)
    before = copy_weights(model)
    loader = DataLoader(Subset(splits.train, range(64)), batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    run_epoch(model, loader, nn.MSELoss(), optimizer)
    for name, value in model.state_dict().items():
        assert not torch.equal(value, before[name]), f"{name} did not change"


def test_network_can_memorize_one_small_batch(splits):
    torch.manual_seed(0)
    model = BikeNet(splits.n_features)
    loader = DataLoader(Subset(splits.train, range(32)), batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    start, _ = run_epoch(model, loader, loss_fn)
    for _ in range(300):
        run_epoch(model, loader, loss_fn, optimizer)
    end, _ = run_epoch(model, loader, loss_fn)
    assert end < 0.05 * start


def test_zero_grad_and_step_happen_once_per_batch(splits, loaders):
    train_loader, _, _ = loaders
    model = BikeNet(splits.n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with (
        mock.patch.object(optimizer, "zero_grad", wraps=optimizer.zero_grad) as zero,
        mock.patch.object(optimizer, "step", wraps=optimizer.step) as step,
    ):
        run_epoch(model, train_loader, nn.MSELoss(), optimizer)
    assert zero.call_count == len(train_loader)
    assert step.call_count == len(train_loader)


def test_validation_epoch_leaves_weights_unchanged(splits, loaders):
    _, val_loader, _ = loaders
    model = BikeNet(splits.n_features)
    before = copy_weights(model)
    run_epoch(model, val_loader, nn.MSELoss())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_train_logs_every_epoch_and_restores_best_weights(splits, loaders):
    train_loader, val_loader, _ = loaders
    torch.manual_seed(0)
    model = BikeNet(splits.n_features)
    history = train(model, train_loader, val_loader, epochs=4, lr=1e-2)
    assert len(history) == 4
    assert {"epoch", "train_loss", "val_loss", "val_mae"} <= set(history[0])
    best = best_epoch(history)
    val_loss, _ = run_epoch(model, val_loader, nn.MSELoss())
    assert abs(val_loss - history[best - 1]["val_loss"]) < 1e-4
