"""Tests for the network."""

import pytest
import torch

from model import BikeNet


@pytest.mark.parametrize("hidden", [4, 32, 256])
def test_network_returns_one_value_per_row(hidden):
    model = BikeNet(n_features=8, hidden=hidden)
    output = model(torch.randn(10, 8))
    assert output.shape == (10,)


def test_network_has_two_hidden_layers_and_no_final_activation():
    model = BikeNet(n_features=8)
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3
    assert isinstance(model.layers[-1], torch.nn.Linear)
