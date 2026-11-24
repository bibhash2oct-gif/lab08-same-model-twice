"""The network: a stack of Linear layers with ReLU, ending in one number."""

import torch
from torch import nn


class BikeNet(nn.Module):
    """Predicts the number of bikes rented. Returns one value per row.

    There is no activation at the end, because the target is a count and not a
    probability.
    """

    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width_in = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(width_in, hidden), nn.ReLU()]
            width_in = hidden
        layers.append(nn.Linear(width_in, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)


if __name__ == "__main__":
    model = BikeNet(n_features=8)
    print(model)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    print("output shape for 5 rows:", model(torch.randn(5, 8)).shape)
