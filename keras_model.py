"""The Lab 07 network written again in Keras, and the weight copy from PyTorch.

Run `python keras_model.py` to see the Keras network and the equivalence check.
"""

import keras
import numpy as np
import torch
from keras import layers
from torch import nn

from model import BikeNet
from prepare import SEED, make_splits

# Float32 networks that compute the same function agree to about 1e-6 here.
# A copy that forgot to transpose a kernel is off by whole rentals.
TOLERANCE = 1e-4


def build_keras_net(n_features: int, hidden: int = 64, n_layers: int = 2):
    """The same layers as BikeNet: Dense and ReLU, then one output, no activation."""
    model = keras.Sequential(name="bike_net")
    model.add(keras.Input(shape=(n_features,)))
    for _ in range(n_layers):
        model.add(layers.Dense(hidden, activation="relu"))
    model.add(layers.Dense(1))
    return model


def count_torch_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def copy_torch_into_keras(torch_net: BikeNet, keras_net: keras.Model) -> None:
    """Copy every Linear layer into the matching Dense layer.

    PyTorch stores a weight as (outputs, inputs). Keras stores a kernel as
    (inputs, outputs). So each weight is transposed with .T on the way over.
    """
    linears = [m for m in torch_net.modules() if isinstance(m, nn.Linear)]
    denses = [layer for layer in keras_net.layers if isinstance(layer, layers.Dense)]
    for linear, dense in zip(linears, denses, strict=True):
        kernel = linear.weight.detach().cpu().numpy().T
        bias = linear.bias.detach().cpu().numpy()
        dense.set_weights([kernel, bias])


def predict_keras(keras_net: keras.Model, x: np.ndarray) -> np.ndarray:
    """Keras returns one column, shape (N, 1). Flatten it to (N,) like the targets.

    Calling the model directly gives the same numbers as model.predict, without
    TensorFlow building a new graph for every new model.
    """
    output = keras_net(x, training=False)  # a TensorFlow tensor
    return output.numpy().reshape(-1)


def predict_torch(torch_net: BikeNet, x: np.ndarray) -> np.ndarray:
    torch_net.eval()
    with torch.no_grad():
        return torch_net(torch.from_numpy(x)).numpy()


def largest_gap(torch_net: BikeNet, keras_net: keras.Model, x: np.ndarray) -> float:
    """The largest difference, in rentals, between the two networks' predictions."""
    gap = np.abs(predict_torch(torch_net, x) - predict_keras(keras_net, x))
    return float(gap.max())


def equivalence_check(torch_net: BikeNet, x: np.ndarray) -> float:
    """Build a Keras twin of torch_net, copy the weights and measure the gap."""
    width = torch_net.layers[0].out_features
    n_layers = sum(isinstance(m, nn.Linear) for m in torch_net.modules()) - 1
    keras_net = build_keras_net(x.shape[1], width, n_layers)
    copy_torch_into_keras(torch_net, keras_net)
    return largest_gap(torch_net, keras_net, x)


if __name__ == "__main__":
    splits = make_splits(SEED)
    keras_net = build_keras_net(splits.n_features)
    keras_net.summary()
    torch.manual_seed(SEED)
    torch_net = BikeNet(splits.n_features)
    print("PyTorch parameters:", count_torch_parameters(torch_net))
    print("Keras parameters:  ", keras_net.count_params())
    gap = equivalence_check(torch_net, splits.val.x.numpy())
    print(f"Same weights, largest gap on the validation rows: {gap:.1e} rentals")
    print("Same function:", gap < TOLERANCE)
