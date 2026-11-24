"""Train the same network in Keras, with compile and fit, and write its reports.

Run `python train_keras.py`. It uses the same rows, loss, optimizer, learning
rate, batch size, number of epochs and seed as the PyTorch main run in train.py,
and recreates reports/history_keras.csv, reports/learning_curves_keras.png and
reports/metrics_keras.json. The best model is saved to models/ (not committed).
"""

import json
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from keras_model import build_keras_net, predict_keras
from prepare import SEED, Splits, make_splits
from train import MAIN_RUN, REPORTS, best_epoch, plot_curves

CHECKPOINT = Path(__file__).parent / "models" / "bike_net_keras.keras"
# PyTorch's Adam uses eps=1e-8 and Keras's Adam uses 1e-7, so set it to match.
TORCH_ADAM_EPSILON = 1e-8
KERAS_TO_TORCH_NAMES = {
    "loss": "train_loss",
    "mae": "train_mae",
    "val_loss": "val_loss",
    "val_mae": "val_mae",
}


def keras_arrays(splits: Splits) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The arrays behind the PyTorch tensors. The data are not prepared again."""
    parts = {"train": splits.train, "val": splits.val, "test": splits.test}
    return {name: (data.x.numpy(), data.y.numpy()) for name, data in parts.items()}


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error. The prediction is flattened first: (N, 1) becomes (N,).

    Without the flatten, (N, 1) minus (N,) broadcasts to an N by N table.
    """
    return float(np.mean(np.abs(prediction.reshape(-1) - target)))


def to_history(logs: dict[str, list[float]]) -> list[dict]:
    """Turn Keras's logs into the same rows and columns as the PyTorch history."""
    return [
        {"epoch": epoch + 1}
        | {new: float(logs[old][epoch]) for old, new in KERAS_TO_TORCH_NAMES.items()}
        for epoch in range(len(logs["loss"]))
    ]


def print_every(n: int) -> keras.callbacks.Callback:
    """Print the same progress line as train.py, on epoch 1 and every n epochs."""

    def show(epoch: int, logs: dict) -> None:
        if epoch == 0 or (epoch + 1) % n == 0:
            print(
                f"epoch {epoch + 1:3d}  train loss {logs['loss']:8.2f}  "
                f"val loss {logs['val_loss']:8.2f}  val MAE {logs['val_mae']:6.2f}"
            )

    return keras.callbacks.LambdaCallback(on_epoch_end=show)


def train_keras(
    settings: dict,
    splits: Splits,
    seed: int = SEED,
    checkpoint: Path | None = None,
    verbose: bool = False,
) -> tuple[keras.Model, list[dict]]:
    """Build a fresh, seeded Keras network and train it with compile and fit.

    EarlyStopping has a patience equal to the number of epochs, so it never
    stops early. It only puts back the weights of the best epoch at the end.
    """
    keras.utils.set_random_seed(seed)
    data = keras_arrays(splits)
    model = build_keras_net(splits.n_features, settings["hidden"], settings["n_layers"])
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=settings["lr"], epsilon=TORCH_ADAM_EPSILON
        ),
        loss="mse",
        metrics=["mae"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=settings["epochs"], restore_best_weights=True
        )
    ]
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint, monitor="val_loss", save_best_only=True
            )
        )
    if verbose:
        callbacks.append(print_every(10))
    x_train, y_train = data["train"]
    fit = model.fit(
        x_train,
        y_train,
        validation_data=data["val"],
        epochs=settings["epochs"],
        batch_size=settings["batch_size"],
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    return model, to_history(fit.history)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    splits = make_splits(SEED)
    print("Main run (Keras):", MAIN_RUN)
    started = time.perf_counter()
    model, history = train_keras(MAIN_RUN, splits, checkpoint=CHECKPOINT, verbose=True)
    seconds = time.perf_counter() - started

    pd.DataFrame(history).round(4).to_csv(REPORTS / "history_keras.csv", index=False)
    plot_curves(history, "Main run (Keras)", REPORTS / "learning_curves_keras.png")

    # The test set, used once, for the weights of the best epoch.
    x_test, y_test = keras_arrays(splits)["test"]
    best = best_epoch(history)
    metrics = {
        "val_mae": round(history[best - 1]["val_mae"], 3),
        "test_mae": round(mae(predict_keras(model, x_test), y_test), 3),
        "best_epoch": best,
        "seconds": round(seconds, 1),
        "parameters": model.count_params(),
        "checkpoint_kb": round(CHECKPOINT.stat().st_size / 1024, 1),
        "keras_version": keras.__version__,
    }
    (REPORTS / "metrics_keras.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
