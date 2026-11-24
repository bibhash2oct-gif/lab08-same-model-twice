"""Train the bike-demand network with a hand-written loop and write the reports.

Run `python train.py`. It recreates every file in reports/: the learning
curves, the history of the main run, the experiments table and the metrics.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader

from model import BikeNet
from prepare import SEED, Splits, make_loaders, make_splits

REPORTS = Path(__file__).parent / "reports"
TRAIN_COLOUR, VAL_COLOUR = "#2a78d6", "#eb6834"

# The network I believe in. Each experiment changes exactly one of these.
MAIN_RUN = {"hidden": 64, "n_layers": 2, "lr": 3e-3, "batch_size": 64, "epochs": 150}
EXPERIMENTS = {
    "lr_too_small": {"lr": 1e-5},
    "lr_too_large": {"lr": 0.3},
    "very_wide": {"hidden": 512},
}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over a loader. Trains if given an optimizer, otherwise measures.

    Returns the mean loss and the mean absolute error, in rentals.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, total_abs_error, count = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:  # 1. a batch
            prediction = model(x)  # 2. forward
            loss = loss_fn(prediction, y)  # 3. loss
            if training:
                optimizer.zero_grad()
                loss.backward()  # 4. backward
                optimizer.step()  # 5. step
            total_loss += loss.item() * len(y)
            total_abs_error += (prediction - y).abs().sum().item()
            count += len(y)
    return total_loss / count, total_abs_error / count


def best_epoch(history: list[dict]) -> int:
    """The epoch with the lowest validation loss (ignoring any nan)."""
    finite = [row for row in history if np.isfinite(row["val_loss"])]
    return min(finite, key=lambda row: row["val_loss"])["epoch"]


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    verbose: bool = False,
) -> list[dict]:
    """Train for many epochs, log every epoch, and put back the best weights."""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_loss, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"epoch {epoch:3d}  train loss {train_loss:8.2f}  "
                f"val loss {val_loss:8.2f}  val MAE {val_mae:6.2f}"
            )
    if best_state is not None:
        model.load_state_dict(best_state)  # go back to the best epoch
    return history


def run(settings: dict, splits: Splits, verbose: bool = False):
    """Build a fresh, seeded network and train it with the given settings."""
    torch.manual_seed(SEED)
    train_loader, val_loader, _ = make_loaders(
        splits, batch_size=settings["batch_size"], seed=SEED
    )
    model = BikeNet(splits.n_features, settings["hidden"], settings["n_layers"])
    history = train(
        model, train_loader, val_loader, settings["epochs"], settings["lr"], verbose
    )
    return model, history


def summarize(name: str, change: str, history: list[dict]) -> dict:
    """One row of the experiments table."""
    best = best_epoch(history)
    return {
        "run": name,
        "change": change,
        "best_epoch": best,
        "best_val_mae": round(history[best - 1]["val_mae"], 3),
        "final_train_loss": round(history[-1]["train_loss"], 3),
        "final_val_loss": round(history[-1]["val_loss"], 3),
    }


def baselines(splits: Splits) -> dict:
    """Two simple models on exactly the same rows and features."""
    x_train, y_train = splits.train.x.numpy(), splits.train.y.numpy()
    scored = {"val": splits.val, "test": splits.test}

    mean = float(y_train.mean())
    linear = LinearRegression().fit(x_train, y_train)
    results = {"training_mean": {}, "linear_regression": {}}
    for split, data in scored.items():
        y_true = data.y.numpy()
        results["training_mean"][f"{split}_mae"] = mean_absolute_error(
            y_true, np.full(len(y_true), mean)
        )
        results["linear_regression"][f"{split}_mae"] = mean_absolute_error(
            y_true, linear.predict(data.x.numpy())
        )
    return results


def plot_curves(history: list[dict], title: str, path: Path) -> None:
    """Training and validation loss for every epoch, with the best epoch marked."""
    frame = pd.DataFrame(history)
    best = best_epoch(history)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame["epoch"], frame["train_loss"], color=TRAIN_COLOUR, linewidth=2)
    ax.plot(frame["epoch"], frame["val_loss"], color=VAL_COLOUR, linewidth=2)
    ax.axvline(best, color="#52514e", linestyle="--", linewidth=1)
    ax.legend(["training loss", "validation loss", f"best epoch ({best})"])
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error (log scale)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    splits = make_splits(SEED)
    metrics = baselines(splits)

    print("Main run:", MAIN_RUN)
    model, history = run(MAIN_RUN, splits, verbose=True)
    pd.DataFrame(history).round(4).to_csv(REPORTS / "history.csv", index=False)
    plot_curves(history, "Main run", REPORTS / "learning_curves.png")
    rows = [summarize("main", "none (the main run)", history)]

    for name, change in EXPERIMENTS.items():
        settings = {**MAIN_RUN, **change}
        print(f"Experiment {name}: {change}")
        _, run_history = run(settings, splits)
        label = ", ".join(f"{key}={value}" for key, value in change.items())
        plot_curves(run_history, f"{name}: {label}", REPORTS / f"curves_{name}.png")
        rows.append(summarize(name, label, run_history))
    experiments = pd.DataFrame(rows)
    experiments.to_csv(REPORTS / "experiments.csv", index=False)

    # The test set, used once, for the network chosen on validation rows.
    _, _, test_loader = make_loaders(splits)
    _, test_mae = run_epoch(model, test_loader, nn.MSELoss())
    best = best_epoch(history)
    metrics["network"] = {
        "val_mae": history[best - 1]["val_mae"],
        "test_mae": test_mae,
        "best_epoch": best,
    }
    metrics = {
        name: {key: round(value, 3) for key, value in scores.items()}
        for name, scores in metrics.items()
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print()
    print(experiments.to_string(index=False))
    print()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
