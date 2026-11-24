"""Train the main configuration in both frameworks, on several seeds, and compare.

Run `python compare.py`. First it checks that the two networks compute the same
function. Then it trains six models (two frameworks, three seeds) on the same
rows with the same settings, scores each final model on the test rows once, and
writes reports/comparison.csv and reports/equivalence.json. It takes a few
minutes.
"""

import json
import time

import pandas as pd
import torch
from torch import nn

from keras_model import TOLERANCE, equivalence_check, predict_keras
from model import BikeNet
from prepare import SEED, Splits, make_loaders, make_splits
from train import MAIN_RUN, REPORTS, best_epoch, run, run_epoch
from train_keras import keras_arrays, mae, train_keras

SEEDS = (42, 1, 2)
COLUMNS = ["framework", "seed", "best_epoch", "val_mae", "test_mae", "seconds"]


def pytorch_run(settings: dict, splits: Splits, seed: int) -> tuple[BikeNet, dict]:
    """Train with the Lab 07 loop, then score the best weights on the test rows."""
    started = time.perf_counter()
    model, history = run(settings, splits, seed=seed)
    seconds = time.perf_counter() - started
    _, _, test_loader = make_loaders(splits)
    _, test_mae = run_epoch(model, test_loader, nn.MSELoss())
    best = best_epoch(history)
    row = {
        "framework": "pytorch",
        "seed": seed,
        "best_epoch": best,
        "val_mae": history[best - 1]["val_mae"],
        "test_mae": test_mae,
        "seconds": seconds,
    }
    return model, row


def keras_run(settings: dict, splits: Splits, seed: int) -> dict:
    """Train with compile and fit, then score the best weights on the test rows."""
    started = time.perf_counter()
    model, history = train_keras(settings, splits, seed=seed)
    seconds = time.perf_counter() - started
    x_test, y_test = keras_arrays(splits)["test"]
    best = best_epoch(history)
    return {
        "framework": "keras",
        "seed": seed,
        "best_epoch": best,
        "val_mae": history[best - 1]["val_mae"],
        "test_mae": mae(predict_keras(model, x_test), y_test),
        "seconds": seconds,
    }


def equivalence(splits: Splits, trained: BikeNet) -> dict:
    """The largest prediction gap on the validation rows, before and after training."""
    x_val = splits.val.x.numpy()
    torch.manual_seed(SEED)
    untrained = BikeNet(splits.n_features, MAIN_RUN["hidden"], MAIN_RUN["n_layers"])
    untrained_gap = equivalence_check(untrained, x_val)
    trained_gap = equivalence_check(trained, x_val)
    return {
        "untrained_gap": untrained_gap,
        "trained_gap": trained_gap,
        "tolerance": TOLERANCE,
        "same_function": max(untrained_gap, trained_gap) < TOLERANCE,
    }


def compare(
    settings: dict = MAIN_RUN, seeds: tuple[int, ...] = SEEDS
) -> tuple[pd.DataFrame, dict]:
    """Both frameworks on every seed, one after the other, on the same rows."""
    splits = make_splits(SEED)
    rows, check = [], None
    for seed in seeds:
        print(f"seed {seed}: PyTorch ...", flush=True)
        model, row = pytorch_run(settings, splits, seed)
        rows.append(row)
        if check is None:
            check = equivalence(splits, model)
        print(f"seed {seed}: Keras ...", flush=True)
        rows.append(keras_run(settings, splits, seed))
    return pd.DataFrame(rows, columns=COLUMNS), check


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    print("Main configuration:", MAIN_RUN, "seeds:", SEEDS)
    table, check = compare()
    table = table.round({"val_mae": 3, "test_mae": 3, "seconds": 1})
    table.to_csv(REPORTS / "comparison.csv", index=False)
    (REPORTS / "equivalence.json").write_text(json.dumps(check, indent=2) + "\n")

    print()
    print(
        f"Same weights, largest gap: untrained {check['untrained_gap']:.1e}, "
        f"trained {check['trained_gap']:.1e} rentals "
        f"(tolerance {TOLERANCE:.0e}). Same function: {check['same_function']}"
    )
    print()
    print(table.to_string(index=False))
    print()
    summary = table.groupby("framework")[["val_mae", "test_mae", "seconds"]]
    print(summary.agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
