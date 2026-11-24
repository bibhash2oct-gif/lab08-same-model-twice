"""Split the table, prepare the features, and hand them to PyTorch.

The split happens first. The scaler and the encoder are fitted on the
training rows only, then applied to all three splits.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import STATION_TYPES, TARGET, make_rentals

NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL = ["station_type"]
FEATURES = NUMERIC + CATEGORICAL
SEED = 42


class BikeDataset(Dataset):
    """Holds prepared features and targets as float32 tensors, one row at a time."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


@dataclass
class Splits:
    """The three prepared splits, the raw rows behind them and the preprocessor."""

    train: BikeDataset
    val: BikeDataset
    test: BikeDataset
    train_rows: pd.DataFrame
    val_rows: pd.DataFrame
    test_rows: pd.DataFrame
    preprocessor: ColumnTransformer

    @property
    def n_features(self) -> int:
        return self.train.x.shape[1]


def split_rows(
    frame: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into 60% training, 20% validation and 20% test rows."""
    train_rows, rest = train_test_split(frame, test_size=0.4, random_state=seed)
    val_rows, test_rows = train_test_split(rest, test_size=0.5, random_state=seed)
    return train_rows, val_rows, test_rows


def make_preprocessor() -> ColumnTransformer:
    """Standardize the numbers and one-hot encode the station type."""
    return ColumnTransformer(
        [
            ("numbers", StandardScaler(), NUMERIC),
            (
                "station",
                OneHotEncoder(categories=[STATION_TYPES], sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )


def make_splits(seed: int = SEED) -> Splits:
    """Split the table, then fit the preparation on the training rows only."""
    train_rows, val_rows, test_rows = split_rows(make_rentals(), seed)
    preprocessor = make_preprocessor()
    preprocessor.fit(train_rows[FEATURES])

    def to_dataset(rows: pd.DataFrame) -> BikeDataset:
        features = preprocessor.transform(rows[FEATURES])
        return BikeDataset(features, rows[TARGET].to_numpy())

    return Splits(
        train=to_dataset(train_rows),
        val=to_dataset(val_rows),
        test=to_dataset(test_rows),
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        preprocessor=preprocessor,
    )


def make_loaders(
    splits: Splits, batch_size: int = 64, seed: int = SEED
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Shuffle the training batches (repeatably), never the other two."""
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        splits.train, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(splits.val, batch_size=256)
    test_loader = DataLoader(splits.test, batch_size=256)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    splits = make_splits()
    print(
        "rows train / validation / test:",
        len(splits.train),
        len(splits.val),
        len(splits.test),
    )
    print("features per row:", splits.n_features)
    train_loader, _, _ = make_loaders(splits)
    x_batch, y_batch = next(iter(train_loader))
    print("one batch:", x_batch.shape, x_batch.dtype, y_batch.shape, y_batch.dtype)
    print("training batches per epoch:", len(train_loader))
