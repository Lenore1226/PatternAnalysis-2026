
"""
COMP3710 Project 7 - TimeGAN
LOBSTER Level-10 data loading and preprocessing.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_lobster(data_dir):
    """Load matching order book and message CSV files."""
    data_dir = Path(data_dir)

    orderbook_files = list(data_dir.glob("*orderbook_10.csv"))
    message_files = list(data_dir.glob("*message_10.csv"))

    if len(orderbook_files) != 1 or len(message_files) != 1:
        raise FileNotFoundError(
            "Expected one Level-10 orderbook and message CSV pair."
        )

    orderbook = pd.read_csv(orderbook_files[0], header=None)
    messages = pd.read_csv(message_files[0], header=None)

    if orderbook.shape[1] != 40:
        raise ValueError("Level-10 orderbook must have 40 columns.")

    if messages.shape[1] != 6:
        raise ValueError("LOBSTER message file must have 6 columns.")

    if len(orderbook) != len(messages):
        raise ValueError("Orderbook and message row counts differ.")

    return orderbook, messages


def chronological_split(data, train_ratio=0.70, val_ratio=0.15):
    """Split observations in chronological order."""
    n = len(data)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        data[:train_end],
        data[train_end:val_end],
        data[val_end:],
    )


class MinMaxNormalizer:
    """Fit normalization parameters on training data only."""

    def fit(self, train):
        self.minimum = train.min(axis=0)
        maximum = train.max(axis=0)
        self.scale = np.maximum(maximum - self.minimum, 1e-8)
        return self

    def transform(self, data):
        return (data - self.minimum) / self.scale

    def inverse_transform(self, data):
        return data * self.scale + self.minimum


def make_sequences(data, seq_len=24, stride=24):
    """Construct fixed-length temporal windows."""
    if seq_len <= 0 or stride <= 0:
        raise ValueError("seq_len and stride must be positive.")

    if len(data) < seq_len:
        return np.empty(
            (0, seq_len, data.shape[1]), dtype=np.float32
        )

    starts = range(0, len(data) - seq_len + 1, stride)

    return np.stack(
        [data[i:i + seq_len] for i in starts]
    ).astype(np.float32)


def prepare_dataset(data_dir, seq_len=24, stride=24):
    """Build chronologically separated TimeGAN sequences."""
    orderbook, messages = load_lobster(data_dir)

    # LOBSTER stores prices multiplied by 10,000.
    features = orderbook.to_numpy(dtype=np.float64)
    features[:, 0::2] /= 10000.0

    # Compress the right-skewed volume distribution.
    features[:, 1::2] = np.log1p(features[:, 1::2])

    train, val, test = chronological_split(features)

    normalizer = MinMaxNormalizer().fit(train)

    train_seq = make_sequences(
        normalizer.transform(train), seq_len, stride
    )
    val_seq = make_sequences(
        normalizer.transform(val), seq_len, stride
    )
    test_seq = make_sequences(
        normalizer.transform(test), seq_len, stride
    )

    return {
        "train": train_seq,
        "val": val_seq,
        "test": test_seq,
        "normalizer": normalizer,
        "timestamps": messages.iloc[:, 0].to_numpy(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=24)
    args = parser.parse_args()

    result = prepare_dataset(
        args.data_dir, seq_len=args.seq_len
    )

    for split in ("train", "val", "test"):
        data = result[split]
        print(f"{split}: {data.shape}")
        print(f"NaN count: {np.isnan(data).sum()}")

    print("Dataset preprocessing completed successfully!")
