
"""
COMP3710 Project 7 - TimeGAN Training
Stage 1: Embedder and Recovery pretraining
Stage 2: Supervisor pretraining
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import prepare_dataset
from modules import TimeGAN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loader(data, batch_size, shuffle=False):
    tensor = torch.from_numpy(data).float()
    return DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def reconstruction_loss(model, batch, criterion):
    reconstructed = model.reconstruct(batch)
    return criterion(reconstructed, batch)


@torch.no_grad()
def validate_embedding(model, loader, device, criterion):
    model.eval()
    total = 0.0
    count = 0

    for (batch,) in loader:
        batch = batch.to(device)
        loss = reconstruction_loss(model, batch, criterion)
        total += loss.item() * len(batch)
        count += len(batch)

    return total / count


def train_embedding(model, train_loader, val_loader,
                    device, epochs=10, lr=0.001):

    optimizer = torch.optim.Adam(
        list(model.embedder.parameters())
        + list(model.recovery.parameters()),
        lr=lr,
    )
    criterion = nn.MSELoss()

    history = []

    print("\n=== Stage 1: Embedding Pretraining ===",
          flush=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for (batch,) in train_loader:
            batch = batch.to(device)

            loss = reconstruction_loss(
                model, batch, criterion
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * len(batch)

        train_loss = total / len(train_loader.dataset)
        val_loss = validate_embedding(
            model, val_loader, device, criterion
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}",
            flush=True,
        )

    return history, optimizer


def supervised_loss(model, batch, criterion):
    # Freeze Embedder outputs for Supervisor pretraining.
    with torch.no_grad():
        h = model.embedder(batch)

    h_pred = model.supervisor(h)

    return criterion(
        h_pred[:, :-1, :],
        h[:, 1:, :],
    )


@torch.no_grad()
def validate_supervisor(model, loader, device, criterion):
    model.eval()
    total = 0.0
    count = 0

    for (batch,) in loader:
        batch = batch.to(device)
        loss = supervised_loss(model, batch, criterion)
        total += loss.item() * len(batch)
        count += len(batch)

    return total / count


def train_supervisor(model, train_loader, val_loader,
                     device, epochs=10, lr=0.001):

    optimizer = torch.optim.Adam(
        model.supervisor.parameters(),
        lr=lr,
    )
    criterion = nn.MSELoss()

    history = []

    print("\n=== Stage 2: Supervisor Pretraining ===",
          flush=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for (batch,) in train_loader:
            batch = batch.to(device)

            loss = supervised_loss(
                model, batch, criterion
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * len(batch)

        train_loss = total / len(train_loader.dataset)
        val_loss = validate_supervisor(
            model, val_loader, device, criterion
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}",
            flush=True,
        )

    return history, optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--supervisor-epochs",
                        type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir",
                        default="checkpoints")

    args = parser.parse_args()

    set_seed(42)
    start = time.time()

    device = torch.device(args.device)

    result = prepare_dataset(args.data_dir)

    train_loader = create_loader(
        result["train"], args.batch_size, shuffle=True
    )
    val_loader = create_loader(
        result["val"], args.batch_size
    )

    feature_dim = result["train"].shape[2]

    model = TimeGAN(
        feature_dim=feature_dim,
        hidden_dim=24,
        noise_dim=feature_dim,
    ).to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Train sequences:", len(result["train"]))
    print("Validation sequences:", len(result["val"]))

    embedding_history, embedding_optimizer = train_embedding(
        model, train_loader, val_loader,
        device, epochs=args.epochs
    )

    # Save Stage 1 before starting Stage 2.
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": embedding_optimizer.state_dict(),
        "history": embedding_history,
        "feature_dim": feature_dim,
        "hidden_dim": 24,
        "noise_dim": feature_dim,
        "num_layers": 2,
        "stage": "embedding",
    }, output_dir / "embedding.pt")

    supervisor_history, supervisor_optimizer = train_supervisor(
        model, train_loader, val_loader,
        device, epochs=args.supervisor_epochs
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": supervisor_optimizer.state_dict(),
        "embedding_history": embedding_history,
        "supervisor_history": supervisor_history,
        "feature_dim": feature_dim,
        "hidden_dim": 24,
        "noise_dim": feature_dim,
        "num_layers": 2,
        "stage": "supervisor",
    }, output_dir / "supervisor.pt")

    print("\nSaved checkpoints:")
    print(output_dir / "embedding.pt")
    print(output_dir / "supervisor.pt")
    print("Runtime (seconds):",
          round(time.time() - start, 2))
    print("Stage 1 and Stage 2 completed!")


if __name__ == "__main__":
    main()
