"""COMP3710 Project 7: generate and visualise V2 synthetic order books.

Uses the architecture-aware load_model() in evaluate.py and the V2
preprocessing used when the checkpoint was trained.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_v2 import prepare_dataset_v2
from evaluate import generate_samples, load_model


def depth_image(decoded_sequence):
    """Return (time, 20) order-size heatmap in log1p(volume) units."""
    volumes = np.maximum(decoded_sequence[:, 1::2], 0.0)
    return np.log1p(volumes)


def plot_comparison(real, fake, destination):
    """Use a shared colour scale; samples are NOT time-aligned."""
    real_img = depth_image(real)
    fake_img = depth_image(fake)
    vmax = max(float(np.max(real_img)), float(np.max(fake_img)), 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    im = axes[0, 0].imshow(real_img.T, origin="lower", aspect="auto",
                           vmin=0, vmax=vmax, interpolation="nearest")
    axes[0, 0].set_title("Real test sequence — depth")
    axes[0, 1].imshow(fake_img.T, origin="lower", aspect="auto",
                      vmin=0, vmax=vmax, interpolation="nearest")
    axes[0, 1].set_title("Generated sequence — depth")
    fig.colorbar(im, ax=[axes[0, 0], axes[0, 1]], label="log(1 + volume)")

    real_spread = real[:, 0] - real[:, 2]
    fake_spread = fake[:, 0] - fake[:, 2]
    axes[1, 0].plot(real_spread, label="Real")
    axes[1, 0].plot(fake_spread, label="Generated")
    axes[1, 0].set_title("Best bid–ask spread")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].set_ylabel("Price difference (USD)")

    axes[1, 1].plot(real_img.mean(axis=1), label="Real")
    axes[1, 1].plot(fake_img.mean(axis=1), label="Generated")
    axes[1, 1].set_title("Mean log-volume across 20 sides/levels")
    axes[1, 1].set_xlabel("Time step")
    axes[1, 1].legend()
    fig.suptitle("Independent real and synthetic sequences (NOT aligned forecasts)")
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs_predict")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be >= 1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    version = checkpoint.get("dataset_version", "v1")
    if version != "v2":
        raise ValueError("This prediction visualisation requires a V2 checkpoint")
    dataset = prepare_dataset_v2(args.data_dir)

    test = dataset["test"]
    if test.shape[-1] != checkpoint["feature_dim"]:
        raise ValueError("Checkpoint and dataset feature counts do not match")
    generated = generate_samples(model, args.count, test.shape[1], device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "generated_normalized.npy", generated)

    # decode to relative price offsets and raw volumes for V2
    transform = dataset["transform"]
    real_decoded = transform.inverse_transform(test[:1])[0]
    fake_decoded = transform.inverse_transform(generated[:1])[0]
    plot_comparison(real_decoded, fake_decoded, output_dir / "prediction_comparison.png")

    print("Checkpoint:", args.checkpoint)
    print("Model version:", checkpoint.get("model_version", "original"))
    print("Dataset version:", version)
    print("Generated shape:", generated.shape)
    print("Mean generated best spread:", float(np.mean(fake_decoded[:, 0] - fake_decoded[:, 2])))
    print("Saved:", output_dir / "generated_normalized.npy")
    print("Saved:", output_dir / "prediction_comparison.png")
    if version == "v2":
        print("Note: V2 prices are mid-price offsets, not absolute price trajectories.")


if __name__ == "__main__":
    main()
