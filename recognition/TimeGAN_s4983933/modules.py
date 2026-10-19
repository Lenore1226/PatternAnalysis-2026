
"""
COMP3710 Project 7 - TimeGAN network modules.

Implements the five neural network components required
for TimeGAN-based limit order book generation.
"""

import torch
from torch import nn


class GRUNetwork(nn.Module):
    """GRU followed by a linear output projection."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        output_activation="sigmoid",
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.projection = nn.Linear(
            hidden_dim, output_dim
        )

        if output_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif output_activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(
                f"Unknown activation: {output_activation}"
            )

    def forward(self, x):
        hidden_states, _ = self.gru(x)
        output = self.projection(hidden_states)
        return self.activation(output)


class TimeGAN(nn.Module):
    """TimeGAN network components and forward operations."""

    def __init__(
        self,
        feature_dim=40,
        hidden_dim=24,
        noise_dim=40,
        num_layers=2,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim

        # Encode real observations into latent space.
        self.embedder = GRUNetwork(
            feature_dim, hidden_dim, hidden_dim,
            num_layers=num_layers,
        )

        # Recover observations from latent states.
        self.recovery = GRUNetwork(
            hidden_dim, hidden_dim, feature_dim,
            num_layers=num_layers,
        )

        # Generate latent states from random noise.
        self.generator = GRUNetwork(
            noise_dim, hidden_dim, hidden_dim,
            num_layers=num_layers,
        )

        # Learn temporal dynamics in latent space.
        self.supervisor = GRUNetwork(
            hidden_dim, hidden_dim, hidden_dim,
            num_layers=num_layers - 1,
        )

        # Distinguish real and generated latent states.
        self.discriminator = GRUNetwork(
            hidden_dim, hidden_dim, 1,
            num_layers=num_layers,
            output_activation="none",
        )

    def encode(self, x):
        return self.embedder(x)

    def reconstruct(self, x):
        h = self.embedder(x)
        return self.recovery(h)

    def generate(self, z):
        e_hat = self.generator(z)
        h_hat = self.supervisor(e_hat)
        return self.recovery(h_hat)

    def discriminate(self, h):
        return self.discriminator(h)


if __name__ == "__main__":
    torch.manual_seed(42)

    model = TimeGAN(
        feature_dim=40,
        hidden_dim=24,
        noise_dim=40,
    )

    batch_size = 8
    seq_len = 24

    x = torch.rand(batch_size, seq_len, 40)
    z = torch.rand(batch_size, seq_len, 40)

    reconstructed = model.reconstruct(x)
    generated = model.generate(z)

    latent = model.encode(x)
    logits = model.discriminate(latent)

    print("Input:", tuple(x.shape))
    print("Reconstructed:", tuple(reconstructed.shape))
    print("Generated:", tuple(generated.shape))
    print("Discriminator:", tuple(logits.shape))

    assert reconstructed.shape == x.shape
    assert generated.shape == x.shape
    assert logits.shape == (batch_size, seq_len, 1)

    print("All TimeGAN module shape checks passed!")
