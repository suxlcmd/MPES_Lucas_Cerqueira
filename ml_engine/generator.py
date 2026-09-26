"""Gerador condicional do STEP-GAN (Camada 3)."""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    G_k(z, τ): gera pseudo-anomalias cuja saída no Discriminador deve ficar no
    limiar τ do degrau k. O limiar entra como condição por uma pequena projeção.
    """

    def __init__(self, input_dim, latent_dim, cond_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition = nn.Sequential(nn.Linear(1, cond_dim), nn.LeakyReLU(0.2))
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, input_dim),
            nn.Tanh(),
        )

    def forward(self, z, tau):
        return self.net(torch.cat([z, self.condition(tau)], dim=1))
