"""Discriminador do STEP-GAN (Camada 3): é o detector de anomalias usado na inferência."""

import torch.nn as nn


class Discriminator(nn.Module):
    """Saída ≈ 0.9 para transação normal e ≈ 0.1 para anomalia; o escore de risco é 1 - D(x)."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
