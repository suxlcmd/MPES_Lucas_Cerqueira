"""
Compartilhamento aditivo de segredos em Z_p (Camada 4), igual ao validado no Teste 4 (v5) da PoC.

- p = 2^61 - 1 (primo de Mersenne): cada parte cabe em int64 e a soma de duas partes (< 2^62) nunca estoura;
- ponto fixo com arredondamento (resolução 2^-24); negativos representados como p - |x|;
- máscaras geradas por CSPRNG (os.urandom): como se cancelam exatamente, o agregado não depende delas;
- checagem explícita de estouro antes de compartilhar: com |v_k| <= LIMITE / n, |Σ v_k| < p / 2.
"""

import os

import numpy as np
import torch

MODULUS = 2**61 - 1
PRECISION_BITS = 24
SCALING_FACTOR = 2**PRECISION_BITS
SUM_LIMIT = (MODULUS // 2) / SCALING_FACTOR  # |Σ| representável ≈ 6.9e10


class SMPCOverflowError(ValueError):
    """O valor excede a faixa representável no campo finito."""


def encode_fixed_point(values):
    """float64 -> Z_p com arredondamento; negativos viram p - |x|."""
    return torch.remainder(torch.round(values.to(torch.float64) * SCALING_FACTOR).to(torch.int64), MODULUS)


def decode_fixed_point(values):
    """Z_p -> float64; valores acima de p/2 são interpretados como negativos."""
    values = torch.where(values > MODULUS // 2, values - MODULUS, values)
    return values.to(torch.float64) / SCALING_FACTOR


def random_field_elements(shape):
    """Inteiros uniformes em [0, p) a partir do CSPRNG do sistema operacional."""
    n = int(np.prod(shape))
    raw = np.frombuffer(os.urandom(8 * n), dtype=np.uint64) & np.uint64(MODULUS)
    raw = raw % np.uint64(MODULUS)
    return torch.from_numpy(raw.astype(np.int64)).view(shape)


def check_range(values, num_parties):
    maximum = float(values.abs().max()) if values.numel() else 0.0
    if maximum > SUM_LIMIT / num_parties:
        raise SMPCOverflowError(
            f"Valor {maximum:.3e} excede o limite {SUM_LIMIT / num_parties:.3e} do campo SMPC."
        )


class SecretSharing:
    """Lado do cliente: codifica o vetor de pesos e o divide em partes aleatórias."""

    def __init__(self, num_parties):
        if num_parties < 2:
            raise ValueError("O SMPC exige pelo menos 2 partes.")
        self.num_parties = num_parties

    def split(self, values):
        """Devolve num_parties partes com Σ partes ≡ encode(values) (mod p); cada parte isolada é uniforme."""
        check_range(values, self.num_parties)
        secret = encode_fixed_point(values)
        masks = [random_field_elements(secret.shape) for _ in range(self.num_parties - 1)]
        last = secret
        for mask in masks:
            last = torch.remainder(last - mask, MODULUS)  # diferença em (-p, p): sem estouro
        return masks + [last]
