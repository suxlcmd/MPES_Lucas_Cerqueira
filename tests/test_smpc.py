import pytest
import torch

from privacy.secret_sharing import (
    MODULUS,
    SCALING_FACTOR,
    SUM_LIMIT,
    SecretSharing,
    SMPCOverflowError,
    decode_fixed_point,
    encode_fixed_point,
)
from privacy.smpc_aggregator import SMPCAggregator


def _agregar(vetores, pesos):
    n = len(vetores)
    sharing, aggregator = SecretSharing(n), SMPCAggregator()
    caixas = [[] for _ in range(n)]
    for v, p in zip(vetores, pesos):
        for j, parte in enumerate(sharing.split(v * p)):
            caixas[j].append(parte)
    return aggregator.reveal([aggregator.partial_sum(c) for c in caixas])


def test_negativos_e_valores_grandes():
    base = torch.tensor([-1.5, 30.0, -30.0, 1595.5, -2.0e7, 7e-8, 0.123456789], dtype=torch.float64)
    vetores = [base, base * 0.5, base * -0.25]
    pesos = [0.5, 0.3, 0.2]
    esperado = sum(v * p for v, p in zip(vetores, pesos))
    obtido = _agregar(vetores, pesos)
    assert (obtido - esperado).abs().max() <= len(vetores) * 0.5 / SCALING_FACTOR


def test_configuracao_antiga_estourava():
    """Com p = 2^31 - 1 a média de três pesos iguais a 30 virava -12,67; com p = 2^61 - 1 é exata."""
    obtido = _agregar([torch.tensor([30.0], dtype=torch.float64)] * 3, [1 / 3] * 3)
    assert abs(obtido.item() - 30.0) < 1e-6


def test_estouro_detectado_antes_de_compartilhar():
    with pytest.raises(SMPCOverflowError):
        SecretSharing(3).split(torch.tensor([SUM_LIMIT], dtype=torch.float64))


def test_partes_individuais_nao_revelam_o_segredo():
    segredo = torch.full((1000,), 0.25, dtype=torch.float64)
    partes = SecretSharing(3).split(segredo)
    assert all(((p >= 0) & (p < MODULUS)).all() for p in partes)
    # Uma parte isolada é uniforme em Z_p: sua média fica perto de p/2, longe do segredo codificado
    assert abs(partes[0].double().mean().item() / MODULUS - 0.5) < 0.05


def test_codificacao_de_negativos():
    valores = torch.tensor([-1.5, 0.0, 2.25], dtype=torch.float64)
    codificados = encode_fixed_point(valores)
    assert codificados[0] == MODULUS - round(1.5 * SCALING_FACTOR)
    assert torch.allclose(decode_fixed_point(codificados), valores)
