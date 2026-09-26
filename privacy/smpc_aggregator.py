"""
Agregação SMPC (Camada 4): nós somam às cegas e o servidor revela apenas o agregado.

Fluxo: cliente k divide seu vetor ponderado em n partes e envia a parte j ao nó j;
cada nó j soma (mod p) só as partes que recebeu; o servidor soma as n somas parciais e decodifica.
"""

import torch

from .secret_sharing import MODULUS, decode_fixed_point


def modular_sum(tensors):
    """Soma reduzindo mod p a cada passo (parcelas < 2^61, logo somas intermediárias < 2^62)."""
    total = torch.zeros_like(tensors[0])
    for tensor in tensors:
        total = torch.remainder(total + tensor, MODULUS)
    return total


class SMPCAggregator:
    def partial_sum(self, received_shares):
        """Executado por cada nó: soma das partes recebidas, que individualmente não revelam nada."""
        return modular_sum(received_shares)

    def reveal(self, partial_sums):
        """Executado pelo servidor: soma das parciais e decodificação do agregado."""
        return decode_fixed_point(modular_sum(partial_sums))
