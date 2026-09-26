"""
Coordenação do Aprendizado Federado com agregação SMPC (Camada 2).

Cada rodada: todos os clientes partem do MESMO estado global, treinam localmente, e o Discriminador e os
K geradores são agregados por FedAvg ponderado (n_k / N) via compartilhamento aditivo de segredos.
O modelo global devolvido é o da rodada com melhor PR-AUC na validação.
"""

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from privacy.secret_sharing import SecretSharing
from privacy.smpc_aggregator import SMPCAggregator


def flatten_state(state):
    """Concatena todos os tensores de ponto flutuante (D e geradores, inclusive buffers do BatchNorm)."""
    partes = [t.to(torch.float64).flatten() for t in state["D"].values() if t.is_floating_point()]
    for g in state["G"]:
        partes += [t.to(torch.float64).flatten() for t in g.values() if t.is_floating_point()]
    return torch.cat(partes)


def unflatten_state(vector, template, local_states):
    """Reconstrói o estado a partir do vetor agregado; contadores inteiros do BatchNorm usam o máximo."""
    pos = 0

    def preencher(modelo, chave_modelo):
        nonlocal pos
        novo = {}
        for chave, tensor in modelo.items():
            if tensor.is_floating_point():
                novo[chave] = vector[pos:pos + tensor.numel()].view(tensor.shape).to(tensor.dtype)
                pos += tensor.numel()
            else:
                novo[chave] = torch.stack([chave_modelo(s)[chave] for s in local_states]).max()
        return novo

    estado = {"D": preencher(template["D"], lambda s: s["D"]), "G": []}
    for k, g in enumerate(template["G"]):
        estado["G"].append(preencher(g, lambda s, k=k: s["G"][k]))
    return estado


def _metricas_validacao(y_val, scores):
    if y_val is None or len(np.unique(y_val)) < 2:
        return float("nan"), float("nan")
    return float(average_precision_score(y_val, scores)), float(roc_auc_score(y_val, scores))


@dataclass
class FitResult:
    best_state: dict
    best_round: int
    history: list = field(default_factory=list)


class FederatedCoordinator:
    def __init__(self, trainer, num_clients, verify_smpc=True):
        self.trainer = trainer
        self.num_clients = num_clients
        self.sharing = SecretSharing(num_clients)
        self.aggregator = SMPCAggregator()
        # Em simulação, compara com o FedAvg em claro para comprovar a exatidão do SMPC.
        # Numa implantação real essa verificação não existe (o servidor nunca vê os pesos individuais).
        self.verify_smpc = verify_smpc

    def aggregate(self, local_states, weights):
        inicio = time.perf_counter()
        caixas = [[] for _ in range(self.num_clients)]  # caixas[j]: partes recebidas pelo nó j
        for estado, peso in zip(local_states, weights):
            for j, parte in enumerate(self.sharing.split(flatten_state(estado) * peso)):
                caixas[j].append(parte)
        parciais = [self.aggregator.partial_sum(caixa) for caixa in caixas]
        agregado = self.aggregator.reveal(parciais)
        tempo = time.perf_counter() - inicio

        info = {"smpc_seconds": tempo, "bytes_per_client": int(self.num_clients * agregado.numel() * 8)}
        if self.verify_smpc:
            referencia = sum(flatten_state(e) * p for e, p in zip(local_states, weights))
            info["smpc_max_error"] = float((agregado - referencia).abs().max())
        return unflatten_state(agregado, local_states[0], local_states), info

    def fit(self, clients, X_val, y_val, num_rounds, seed, progress=None):
        """clients: lista de (X_k, y_k). progress(rodada, cliente, total_rodadas) é chamado a cada etapa."""
        tamanhos = np.array([len(X) for X, _ in clients], dtype=np.float64)
        pesos = list(tamanhos / tamanhos.sum())
        estado_global = self.trainer.init_global_state(seed)
        historico, melhor = [], None

        for rodada in range(1, num_rounds + 1):
            inicio = time.perf_counter()
            locais, perdas = [], []
            for k, (X_k, y_k) in enumerate(clients):
                if progress:
                    progress(rodada, k + 1, num_rounds)
                estado_local, stats = self.trainer.train_local(
                    estado_global, X_k, y_k, seed=seed * 1000 + rodada * 10 + k
                )
                locais.append(estado_local)
                perdas.append(stats)
            tempo_treino = time.perf_counter() - inicio

            estado_global, info = self.aggregate(locais, pesos)
            pr_auc, roc_auc = _metricas_validacao(y_val, self.trainer.anomaly_scores(estado_global, X_val))
            registro = {
                "rodada": rodada,
                "pr_auc_val": pr_auc,
                "roc_auc_val": roc_auc,
                "loss_d": float(np.nanmean([p["loss_d"] for p in perdas])),
                "loss_g": float(np.nanmean([p["loss_g"] for p in perdas])),
                "train_seconds": tempo_treino,
                **info,
            }
            for k, p in enumerate(perdas):
                registro[f"loss_d_cliente_{k + 1}"] = p["loss_d"]
            historico.append(registro)

            criterio = pr_auc if not np.isnan(pr_auc) else -rodada  # sem rótulos na validação: última rodada
            if melhor is None or criterio > melhor[0] or np.isnan(pr_auc):
                melhor = (criterio, rodada, copy.deepcopy(estado_global))

        return FitResult(best_state=melhor[2], best_round=melhor[1], history=historico)
