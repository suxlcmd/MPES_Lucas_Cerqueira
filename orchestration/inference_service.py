"""
Inferência para a auditoria (Camada 2).

- Escore de risco = 1 - D(x), sem normalização pelo lote: o resultado de uma transação não depende das
  outras transações do arquivo.
- Limiar fixo, calibrado na validação durante o treino e salvo no checkpoint.
- Nível de risco e fatores mais atípicos (desvio em relação às transações normais do treino) para triagem.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from orchestration.training_service import classification_metrics

RISK_LEVELS = ["Crítico", "Alto", "Médio", "Baixo"]
TRIAGE_STATUS = ["Pendente", "Em análise", "Fraude confirmada", "Falso positivo"]


def risk_levels(scores, threshold):
    """Alto/Crítico geram alerta; Médio fica abaixo do limiar, mas acima da metade dele."""
    critico = threshold + (1.0 - threshold) / 2
    return np.select(
        [scores >= critico, scores >= threshold, scores >= threshold / 2],
        ["Crítico", "Alto", "Médio"],
        default="Baixo",
    )


def atypical_factors(X, reference, feature_names, top=3):
    """Features mais distantes do perfil normal do treino, em unidades de intervalo interquartil."""
    desvio = (X - np.asarray(reference["median"])) / np.asarray(reference["iqr"])
    ordem = np.argsort(-np.abs(desvio), axis=1)[:, :top]
    textos = []
    for linha, indices in zip(desvio, ordem):
        partes = [f"{feature_names[i]} {'↑' if linha[i] > 0 else '↓'}{abs(linha[i]):.1f}" for i in indices]
        textos.append(", ".join(partes))
    return textos


@dataclass
class InferenceResult:
    table: pd.DataFrame
    summary: dict
    metrics: dict = field(default_factory=dict)


def score_transactions(bundle, df_raw):
    pre = bundle.build_preprocessor()
    trainer = bundle.build_trainer()
    df_limpo, features, y = pre.prepare(df_raw, training=False)
    if len(features) == 0:
        raise ValueError("O arquivo não contém transações.")
    X = pre.transform(features)

    inicio = time.perf_counter()
    scores = trainer.anomaly_scores(bundle.state, X)
    tempo = time.perf_counter() - inicio

    niveis = risk_levels(scores, bundle.threshold)
    tabela = df_limpo.copy()
    tabela.insert(0, "ID", np.arange(1, len(tabela) + 1))
    tabela["Escore de risco"] = np.round(scores * 100, 2)
    tabela["Nível de risco"] = pd.Categorical(niveis, categories=RISK_LEVELS, ordered=True)
    tabela["Alerta"] = scores >= bundle.threshold
    tabela["Fatores mais atípicos"] = atypical_factors(X, bundle.reference, bundle.feature_names)
    tabela["Status da análise"] = "Pendente"
    tabela["Observação do auditor"] = ""

    alertas = int(tabela["Alerta"].sum())
    resumo = {
        "transactions": int(len(tabela)),
        "alerts": alertas,
        "alert_rate": alertas / len(tabela),
        "threshold": float(bundle.threshold),
        "tps": len(tabela) / tempo if tempo > 0 else float("nan"),
        "levels": {nivel: int((niveis == nivel).sum()) for nivel in RISK_LEVELS},
        "model_id": bundle.metadata.get("model_id", "-"),
    }

    metricas = {}
    if y is not None and y.notna().all():
        metricas = classification_metrics(y.astype(int).values, scores, bundle.threshold)
    return InferenceResult(table=tabela, summary=resumo, metrics=metricas)
