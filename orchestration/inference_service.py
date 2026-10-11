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


RARE_FREQUENCY = 0.05   # categoria (ou valor 0/1) presente em menos de 5% das transações normais = incomum


def _readable_feature(name):
    """'amount_log' -> 'amount'; 'payment_type=AB' -> ('payment_type', 'AB'); 'type_TRANSFER' (PaySim) -> 'type'."""
    if name.endswith("_log"):
        return name[:-4], None
    if "=" in name:
        coluna, categoria = name.split("=", 1)
        return coluna, categoria
    if name.startswith("type_") and name[5:].isupper():
        return "type", name[5:]
    return name, None


def describe_deviation(name, deviation, categories=None, binary=False):
    """
    Frase curta para o auditor: em que a transação foge do perfil normal do treino.
    categories: categorias da coluna (para dizer 'possui_nota_fiscal = Não' em vez de 'diferente de Sim').
    """
    coluna, categoria = _readable_feature(name)
    if categoria is not None:
        if deviation > 0:
            return f"{coluna} = {categoria} (pouco comum)"
        outras = [c for c in (categories or []) if c != categoria]
        return f"{coluna} = {outras[0]} (pouco comum)" if len(outras) == 1 else f"{coluna} diferente de {categoria}"
    if binary:
        return f"{coluna} com valor pouco comum"
    intensidade = "muito " if abs(deviation) >= 3 else ""
    return f"{coluna} {intensidade}{'acima' if deviation > 0 else 'abaixo'} do comum"


def deviation_scores(X, reference):
    """
    Quanto cada valor foge do perfil normal do treino. Colunas numéricas: distância à mediana em intervalos
    interquartis. Colunas de dois valores: raridade do valor entre as transações normais (0 se for comum).
    """
    desvio = (X - np.asarray(reference["median"])) / np.asarray(reference["iqr"])
    if "binary" not in reference:  # checkpoints antigos
        return desvio
    binaria = np.asarray(reference["binary"], dtype=bool)
    estado_alto = X >= np.asarray(reference["high"]) - 1e-6
    freq_alto = np.asarray(reference["freq_high"])
    frequencia = np.clip(np.where(estado_alto, freq_alto, 1.0 - freq_alto), 1e-3, 1.0)
    raridade = np.where(frequencia < RARE_FREQUENCY, -2.0 * np.log10(frequencia), 0.0)  # p=5% -> 2,6; p=0,1% -> 6
    desvio[:, binaria] = (np.where(estado_alto, 1.0, -1.0) * raridade)[:, binaria]
    return desvio


def atypical_factors(X, reference, feature_names, top=3, categories=None):
    """Features mais distantes do perfil normal do treino, em linguagem simples."""
    desvio = deviation_scores(X, reference)
    binaria = reference.get("binary", [False] * len(feature_names))
    categorias = categories or {}
    ordem = np.argsort(-np.abs(desvio), axis=1)[:, :top]
    textos = []
    for linha, indices in zip(desvio, ordem):
        partes = []
        for i in indices:
            if abs(linha[i]) < 1:
                continue
            coluna, _ = _readable_feature(feature_names[i])
            partes.append(describe_deviation(feature_names[i], linha[i], categorias.get(coluna), binaria[i]))
        textos.append("; ".join(partes) if partes else "combinação incomum de características")
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
    plano = bundle.preprocessor.get("plan") or {}
    tabela["Por que foi marcada"] = atypical_factors(X, bundle.reference, bundle.feature_names,
                                                     categories=plano.get("categorical"))
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
