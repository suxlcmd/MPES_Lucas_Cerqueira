"""
Pipeline de treinamento ponta a ponta (Camada 2), no mesmo protocolo do Teste 4 (v5) da PoC:
engenharia de features -> amostragem -> split 60/20/20 -> scaler ajustado no treino -> FL + SMPC
-> limiar calibrado na validação -> métricas no teste reservado -> checkpoint completo.
"""

import datetime
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_engine.step_gan_trainer import STEPGANConfig, STEPGANTrainer
from orchestration.data_preprocessor import (
    SCHEMA_LABELS,
    DataPreprocessor,
    balance_sample,
    split_train_val_test,
)
from orchestration.federated_coordinator import FederatedCoordinator
from orchestration.model_bundle import ModelBundle

APP_VERSION = "2.0.0"


@dataclass
class TrainingSettings:
    num_clients: int = 3
    num_rounds: int = 5
    local_epochs: int = 7
    batch_size: int = 512
    latent_dim: int = 100
    normals_per_fraud: int = 200
    min_precision: float = 0.70
    expected_alert_rate: float = 0.01
    seed: int = 42


def calibrate_threshold(y_val, scores_val, min_precision, expected_alert_rate):
    """Limiar escolhido SÓ na validação. Com rótulos: maior recall com precisão mínima; sem rótulos: quantil."""
    if y_val is not None and len(np.unique(y_val)) == 2:
        precisions, recalls, thresholds = precision_recall_curve(y_val, scores_val)
        validos = np.where(precisions[:-1] >= min_precision)[0]
        if len(validos) > 0:
            idx = validos[np.argmax(recalls[validos])]
            return float(thresholds[idx]), f"Maior recall com precisão ≥ {min_precision:.0%} na validação"
        f1 = np.divide(2 * precisions * recalls, precisions + recalls,
                       out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
        idx = int(np.argmax(f1[:-1]))
        return float(thresholds[idx]), "Máximo F1 na validação (precisão mínima não atingida)"
    quantil = 1.0 - expected_alert_rate
    return float(np.quantile(scores_val, quantil)), f"Quantil {quantil:.1%} dos escores de validação (base sem rótulos)"


def classification_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return {}
    y_pred = (scores >= threshold).astype(int)
    return {
        "ROC-AUC": float(roc_auc_score(y_true, scores)),
        "PR-AUC": float(average_precision_score(y_true, scores)),
        "Precisão": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Acurácia": float(accuracy_score(y_true, y_pred)),
    }


def run_training(df_raw, dataset_name, dataset_sha256, settings, progress=None, user="-"):
    pre = DataPreprocessor()
    _, features, y = pre.prepare(df_raw, training=True)
    if features.shape[1] == 0:
        raise ValueError("Nenhuma coluna numérica utilizável foi encontrada na base.")
    rotulada = y is not None
    if not rotulada:
        y = pd.Series(0, index=features.index)

    linhas_originais = len(features)
    features, y = balance_sample(features, y, settings.normals_per_fraud if rotulada else 0, settings.seed)
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split_train_val_test(
        features, y if rotulada else None, settings.seed
    )
    if len(X_train_df) < settings.num_clients * 20:
        raise ValueError("A base é pequena demais para o número de clientes escolhido.")

    pre.fit(X_train_df)
    X_train, X_val, X_test = (pre.transform(d) for d in (X_train_df, X_val_df, X_test_df))
    y_train, y_val, y_test = (np.asarray(v).astype(int) for v in (y_train, y_val, y_test))

    config = STEPGANConfig(
        latent_dim=settings.latent_dim, batch_size=settings.batch_size, local_epochs=settings.local_epochs
    )
    trainer = STEPGANTrainer(X_train.shape[1], config)
    fatias = np.array_split(np.arange(len(X_train)), settings.num_clients)
    clients = [(X_train[i], y_train[i]) for i in fatias]

    coordinator = FederatedCoordinator(trainer, settings.num_clients)
    resultado = coordinator.fit(
        clients, X_val, y_val if rotulada else None, settings.num_rounds, settings.seed, progress=progress
    )

    scores_val = trainer.anomaly_scores(resultado.best_state, X_val)
    threshold, estrategia = calibrate_threshold(
        y_val if rotulada else None, scores_val, settings.min_precision, settings.expected_alert_rate
    )
    scores_test = trainer.anomaly_scores(resultado.best_state, X_test)

    normais = X_train[y_train == 0] if (y_train == 0).any() else X_train
    q25, mediana, q75 = np.percentile(normais, [25, 50, 75], axis=0)
    referencia = {
        "median": [float(v) for v in mediana],
        "iqr": [float(max(v, 0.05)) for v in (q75 - q25)],
    }

    agora = datetime.datetime.now(datetime.timezone.utc)
    return ModelBundle(
        state=resultado.best_state,
        config=config.to_dict(),
        preprocessor=pre.to_dict(),
        threshold=threshold,
        threshold_strategy=estrategia,
        reference=referencia,
        metrics={
            "validacao": classification_metrics(y_val, scores_val, threshold) if rotulada else {},
            "teste": classification_metrics(y_test, scores_test, threshold) if rotulada else {},
            "taxa_alerta_teste": float((scores_test >= threshold).mean()),
        },
        history=resultado.history,
        adherence=trainer.conditioning_adherence(resultado.best_state),
        metadata={
            "model_id": f"stepgan_{agora.strftime('%Y%m%d_%H%M%S')}",
            "created_at": agora.isoformat(timespec="seconds"),
            "created_by": user,
            "app_version": APP_VERSION,
            "dataset": dataset_name,
            "dataset_sha256": dataset_sha256,
            "schema": pre.schema,
            "schema_label": SCHEMA_LABELS.get(pre.schema, pre.schema),
            "target": pre.target or "-",
            "labeled": rotulada,
            "rows_original": int(linhas_originais),
            "rows_used": int(len(features)),
            "frauds_used": int(np.asarray(y).sum()) if rotulada else 0,
            "rows_train": int(len(X_train)),
            "rows_val": int(len(X_val)),
            "rows_test": int(len(X_test)),
            "best_round": int(resultado.best_round),
            "settings": asdict(settings),
        },
    )
