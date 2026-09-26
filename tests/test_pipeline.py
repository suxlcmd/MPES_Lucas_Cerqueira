import numpy as np
import pytest
import torch

from conftest import make_creditcard_like
from orchestration.inference_service import RISK_LEVELS, score_transactions
from orchestration.model_bundle import ModelBundle
from orchestration.training_service import TrainingSettings, run_training

SETTINGS = TrainingSettings(num_clients=3, num_rounds=2, local_epochs=1, batch_size=128, normals_per_fraud=0, seed=1)


@pytest.fixture(scope="module")
def bundle():
    return run_training(make_creditcard_like(), "sintetico.csv", "abc123", SETTINGS, user="teste")


def test_treino_gera_checkpoint_completo(bundle):
    assert bundle.metadata["schema"] == "creditcard"
    assert bundle.metadata["rows_train"] + bundle.metadata["rows_val"] + bundle.metadata["rows_test"] == 1500
    assert 0.0 < bundle.threshold < 1.0
    assert len(bundle.state["G"]) == 3
    assert len(bundle.history) == 2
    assert all(r["smpc_max_error"] < 1e-6 for r in bundle.history)
    assert bundle.metrics["teste"]["ROC-AUC"] > 0.8  # fraudes sintéticas bem separadas


def test_checkpoint_roundtrip_seguro(bundle):
    restaurado = ModelBundle.from_bytes(bundle.to_bytes())
    X = np.random.default_rng(0).uniform(-1, 1, size=(50, bundle.input_dim)).astype(np.float32)
    s1 = bundle.build_trainer().anomaly_scores(bundle.state, X)
    s2 = restaurado.build_trainer().anomaly_scores(restaurado.state, X)
    assert np.allclose(s1, s2)
    assert restaurado.metadata == bundle.metadata


def test_checkpoint_invalido_e_rejeitado():
    import io

    buffer = io.BytesIO()
    torch.save({"format": "outro"}, buffer)
    with pytest.raises(ValueError):
        ModelBundle.from_bytes(buffer.getvalue())


def test_escore_nao_depende_do_lote(bundle):
    df = make_creditcard_like()
    completo = score_transactions(bundle, df)
    parcial = score_transactions(bundle, df.iloc[:10])
    assert np.allclose(completo.table["Escore de risco"].values[:10], parcial.table["Escore de risco"].values)


def test_resultado_da_inferencia(bundle):
    df = make_creditcard_like().drop(columns=["Class"])
    resultado = score_transactions(bundle, df)
    tabela = resultado.table
    for coluna in ["ID", "Escore de risco", "Nível de risco", "Alerta", "Fatores mais atípicos", "Status da análise"]:
        assert coluna in tabela
    assert set(tabela["Nível de risco"].astype(str)) <= set(RISK_LEVELS)
    assert resultado.summary["alerts"] == int(tabela["Alerta"].sum())
    assert resultado.metrics == {}  # sem rótulos, sem métricas


def test_treino_sem_rotulos_usa_quantil():
    df = make_creditcard_like().drop(columns=["Class"])
    bundle = run_training(df, "sem_rotulos.csv", "def456", SETTINGS, user="teste")
    assert not bundle.metadata["labeled"]
    assert bundle.threshold_strategy.startswith("Quantil")
    assert bundle.metrics["teste"] == {}
    assert 0.0 < bundle.metrics["taxa_alerta_teste"] < 0.05
