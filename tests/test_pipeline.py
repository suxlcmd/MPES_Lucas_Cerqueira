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
    for coluna in ["ID", "Escore de risco", "Nível de risco", "Alerta", "Por que foi marcada", "Status da análise"]:
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


def test_qualidade_em_linguagem_simples(bundle):
    from orchestration.training_service import quality_summary
    nota, tipo, frases = quality_summary(bundle.metrics["teste"])
    assert nota in {"Muito bom", "Bom", "Razoável", "Fraco"} and tipo in {"success", "warning", "danger"}
    assert any("de cada 100 fraudes" in f for f in frases)
    assert quality_summary({}) is None


def test_motivos_legiveis():
    from orchestration.inference_service import describe_deviation
    assert describe_deviation("amount_log", 4.0) == "amount muito acima do comum"
    assert describe_deviation("idade", -1.5) == "idade abaixo do comum"
    assert describe_deviation("payment_type=AB", 2.0) == "payment_type = AB (pouco comum)"
    assert describe_deviation("type_TRANSFER", 2.0) == "type = TRANSFER (pouco comum)"


def test_base_generica_com_rotulo_em_texto():
    from conftest import make_generic_like
    df = make_generic_like()
    bundle = run_training(df, "seguros.csv", "ghi789", SETTINGS, user="teste")
    assert bundle.metadata["target"] == "fraud_reported" and bundle.metadata["labeled"]
    assert {"policy_number", "incident_date"} <= set(bundle.metadata["dropped_columns"])
    assert any(n.startswith("incident_severity=") for n in bundle.feature_names)
    # Análise de um arquivo sem a coluna de fraude e com uma categoria nunca vista no treino
    novo = df.drop(columns=["fraud_reported"]).head(30).copy()
    novo.loc[novo.index[0], "incident_severity"] = "Categoria nova"
    resultado = score_transactions(bundle, novo)
    assert len(resultado.table) == 30 and resultado.metrics == {}


def test_coluna_escolhida_pelo_usuario_e_base_sem_rotulo():
    from conftest import make_generic_like
    df = make_generic_like().rename(columns={"fraud_reported": "conferido"})
    com_coluna = run_training(df, "a.csv", "1", SETTINGS, user="teste", target="conferido")
    assert com_coluna.metadata["target"] == "conferido" and com_coluna.metadata["labeled"]
    sem_rotulo = run_training(df, "b.csv", "2", SETTINGS, user="teste", target="")
    assert not sem_rotulo.metadata["labeled"]


def test_base_com_poucas_fraudes_gera_erro_claro():
    df = make_creditcard_like()
    df["Class"] = 0
    df.loc[df.index[:2], "Class"] = 1
    with pytest.raises(ValueError, match="menos de 5 fraudes"):
        run_training(df, "poucas.csv", "3", SETTINGS, user="teste")


def test_limiar_nao_fica_inutil():
    """Se a precisão mínima só é atingida alertando quase nada, o limiar passa a equilibrar acertos e alarmes."""
    from orchestration.training_service import calibrate_threshold
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(50), np.zeros(950)].astype(int)
    scores = np.r_[0.99, rng.uniform(0.3, 0.7, 49), rng.uniform(0.0, 0.7, 950)]  # só 1 fraude bem separada
    limiar, estrategia = calibrate_threshold(y, scores, 0.70, 0.01)
    assert estrategia.startswith("Máximo F1")
    assert ((scores >= limiar) & (y == 1)).sum() / y.sum() >= 0.2


def test_categoria_comum_nao_vira_motivo():
    """Uma categoria frequente (ex.: 'dia = Sexta') não é motivo; uma rara (ex.: 'nota duplicada = Sim') é."""
    from orchestration.inference_service import atypical_factors
    # Escala (-1, 1) como no pré-processador: coluna 0 = categoria comum (30%), coluna 1 = indicador raro (1%)
    referencia = {"median": [-1.0, -1.0], "iqr": [0.05, 0.05], "binary": [True, True], "high": [1.0, 1.0],
                  "freq_high": [0.30, 0.01]}
    nomes = ["dia_semana=Sexta", "nota_duplicada=Sim"]
    X = np.array([[1.0, -1.0], [1.0, 1.0]])
    motivos = atypical_factors(X, referencia, nomes, categories={"nota_duplicada": ["Não", "Sim"]})
    assert motivos[0] == "combinação incomum de características"
    assert motivos[1] == "nota_duplicada = Sim (pouco comum)"
    # Valor ausente da categoria quase sempre presente vira "= outra categoria"
    referencia["freq_high"] = [0.30, 0.99]
    assert atypical_factors(np.array([[-1.0, -1.0]]), referencia, nomes,
                            categories={"nota_duplicada": ["Não", "Sim"]})[0] == "nota_duplicada = Não (pouco comum)"
