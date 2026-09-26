import os

import pytest
from streamlit.testing.v1 import AppTest

from conftest import make_creditcard_like
from orchestration.inference_service import score_transactions
from orchestration.training_service import TrainingSettings, run_training

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


@pytest.fixture(scope="module")
def estado_com_analise():
    bundle = run_training(make_creditcard_like(), "sintetico.csv", "abc", TrainingSettings(
        num_rounds=1, local_epochs=1, batch_size=128, normals_per_fraud=0))
    resultado = score_transactions(bundle, make_creditcard_like())
    inferencia = {"id": "t1", "result": resultado, "source": "sintetico.csv", "sha256": "abc" * 20,
                  "at": "2026-01-01T00:00:00+00:00"}
    return bundle, inferencia


def _abrir(subvisao=None, estado=None):
    at = AppTest.from_file(APP, default_timeout=60)
    if estado:
        at.session_state["model_bundle"], at.session_state["inference"] = estado
    at.run()
    if subvisao:
        at.sidebar.radio(key="auditor_navigation").set_value(subvisao).run()
    assert not at.exception, at.exception
    return at


def _textos(at):
    return " ".join(m.value for m in at.markdown)


@pytest.mark.parametrize("subvisao", ["training", "dashboard", "azure"])
def test_subvisoes_sem_modelo(subvisao):
    _abrir(subvisao)


def test_cabecalho_nao_duplicado():
    at = _abrir()
    assert _textos(at).count('class="mpes-page-header"') == 1


def test_treinamento_com_modelo_ativo(estado_com_analise):
    at = _abrir("training", estado_com_analise)
    textos = _textos(at)
    assert "MODELO ATIVO" in textos and "Fila de triagem dos alertas" in textos


def test_dashboard_com_analise(estado_com_analise):
    at = _abrir("dashboard", estado_com_analise)
    textos = _textos(at)
    assert "Última análise de transações" in textos
    assert "Alertas por cliente federado" not in textos  # gráfico fictício removido


def test_controle_de_acesso(monkeypatch):
    monkeypatch.setenv("MPES_ACCESS_PASSWORD", "segredo")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.sidebar.radio) == 0  # nada além do login antes da senha
    at.text_input[0].input("segredo")
    at.button[0].click().run()
    assert len(at.sidebar.radio) == 1
