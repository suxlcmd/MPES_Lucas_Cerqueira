import streamlit as st

from theme import apply_theme, page_header
from views.azure_manager import view_azure_manager
from views.dashboard import view_dashboard
from views.training import view_training
from cloud.azure_blob_client import AzureBlobClient
from orchestration.data_preprocessor import DataPreprocessor
from orchestration.federated_coordinator import FederatedCoordinator
from ml_engine.step_gan_trainer import STEPGANTrainer

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


st.set_page_config(
    page_title="MPES | STEP-GAN Fraud Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_theme()


if "current_view" not in st.session_state:
    st.session_state.current_view = "home"


# Mantém a etapa atual do auditor mesmo depois do rerun provocado
# pelo upload do arquivo ou pelo clique do botão de teste.
if "auditor_stage" not in st.session_state:
    st.session_state.auditor_stage = "setup"


def go(view_name):
    st.session_state.current_view = view_name
    st.rerun()


def render_home():
    st.markdown(
        """
        <style>
        .home-wrap {
            max-width: 1080px;
            margin: 26px auto 0 auto;
        }
        .hero {
            text-align: center;
            padding: 38px 20px 28px 20px;
        }
        .hero .shield {
            font-size: 56px;
            margin-bottom: 10px;
        }
        .hero h1 {
            font-size: 38px;
            font-weight: 800;
            letter-spacing: -1px;
            margin: 0;
        }
        .hero p {
            max-width: 700px;
            margin: 12px auto 0 auto;
            color: #8A93B8;
            font-size: 16px;
            line-height: 1.6;
        }
        .role-card {
            min-height: 248px;
            padding: 28px;
            border: 1px solid rgba(138, 147, 184, .18);
            border-radius: 18px;
            background: linear-gradient(145deg, #12172B, #1B2140);
            box-shadow: 0 12px 28px rgba(0, 0, 0, .12);
            margin-bottom: 14px;
        }
        .role-card .role-icon {
            font-size: 38px;
        }
        .role-card h2 {
            font-size: 21px;
            margin: 12px 0 8px 0;
        }
        .role-card p {
            color: #8A93B8;
            font-size: 13px;
            min-height: 48px;
        }
        .role-card ul {
            color: #B9C0DC;
            font-size: 12px;
            line-height: 1.8;
            padding-left: 18px;
        }
        .trust-strip {
            text-align: center;
            color: #8A93B8;
            font-size: 12px;
            margin-top: 22px;
        }
        </style>
        <div class="home-wrap">
            <div class="hero">
                <div class="shield">🛡️</div>
                <h1>MPES Fraud Detection Platform</h1>
                <p>
                    Plataforma segura para detecção de fraudes com STEP-GAN,
                    treinamento federado e agregação protegida por SMPC.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    auditor_col, tecnico_col = st.columns(2, gap="large")

    with auditor_col:
        st.markdown(
            """
            <div class="role-card">
                <div class="role-icon">🔎</div>
                <h2>Visão do Auditor</h2>
                <p>
                    Importe uma base e obtenha resultados confiáveis
                    sem configurar o modelo.
                </p>
                <ul>
                    <li>Escolha apenas a base de dados.</li>
                    <li>Hiperparâmetros selecionados automaticamente.</li>
                    <li>Resultados, métricas e gráficos de auditoria.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Entrar como Auditor",
            width="stretch",
            key="btn_auditor",
        ):
            st.session_state.auditor_stage = "setup"
            go("auditor")

    with tecnico_col:
        st.markdown(
            """
            <div class="role-card">
                <div class="role-icon">⚙️</div>
                <h2>Visão Técnica</h2>
                <p>
                    Ambiente completo para profissionais de dados e TI
                    controlarem o pipeline experimental.
                </p>
                <ul>
                    <li>Gestão de artefatos no Azure Blob Storage.</li>
                    <li>Configuração dos hiperparâmetros do STEP-GAN.</li>
                    <li>Dashboard com os valores do treinamento.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Entrar como Técnico",
            width="stretch",
            key="btn_tecnico",
        ):
            go("tecnico")

    st.markdown(
        """
        <div class="trust-strip">
            Dados organizados por cliente · Agregação federada com SMPC ·
            Artefatos rastreáveis no Azure
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_topbar(title, subtitle):
    left, right = st.columns([5, 1])

    with left:
        page_header("🛡️", title, subtitle)

    with right:
        st.write("")
        st.write("")

        if st.button(
            "Voltar ao início",
            width="stretch",
            key=f"home_{title}",
        ):
            st.session_state.auditor_stage = "setup"
            go("home")


def render_auditor():
    """Renderiza auditor com estado persistente após o treinamento."""
    render_topbar(
        "Visão do Auditor",
        "Importe uma base e o sistema selecionará automaticamente "
        "uma configuração adequada para o treinamento.",
    )

    st.info(
        "O modo auditor seleciona automaticamente os hiperparâmetros "
        "com base no tamanho e na dimensão da base."
    )

    st.warning(
        "A configuração automática é uma heurística operacional. "
        "Para uso científico, valide configurações adicionais."
    )

    if st.session_state.auditor_stage == "setup":
        _render_auditor_setup()
    else:
        _render_auditor_test()


def _render_auditor_setup():
    st.markdown("### 1. Selecione a base de treinamento")

    azure = AzureBlobClient()
    files = azure.list_blobs("raw-data")

    if not files:
        st.warning(
            "Nenhuma base disponível em raw-data. "
            "Solicite o upload no ambiente técnico."
        )
        return

    selected_file = st.selectbox(
        "Base para auditoria",
        files,
        key="auditor_dataset",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        clients = st.selectbox(
            "Clientes federados",
            [2, 3, 4, 5],
            index=1,
            key="auditor_clients",
        )

    with col2:
        rounds = st.selectbox(
            "Rodadas automáticas",
            [3, 5, 8, 10],
            index=1,
            key="auditor_rounds",
        )

    with col3:
        st.markdown("**Configuração automática**")
        st.caption(
            "A dimensão latente, o lote e as taxas de aprendizado "
            "serão definidos pelo sistema."
        )

    if not st.button(
        "Importar e treinar automaticamente",
        width="stretch",
        key="start_auditor_training",
    ):
        return

    _run_auditor_training(
        azure=azure,
        selected_file=selected_file,
        clients=clients,
        rounds=rounds,
    )


def _run_auditor_training(azure, selected_file, clients, rounds):
    with st.spinner("Importando e analisando a base..."):
        df = azure.download_blob_as_dataframe(
            "raw-data",
            selected_file,
        )

    if df is None or df.empty:
        st.error("Não foi possível carregar a base de dados.")
        return

    with st.spinner("Pré-processando dados..."):
        preprocessor = DataPreprocessor()
        X, y, _ = preprocessor.process_and_scale(
            df,
            is_training=True,
        )

    if len(X) == 0:
        st.error("A base ficou vazia após o pré-processamento.")
        return

    number_rows = len(X)
    input_dim = X.shape[1]

    latent_dim = (
        64
        if input_dim <= 30
        else 100
        if input_dim <= 100
        else 128
    )

    batch_size = (
        128
        if number_rows < 10000
        else 256
        if number_rows < 100000
        else 512
    )

    config = {
        "latent_dim": latent_dim,
        "batch": batch_size,
        "lr_g": 0.0002,
        "lr_d": 0.0001,
    }

    st.session_state.preprocessor = preprocessor
    st.session_state.auditor_config = config
    st.session_state.input_dim = input_dim

    st.markdown("### 2. Configuração escolhida pelo sistema")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Dimensão latente", latent_dim)
    with c2:
        st.metric("Batch size", batch_size)
    with c3:
        st.metric("Learning rate G", config["lr_g"])
    with c4:
        st.metric("Learning rate D", config["lr_d"])

    indexes = np.array_split(
        np.arange(len(X)),
        clients,
    )

    clients_X = [X[index] for index in indexes if len(index) > 0]
    clients_y = [y[index] for index in indexes if len(index) > 0]

    coordinator = FederatedCoordinator(len(clients_X))
    trainer = STEPGANTrainer(input_dim, config)

    progress = st.progress(0)
    status = st.empty()

    for round_number in range(1, rounds + 1):
        status.info(
            f"Processando rodada {round_number}/{rounds}: "
            "treino local e agregação SMPC..."
        )

        coordinator.run_federated_round(
            round_number,
            trainer,
            clients_X,
            clients_y,
        )

        progress.progress(round_number / rounds)

    st.session_state.trained_model = trainer.D
    st.session_state.training_metadata = {
        "num_rounds": rounds,
        "num_clients": len(clients_X),
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "lr_g": config["lr_g"],
        "lr_d": config["lr_d"],
        "input_dim": input_dim,
        "dataset": selected_file,
    }
    st.session_state.training_logs = trainer.logs
    st.session_state.auditor_stage = "test"

    st.success(
        "Treinamento concluído. Agora envie o CSV de teste abaixo."
    )

    # O script continua nesta execução e já renderiza a etapa de teste.
    _render_auditor_test()


def _render_auditor_test():
    if "trained_model" not in st.session_state:
        st.warning(
            "O treinamento ainda não foi concluído. "
            "Execute o treinamento primeiro."
        )
        st.session_state.auditor_stage = "setup"
        return

    st.markdown("### 3. Avaliar uma base de teste")

    metadata = st.session_state.get(
        "training_metadata",
        {},
    )

    st.success(
        "Modelo treinado carregado: "
        f"{metadata.get('dataset', 'dataset selecionado')}"
    )

    test_file = st.file_uploader(
        "CSV de teste",
        type=["csv"],
        key="auditor_test_file",
    )

    if test_file is None:
        return

    if not st.button(
        "Executar auditoria",
        width="stretch",
        key="run_auditor_inference",
    ):
        return

    try:
        with st.spinner("Calculando escores e métricas..."):
            test_df = pd.read_csv(
                test_file,
                sep=None,
                engine="python",
            )

            X_test, y_test, clean_df = (
                st.session_state.preprocessor.process_and_scale(
                    test_df,
                    is_training=False,
                )
            )

            if len(X_test) == 0:
                st.error(
                    "A base de teste ficou vazia após o pré-processamento."
                )
                return

            model = st.session_state.trained_model
            model.eval()

            with torch.no_grad():
                scores = (
                    1.0
                    - model(torch.FloatTensor(X_test))
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )

        if scores.max() > scores.min():
            scores = (
                scores - scores.min()
            ) / (scores.max() - scores.min())

        predictions = (scores >= 0.85).astype(int)

        clean_df = clean_df.copy()
        clean_df["Score Anomalia"] = scores * 100
        clean_df["Alerta Fraude"] = predictions

        st.session_state.inference_results = clean_df

        alerts = int(predictions.sum())

        if alerts > 0:
            st.error(
                f"{alerts} transação(ões) suspeita(s) detectada(s)."
            )
        else:
            st.success("Nenhuma anomalia detectada.")

        if (
            "Class" in test_df.columns
            and len(np.unique(y_test)) > 1
            and len(y_test) == len(predictions)
        ):
            metrics = {
                "Acurácia": accuracy_score(y_test, predictions),
                "Precisão": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "F1-Score": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "ROC-AUC": roc_auc_score(y_test, scores),
                "PR-AUC": average_precision_score(y_test, scores),
            }

            st.session_state.inference_metrics = metrics

        st.markdown("### Todas as métricas de auditoria")

        metrics = st.session_state.get(
            "inference_metrics",
            {},
        )

        if metrics:
            metric_columns = st.columns(len(metrics))

            for column, (name, value) in zip(
                metric_columns,
                metrics.items(),
            ):
                with column:
                    st.metric(name, f"{value:.4f}")
        else:
            st.info(
                "O CSV não possui rótulos suficientes para calcular "
                "as métricas de classificação."
            )

        st.markdown("### Distribuição dos escores")

        counts, bins = np.histogram(scores, bins=30)

        st.area_chart(
            pd.DataFrame(
                {"Transações": counts},
                index=bins[:-1],
            )
        )

        st.markdown("### Transações priorizadas")

        st.dataframe(
            clean_df.sort_values(
                "Score Anomalia",
                ascending=False,
            ).head(1000),
            width="stretch",
        )

    except Exception as error:
        st.error(f"Erro durante a auditoria: {error}")


def render_tecnico():
    render_topbar(
        "Visão Técnica",
        "Ambiente completo para configurar, executar e diagnosticar "
        "o pipeline STEP-GAN.",
    )

    st.warning(
        "Modo técnico: altere os hiperparâmetros somente se compreender "
        "o impacto em estabilidade, custo e métricas."
    )

    tab_training, tab_azure, tab_dashboard = st.tabs(
        [
            "Treinamento STEP-GAN",
            "Azure Storage",
            "Dashboard",
        ]
    )

    with tab_training:
        view_training()

    with tab_azure:
        view_azure_manager()

    with tab_dashboard:
        view_dashboard()


def main():
    current_view = st.session_state.current_view

    if current_view == "home":
        render_home()
    elif current_view == "auditor":
        render_auditor()
    elif current_view == "tecnico":
        render_tecnico()
    else:
        st.session_state.current_view = "home"
        st.rerun()


if __name__ == "__main__":
    main()