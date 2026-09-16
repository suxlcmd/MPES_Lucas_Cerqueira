import streamlit as st

from theme import apply_theme, page_header
from views.azure_manager import view_azure_manager
from views.dashboard import view_dashboard
from views.training import view_training

st.set_page_config(
    page_title="MPES | STEP-GAN Fraud Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_theme()

if "current_view" not in st.session_state:
    st.session_state.current_view = "home"


def go(view_name):
    st.session_state.current_view = view_name
    st.rerun()


def render_home():
    st.markdown(
        """
        <style>
            .home-wrap { max-width: 1080px; margin: 26px auto 0 auto; }
            .hero { text-align: center; padding: 38px 20px 28px 20px; }
            .hero .shield { font-size: 56px; margin-bottom: 10px; }
            .hero h1 { font-size: 38px; font-weight: 800; letter-spacing: -1px; margin: 0; }
            .hero p { max-width: 700px; margin: 12px auto 0 auto; color: #8A93B8; font-size: 16px; line-height: 1.6; }
            .role-card { min-height: 248px; padding: 28px; border: 1px solid rgba(138,147,184,.18); border-radius: 18px; background: linear-gradient(145deg, #12172B, #1B2140); box-shadow: 0 12px 28px rgba(0,0,0,.12); margin-bottom: 14px; }
            .role-card .role-icon { font-size: 38px; }
            .role-card h2 { font-size: 21px; margin: 12px 0 8px 0; }
            .role-card p { color: #8A93B8; font-size: 13px; min-height: 48px; }
            .role-card ul { color: #B9C0DC; font-size: 12px; line-height: 1.8; padding-left: 18px; }
            .trust-strip { text-align: center; color: #8A93B8; font-size: 12px; margin-top: 22px; }
        </style>
        <div class="home-wrap">
            <div class="hero">
                <div class="shield">🛡️</div>
                <h1>MPES Fraud Detection Platform</h1>
                <p>Plataforma segura para detecção de fraudes com STEP-GAN, treinamento federado e agregação protegida por SMPC.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_auditor, col_tecnico = st.columns(2, gap="large")
    with col_auditor:
        st.markdown(
            """<div class="role-card"><div class="role-icon">🔎</div><h2>Visão do Auditor</h2><p>Importe uma base e obtenha resultados confiáveis sem configurar o modelo.</p><ul><li>Escolha apenas a base de dados</li><li>Hiperparâmetros selecionados automaticamente</li><li>Resultados, métricas e gráficos de auditoria</li></ul></div>""",
            unsafe_allow_html=True,
        )
        if st.button("Entrar como Auditor", use_container_width=True, key="btn_auditor"):
            go("auditor")

    with col_tecnico:
        st.markdown(
            """<div class="role-card"><div class="role-icon">⚙️</div><h2>Visão Técnica</h2><p>Ambiente completo para profissionais de dados e TI controlarem o pipeline experimental.</p><ul><li>Gestão de artefatos no Azure Blob Storage</li><li>Configuração dos hiperparâmetros do STEP-GAN</li><li>Dashboard com todos os valores do treinamento</li></ul></div>""",
            unsafe_allow_html=True,
        )
        if st.button("Entrar como Técnico", use_container_width=True, key="btn_tecnico"):
            go("tecnico")

    st.markdown("<div class='trust-strip'>🔐 Dados organizados por cliente · Agregação federada com SMPC · Artefatos rastreáveis no Azure</div>", unsafe_allow_html=True)


def render_topbar(title, subtitle):
    left, right = st.columns([5, 1])
    with left:
        page_header("🛡️", title, subtitle)
    with right:
        st.write("")
        if st.button("← Trocar visão", use_container_width=True):
            go("home")


def render_auditor():
    render_topbar("Visão do Auditor", "Importe uma base; o sistema selecionará automaticamente uma configuração adequada para o treinamento.")
    st.info("O modo auditor prioriza simplicidade: você escolhe a base e o sistema gerencia os hiperparâmetros.")
    st.warning("A configuração automática é uma heurística operacional. Para uso científico, valide diferentes configurações em um conjunto separado.")
    view_auditor_training()


def view_auditor_training():
    from orchestration.data_preprocessor import DataPreprocessor
    from orchestration.federated_coordinator import FederatedCoordinator
    from ml_engine.step_gan_trainer import STEPGANTrainer
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

    st.markdown("### 1. Selecione a base de treinamento")
    from cloud.azure_blob_client import AzureBlobClient
    azure = AzureBlobClient()
    files = azure.list_blobs("raw-data")
    if not files:
        st.warning("Nenhuma base disponível em `raw-data/`. Solicite o upload no ambiente técnico.")
        return

    selected = st.selectbox("Base para auditoria", files)
    col1, col2, col3 = st.columns(3)
    with col1:
        clients = st.selectbox("Clientes federados", [2, 3, 4, 5], index=1)
    with col2:
        rounds = st.selectbox("Rodadas automáticas", [3, 5, 8, 10], index=1)
    with col3:
        st.markdown("**Configuração automática**")
        st.caption("Dimensão latente, lote e taxas de aprendizado são definidos pelo sistema.")

    if st.button("🚀 Importar e treinar automaticamente", use_container_width=True):
        with st.spinner("Importando e analisando a base..."):
            df = azure.download_blob_as_dataframe("raw-data", selected)
            if df is None:
                return
            pre = DataPreprocessor()
            X, y, _ = pre.process_and_scale(df, is_training=True)

        if len(X) == 0:
            st.error("A base ficou vazia após o pré-processamento.")
            return

        n_rows, input_dim = len(X), X.shape[1]
        latent = 64 if input_dim <= 30 else 100 if input_dim <= 100 else 128
        batch = 128 if n_rows < 10000 else 256 if n_rows < 100000 else 512
        config = {"latent_dim": latent, "batch": batch, "lr_g": 0.0002, "lr_d": 0.0001}
        st.session_state["preprocessor"] = pre
        st.session_state["auditor_config"] = config
        st.session_state["input_dim"] = input_dim

        st.markdown("### 2. Configuração escolhida pelo sistema")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dimensão latente", latent)
        c2.metric("Batch size", batch)
        c3.metric("Learning rate G", config["lr_g"])
        c4.metric("Learning rate D", config["lr_d"])

        chunk = len(X) // clients
        clientes_X = [X[i * chunk: (i + 1) * chunk] for i in range(clients)]
        clientes_y = [y[i * chunk: (i + 1) * chunk] for i in range(clients)]
        coordinator = FederatedCoordinator(clients)
        trainer = STEPGANTrainer(input_dim, config)
        progress = st.progress(0)
        status = st.empty()

        for r in range(1, rounds + 1):
            status.info(f"Processando rodada {r}/{rounds}: treino local e agregação SMPC...")
            coordinator.run_federated_round(r, trainer, clientes_X, clientes_y)
            progress.progress(r / rounds)

        st.session_state["trained_model"] = trainer.D
        st.session_state["training_metadata"] = {"num_rounds": rounds, "num_clients": clients, "latent_dim": latent, "batch_size": batch, "lr_g": config["lr_g"], "lr_d": config["lr_d"], "input_dim": input_dim, "dataset": selected}
        st.session_state["training_logs"] = trainer.logs
        status.success("Treinamento concluído. O modelo está pronto para auditoria.")

        st.markdown("### 3. Métricas e gráficos do treinamento")
        logs = pd.DataFrame(trainer.logs)
        st.dataframe(logs.rename(columns={"client": "Cliente", "round": "Rodada", "loss_g": "Loss Gerador", "loss_d": "Loss Discriminador"}).round(6), use_container_width=True, hide_index=True)
        st.line_chart(logs.groupby("round")[["loss_g", "loss_d"]].mean().rename(columns={"loss_g": "Loss Gerador", "loss_d": "Loss Discriminador"}))
        st.caption("A seleção automática é uma heurística operacional inicial; valide configurações em um conjunto separado para uso científico.")

    if "trained_model" in st.session_state:
        st.markdown("### 4. Avaliar uma base de teste")
        test_file = st.file_uploader("CSV de teste", type=["csv"], key="auditor_test")
        if test_file is not None and st.button("🔍 Executar auditoria", use_container_width=True):
            with st.spinner("Calculando escores e métricas..."):
                test_df = pd.read_csv(test_file, sep=None, engine="python")
                X_test, y_test, clean = st.session_state["preprocessor"].process_and_scale(test_df, is_training=False)
                model = st.session_state["trained_model"]
                model.eval()
                with torch.no_grad():
                    scores = 1.0 - model(torch.FloatTensor(X_test)).cpu().numpy().flatten()
                if scores.max() > scores.min():
                    scores = (scores - scores.min()) / (scores.max() - scores.min())
                pred = (scores >= 0.85).astype(int)
                clean["Score_Anomalia"] = scores * 100
                clean["Alerta_Fraude"] = pred

            alerts = int(pred.sum())
            st.error(f"⚠️ {alerts} transações suspeitas detectadas.") if alerts else st.success("✅ Nenhuma anomalia detectada.")
            if "Class" in test_df.columns and len(np.unique(y_test)) > 1:
                metrics = {"Acurácia": accuracy_score(y_test, pred), "Precisão": precision_score(y_test, pred, zero_division=0), "Recall": recall_score(y_test, pred, zero_division=0), "F1-Score": f1_score(y_test, pred, zero_division=0), "ROC-AUC": roc_auc_score(y_test, scores), "PR-AUC": average_precision_score(y_test, scores)}
                st.markdown("#### Todas as métricas de auditoria")
                cols = st.columns(6)
                for col, (name, value) in zip(cols, metrics.items()):
                    col.metric(name, f"{value:.4f}")
            st.markdown("#### Distribuição dos escores")
            counts, bins = np.histogram(scores, bins=30)
            st.area_chart(pd.DataFrame({"Transações": counts}, index=bins[:-1]))
            st.markdown("#### Transações priorizadas")
            st.dataframe(clean.sort_values("Score_Anomalia", ascending=False).head(1000), use_container_width=True)


def render_tecnico():
    render_topbar("Visão Técnica", "Ambiente completo para configurar, executar e diagnosticar o pipeline STEP-GAN.")
    st.warning("Modo técnico: altere os hiperparâmetros somente se compreender o impacto em estabilidade, custo e métricas.")
    # Treinamento é deliberadamente a primeira aba.
    tabs = st.tabs(["⚙️ Treinamento STEP-GAN", "☁️ Azure Storage", "📊 Dashboard"])
    with tabs[0]:
        view_training()
    with tabs[1]:
        view_azure_manager()
    with tabs[2]:
        view_dashboard()


if st.session_state.current_view == "home":
    render_home()
elif st.session_state.current_view == "auditor":
    render_auditor()
elif st.session_state.current_view == "tecnico":
    render_tecnico()
else:
    st.session_state.current_view = "home"
    st.rerun()