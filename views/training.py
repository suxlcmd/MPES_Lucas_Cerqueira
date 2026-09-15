import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, precision_recall_curve)

from cloud.azure_blob_client import AzureBlobClient
from orchestration.data_preprocessor import DataPreprocessor
from orchestration.federated_coordinator import FederatedCoordinator
from ml_engine.step_gan_trainer import STEPGANTrainer
from theme import apply_theme, page_header, section_header, status_badge, metric_card

PRIMARY = "#4F7CFF"
ACCENT = "#00C2A8"
DANGER = "#FF4D6D"

try:
    import torch
    TORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    device = "cpu"


def _clean_test_csv(df_teste):
    if len(df_teste.columns) == 1:
        col_name = df_teste.columns[0]
        if df_teste[col_name].astype(str).str.contains(',').any():
            df_expanded = df_teste[col_name].astype(str).str.split(',', expand=True)
            novas_cols = str(col_name).split(',')
            if len(novas_cols) == df_expanded.shape[1]:
                df_expanded.columns = novas_cols
                df_teste = df_expanded

    df_teste.columns = [str(col).strip().replace('"', '').replace("'", '') for col in df_teste.columns]
    for col in df_teste.columns:
        if df_teste[col].dtype == object:
            df_teste[col] = df_teste[col].astype(str).str.strip(' "\'')
    df_teste = df_teste.apply(pd.to_numeric, errors='coerce')
    return df_teste


def view_training():
    apply_theme()
    page_header(
        "⚙️",
        "Treinamento Federado STEP-GAN",
        "Orquestração segura via SMPC e detecção de anomalias em transações",
    )

    azure_client = AzureBlobClient()

    # ============================================================
    # ETAPA 1 — Seleção de dados
    # ============================================================
    section_header("1", "Seleção de Dados de Treinamento")
    arquivos_raw = azure_client.list_blobs("raw-data")

    if not arquivos_raw:
        st.markdown(
            f"""<div class="mpes-card" style="border-left:3px solid #FFB020;">
                {status_badge("SEM DADOS", "warning")}
                <p style="margin-top:10px; color:#8A93B8; font-size:13px;">
                Nenhum arquivo encontrado em <code>raw-data/</code>. Vá até o Azure Storage Manager e envie um CSV.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="mpes-card">', unsafe_allow_html=True)
    selected_file = st.selectbox("Dataset de treinamento:", arquivos_raw)
    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # ETAPA 2 — Configuração e orquestração
    # ============================================================
    section_header("2", "Configuração da Orquestração Federada")

    with st.form("training_config_form"):
        st.markdown('<div class="mpes-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            num_clients = st.slider("👥 Número de Clientes (Silos)", 2, 10, 3)
            num_rounds = st.number_input("🔁 Rodadas Federadas", min_value=1, max_value=50, value=3)
        with col2:
            latent_dim = st.selectbox("🧬 Dimensão Latente (Z)", [64, 100, 128], index=1)
            batch_size = st.selectbox("📦 Tamanho do Lote", [128, 256, 512], index=2)
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🚀 Iniciar Orquestração Federada")

    if submitted:
        if not TORCH_AVAILABLE:
            st.error("❌ O treinamento requer o PyTorch instalado no ambiente.")
            return

        log_container = st.empty()

        with st.spinner(f"Baixando {selected_file} do Azure..."):
            df_train = azure_client.download_blob_as_dataframe("raw-data", selected_file)

        if df_train is not None:
            with st.spinner("Pré-processando dados (limpeza robusta e MinMaxScaler)..."):
                preprocessor = DataPreprocessor()
                X_scaled, y, _ = preprocessor.process_and_scale(df_train, is_training=True)

            if len(X_scaled) == 0:
                st.error("⚠️ A base de dados ficou vazia após o tratamento de limpeza. Verifique se o arquivo possui valores numéricos extraíveis.")
                return

            st.session_state['preprocessor'] = preprocessor
            st.session_state['input_dim'] = X_scaled.shape[1]

            csv_buffer = io.BytesIO()
            pd.DataFrame(X_scaled).to_csv(csv_buffer, index=False)
            azure_client.upload_blob("processed-data", f"processed_{selected_file}", csv_buffer.getvalue())

            tamanho_chunk = len(X_scaled) // num_clients
            clientes_X = [X_scaled[i * tamanho_chunk: (i + 1) * tamanho_chunk] for i in range(num_clients)]
            clientes_y = [y[i * tamanho_chunk: (i + 1) * tamanho_chunk] for i in range(num_clients)]

            coordinator = FederatedCoordinator(num_clients)
            trainer = STEPGANTrainer(
                X_scaled.shape[1],
                {"latent_dim": latent_dim, "batch": batch_size, "lr_g": 0.0002, "lr_d": 0.0001},
            )

            progress_bar = st.progress(0)
            for r in range(1, num_rounds + 1):
                with log_container.container():
                    st.markdown(
                        f"""<div class="mpes-card-light">
                            {status_badge(f"Rodada {r}/{num_rounds}", "ok")}
                            <span style="margin-left:8px; color:#8A93B8; font-size:13px;">
                                Treinamento local + agregação segura via SMPC em andamento...
                            </span>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                coordinator.run_federated_round(r, trainer, clientes_X, clientes_y)
                progress_bar.progress(r / num_rounds)

            st.session_state['trained_model'] = trainer.D
            st.session_state['training_metadata'] = {'num_rounds': num_rounds, 'num_clients': num_clients}
            st.session_state['training_logs'] = trainer.logs

            agora_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            nome_arquivo_checkpoint = f"{agora_str}.txt"
            conteudo_checkpoint = f"""=== CHECKPOINT DO MODELO GLOBAL STEP-GAN ===
Data de Geração: {agora_str}
Rodadas Federadas: {num_rounds}
Clientes Participantes: {num_clients}
Dimensão Latente: {latent_dim}
Tamanho do Lote: {batch_size}
Algoritmo de Agregação: SMPC Aritmética Modular
Status: Treinamento Concluído.
"""
            azure_client.upload_blob("model-checkpoints", nome_arquivo_checkpoint, conteudo_checkpoint.encode('utf-8'))

            st.markdown(
                f"""<div class="mpes-card" style="border-left:3px solid #00C2A8;">
                    {status_badge("TREINAMENTO CONCLUÍDO", "ok")}
                    <p style="margin-top:10px; font-size:13px; color:#8A93B8;">
                        Checkpoint <code>{nome_arquivo_checkpoint}</code> salvo em <code>model-checkpoints/</code>.
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )

    # ============================================================
    # ETAPA 3 — Inferência
    # ============================================================
    if 'trained_model' in st.session_state:
        section_header("3", "Detecção de Anomalias (Inferência)")
        st.markdown(
            '<div class="mpes-card">'
            '<span style="color:#8A93B8; font-size:13px;">Faça upload do arquivo de transações para verificar fraudes usando o modelo global.</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        arquivo_teste = st.file_uploader("Upload de arquivo CSV de teste", type=['csv'])

        if arquivo_teste is not None:
            if st.button("🔍 Executar Inferência do Discriminador"):
                with st.spinner("Avaliando transações e calculando anomalias..."):
                    df_teste = pd.read_csv(arquivo_teste, sep=None, engine='python')
                    df_teste = _clean_test_csv(df_teste)

                    tem_gabarito = 'Class' in df_teste.columns
                    y_test_real = df_teste['Class'].dropna().values if tem_gabarito else None

                    inicio_infer = time.time()
                    preprocessor = st.session_state['preprocessor']
                    X_test_scaled, _, df_limpo = preprocessor.process_and_scale(df_teste, is_training=False)

                    if len(X_test_scaled) == 0:
                        st.error("⚠️ O arquivo de teste submetido ficou vazio após a limpeza.")
                        return

                    D_global = st.session_state['trained_model']
                    D_global.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_test_scaled).to(device)
                        prob_normal = D_global(X_tensor).cpu().numpy().flatten()
                        anomaly_scores = (1.0 - prob_normal)

                    sc_min, sc_max = anomaly_scores.min(), anomaly_scores.max()
                    if sc_max > sc_min:
                        anomaly_scores = (anomaly_scores - sc_min) / (sc_max - sc_min)
                    anomaly_scores = np.clip(anomaly_scores, 0.0, 1.0)

                    tempo_infer = time.time() - inicio_infer
                    tps = len(df_limpo) / tempo_infer if tempo_infer > 0 else 0

                    optimal_threshold = 0.85
                    if tem_gabarito:
                        precision, recall, thresholds = precision_recall_curve(y_test_real, anomaly_scores)
                        valid_idx = np.where(precision[:-1] >= 0.70)[0]
                        if len(valid_idx) > 0:
                            best_idx = valid_idx[np.argmax(recall[valid_idx])]
                            optimal_threshold = thresholds[best_idx]

                    y_pred = (anomaly_scores >= optimal_threshold).astype(int)
                    df_limpo['Score_Anomalia'] = anomaly_scores * 100
                    df_limpo['Alerta_Fraude'] = y_pred

                    qtd_alertas = int(df_limpo['Alerta_Fraude'].sum())

                    if qtd_alertas > 0:
                        st.markdown(
                            f"""<div class="mpes-card" style="border-left:3px solid {DANGER};">
                                {status_badge(f"⚠️ {qtd_alertas} TRANSAÇÕES SUSPEITAS", "danger")}
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""<div class="mpes-card" style="border-left:3px solid {ACCENT};">
                                {status_badge("✅ NENHUMA ANOMALIA DETECTADA", "ok")}
                            </div>""",
                            unsafe_allow_html=True,
                        )

                    st.session_state['inference_results'] = df_limpo

                    if tem_gabarito:
                        acc = accuracy_score(y_test_real, y_pred)
                        prec = precision_score(y_test_real, y_pred, zero_division=0)
                        rec = recall_score(y_test_real, y_pred, zero_division=0)
                        f1 = f1_score(y_test_real, y_pred, zero_division=0)
                        roc = roc_auc_score(y_test_real, anomaly_scores)
                        pr = average_precision_score(y_test_real, anomaly_scores)

                        st.session_state['inference_metrics'] = {
                            "Acurácia": f"{acc * 100:.2f}%", "Precisão": f"{prec * 100:.2f}%",
                            "Recall": f"{rec * 100:.2f}%", "F1-Score": f"{f1 * 100:.2f}%",
                            "ROC-AUC": f"{roc:.4f}", "PR-AUC": f"{pr:.4f}", "TPS": f"{tps:.0f}/s",
                        }

                        section_header("4", "Métricas de Qualidade")
                        m_keys = ["ROC-AUC", "PR-AUC", "F1-Score", "Recall", "Precisão", "Acurácia", "TPS"]
                        cols = st.columns(len(m_keys))
                        for col, k in zip(cols, m_keys):
                            with col:
                                st.markdown(metric_card(k, st.session_state['inference_metrics'][k], "", "#E4E8FA"), unsafe_allow_html=True)

                    section_header("5", "Transações Priorizadas por Risco")
                    st.markdown('<div class="mpes-card">', unsafe_allow_html=True)
                    df_view = df_limpo.sort_values(by='Score_Anomalia', ascending=False).head(1000)
                    st.dataframe(df_view, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
