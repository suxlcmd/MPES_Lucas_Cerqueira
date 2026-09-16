import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import datetime
import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

from cloud.azure_blob_client import AzureBlobClient
from orchestration.data_preprocessor import DataPreprocessor
from orchestration.federated_coordinator import FederatedCoordinator
from ml_engine.step_gan_trainer import STEPGANTrainer
from theme import apply_theme, page_header, section_header, status_badge, metric_card
import streamlit as st


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
        if df_teste[col_name].astype(str).str.contains(",").any():
            df_expanded = df_teste[col_name].astype(str).str.split(",", expand=True)
            novas_cols = str(col_name).split(",")
            if len(novas_cols) == df_expanded.shape[1]:
                df_expanded.columns = novas_cols
                df_teste = df_expanded

    df_teste.columns = [
        str(col).strip().replace('"', "").replace("'", "")
        for col in df_teste.columns
    ]
    for col in df_teste.columns:
        if df_teste[col].dtype == object:
            df_teste[col] = df_teste[col].astype(str).str.strip(' "\'')
    return df_teste.apply(pd.to_numeric, errors="coerce")


def _show_training_metrics(logs_df):
    """Mostra todos os valores de loss registrados no treinamento."""
    if logs_df.empty:
        st.info("Nenhum registro de treinamento disponível.")
        return

    st.markdown("#### Valores das métricas por cliente e rodada")
    table = logs_df.copy()
    table = table.rename(
        columns={
            "client": "Cliente",
            "round": "Rodada",
            "loss_g": "Loss Gerador",
            "loss_d": "Loss Discriminador",
        }
    )
    for col in ["Loss Gerador", "Loss Discriminador"]:
        if col in table.columns:
            table[col] = table[col].astype(float).round(6)

    st.dataframe(table, use_container_width=True, hide_index=True)

    summary = logs_df.groupby("round")[["loss_g", "loss_d"]].agg(
        ["mean", "min", "max"]
    )
    summary.columns = [
        f"{metric} - {stat}"
        for metric, stat in summary.columns.to_flat_index()
    ]
    summary.index.name = "Rodada"
    st.markdown("#### Resumo estatístico por rodada")
    st.dataframe(summary.round(6), use_container_width=True)


def _show_training_charts(logs_df):
    if logs_df.empty:
        return

    st.markdown("#### Evolução das perdas")
    loss_summary = logs_df.groupby("round")[["loss_g", "loss_d"]].mean()
    loss_summary = loss_summary.rename(
        columns={
            "loss_g": "Loss Gerador",
            "loss_d": "Loss Discriminador",
        }
    )
    st.line_chart(loss_summary, color=[PRIMARY, DANGER])

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Loss médio do gerador por rodada**")
        st.bar_chart(loss_summary[["Loss Gerador"]], color=PRIMARY)
    with chart_col2:
        st.markdown("**Loss médio do discriminador por rodada**")
        st.bar_chart(loss_summary[["Loss Discriminador"]], color=DANGER)


def view_training():
    apply_theme()
    page_header(
        "⚙️",
        "Treinamento Federado STEP-GAN",
        "Orquestração segura via SMPC e detecção de anomalias em transações",
    )

    azure_client = AzureBlobClient()

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
                st.error("⚠️ A base de dados ficou vazia após o tratamento de limpeza.")
                return

            st.session_state["preprocessor"] = preprocessor
            st.session_state["input_dim"] = X_scaled.shape[1]

            csv_buffer = io.BytesIO()
            pd.DataFrame(X_scaled).to_csv(csv_buffer, index=False)
            azure_client.upload_blob(
                "processed-data",
                f"processed_{selected_file}",
                csv_buffer.getvalue(),
            )

            tamanho_chunk = len(X_scaled) // num_clients
            clientes_X = [
                X_scaled[i * tamanho_chunk: (i + 1) * tamanho_chunk]
                for i in range(num_clients)
            ]
            clientes_y = [
                y[i * tamanho_chunk: (i + 1) * tamanho_chunk]
                for i in range(num_clients)
            ]

            coordinator = FederatedCoordinator(num_clients)
            trainer = STEPGANTrainer(
                X_scaled.shape[1],
                {
                    "latent_dim": latent_dim,
                    "batch": batch_size,
                    "lr_g": 0.0002,
                    "lr_d": 0.0001,
                },
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
                coordinator.run_federated_round(
                    r,
                    trainer,
                    clientes_X,
                    clientes_y,
                )
                progress_bar.progress(r / num_rounds)

            logs_df = pd.DataFrame(trainer.logs)
            st.session_state["trained_model"] = trainer.D
            st.session_state["training_metadata"] = {
                "num_rounds": num_rounds,
                "num_clients": num_clients,
                "latent_dim": latent_dim,
                "batch_size": batch_size,
                "lr_g": 0.0002,
                "lr_d": 0.0001,
                "input_dim": X_scaled.shape[1],
                "dataset": selected_file,
            }
            st.session_state["training_logs"] = trainer.logs

            agora_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            nome_arquivo_checkpoint = f"{agora_str}.txt"
            conteudo_checkpoint = f"""=== CHECKPOINT DO MODELO GLOBAL STEP-GAN ===
Data de Geração: {agora_str}
Dataset: {selected_file}
Rodadas Federadas: {num_rounds}
Clientes Participantes: {num_clients}
Dimensão de Entrada: {X_scaled.shape[1]}
Dimensão Latente: {latent_dim}
Tamanho do Lote: {batch_size}
Learning Rate Gerador: 0.0002
Learning Rate Discriminador: 0.0001
Algoritmo de Agregação: SMPC Aritmética Modular
Status: Treinamento Concluído.
"""
            azure_client.upload_blob(
                "model-checkpoints",
                nome_arquivo_checkpoint,
                conteudo_checkpoint.encode("utf-8"),
            )

            st.markdown(
                f"""<div class="mpes-card" style="border-left:3px solid #00C2A8;">
                    {status_badge("TREINAMENTO CONCLUÍDO", "ok")}
                    <p style="margin-top:10px; font-size:13px; color:#8A93B8;">
                    Checkpoint <code>{nome_arquivo_checkpoint}</code> salvo em <code>model-checkpoints/</code>.
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )

    if "training_logs" in st.session_state:
        section_header("3", "Métricas do Treinamento")
        logs_df = pd.DataFrame(st.session_state["training_logs"])
        _show_training_metrics(logs_df)
        _show_training_charts(logs_df)

        metadata = st.session_state.get("training_metadata", {})
        st.markdown("#### Configuração utilizada")
        config_cols = st.columns(8)
        config_items = [
            ("Dataset", metadata.get("dataset", "-")),
            ("Clientes", metadata.get("num_clients", "-")),
            ("Rodadas", metadata.get("num_rounds", "-")),
            ("Input dim", metadata.get("input_dim", "-")),
            ("Latent dim", metadata.get("latent_dim", "-")),
            ("Batch", metadata.get("batch_size", "-")),
            ("LR G", metadata.get("lr_g", "-")),
            ("LR D", metadata.get("lr_d", "-")),
        ]
        for col, (label, value) in zip(config_cols, config_items):
            with col:
                st.markdown(metric_card(label, str(value), "", "#E4E8FA"), unsafe_allow_html=True)

    if "trained_model" in st.session_state:
        section_header("4", "Detecção de Anomalias (Inferência)")
        st.markdown(
            '<div class="mpes-card"><span style="color:#8A93B8; font-size:13px;">'
            "Faça upload do arquivo de transações para verificar fraudes usando o modelo global."
            "</span></div>",
            unsafe_allow_html=True,
        )

        arquivo_teste = st.file_uploader("Upload de arquivo CSV de teste", type=["csv"])
        if arquivo_teste is not None and st.button("🔍 Executar Inferência do Discriminador"):
            with st.spinner("Avaliando transações e calculando anomalias..."):
                df_teste = _clean_test_csv(
                    pd.read_csv(arquivo_teste, sep=None, engine="python")
                )
                tem_gabarito = "Class" in df_teste.columns
                preprocessor = st.session_state["preprocessor"]
                X_test_scaled, _, df_limpo = preprocessor.process_and_scale(
                    df_teste,
                    is_training=False,
                )

                if len(X_test_scaled) == 0:
                    st.error("⚠️ O arquivo de teste submetido ficou vazio após a limpeza.")
                    return

                D_global = st.session_state["trained_model"]
                D_global.eval()
                inicio_infer = time.time()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test_scaled).to(device)
                    prob_normal = D_global(X_tensor).cpu().numpy().flatten()
                    anomaly_scores = 1.0 - prob_normal

                sc_min, sc_max = anomaly_scores.min(), anomaly_scores.max()
                if sc_max > sc_min:
                    anomaly_scores = (anomaly_scores - sc_min) / (sc_max - sc_min)
                anomaly_scores = np.clip(anomaly_scores, 0.0, 1.0)
                tempo_infer = time.time() - inicio_infer
                tps = len(df_limpo) / tempo_infer if tempo_infer > 0 else 0

                optimal_threshold = 0.85
                y_test_real = df_teste["Class"].dropna().values if tem_gabarito else None
                if tem_gabarito and len(np.unique(y_test_real)) > 1:
                    precision, recall, thresholds = precision_recall_curve(
                        y_test_real,
                        anomaly_scores,
                    )
                    valid_idx = np.where(precision[:-1] >= 0.70)[0]
                    if len(valid_idx) > 0:
                        best_idx = valid_idx[np.argmax(recall[valid_idx])]
                        optimal_threshold = thresholds[best_idx]

                y_pred = (anomaly_scores >= optimal_threshold).astype(int)
                df_limpo["Score_Anomalia"] = anomaly_scores * 100
                df_limpo["Alerta_Fraude"] = y_pred
                qtd_alertas = int(df_limpo["Alerta_Fraude"].sum())

                st.session_state["inference_results"] = df_limpo
                st.session_state["inference_metadata"] = {
                    "threshold": optimal_threshold,
                    "tps": tps,
                    "transactions": len(df_limpo),
                    "alerts": qtd_alertas,
                }

                if qtd_alertas > 0:
                    st.error(f"⚠️ {qtd_alertas} transações suspeitas detectadas.")
                else:
                    st.success("✅ Nenhuma anomalia detectada.")

                if tem_gabarito and len(np.unique(y_test_real)) > 1:
                    metrics = {
                        "Acurácia": accuracy_score(y_test_real, y_pred),
                        "Precisão": precision_score(y_test_real, y_pred, zero_division=0),
                        "Recall": recall_score(y_test_real, y_pred, zero_division=0),
                        "F1-Score": f1_score(y_test_real, y_pred, zero_division=0),
                        "ROC-AUC": roc_auc_score(y_test_real, anomaly_scores),
                        "PR-AUC": average_precision_score(y_test_real, anomaly_scores),
                        "TPS": tps,
                        "Threshold": optimal_threshold,
                    }
                    st.session_state["inference_metrics"] = metrics

            if "inference_metrics" in st.session_state:
                section_header("5", "Métricas de Inferência")
                metrics = st.session_state["inference_metrics"]
                metric_cols = st.columns(8)
                for col, (name, value) in zip(metric_cols, metrics.items()):
                    display = f"{value:.4f}" if isinstance(value, float) else str(value)
                    with col:
                        st.markdown(metric_card(name, display, "", "#E4E8FA"), unsafe_allow_html=True)

            if "inference_results" in st.session_state:
                results = st.session_state["inference_results"]
                section_header("6", "Distribuição e Transações Priorizadas")
                chart_col, table_col = st.columns([1, 2], gap="large")
                with chart_col:
                    counts, bins = np.histogram(results["Score_Anomalia"], bins=30)
                    st.area_chart(
                        pd.DataFrame({"Transações": counts}, index=bins[:-1]),
                        color=PRIMARY,
                    )
                with table_col:
                    st.dataframe(
                        results.sort_values("Score_Anomalia", ascending=False).head(1000),
                        use_container_width=True,
                    )