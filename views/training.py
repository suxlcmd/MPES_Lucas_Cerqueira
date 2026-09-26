import datetime
import io
import time

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cloud.azure_blob_client import AzureBlobClient
from ml_engine.step_gan_trainer import STEPGANTrainer
from orchestration.data_preprocessor import DataPreprocessor
from orchestration.federated_coordinator import FederatedCoordinator
from theme import (
    apply_theme,
    metric_card,
    page_header,
    section_header,
    status_badge,
)


PRIMARY = "#2563EB"
ACCENT = "#0F766E"
DANGER = "#B91C1C"
SUCCESS = "#15803D"
WARNING = "#B45309"


try:
    import torch

    TORCH_AVAILABLE = True
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"


def _clean_test_csv(df_test):
    """Limpa e converte o CSV usado na inferência."""
    df_test = df_test.copy()

    if len(df_test.columns) == 1:
        column_name = df_test.columns[0]

        if (
            df_test[column_name]
            .astype(str)
            .str.contains(",", regex=False)
            .any()
        ):
            expanded = (
                df_test[column_name]
                .astype(str)
                .str.split(",", expand=True)
            )

            new_columns = str(column_name).split(",")

            if len(new_columns) == expanded.shape[1]:
                expanded.columns = new_columns
                df_test = expanded

    df_test.columns = [
        str(column).strip().replace('"', "").replace("'", "")
        for column in df_test.columns
    ]

    for column in df_test.columns:
        if df_test[column].dtype == object:
            df_test[column] = (
                df_test[column]
                .astype(str)
                .str.strip(' "\'')
            )

    return df_test.apply(pd.to_numeric, errors="coerce")


def _show_training_metrics(logs_df):
    """Exibe tabela e resumo das perdas de treinamento."""
    if logs_df.empty:
        st.info("Nenhum registro de treinamento disponível.")
        return

    st.markdown(
        "<div class='card shadow-sm border-0 p-3 mb-3'>"
        "<div class='fw-semibold text-primary mb-2'>"
        "Valores das métricas por cliente e rodada"
        "</div>",
        unsafe_allow_html=True,
    )

    table = logs_df.rename(
        columns={
            "client": "Cliente",
            "round": "Rodada",
            "loss_g": "Loss Gerador",
            "loss_d": "Loss Discriminador",
        }
    ).copy()

    for column in ["Loss Gerador", "Loss Discriminador"]:
        if column in table.columns:
            table[column] = table[column].astype(float).round(6)

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    summary = (
        logs_df
        .groupby("round")[["loss_g", "loss_d"]]
        .agg(["mean", "min", "max"])
    )

    summary.columns = [
        f"{metric} - {stat}"
        for metric, stat in summary.columns.to_flat_index()
    ]
    summary.index.name = "Rodada"

    st.markdown(
        "<div class='card shadow-sm border-0 p-3 mb-3'>"
        "<div class='fw-semibold text-primary mb-2'>"
        "Resumo estatístico por rodada"
        "</div>",
        unsafe_allow_html=True,
    )

    st.dataframe(
        summary.round(6),
        width="stretch",
    )

    st.markdown("</div>", unsafe_allow_html=True)


def _show_training_charts(logs_df):
    """Exibe os gráficos de evolução das perdas."""
    if logs_df.empty:
        return

    loss_summary = (
        logs_df
        .groupby("round")[["loss_g", "loss_d"]]
        .mean()
        .rename(
            columns={
                "loss_g": "Loss Gerador",
                "loss_d": "Loss Discriminador",
            }
        )
    )

    st.markdown(
        "<div class='card shadow-sm border-0 p-3 mb-3'>"
        "<div class='fw-semibold text-primary mb-3'>"
        "Evolução das perdas"
        "</div>",
        unsafe_allow_html=True,
    )

    st.line_chart(
        loss_summary,
        color=[PRIMARY, DANGER],
    )

    st.markdown("</div>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown(
            "<div class='card shadow-sm border-0 p-3 mb-3'>"
            "<div class='small text-secondary mb-2'>"
            "Loss médio do gerador"
            "</div>",
            unsafe_allow_html=True,
        )

        st.bar_chart(
            loss_summary[["Loss Gerador"]],
            color=PRIMARY,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown(
            "<div class='card shadow-sm border-0 p-3 mb-3'>"
            "<div class='small text-secondary mb-2'>"
            "Loss médio do discriminador"
            "</div>",
            unsafe_allow_html=True,
        )

        st.bar_chart(
            loss_summary[["Loss Discriminador"]],
            color=DANGER,
        )

        st.markdown("</div>", unsafe_allow_html=True)


def _prepare_client_data(X_scaled, y, num_clients):
    """Divide os dados entre os clientes federados."""
    if num_clients <= 0:
        raise ValueError(
            "O número de clientes deve ser maior que zero."
        )

    if len(X_scaled) == 0:
        raise ValueError("A base de dados está vazia.")

    indexes_by_client = np.array_split(
        np.arange(len(X_scaled)),
        num_clients,
    )

    clients_X = [
        X_scaled[indexes]
        for indexes in indexes_by_client
        if len(indexes) > 0
    ]

    clients_y = [
        y[indexes]
        for indexes in indexes_by_client
        if len(indexes) > 0
    ]

    return clients_X, clients_y


def _save_processed_data(azure_client, original_filename, X_scaled):
    """Salva os dados processados no Azure."""
    csv_buffer = io.BytesIO()
    pd.DataFrame(X_scaled).to_csv(csv_buffer, index=False)

    return azure_client.upload_blob(
        "processed-data",
        f"processed_{original_filename}",
        csv_buffer.getvalue(),
    )


def _calculate_threshold(y_true, anomaly_scores):
    """Calcula um limiar usando a curva Precision-Recall."""
    default_threshold = 0.85

    if y_true is None:
        return default_threshold

    y_true = np.asarray(y_true)

    if len(np.unique(y_true)) < 2:
        return default_threshold

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        anomaly_scores,
    )

    if len(thresholds) == 0:
        return default_threshold

    valid_indexes = np.where(precision[:-1] >= 0.70)[0]

    if len(valid_indexes) == 0:
        return default_threshold

    best_index = valid_indexes[np.argmax(recall[valid_indexes])]
    return float(thresholds[best_index])


def _resolve_discriminator(model_or_trainer):
    """Aceita um trainer ou um Discriminator diretamente."""
    discriminator = (
        model_or_trainer.D
        if hasattr(model_or_trainer, "D")
        else model_or_trainer
    )

    if not callable(discriminator):
        raise TypeError(
            "O modelo salvo não é um Discriminator válido."
        )

    return discriminator


def _run_inference(df_test, model_or_trainer, preprocessor):
    """Executa a inferência usando o discriminador global."""
    df_test = _clean_test_csv(df_test)
    has_labels = "Class" in df_test.columns

    X_test_scaled, _, df_clean = preprocessor.process_and_scale(
        df_test,
        is_training=False,
    )

    if len(X_test_scaled) == 0:
        raise ValueError(
            "O arquivo de teste ficou vazio após a limpeza."
        )

    discriminator = _resolve_discriminator(model_or_trainer)
    discriminator.eval()

    start_time = time.time()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
        probability_normal = (
            discriminator(X_tensor)
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )

    anomaly_scores = 1.0 - probability_normal
    score_min = anomaly_scores.min()
    score_max = anomaly_scores.max()

    if score_max > score_min:
        anomaly_scores = (
            anomaly_scores - score_min
        ) / (score_max - score_min)

    anomaly_scores = np.clip(
        anomaly_scores,
        0.0,
        1.0,
    )

    elapsed_time = time.time() - start_time
    transactions_per_second = (
        len(df_clean) / elapsed_time
        if elapsed_time > 0
        else 0
    )

    y_test = None

    if has_labels:
        y_test = (
            df_test["Class"]
            .dropna()
            .astype(int)
            .values
        )

    threshold = _calculate_threshold(
        y_test,
        anomaly_scores,
    )

    predictions = (
        anomaly_scores >= threshold
    ).astype(int)

    result_df = df_clean.copy()
    result_df["Score Anomalia"] = anomaly_scores * 100
    result_df["Alerta Fraude"] = predictions

    metrics = {}

    if (
        y_test is not None
        and len(np.unique(y_test)) >= 2
        and len(y_test) == len(predictions)
    ):
        metrics = {
            "Acurácia": accuracy_score(
                y_test,
                predictions,
            ),
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
            "ROC-AUC": roc_auc_score(
                y_test,
                anomaly_scores,
            ),
            "PR-AUC": average_precision_score(
                y_test,
                anomaly_scores,
            ),
            "TPS": transactions_per_second,
            "Threshold": threshold,
        }

    return (
        result_df,
        metrics,
        threshold,
        transactions_per_second,
    )


def _show_inference_metrics(metrics):
    """Exibe as métricas da inferência em cards corporativos."""
    if not metrics:
        return

    st.markdown(
        "<div class='row g-3 mb-3'>",
        unsafe_allow_html=True,
    )

    columns = st.columns(min(len(metrics), 4))

    for column, (name, value) in zip(columns, metrics.items()):
        display_value = (
            f"{value:.4f}"
            if isinstance(value, (float, np.floating))
            else str(value)
        )

        with column:
            st.markdown(
                metric_card(
                    name,
                    display_value,
                    "métrica calculada",
                    PRIMARY,
                ),
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


def _render_inference_section():
    """Renderiza a seção de teste e inferência."""
    if "trained_model" not in st.session_state:
        return

    section_header(
        "4",
        "Detecção de anomalias e inferência",
    )

    st.markdown(
        """
        <div class="alert alert-primary shadow-sm" role="alert">
            <i class="bi bi-shield-check"></i>
            Faça upload de um CSV de transações para avaliar o modelo global.
        </div>
        """,
        unsafe_allow_html=True,
    )

    test_file = st.file_uploader(
        "Upload de arquivo CSV de teste",
        type=["csv"],
        key="technical_test_csv",
    )

    if test_file is None:
        return

    if not st.button(
        "Executar inferência do discriminador",
        width="stretch",
        key="technical_inference_button",
    ):
        return

    try:
        with st.spinner(
            "Avaliando transações e calculando anomalias..."
        ):
            df_test = pd.read_csv(
                test_file,
                sep=None,
                engine="python",
            )

            result_df, metrics, threshold, tps = _run_inference(
                df_test,
                st.session_state.trained_model,
                st.session_state.preprocessor,
            )

        alerts = int(result_df["Alerta Fraude"].sum())

        st.session_state.inference_results = result_df
        st.session_state.inference_metadata = {
            "threshold": threshold,
            "tps": tps,
            "transactions": len(result_df),
            "alerts": alerts,
        }
        st.session_state.inference_metrics = metrics

        if alerts > 0:
            st.warning(
                f"{alerts} transação(ões) suspeita(s) detectada(s)."
            )
        else:
            st.success("Nenhuma anomalia detectada.")

    except Exception as error:
        st.error(f"Erro durante a inferência: {error}")
        return

    if metrics:
        section_header("5", "Métricas de inferência")
        _show_inference_metrics(metrics)

    section_header(
        "6",
        "Distribuição e transações priorizadas",
    )

    chart_col, table_col = st.columns([1, 2], gap="large")

    with chart_col:
        st.markdown(
            "<div class='card shadow-sm border-0 p-3'>"
            "<div class='fw-semibold text-primary mb-3'>"
            "Distribuição do escore"
            "</div>",
            unsafe_allow_html=True,
        )

        counts, bins = np.histogram(
            result_df["Score Anomalia"],
            bins=30,
        )

        chart_df = pd.DataFrame(
            {"Transações": counts},
            index=bins[:-1],
        )

        st.area_chart(
            chart_df,
            color=PRIMARY,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with table_col:
        st.markdown(
            "<div class='card shadow-sm border-0 p-3'>"
            "<div class='fw-semibold text-primary mb-3'>"
            "Transações priorizadas"
            "</div>",
            unsafe_allow_html=True,
        )

        st.dataframe(
            result_df.sort_values(
                "Score Anomalia",
                ascending=False,
            ).head(1000),
            width="stretch",
        )

        st.markdown("</div>", unsafe_allow_html=True)


def view_training():
    """Renderiza a view corporativa de treinamento e inferência."""
    apply_theme()

    page_header(
        "⚙️",
        "Treinamento e Inferência",
        "Treinamento federado, análise de transações e detecção de anomalias.",
    )

    azure_client = AzureBlobClient()

    section_header(
        "1",
        "Seleção de dados de treinamento",
    )

    raw_files = azure_client.list_blobs("raw-data")

    if not raw_files:
        st.markdown(
            """
            <div class="alert alert-warning shadow-sm" role="alert">
                <i class="bi bi-database-exclamation"></i>
                <strong>Nenhum arquivo encontrado.</strong>
                <p class="mb-0 mt-2">
                    Envie um CSV pelo Azure Blob Storage antes de iniciar.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    selected_file = st.selectbox(
        "Dataset de treinamento",
        raw_files,
        key="technical_training_dataset",
    )

    section_header(
        "2",
        "Configuração da orquestração federada",
    )

    st.markdown(
        "<div class='card shadow-sm border-0 p-3 mb-3'>",
        unsafe_allow_html=True,
    )

    with st.form("training_config_form"):
        col1, col2 = st.columns(2)

        with col1:
            num_clients = st.slider(
                "Número de clientes/silos",
                min_value=2,
                max_value=10,
                value=3,
            )

            num_rounds = st.number_input(
                "Rodadas federadas",
                min_value=1,
                max_value=50,
                value=3,
            )

        with col2:
            latent_dim = st.selectbox(
                "Dimensão latente Z",
                [64, 100, 128],
                index=1,
            )

            batch_size = st.selectbox(
                "Tamanho do lote",
                [128, 256, 512],
                index=2,
            )

        submitted = st.form_submit_button(
            "Iniciar orquestração federada",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        _run_training(
            azure_client=azure_client,
            selected_file=selected_file,
            num_clients=num_clients,
            num_rounds=num_rounds,
            latent_dim=latent_dim,
            batch_size=batch_size,
        )

    if "training_logs" in st.session_state:
        section_header("3", "Métricas do treinamento")

        logs_df = pd.DataFrame(
            st.session_state.training_logs
        )

        _show_training_metrics(logs_df)
        _show_training_charts(logs_df)

    _render_inference_section()


def _run_training(
    azure_client,
    selected_file,
    num_clients,
    num_rounds,
    latent_dim,
    batch_size,
):
    """Executa o treinamento federado."""
    if not TORCH_AVAILABLE:
        st.error("O treinamento requer o PyTorch instalado.")
        return

    with st.spinner(
        f"Baixando {selected_file} do Azure..."
    ):
        df_train = azure_client.download_blob_as_dataframe(
            "raw-data",
            selected_file,
        )

    if df_train is None or df_train.empty:
        st.error("Não foi possível carregar o dataset.")
        return

    with st.spinner("Pré-processando dados..."):
        preprocessor = DataPreprocessor()
        X_scaled, y, _ = preprocessor.process_and_scale(
            df_train,
            is_training=True,
        )

    if len(X_scaled) == 0:
        st.error("A base ficou vazia após a limpeza.")
        return

    st.session_state.preprocessor = preprocessor
    st.session_state.input_dim = X_scaled.shape[1]

    _save_processed_data(
        azure_client,
        selected_file,
        X_scaled,
    )

    clients_X, clients_y = _prepare_client_data(
        X_scaled,
        y,
        num_clients,
    )

    coordinator = FederatedCoordinator(len(clients_X))

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
    status_container = st.empty()

    for round_number in range(1, num_rounds + 1):
        with status_container.container():
            st.markdown(
                f"""
                <div class="alert alert-primary shadow-sm" role="alert">
                    <i class="bi bi-arrow-repeat"></i>
                    <strong>Rodada {round_number}/{num_rounds}</strong>
                    <span class="ms-2">
                        Treinamento local e agregação SMPC em andamento.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        coordinator.run_federated_round(
            round_number,
            trainer,
            clients_X,
            clients_y,
        )

        progress_bar.progress(round_number / num_rounds)

    # O estado guarda o discriminador diretamente.
    # A inferência trata também o caso de trainer completo.
    st.session_state.trained_model = trainer.D
    st.session_state.preprocessor = preprocessor
    st.session_state.training_metadata = {
        "num_rounds": num_rounds,
        "num_clients": len(clients_X),
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "lr_g": 0.0002,
        "lr_d": 0.0001,
        "input_dim": X_scaled.shape[1],
        "dataset": selected_file,
    }
    st.session_state.training_logs = trainer.logs

    now_text = datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    checkpoint_filename = (
        f"step_gan_checkpoint_{now_text}.txt"
    )

    checkpoint_content = f"""
CHECKPOINT DO MODELO GLOBAL STEP-GAN

Data de Geração: {now_text}
Dataset: {selected_file}
Rodadas Federadas: {num_rounds}
Clientes Participantes: {len(clients_X)}
Dimensão de Entrada: {X_scaled.shape[1]}
Dimensão Latente: {latent_dim}
Tamanho do Lote: {batch_size}
Learning Rate Gerador: 0.0002
Learning Rate Discriminador: 0.0001
Algoritmo de Agregação: SMPC
Aritmética: Modular
Status: Treinamento Concluído
"""

    uploaded = azure_client.upload_blob(
        "model-checkpoints",
        checkpoint_filename,
        checkpoint_content.encode("utf-8"),
    )

    st.markdown(
        "<div class='card shadow-sm border-0 p-3 mb-3'>",
        unsafe_allow_html=True,
    )

    if uploaded:
        st.markdown(
            status_badge(
                "TREINAMENTO CONCLUÍDO",
                "success",
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Checkpoint salvo em model-checkpoints/"
            f"{checkpoint_filename}."
        )
    else:
        st.markdown(
            status_badge(
                "TREINAMENTO CONCLUÍDO SEM CHECKPOINT",
                "warning",
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            "O treinamento terminou, mas o checkpoint não pôde ser salvo."
        )

    st.markdown("</div>", unsafe_allow_html=True)


# Compatibilidade com o nome usado anteriormente.
render_training = view_training


if __name__ == "__main__":
    view_training()