import numpy as np
import pandas as pd
import streamlit as st

from theme import (
    apply_theme,
    metric_card,
    page_header,
    section_header,
)


PRIMARY = "#2563EB"
ACCENT = "#0F766E"
DANGER = "#B91C1C"
WARNING = "#B45309"


def _format_metric(value):
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def _show_metric_grid(metrics):
    if not metrics:
        return

    items = list(metrics.items())

    for start in range(0, len(items), 4):
        row = items[start:start + 4]
        columns = st.columns(len(row))

        for column, (name, value) in zip(columns, row):
            with column:
                st.markdown(
                    metric_card(
                        name,
                        _format_metric(value),
                        "",
                        PRIMARY,
                    ),
                    unsafe_allow_html=True,
                )


def _show_general_indicators(metadata):
    section_header("1", "Indicadores gerais")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            metric_card(
                "Rodadas federadas",
                str(metadata.get("num_rounds", 0)),
                "rodadas concluídas",
                PRIMARY,
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            metric_card(
                "Clientes ativos",
                str(metadata.get("num_clients", 0)),
                "silos participantes",
                PRIMARY,
            ),
            unsafe_allow_html=True,
        )

    if "inference_results" not in st.session_state:
        with col3:
            st.markdown(
                metric_card(
                    "Transações avaliadas",
                    "0",
                    "aguardando inferência",
                    "#64748B",
                ),
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                metric_card(
                    "Anomalias detectadas",
                    "0",
                    "aguardando inferência",
                    "#64748B",
                ),
                unsafe_allow_html=True,
            )
        return

    result_df = st.session_state.inference_results
    total_transactions = len(result_df)
    total_alerts = int(result_df["Alerta Fraude"].sum())

    alert_rate = (
        total_alerts / total_transactions * 100
        if total_transactions
        else 0
    )

    alert_color = (
        DANGER
        if alert_rate >= 5
        else WARNING
        if alert_rate > 0
        else ACCENT
    )

    with col3:
        st.markdown(
            metric_card(
                "Transações avaliadas",
                f"{total_transactions:,}".replace(",", "."),
                "lote de teste",
                PRIMARY,
            ),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            metric_card(
                "Anomalias detectadas",
                str(total_alerts),
                f"{alert_rate:.2f}% do total",
                alert_color,
            ),
            unsafe_allow_html=True,
        )


def _show_configuration(metadata):
    section_header("2", "Configuração do treinamento")

    config_items = [
        ("Dataset", metadata.get("dataset", "-")),
        ("Dimensão de entrada", metadata.get("input_dim", "-")),
        ("Dimensão latente", metadata.get("latent_dim", "-")),
        ("Batch size", metadata.get("batch_size", "-")),
        ("Learning rate G", metadata.get("lr_g", "-")),
        ("Learning rate D", metadata.get("lr_d", "-")),
    ]

    for start in range(0, len(config_items), 3):
        row = config_items[start:start + 3]
        columns = st.columns(len(row))

        for column, (label, value) in zip(columns, row):
            with column:
                st.markdown(
                    metric_card(
                        label,
                        str(value),
                        "",
                        PRIMARY,
                    ),
                    unsafe_allow_html=True,
                )


def _show_training_metrics(logs_df):
    section_header("3", "Métricas do treinamento")

    if logs_df.empty:
        st.info("Nenhum log de treinamento disponível.")
        return

    logs_table = logs_df.rename(
        columns={
            "client": "Cliente",
            "round": "Rodada",
            "loss_g": "Loss Gerador",
            "loss_d": "Loss Discriminador",
        }
    ).copy()

    for column in ["Loss Gerador", "Loss Discriminador"]:
        if column in logs_table.columns:
            logs_table[column] = logs_table[column].astype(float).round(6)

    st.dataframe(
        logs_table,
        width="stretch",
        hide_index=True,
    )

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

    st.markdown("#### Resumo estatístico por rodada")
    st.dataframe(summary.round(6), width="stretch")

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

    st.markdown("#### Convergência das perdas")
    st.line_chart(loss_summary, color=[PRIMARY, DANGER])

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Loss médio do gerador**")
        st.bar_chart(
            loss_summary[["Loss Gerador"]],
            color=PRIMARY,
        )

    with chart_col2:
        st.markdown("**Loss médio do discriminador**")
        st.bar_chart(
            loss_summary[["Loss Discriminador"]],
            color=DANGER,
        )


def _show_inference_metrics():
    section_header("4", "Métricas de inferência")

    metrics = st.session_state.get(
        "inference_metrics",
        {},
    )

    if metrics:
        _show_metric_grid(metrics)
    else:
        st.info(
            "As métricas aparecerão após a avaliação de um CSV "
            "de teste com rótulos."
        )


def _show_anomaly_distribution(metadata):
    if "inference_results" not in st.session_state:
        return

    result_df = st.session_state.inference_results

    section_header("5", "Distribuição do escore de anomalia")

    counts, bins = np.histogram(
        result_df["Score Anomalia"],
        bins=40,
    )

    chart_df = pd.DataFrame(
        {"Transações": counts},
        index=bins[:-1],
    )

    st.area_chart(chart_df, color=PRIMARY)

    section_header("6", "Alertas por cliente federado")

    total_alerts = int(result_df["Alerta Fraude"].sum())

    if total_alerts == 0:
        st.success("Nenhum alerta detectado no conjunto avaliado.")
        return

    number_clients = max(
        int(metadata.get("num_clients", 1)),
        1,
    )

    np.random.seed(42)

    alerts_by_client = np.random.multinomial(
        total_alerts,
        [1 / number_clients] * number_clients,
    )

    labels = [
        f"Cliente {index + 1}"
        for index in range(number_clients)
    ]

    alerts_df = pd.DataFrame(
        {"Alertas": alerts_by_client},
        index=labels,
    )

    st.bar_chart(alerts_df, color=DANGER)

    st.caption(
        "Distribuição ilustrativa: o modelo global não rastreia "
        "a origem exata do alerta por cliente."
    )


def view_dashboard():
    """Renderiza a subvisão corporativa do dashboard."""
    apply_theme()

    page_header(
        "📊",
        "Dashboard de Auditoria",
        "Indicadores, métricas, perdas e alertas identificados.",
    )

    if "training_metadata" not in st.session_state:
        st.markdown(
            """
            <div class="alert alert-info shadow-sm" role="alert">
                <i class="bi bi-info-circle-fill"></i>
                <strong>Nenhum treinamento executado</strong>
                <p class="mb-0 mt-2">
                    Execute o treinamento para alimentar o dashboard.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    metadata = st.session_state.training_metadata
    logs_df = pd.DataFrame(
        st.session_state.get("training_logs", [])
    )

    _show_general_indicators(metadata)
    _show_configuration(metadata)
    _show_training_metrics(logs_df)
    _show_inference_metrics()
    _show_anomaly_distribution(metadata)


# Compatibilidade com o nome utilizado anteriormente.
render_auditor = view_dashboard


if __name__ == "__main__":
    view_dashboard()    