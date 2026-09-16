import streamlit as st
import pandas as pd
import numpy as np
from theme import apply_theme, page_header, section_header, metric_card

ACCENT = "#00C2A8"
DANGER = "#FF4D6D"
PRIMARY = "#4F7CFF"
WARNING = "#FFB020"


def _format_metric(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _show_metric_grid(metrics):
    """Renderiza todos os valores disponíveis, sem ocultar métricas extras."""
    if not metrics:
        return

    items = list(metrics.items())
    for start in range(0, len(items), 6):
        row = items[start:start + 6]
        cols = st.columns(len(row))
        for col, (name, value) in zip(cols, row):
            with col:
                st.markdown(
                    metric_card(name, _format_metric(value), "", "#E4E8FA"),
                    unsafe_allow_html=True,
                )


def view_dashboard():
    apply_theme()
    page_header(
        "📊",
        "Dashboard de Auditoria e Anomalias",
        "Monitoramento completo do treinamento federado STEP-GAN e resultados de inferência",
    )

    if "training_metadata" not in st.session_state:
        st.markdown(
            """<div class="mpes-card" style="text-align:center; padding:40px;">
                <div style="font-size:40px; margin-bottom:12px;">🛰️</div>
                <div style="font-weight:700; font-size:15px; margin-bottom:6px;">Nenhum treinamento executado ainda</div>
                <div style="color:#8A93B8; font-size:13px;">Execute o treinamento para alimentar este painel.</div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    meta = st.session_state["training_metadata"]
    logs_df = pd.DataFrame(st.session_state.get("training_logs", []))
    has_test_data = "inference_results" in st.session_state

    section_header("1", "Indicadores Gerais")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Rodadas Federadas", str(meta.get("num_rounds", "-")), "concluídas", PRIMARY), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Clientes Ativos", str(meta.get("num_clients", "-")), "silos participantes", PRIMARY), unsafe_allow_html=True)

    if has_test_data:
        df_teste = st.session_state["inference_results"]
        total_trans = len(df_teste)
        total_alerts = int(df_teste["Alerta_Fraude"].sum())
        alert_rate = total_alerts / total_trans * 100 if total_trans else 0
        with c3:
            st.markdown(metric_card("Transações Avaliadas", f"{total_trans:,}".replace(",", "."), "lote de teste", PRIMARY), unsafe_allow_html=True)
        with c4:
            color = DANGER if alert_rate > 5 else WARNING if alert_rate > 0 else ACCENT
            st.markdown(metric_card("Anomalias Detectadas", str(total_alerts), f"{alert_rate:.2f}% do total", color), unsafe_allow_html=True)
    else:
        with c3:
            st.markdown(metric_card("Transações Avaliadas", "0", "aguardando inferência", "#8A93B8"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Anomalias Detectadas", "0", "aguardando inferência", "#8A93B8"), unsafe_allow_html=True)

    section_header("2", "Configuração Utilizada no Treinamento")
    config_items = [
        ("Dataset", meta.get("dataset", "-")),
        ("Dimensão de entrada", meta.get("input_dim", "-")),
        ("Dimensão latente", meta.get("latent_dim", "-")),
        ("Batch size", meta.get("batch_size", "-")),
        ("Learning rate G", meta.get("lr_g", "-")),
        ("Learning rate D", meta.get("lr_d", "-")),
    ]
    for start in range(0, len(config_items), 6):
        row = config_items[start:start + 6]
        cols = st.columns(len(row))
        for col, (label, value) in zip(cols, row):
            with col:
                st.markdown(metric_card(label, str(value), "", "#E4E8FA"), unsafe_allow_html=True)

    section_header("3", "Todas as Métricas do Treinamento")
    if logs_df.empty:
        st.info("Nenhum log de treinamento disponível.")
    else:
        logs_table = logs_df.rename(
            columns={
                "client": "Cliente",
                "round": "Rodada",
                "loss_g": "Loss Gerador",
                "loss_d": "Loss Discriminador",
            }
        ).copy()
        for col in ["Loss Gerador", "Loss Discriminador"]:
            if col in logs_table.columns:
                logs_table[col] = logs_table[col].astype(float).round(6)
        st.dataframe(logs_table, use_container_width=True, hide_index=True)

        summary = logs_df.groupby("round")[["loss_g", "loss_d"]].agg(["mean", "min", "max"])
        summary.columns = [f"{metric} - {stat}" for metric, stat in summary.columns.to_flat_index()]
        summary.index.name = "Rodada"
        st.markdown("#### Resumo estatístico por rodada")
        st.dataframe(summary.round(6), use_container_width=True)

        loss_summary = logs_df.groupby("round")[["loss_g", "loss_d"]].mean()
        loss_summary = loss_summary.rename(columns={"loss_g": "Loss Gerador", "loss_d": "Loss Discriminador"})
        st.markdown("#### Convergência das perdas")
        st.line_chart(loss_summary, color=[PRIMARY, DANGER])

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**Loss médio do gerador por rodada**")
            st.bar_chart(loss_summary[["Loss Gerador"]], color=PRIMARY)
        with chart_col2:
            st.markdown("**Loss médio do discriminador por rodada**")
            st.bar_chart(loss_summary[["Loss Discriminador"]], color=DANGER)

    section_header("4", "Todas as Métricas de Inferência")
    if "inference_metrics" in st.session_state:
        _show_metric_grid(st.session_state["inference_metrics"])
    else:
        st.info("As métricas de inferência aparecerão depois da avaliação de um arquivo de teste com rótulos.")

    if has_test_data:
        section_header("5", "Distribuição do Escore de Anomalia")
        counts, bins = np.histogram(df_teste["Score_Anomalia"], bins=40)
        st.area_chart(pd.DataFrame(counts, index=bins[:-1], columns=["Transações"]), color=PRIMARY)

        section_header("6", "Alertas por Cliente Federado")
        if total_alerts > 0:
            np.random.seed(42)
            alerts_array = np.random.multinomial(
                total_alerts,
                [1 / meta["num_clients"]] * meta["num_clients"],
            )
            clientes_labels = [f"Cliente {i + 1}" for i in range(meta["num_clients"])]
            alerts_data = pd.DataFrame({"Alertas": alerts_array}, index=clientes_labels)
            st.bar_chart(alerts_data, color=DANGER)
            st.caption("Distribuição ilustrativa: o modelo global não rastreia a origem exata do alerta por cliente.")
        else:
            st.success("Nenhum alerta detectado no conjunto avaliado.")