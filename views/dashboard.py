import streamlit as st
import pandas as pd
import numpy as np
from theme import apply_theme, page_header, section_header, status_badge, metric_card

ACCENT = "#00C2A8"
DANGER = "#FF4D6D"
PRIMARY = "#4F7CFF"
WARNING = "#FFB020"


def view_dashboard():
    apply_theme()
    page_header(
        "📊",
        "Dashboard de Auditoria e Anomalias",
        "Monitoramento do treinamento federado STEP-GAN e resultados de inferência",
    )

    if 'training_metadata' not in st.session_state:
        st.markdown(
            f"""<div class="mpes-card" style="text-align:center; padding:40px;">
                <div style="font-size:40px; margin-bottom:12px;">🛰️</div>
                <div style="font-weight:700; font-size:15px; margin-bottom:6px;">Nenhum treinamento executado ainda</div>
                <div style="color:#8A93B8; font-size:13px;">
                    Vá até a aba <b>Treinamento Federado</b> para alimentar este painel com dados consistentes.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    meta = st.session_state['training_metadata']
    logs_df = pd.DataFrame(st.session_state['training_logs'])
    has_test_data = 'inference_results' in st.session_state

    # ---------- Indicadores principais ----------
    section_header("1", "Indicadores Gerais")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(metric_card("Rodadas Federadas", str(meta['num_rounds']), "concluídas", PRIMARY), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Clientes Ativos", str(meta['num_clients']), "silos participantes", PRIMARY), unsafe_allow_html=True)

    if has_test_data:
        df_teste = st.session_state['inference_results']
        total_trans = len(df_teste)
        total_alerts = int(df_teste['Alerta_Fraude'].sum())
        alert_rate = (total_alerts / total_trans * 100) if total_trans else 0
        with c3:
            st.markdown(metric_card("Transações Avaliadas", f"{total_trans:,}".replace(',', '.'), "lote de teste", PRIMARY), unsafe_allow_html=True)
        with c4:
            color = DANGER if alert_rate > 5 else WARNING if alert_rate > 0 else ACCENT
            st.markdown(metric_card("Anomalias Detectadas", str(total_alerts), f"{alert_rate:.2f}% do total", color), unsafe_allow_html=True)
    else:
        with c3:
            st.markdown(metric_card("Transações Avaliadas", "0", "aguardando inferência", "#8A93B8"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Anomalias Detectadas", "0", "aguardando inferência", "#8A93B8"), unsafe_allow_html=True)

    # ---------- Métricas de qualidade ----------
    if 'inference_metrics' in st.session_state:
        section_header("2", "Desempenho do Modelo (Conjunto de Teste)")
        metrics = st.session_state['inference_metrics']
        keys = ["ROC-AUC", "PR-AUC", "F1-Score", "Recall", "Precisão", "Acurácia"]
        cols = st.columns(len(keys))
        for col, k in zip(cols, keys):
            with col:
                st.markdown(metric_card(k, metrics[k], "", "#E4E8FA"), unsafe_allow_html=True)

    # ---------- Gráficos ----------
    if has_test_data:
        section_header("3", "Distribuição do Escore de Anomalia")
        st.markdown('<div class="mpes-card">', unsafe_allow_html=True)
        counts, bins = np.histogram(df_teste['Score_Anomalia'], bins=40)
        st.area_chart(pd.DataFrame(counts, index=bins[:-1], columns=["Transações"]), color=PRIMARY)
        st.markdown("</div>", unsafe_allow_html=True)

    section_header("4", "Convergência e Distribuição por Cliente")
    col_chart1, col_chart2 = st.columns(2, gap="large")

    with col_chart1:
        st.markdown('<div class="mpes-card">', unsafe_allow_html=True)
        st.markdown("**📉 Convergência da Perda (Loss)**")
        loss_summary = logs_df.groupby('round')[['loss_g', 'loss_d']].mean()
        loss_summary.rename(columns={'loss_g': 'Loss Gerador', 'loss_d': 'Loss Discriminador'}, inplace=True)
        st.line_chart(loss_summary, color=[PRIMARY, DANGER])
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown('<div class="mpes-card">', unsafe_allow_html=True)
        st.markdown("**🚨 Alertas por Cliente Federado**")
        if has_test_data and total_alerts > 0:
            np.random.seed(42)
            alerts_array = np.random.multinomial(total_alerts, [1 / meta['num_clients']] * meta['num_clients'])
            clientes_labels = [f"Cliente {i+1}" for i in range(meta['num_clients'])]
            alerts_data = pd.DataFrame({"Alertas": alerts_array}, index=clientes_labels)
            st.bar_chart(alerts_data, color=DANGER)
            st.caption("⚠️ Distribuição ilustrativa — o modelo global não rastreia a origem exata do alerta por cliente.")
        else:
            st.markdown(
                """<div style="text-align:center; padding:30px 0; color:#8A93B8; font-size:13px;">
                    ✅ Nenhum alerta detectado no conjunto avaliado.
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)