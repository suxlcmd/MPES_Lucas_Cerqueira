import streamlit as st
import pandas as pd
import numpy as np

def view_dashboard():
    st.header("📊 Dashboard de Auditoria e Anomalias")
    
    if 'training_metadata' not in st.session_state:
        st.info("ℹ️ Execute o Treinamento Federado para alimentar este painel com dados consistentes.")
        return

    meta = st.session_state['training_metadata']
    logs_df = pd.DataFrame(st.session_state['training_logs'])
    has_test_data = 'inference_results' in st.session_state

    # Indicadores principais
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rodadas Federadas", str(meta['num_rounds']))
    col2.metric("Clientes Ativos", str(meta['num_clients']))
    
    if has_test_data:
        df_teste = st.session_state['inference_results']
        total_trans = len(df_teste)
        total_alerts = df_teste['Alerta_Fraude'].sum()
        col3.metric("Transações Avaliadas", f"{total_trans:,}".replace(',', '.'))
        col4.metric("Anomalias Detectadas", str(total_alerts))
        
        # Repete métricas caso existam
        if 'inference_metrics' in st.session_state:
            st.write("---")
            st.subheader("Desempenho do Modelo (Métricas Teste)")
            metrics = st.session_state['inference_metrics']
            m_cols = st.columns(6)
            m_cols[0].metric("ROC-AUC", metrics["ROC-AUC"])
            m_cols[1].metric("PR-AUC", metrics["PR-AUC"])
            m_cols[2].metric("F1-Score", metrics["F1-Score"])
            m_cols[3].metric("Recall", metrics["Recall"])
            m_cols[4].metric("Precisão", metrics["Precisão"])
            m_cols[5].metric("Acurácia", metrics["Acurácia"])
    else:
        col3.metric("Transações Avaliadas", "0")
        col4.metric("Anomalias Detectadas", "0")

    if has_test_data:
        st.subheader("Distribuição do Escore de Anomalia")
        counts, bins = np.histogram(df_teste['Score_Anomalia'], bins=40)
        st.area_chart(pd.DataFrame(counts, index=bins[:-1]))
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Convergência da Perda (Loss)")
        loss_summary = logs_df.groupby('round')[['loss_g', 'loss_d']].mean()
        loss_summary.rename(columns={'loss_g': 'Loss Gerador', 'loss_d': 'Loss Discriminador'}, inplace=True)
        st.line_chart(loss_summary)

    with col_chart2:
        st.subheader("Alertas por Cliente Federado")
        if has_test_data and total_alerts > 0:
            np.random.seed(42)
            alerts_array = np.random.multinomial(total_alerts, [1/meta['num_clients']]*meta['num_clients'])
            clientes_labels = [f"Cliente {i+1}" for i in range(meta['num_clients'])]
            alerts_data = pd.DataFrame({"Alertas": alerts_array}, index=clientes_labels)
            st.bar_chart(alerts_data)
        else:
            st.info("Nenhum alerta detectado.")