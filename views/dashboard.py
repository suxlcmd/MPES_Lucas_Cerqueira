"""Dashboard de auditoria (Camada 1): somente dados reais do modelo ativo, da última análise e da trilha."""

import pandas as pd
import streamlit as st

from orchestration.inference_service import RISK_LEVELS, TRIAGE_STATUS
from theme import note, page_header, section_header
from views.common import ACCENT, DANGER, MUTED, PRIMARY, WARNING, metric_grid, model_metric_items


def _indicadores(bundle, inferencia):
    meta = bundle.metadata
    settings = meta.get("settings", {})
    itens = [
        ("Modelo", meta.get("model_id", "-"), meta.get("created_at", "")[:10], PRIMARY),
        ("Rodadas federadas", str(settings.get("num_rounds", "-")), f"melhor: {meta.get('best_round', '-')}", PRIMARY),
        ("Clientes / silos", str(settings.get("num_clients", "-")), "agregação SMPC", PRIMARY),
        ("Limiar de alerta", f"{bundle.threshold * 100:.1f}", "escore 0–100", DANGER),
    ]
    if inferencia:
        resumo = inferencia["result"].summary
        itens += [
            ("Transações avaliadas", f"{resumo['transactions']:,}".replace(",", "."), inferencia["source"], PRIMARY),
            ("Alertas", str(resumo["alerts"]), f"{resumo['alert_rate'] * 100:.2f}% do total",
             DANGER if resumo["alert_rate"] >= 0.05 else WARNING if resumo["alerts"] else ACCENT),
        ]
    else:
        itens += [
            ("Transações avaliadas", "0", "aguardando análise", MUTED),
            ("Alertas", "0", "aguardando análise", MUTED),
        ]
    metric_grid(itens, per_row=3)


def _desempenho(bundle):
    teste = bundle.metrics.get("teste", {})
    if not teste:
        st.info("O modelo foi treinado sem rótulos: não há métricas supervisionadas.")
        return
    metric_grid(model_metric_items(teste), per_row=6)
    validacao = bundle.metrics.get("validacao", {})
    if validacao:
        st.dataframe(
            pd.DataFrame({"Validação (20%)": validacao, "Teste reservado (20%)": teste}).round(4),
            width="stretch",
        )
    note(f"Limiar: {bundle.threshold_strategy}. As métricas de teste usam dados que não participaram do treino "
         "nem da escolha do limiar.")


def _convergencia(bundle):
    historico = pd.DataFrame(bundle.history)
    if historico.empty:
        return
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Validação por rodada federada**")
        st.line_chart(
            historico.set_index("rodada")[["pr_auc_val", "roc_auc_val"]]
            .rename(columns={"pr_auc_val": "PR-AUC", "roc_auc_val": "ROC-AUC"}),
            color=[PRIMARY, ACCENT],
        )
    with col2:
        st.markdown("**Perdas médias dos clientes por rodada**")
        st.line_chart(
            historico.set_index("rodada")[["loss_d", "loss_g"]]
            .rename(columns={"loss_d": "Loss Discriminador", "loss_g": "Loss Geradores"}),
            color=[DANGER, PRIMARY],
        )


def _privacidade(bundle):
    historico = pd.DataFrame(bundle.history)
    if historico.empty:
        return
    erro = historico.get("smpc_max_error")
    metric_grid([
        ("Tempo SMPC por rodada", f"{historico['smpc_seconds'].mean():.3f} s", "divisão + somas + revelação", PRIMARY),
        ("Enviado por cliente", f"{historico['bytes_per_client'].mean() / 1e6:.1f} MB", "por rodada", PRIMARY),
        ("Erro máx. vs FedAvg", f"{erro.max():.1e}" if erro is not None else "-", "arredondamento do ponto fixo", ACCENT),
    ], per_row=3)
    note("Cada cliente divide seus pesos em partes aleatórias em Z_p (p = 2^61−1); cada nó soma apenas as partes "
         "recebidas e o servidor conhece somente o agregado. O erro vs FedAvg em claro é medido apenas nesta "
         "simulação, para comprovar a exatidão.")
    if bundle.adherence:
        st.markdown("**Condicionamento por limiares do STEP-GAN**")
        st.dataframe(pd.DataFrame(bundle.adherence).round(3), hide_index=True, width="stretch")


def _ultima_analise(inferencia):
    if not inferencia:
        st.info("Execute uma análise de transações na subvisão de Treinamento e Inferência.")
        return
    tabela = inferencia["result"].table
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Transações por nível de risco**")
        st.bar_chart(tabela["Nível de risco"].value_counts().reindex(RISK_LEVELS, fill_value=0).rename("Transações"),
                     color=DANGER)
    with col2:
        st.markdown("**Status da triagem dos alertas**")
        status = tabela.loc[tabela["Alerta"], "Status da análise"].value_counts().reindex(TRIAGE_STATUS, fill_value=0)
        st.bar_chart(status.rename("Alertas"), color=PRIMARY)
    st.markdown("**10 transações de maior risco**")
    colunas = ["ID", "Escore de risco", "Nível de risco", "Fatores mais atípicos", "Status da análise"]
    st.dataframe(tabela.nlargest(10, "Escore de risco")[colunas], hide_index=True, width="stretch")


def _trilha():
    eventos = st.session_state.get("audit_events", [])
    if not eventos:
        st.info("Nenhum evento registrado nesta sessão.")
        return
    linhas = [{
        "Data/hora (UTC)": e["timestamp"].replace("T", " "),
        "Usuário": e["user"],
        "Evento": e["type"],
        "Resumo": ", ".join(f"{k}={v}" for k, v in e["details"].items() if not isinstance(v, (dict, list)))[:200],
        "Gravado no Azure": "sim" if e.get("persisted") else "não",
    } for e in reversed(eventos)]
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
    note("Os eventos gravados ficam em audit-artifacts/ como JSON (um arquivo por evento, sem sobrescrita).")


def view_dashboard():
    page_header("📊", "Dashboard de Auditoria", "Desempenho do modelo, privacidade, alertas e trilha de auditoria.")
    bundle = st.session_state.get("model_bundle")
    if bundle is None:
        st.info("Nenhum modelo ativo. Treine ou carregue um modelo em Treinamento e Inferência.")
        section_header("1", "Trilha de auditoria da sessão")
        _trilha()
        return
    inferencia = st.session_state.get("inference")

    section_header("1", "Indicadores gerais")
    _indicadores(bundle, inferencia)
    section_header("2", "Desempenho do modelo")
    _desempenho(bundle)
    section_header("3", "Convergência do treinamento federado")
    _convergencia(bundle)
    section_header("4", "Privacidade e STEP-GAN")
    _privacidade(bundle)
    section_header("5", "Última análise de transações")
    _ultima_analise(inferencia)
    section_header("6", "Trilha de auditoria da sessão")
    _trilha()
