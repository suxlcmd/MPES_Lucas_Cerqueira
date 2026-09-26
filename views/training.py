"""Subvisão de treinamento federado, gestão de modelos e análise de transações (Camada 1)."""

import datetime
import json
import uuid

import numpy as np
import pandas as pd
import streamlit as st

from cloud.azure_blob_client import read_csv_bytes
from orchestration.audit_log import sha256_hex
from orchestration.data_preprocessor import SCHEMA_LABELS, clean_raw_dataframe, detect_schema, detect_target
from orchestration.inference_service import RISK_LEVELS, TRIAGE_STATUS, score_transactions
from orchestration.model_bundle import ModelBundle
from orchestration.training_service import TrainingSettings, run_training
from theme import note, page_header, section_header, status_badge
from views.common import (
    ACCENT,
    DANGER,
    MUTED,
    PRIMARY,
    WARNING,
    current_user,
    get_azure_client,
    metric_grid,
    model_metric_items,
    record_event,
)


# ---------------------------------------------------------------------------
# Leitura de arquivos (CSV de transações e checkpoints)
# ---------------------------------------------------------------------------

def _file_picker(label, key, azure_dir, extensions):
    """Escolhe um arquivo do Azure ou do computador. Retorna (nome, bytes) ou (None, None)."""
    azure = get_azure_client()
    origens = ["Upload do computador"]
    if azure.available:
        origens.insert(0, f"Azure Blob ({azure_dir})")
    origem = st.radio(f"Origem — {label}", origens, horizontal=True, key=f"{key}_origem")

    if origem.startswith("Azure"):
        arquivos = [a for a in azure.list_blobs(azure_dir) if a.lower().endswith(tuple(extensions))]
        if not arquivos:
            st.info(f"Nenhum arquivo {', '.join(extensions)} em {azure_dir}.")
            return None, None
        nome = st.selectbox(label, arquivos, key=f"{key}_blob")
        cache = st.session_state.get(f"{key}_cache")
        if cache and cache[0] == ("azure", nome):
            return nome, cache[1]
        if st.button("Carregar arquivo do Azure", key=f"{key}_baixar"):
            with st.spinner(f"Baixando {nome}..."):
                dados = azure.download_blob_bytes(azure_dir, nome)
            if dados is None:
                st.error(azure.error or "Não foi possível baixar o arquivo.")
                return None, None
            st.session_state[f"{key}_cache"] = (("azure", nome), dados)
            return nome, dados
        return None, None

    arquivo = st.file_uploader(label, type=[e.lstrip(".") for e in extensions], key=f"{key}_upload")
    if arquivo is None:
        return None, None
    return arquivo.name, arquivo.getvalue()


def _read_csv_cached(key, nome, dados):
    """Lê o CSV uma única vez por arquivo (o rerun do Streamlit não relê 150 MB a cada clique)."""
    sha = sha256_hex(dados)
    cache = st.session_state.get(f"{key}_df")
    if cache and cache["sha256"] == sha:
        return cache
    df = read_csv_bytes(dados)
    cache = {"name": nome, "sha256": sha, "df": df, "size": len(dados)}
    st.session_state[f"{key}_df"] = cache
    return cache


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

def _dataset_preview(df):
    amostra = clean_raw_dataframe(df.head(50))
    alvo = detect_target(amostra)
    esquema = detect_schema(amostra)
    itens = [
        ("Transações", f"{len(df):,}".replace(",", "."), "linhas no arquivo", PRIMARY),
        ("Colunas", str(df.shape[1]), "no arquivo original", PRIMARY),
        ("Esquema detectado", esquema.upper(), SCHEMA_LABELS[esquema], ACCENT),
    ]
    if alvo:
        # O nome limpo pode diferir do original (ex.: CSV com aspas nos cabeçalhos)
        origem = df if alvo in df.columns else clean_raw_dataframe(df)
        fraudes = int(pd.to_numeric(origem[alvo], errors="coerce").fillna(0).sum())
        itens.append(("Fraudes rotuladas", f"{fraudes:,}".replace(",", "."), f"coluna-alvo: {alvo}", DANGER))
    else:
        itens.append(("Rótulos", "ausentes", "treino não supervisionado", WARNING))
    metric_grid(itens)
    if not alvo:
        note(
            "A base não tem coluna de rótulo (Class, isFraud, ...). O modelo será treinado só com o perfil "
            "normal e o limiar será o quantil definido em 'Taxa de alerta esperada'.",
            "warning",
        )


def _training_form():
    with st.form("training_config_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            num_clients = st.slider("Clientes / silos federados", 2, 10, 3)
            num_rounds = st.number_input("Rodadas federadas", 1, 30, 5)
            local_epochs = st.number_input("Épocas locais por rodada", 1, 30, 7)
        with col2:
            batch_size = st.selectbox("Tamanho do lote", [128, 256, 512], index=2)
            latent_dim = st.selectbox("Dimensão latente Z", [64, 100, 128], index=1)
            seed = st.number_input("Semente (reprodutibilidade)", 0, 10_000, 42)
        with col3:
            normals_per_fraud = st.number_input(
                "Normais por fraude (0 = base completa)", 0, 2000, 200,
                help="Amostragem do protocolo validado na PoC: mantém todas as fraudes e N normais por fraude.",
            )
            min_precision = st.slider(
                "Precisão mínima do alerta", 0.50, 0.95, 0.70, 0.05,
                help="O limiar é o de maior recall que atinge esta precisão na validação.",
            )
            alert_rate = st.slider(
                "Taxa de alerta esperada (%) — base sem rótulos", 0.1, 10.0, 1.0, 0.1,
            )
        st.caption(
            "STEP-GAN com 3 geradores condicionados por limiares e Discriminador graduado; "
            "agregação SMPC em Z_p (p = 2^61−1) — protocolo do Teste 4 (v5) da PoC."
        )
        enviado = st.form_submit_button("Iniciar treinamento federado", width="stretch")
    return enviado, TrainingSettings(
        num_clients=num_clients, num_rounds=int(num_rounds), local_epochs=int(local_epochs),
        batch_size=batch_size, latent_dim=latent_dim, normals_per_fraud=int(normals_per_fraud),
        min_precision=float(min_precision), expected_alert_rate=alert_rate / 100, seed=int(seed),
    )


def _save_checkpoint(bundle):
    azure = get_azure_client()
    if not azure.available:
        return None
    model_id = bundle.metadata["model_id"]
    ok = azure.upload_blob("model-checkpoints", f"{model_id}.pt", bundle.to_bytes())
    resumo = {"metadata": bundle.metadata, "threshold": bundle.threshold,
              "threshold_strategy": bundle.threshold_strategy, "metrics": bundle.metrics}
    azure.upload_blob("model-checkpoints", f"{model_id}.json",
                      json.dumps(resumo, ensure_ascii=False, indent=2, default=str).encode("utf-8"))
    return f"model-checkpoints/{model_id}.pt" if ok else None


def _run_training(dataset, settings):
    total = settings.num_rounds * settings.num_clients
    barra = st.progress(0.0, text="Preparando dados...")

    def progresso(rodada, cliente, total_rodadas):
        feito = (rodada - 1) * settings.num_clients + (cliente - 1)
        barra.progress(feito / total, text=f"Rodada {rodada}/{total_rodadas} · treino local do cliente {cliente}")

    try:
        bundle = run_training(
            dataset["df"], dataset["name"], dataset["sha256"], settings, progress=progresso, user=current_user()
        )
    except Exception as exc:  # noqa: BLE001 - erro exibido ao auditor
        barra.empty()
        st.error(f"O treinamento não pôde ser concluído: {exc}")
        record_event("treinamento_falhou", {"dataset": dataset["name"], "erro": str(exc)})
        return
    barra.progress(1.0, text="Treinamento concluído.")

    caminho = _save_checkpoint(bundle)
    st.session_state.model_bundle = bundle
    st.session_state.pop("inference", None)
    record_event("treinamento", {
        "model_id": bundle.metadata["model_id"],
        "dataset": dataset["name"],
        "dataset_sha256": dataset["sha256"],
        "settings": bundle.metadata["settings"],
        "threshold": bundle.threshold,
        "metrics_test": bundle.metrics.get("teste", {}),
        "checkpoint": caminho,
    })
    if caminho:
        st.success(f"Modelo treinado e salvo em {caminho}.")
    else:
        st.warning("Modelo treinado, mas o checkpoint não foi salvo no Azure. Use o botão de download abaixo.")


def _tab_train():
    nome, dados = _file_picker("Base de treinamento (CSV)", "treino", "raw-data", [".csv"])
    if dados is None:
        return
    try:
        dataset = _read_csv_cached("treino", nome, dados)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Não foi possível ler o CSV: {exc}")
        return
    st.caption(f"Arquivo: {nome} · SHA-256: {dataset['sha256'][:16]}…")
    _dataset_preview(dataset["df"])
    enviado, settings = _training_form()
    if enviado:
        _run_training(dataset, settings)


def _tab_load():
    nome, dados = _file_picker("Checkpoint do modelo (.pt)", "modelo", "model-checkpoints", [".pt"])
    if dados is None:
        return
    if st.button("Usar este modelo", key="carregar_modelo"):
        try:
            bundle = ModelBundle.from_bytes(dados)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Checkpoint inválido: {exc}")
            return
        st.session_state.model_bundle = bundle
        st.session_state.pop("inference", None)
        record_event("modelo_carregado", {"arquivo": nome, "sha256": sha256_hex(dados),
                                          "model_id": bundle.metadata.get("model_id")})
        st.success(f"Modelo {bundle.metadata.get('model_id')} carregado.")


def _model_summary(bundle):
    meta = bundle.metadata
    st.markdown(
        status_badge(f"MODELO ATIVO: {meta.get('model_id', '-')}", "success"), unsafe_allow_html=True
    )
    metric_grid([
        ("Dataset", meta.get("dataset", "-"), meta.get("schema_label", ""), PRIMARY),
        ("Limiar de alerta", f"{bundle.threshold * 100:.1f}", "escore 0–100 · calibrado na validação", DANGER),
        ("Melhor rodada", str(meta.get("best_round", "-")),
         f"de {meta.get('settings', {}).get('num_rounds', '-')} · {meta.get('settings', {}).get('num_clients', '-')} clientes", PRIMARY),
        ("Treinado em", meta.get("created_at", "-")[:16].replace("T", " "), meta.get("created_by", ""), MUTED),
    ])
    note(f"Critério do limiar: {bundle.threshold_strategy}.")
    if bundle.metrics.get("teste"):
        metric_grid(model_metric_items(bundle.metrics["teste"]), per_row=6)

    with st.expander("Detalhes do treinamento, privacidade e rastreabilidade"):
        historico = pd.DataFrame(bundle.history)
        if not historico.empty:
            st.markdown("**Convergência na validação por rodada**")
            st.line_chart(historico.set_index("rodada")[["pr_auc_val", "roc_auc_val"]]
                          .rename(columns={"pr_auc_val": "PR-AUC", "roc_auc_val": "ROC-AUC"}),
                          color=[PRIMARY, ACCENT])
            st.markdown("**Agregação SMPC por rodada**")
            st.dataframe(pd.DataFrame({
                "Rodada": historico["rodada"],
                "Tempo SMPC (s)": historico["smpc_seconds"].round(3),
                "Erro máx. vs FedAvg em claro": historico.get("smpc_max_error", pd.Series(dtype=float)).map("{:.1e}".format),
                "Enviado por cliente (MB)": (historico["bytes_per_client"] / 1e6).round(2),
            }), hide_index=True, width="stretch")
        if bundle.adherence:
            st.markdown("**Condicionamento por limiares (τ pedido × D obtido)**")
            st.dataframe(pd.DataFrame(bundle.adherence).round(3), hide_index=True, width="stretch")
        st.markdown("**Metadados (rastreabilidade)**")
        st.json(meta, expanded=False)

    st.download_button(
        "Baixar checkpoint do modelo (.pt)", bundle.to_bytes(), file_name=f"{meta.get('model_id', 'modelo')}.pt",
        mime="application/octet-stream", key="baixar_checkpoint",
    )


# ---------------------------------------------------------------------------
# Inferência e triagem
# ---------------------------------------------------------------------------

def _run_inference(bundle, nome, dados):
    try:
        with st.spinner("Calculando o escore de risco das transações..."):
            dataset = _read_csv_cached("inferencia", nome, dados)
            resultado = score_transactions(bundle, dataset["df"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erro durante a análise: {exc}")
        return
    st.session_state.inference = {
        "id": uuid.uuid4().hex[:8], "result": resultado, "source": nome, "sha256": dataset["sha256"],
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    record_event("inferencia", {
        "model_id": bundle.metadata.get("model_id"), "arquivo": nome, "sha256": dataset["sha256"],
        "transacoes": resultado.summary["transactions"], "alertas": resultado.summary["alerts"],
        "niveis": resultado.summary["levels"], "metricas": resultado.metrics,
    })


def _inference_overview(resultado):
    resumo = resultado.summary
    metric_grid([
        ("Transações analisadas", f"{resumo['transactions']:,}".replace(",", "."), "no arquivo", PRIMARY),
        ("Alertas", str(resumo["alerts"]), f"{resumo['alert_rate'] * 100:.2f}% do total",
         DANGER if resumo["alerts"] else ACCENT),
        ("Risco crítico", str(resumo["levels"]["Crítico"]), "prioridade máxima", "#7F1D1D"),
        ("Throughput", f"{resumo['tps']:,.0f}".replace(",", "."), "transações/s", MUTED),
    ])
    if resultado.metrics:
        st.markdown("**Desempenho neste arquivo (possui rótulos)**")
        metric_grid([(n, v, "limiar fixo do modelo", PRIMARY) for n, v in resultado.metrics.items()], per_row=6)
        note("O limiar foi calibrado na validação do treino; este arquivo não influenciou a escolha do limiar.")

    col1, col2 = st.columns(2)
    tabela = resultado.table
    with col1:
        st.markdown("**Transações por nível de risco**")
        contagem = tabela["Nível de risco"].value_counts().reindex(RISK_LEVELS, fill_value=0)
        st.bar_chart(contagem.rename("Transações"), color=DANGER)
    with col2:
        st.markdown(f"**Distribuição do escore (limiar = {resumo['threshold'] * 100:.1f})**")
        contagens, bordas = np.histogram(tabela["Escore de risco"], bins=20, range=(0, 100))
        st.bar_chart(pd.DataFrame({"Transações": contagens}, index=[f"{b:.0f}" for b in bordas[:-1]]), color=PRIMARY)


def _triage_queue(inferencia):
    tabela = inferencia["result"].table
    col1, col2 = st.columns([2, 1])
    with col1:
        niveis = st.multiselect("Níveis de risco na fila", RISK_LEVELS, default=["Crítico", "Alto"], key="fila_niveis")
    with col2:
        limite = st.number_input("Máximo de linhas", 10, 5000, 500, key="fila_limite")
    fila = (tabela[tabela["Nível de risco"].isin(niveis)]
            .sort_values("Escore de risco", ascending=False).head(int(limite)))
    if fila.empty:
        st.success("Nenhuma transação nos níveis selecionados.")
        return

    editaveis = ["Status da análise", "Observação do auditor"]
    editado = st.data_editor(
        fila,
        key=f"triagem_{inferencia['id']}",
        hide_index=True,
        width="stretch",
        disabled=[c for c in fila.columns if c not in editaveis],
        column_config={
            "Escore de risco": st.column_config.ProgressColumn("Escore de risco", min_value=0, max_value=100, format="%.1f"),
            "Status da análise": st.column_config.SelectboxColumn("Status da análise", options=TRIAGE_STATUS, required=True),
            "Observação do auditor": st.column_config.TextColumn("Observação do auditor", max_chars=500),
            "Alerta": st.column_config.CheckboxColumn("Alerta"),
        },
    )
    tabela.loc[editado.index, editaveis] = editado[editaveis].values
    contagem = tabela.loc[tabela["Alerta"], "Status da análise"].value_counts()
    st.caption(" · ".join(f"{s}: {int(contagem.get(s, 0))}" for s in TRIAGE_STATUS) + " (alertas)")


def _exports(bundle, inferencia):
    tabela = inferencia["result"].table
    carimbo = inferencia["at"][:19].replace(":", "").replace("-", "")
    base = f"analise_{bundle.metadata.get('model_id', 'modelo')}_{carimbo}"
    csv_completo = tabela.to_csv(index=False).encode("utf-8-sig")
    csv_alertas = tabela[tabela["Alerta"]].to_csv(index=False).encode("utf-8-sig")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("Baixar resultado completo (CSV)", csv_completo, f"{base}.csv", "text/csv", width="stretch")
    with col2:
        st.download_button("Baixar só os alertas (CSV)", csv_alertas, f"{base}_alertas.csv", "text/csv", width="stretch")
    with col3:
        azure = get_azure_client()
        if st.button("Salvar relatório no Azure", disabled=not azure.available, width="stretch", key="salvar_relatorio"):
            ok = azure.upload_blob("audit-artifacts", f"{base}_triagem.csv", csv_alertas)
            status = tabela.loc[tabela["Alerta"], "Status da análise"].value_counts().to_dict()
            record_event("relatorio_triagem", {"arquivo": f"audit-artifacts/{base}_triagem.csv", "salvo": ok,
                                               "origem": inferencia["source"], "status": status})
            if ok:
                st.success(f"Relatório salvo em audit-artifacts/{base}_triagem.csv.")
            else:
                st.error(azure.error or "Falha ao salvar o relatório.")


def _render_inference(bundle):
    section_header("3", "Análise de transações")
    nome, dados = _file_picker("Transações para análise (CSV)", "inferencia", "raw-data", [".csv"])
    if dados is not None and st.button("Executar análise", width="stretch", key="executar_analise"):
        _run_inference(bundle, nome, dados)

    inferencia = st.session_state.get("inference")
    if not inferencia:
        return
    st.caption(f"Arquivo analisado: {inferencia['source']} · SHA-256 {inferencia['sha256'][:16]}… · {inferencia['at']}")
    _inference_overview(inferencia["result"])
    section_header("4", "Fila de triagem dos alertas")
    note("Registre o status e a observação de cada alerta. Os 'fatores mais atípicos' indicam as variáveis mais "
         "distantes do perfil normal do treino (em intervalos interquartis); não são prova de fraude.")
    _triage_queue(inferencia)
    section_header("5", "Exportação e evidências")
    _exports(bundle, inferencia)


def view_training():
    page_header("⚙️", "Treinamento e Inferência",
                "Treinamento federado com SMPC, gestão de modelos e triagem de transações suspeitas.")

    section_header("1", "Modelo")
    aba_treino, aba_carregar = st.tabs(["Treinar novo modelo", "Carregar modelo salvo"])
    with aba_treino:
        _tab_train()
    with aba_carregar:
        _tab_load()

    bundle = st.session_state.get("model_bundle")
    if bundle is None:
        note("Treine um modelo ou carregue um checkpoint para analisar transações.")
        return
    section_header("2", "Modelo ativo")
    _model_summary(bundle)
    _render_inference(bundle)
