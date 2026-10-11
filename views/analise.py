"""
Tela principal (Camada 1): fluxo guiado em 3 passos para quem não é da área de dados.

1. Base de treino: o usuário envia o histórico; a coluna de fraude é detectada sozinha (ou escolhida numa lista).
   O treinamento usa a configuração validada na PoC, sem parâmetros na tela.
2. Transações para analisar: o usuário envia o arquivo e a análise roda automaticamente.
3. Resultados: frase-resumo, transações suspeitas com o motivo em linguagem simples, triagem e download.
"""

import datetime
import json
import uuid

import pandas as pd
import streamlit as st

from cloud.azure_blob_client import read_csv_bytes
from orchestration.audit_log import sha256_hex
from orchestration.data_preprocessor import clean_raw_dataframe, detect_target, encode_target, label_candidates
from orchestration.inference_service import RISK_LEVELS, TRIAGE_STATUS, score_transactions
from orchestration.model_bundle import ModelBundle
from orchestration.training_service import TrainingSettings, quality_summary, run_training
from theme import headline, note, page_header, section_header, status_badge, steps_bar
from views.common import ACCENT, DANGER, MUTED, PRIMARY, WARNING, current_user, get_azure_client, metric_grid, record_event

SEM_ROTULO = "__sem_rotulo__"
PASSOS = ["Base de treino", "Transações para analisar", "Resultados"]
DESCRICAO_NIVEIS = {
    "Crítico": "verificar primeiro",
    "Alto": "suspeita",
    "Médio": "abaixo do limite de alerta",
    "Baixo": "comportamento normal",
}


def _fmt_int(valor):
    return f"{int(valor):,}".replace(",", ".")


# ---------------------------------------------------------------------------
# Entrada de arquivos
# ---------------------------------------------------------------------------

def _file_input(label, key, help_text):
    """Upload do computador; se o Azure estiver configurado, também permite escolher um arquivo salvo na nuvem."""
    arquivo = st.file_uploader(label, type=["csv"], key=f"{key}_upload", help=help_text)
    if arquivo is not None:
        return arquivo.name, arquivo.getvalue()

    azure = get_azure_client()
    if not azure.available:
        return None, None
    with st.expander("Ou escolha um arquivo já salvo na nuvem"):
        arquivos = [a for a in azure.list_blobs("raw-data") if a.lower().endswith(".csv")]
        if not arquivos:
            st.caption("Nenhum arquivo CSV salvo na nuvem.")
            return None, None
        nome = st.selectbox("Arquivo", arquivos, key=f"{key}_blob")
        cache = st.session_state.get(f"{key}_nuvem")
        if cache and cache[0] == nome:
            return cache
        if st.button("Usar este arquivo", key=f"{key}_baixar"):
            with st.spinner(f"Baixando {nome}..."):
                dados = azure.download_blob_bytes("raw-data", nome)
            if dados is None:
                st.error(azure.error or "Não foi possível baixar o arquivo.")
                return None, None
            st.session_state[f"{key}_nuvem"] = (nome, dados)
            return nome, dados
    return None, None


def _read_dataset(key, nome, dados):
    """Lê o CSV uma única vez por arquivo (o Streamlit reexecuta a página a cada clique)."""
    sha = sha256_hex(dados)
    cache = st.session_state.get(f"{key}_df")
    if cache and cache["sha256"] == sha:
        return cache
    df = read_csv_bytes(dados)
    cache = {"name": nome, "sha256": sha, "df": df}
    st.session_state[f"{key}_df"] = cache
    return cache


# ---------------------------------------------------------------------------
# Passo 1: base de treino e treinamento
# ---------------------------------------------------------------------------

def _label_choice(dataset):
    """Única pergunta ao usuário: qual coluna diz se a transação é fraude (já vem preenchida quando detectada)."""
    if "label_info" not in dataset:
        amostra = clean_raw_dataframe(dataset["df"].head(5000))
        dataset["label_info"] = {"detectada": detect_target(amostra), "candidatas": label_candidates(amostra)}
    info = dataset["label_info"]
    opcoes = list(dict.fromkeys(([info["detectada"]] if info["detectada"] else []) + info["candidatas"])) + [SEM_ROTULO]
    escolha = st.selectbox(
        "Qual coluna indica se a transação foi fraude?",
        opcoes,
        index=0 if info["detectada"] else len(opcoes) - 1,
        format_func=lambda c: "A base não indica quais transações foram fraude" if c == SEM_ROTULO else c,
        key=f"rotulo_{dataset['sha256'][:12]}",
        help="Encontrada automaticamente quando possível. Com essa coluna o modelo aprende com exemplos reais "
             "de fraude e fica mais preciso.",
    )
    return None if escolha == SEM_ROTULO else escolha


def _training_preview(dataset, coluna):
    df = dataset["df"]
    itens = [("Transações na base", _fmt_int(len(df)), f"{df.shape[1]} colunas", PRIMARY)]
    if coluna:
        limpa = df if coluna in df.columns else clean_raw_dataframe(df)
        rotulo = encode_target(clean_raw_dataframe(limpa[[coluna]])[coluna])
        fraudes = int((rotulo == 1).sum())
        itens.append(("Fraudes marcadas", _fmt_int(fraudes), f"{fraudes / max(len(df), 1) * 100:.2f}% da base",
                      DANGER if fraudes else WARNING))
    else:
        itens.append(("Fraudes marcadas", "sem informação", "o modelo aprende só o comportamento normal", WARNING))
    metric_grid(itens, per_row=2)
    if not coluna:
        note("Sem a coluna de fraude, o modelo marca como suspeito o 1% de transações mais diferentes do "
             "habitual. Se a base tiver essa informação, selecione a coluna acima.", "warning")


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


def _train(dataset, coluna):
    settings = TrainingSettings()
    total = settings.num_rounds * settings.num_clients
    barra = st.progress(0.0, text="Preparando os dados...")

    def progresso(rodada, cliente, total_rodadas):
        feito = (rodada - 1) * settings.num_clients + (cliente - 1)
        barra.progress(feito / total, text=f"Aprendendo os padrões da base... etapa {feito + 1} de {total}")

    try:
        bundle = run_training(dataset["df"], dataset["name"], dataset["sha256"], settings, progress=progresso,
                              user=current_user(), target=coluna or "")
    except Exception as exc:  # noqa: BLE001 - erro exibido ao usuário
        barra.empty()
        st.error(f"Não foi possível treinar o modelo: {exc}")
        record_event("treinamento_falhou", {"dataset": dataset["name"], "erro": str(exc)})
        return
    barra.empty()
    caminho = _save_checkpoint(bundle)
    st.session_state.model_bundle = bundle
    st.session_state.pop("inference", None)
    record_event("treinamento", {
        "model_id": bundle.metadata["model_id"], "dataset": dataset["name"], "dataset_sha256": dataset["sha256"],
        "coluna_fraude": coluna or "-", "settings": bundle.metadata["settings"], "threshold": bundle.threshold,
        "metrics_test": bundle.metrics.get("teste", {}), "checkpoint": caminho,
    })
    st.rerun()


def _model_card(bundle):
    meta = bundle.metadata
    qualidade = quality_summary(bundle.metrics.get("teste", {}))
    st.markdown(status_badge("MODELO PRONTO", "success"), unsafe_allow_html=True)
    if qualidade:
        nota, tipo, frases = qualidade
        headline(f"Qualidade do modelo: {nota}", " ".join(frases), tipo)
    else:
        headline("Modelo pronto (base sem a coluna de fraude)",
                 "Sem exemplos de fraude não é possível medir a precisão: o modelo marcará como suspeitas as "
                 "transações mais diferentes do comportamento habitual da base de treino.", "warning")
    st.caption(f"Treinado com {meta.get('dataset', '-')} em {meta.get('created_at', '-')[:16].replace('T', ' ')} "
               f"(UTC) · {_fmt_int(meta.get('rows_used', 0))} transações usadas · modelo {meta.get('model_id', '-')}")
    descartadas = meta.get("dropped_columns") or {}
    if descartadas:
        with st.expander(f"Colunas ignoradas automaticamente ({len(descartadas)})"):
            st.dataframe(pd.DataFrame({"Coluna": list(descartadas), "Motivo": list(descartadas.values())}),
                         hide_index=True, width="stretch")
    st.download_button("Baixar o modelo (para usar depois)", bundle.to_bytes(),
                       file_name=f"{meta.get('model_id', 'modelo')}.pt", mime="application/octet-stream",
                       key="baixar_modelo")


def _load_saved_model():
    with st.expander("Já tem um modelo treinado? Carregue aqui"):
        arquivo = st.file_uploader("Arquivo do modelo (.pt)", type=["pt"], key="modelo_salvo")
        dados, nome = (arquivo.getvalue(), arquivo.name) if arquivo is not None else (None, None)
        azure = get_azure_client()
        if dados is None and azure.available:
            salvos = [a for a in azure.list_blobs("model-checkpoints") if a.endswith(".pt")]
            if salvos:
                nome = st.selectbox("Ou escolha um modelo salvo na nuvem", salvos, key="modelo_nuvem")
        if nome and st.button("Usar este modelo", key="usar_modelo"):
            if dados is None:
                dados = azure.download_blob_bytes("model-checkpoints", nome)
            try:
                bundle = ModelBundle.from_bytes(dados)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Arquivo de modelo inválido: {exc}")
                return
            st.session_state.model_bundle = bundle
            st.session_state.pop("inference", None)
            record_event("modelo_carregado", {"arquivo": nome, "sha256": sha256_hex(dados),
                                              "model_id": bundle.metadata.get("model_id")})
            st.rerun()


def _step_training(bundle):
    section_header("1", "Base de treino (histórico de transações)")
    if bundle is not None:
        _model_card(bundle)
        with st.expander("Treinar um novo modelo com outra base"):
            _training_upload()
    else:
        note("Envie um arquivo CSV com transações passadas. Se ele tiver uma coluna dizendo quais foram fraude, "
             "o modelo aprende com esses exemplos. Os dados não precisam de nenhum preparo.")
        _training_upload()
        _load_saved_model()


def _training_upload():
    nome, dados = _file_input("Arquivo de treino (CSV)", "treino",
                              "Histórico de transações, uma por linha, com cabeçalho na primeira linha.")
    if dados is None:
        return
    try:
        dataset = _read_dataset("treino", nome, dados)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Não foi possível ler o arquivo: {exc}")
        return
    coluna = _label_choice(dataset)
    _training_preview(dataset, coluna)
    st.caption("O treinamento costuma levar de 2 a 10 minutos, conforme o tamanho da base. "
               "A configuração é a validada na pesquisa (STEP-GAN com aprendizado federado e SMPC).")
    if st.button("Treinar o modelo", width="stretch", key="treinar"):
        _train(dataset, coluna)


# ---------------------------------------------------------------------------
# Passo 2: transações para analisar
# ---------------------------------------------------------------------------

def _step_analysis(bundle):
    section_header("2", "Transações para analisar")
    if bundle is None:
        st.info("Primeiro treine o modelo no passo 1.")
        return
    note("Envie as transações que você quer verificar, no mesmo formato da base de treino. "
         "Se o arquivo tiver a coluna de fraude, o resultado também mostra quantas o modelo acertou.")
    nome, dados = _file_input("Arquivo para análise (CSV)", "analise", "Mesmas colunas da base de treino.")
    if dados is None:
        return
    sha = sha256_hex(dados)
    atual = st.session_state.get("inference")
    modelo = bundle.metadata.get("model_id")
    if atual and atual["sha256"] == sha and atual.get("model_id") == modelo:
        return  # já analisado com este modelo
    try:
        with st.spinner("Analisando as transações..."):
            dataset = _read_dataset("analise", nome, dados)
            resultado = score_transactions(bundle, dataset["df"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Não foi possível analisar o arquivo: {exc}")
        return
    st.session_state.inference = {
        "id": uuid.uuid4().hex[:8], "result": resultado, "source": nome, "sha256": sha, "model_id": modelo,
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    record_event("inferencia", {
        "model_id": modelo, "arquivo": nome, "sha256": sha, "transacoes": resultado.summary["transactions"],
        "alertas": resultado.summary["alerts"], "niveis": resultado.summary["levels"], "metricas": resultado.metrics,
    })


# ---------------------------------------------------------------------------
# Passo 3: resultados
# ---------------------------------------------------------------------------

def _summary(inferencia):
    resultado = inferencia["result"]
    resumo = resultado.summary
    total, alertas = resumo["transactions"], resumo["alerts"]
    if alertas:
        headline(f"{_fmt_int(alertas)} de {_fmt_int(total)} transações ({alertas / total * 100:.1f}%) foram marcadas "
                 f"como suspeitas.", f"{_fmt_int(resumo['levels']['Crítico'])} delas têm risco crítico e devem ser "
                 "verificadas primeiro. Arquivo analisado: " + inferencia["source"], "danger")
    else:
        headline(f"Nenhuma das {_fmt_int(total)} transações foi marcada como suspeita.",
                 "Arquivo analisado: " + inferencia["source"], "success")
    metric_grid([
        ("Transações analisadas", _fmt_int(total), "no arquivo", PRIMARY),
        ("Suspeitas", _fmt_int(alertas), f"{alertas / total * 100:.2f}% do total", DANGER if alertas else ACCENT),
        ("Risco crítico", _fmt_int(resumo["levels"]["Crítico"]), "verificar primeiro", "#7F1D1D"),
        ("Risco alto", _fmt_int(resumo["levels"]["Alto"]), "suspeitas", DANGER),
    ])
    if resultado.metrics:
        m = resultado.metrics
        note(f"O arquivo tem a coluna de fraude, então dá para conferir: o modelo encontrou {m['Recall'] * 100:.0f} "
             f"de cada 100 fraudes do arquivo, e de cada 10 transações marcadas, {m['Precisão'] * 10:.0f} eram "
             "fraude de fato.")

    niveis = resumo["levels"]
    st.caption("Por nível de risco: " + " · ".join(
        f"{n} ({DESCRICAO_NIVEIS[n]}): {_fmt_int(niveis[n])}" for n in RISK_LEVELS))


def _suspects_table(inferencia):
    tabela = inferencia["result"].table
    suspeitas = tabela[tabela["Alerta"]].sort_values("Escore de risco", ascending=False)
    if suspeitas.empty:
        return
    st.markdown(f"**Transações suspeitas** (da mais para a menos arriscada) — registre o que foi verificado")
    principais = ["ID", "Escore de risco", "Nível de risco", "Por que foi marcada", "Status da análise",
                  "Observação do auditor"]
    exibidas = suspeitas.head(1000)
    editado = st.data_editor(
        exibidas[principais + [c for c in exibidas.columns if c not in principais + ["Alerta"]]],
        key=f"triagem_{inferencia['id']}", hide_index=True, width="stretch",
        disabled=[c for c in exibidas.columns if c not in ("Status da análise", "Observação do auditor")],
        column_config={
            "Escore de risco": st.column_config.ProgressColumn("Risco (0–100)", min_value=0, max_value=100,
                                                               format="%.0f"),
            "Status da análise": st.column_config.SelectboxColumn("Status", options=TRIAGE_STATUS, required=True),
            "Observação do auditor": st.column_config.TextColumn("Observação", max_chars=500),
            "Por que foi marcada": st.column_config.TextColumn("Por que foi marcada", width="large"),
        },
    )
    editaveis = ["Status da análise", "Observação do auditor"]
    tabela.loc[editado.index, editaveis] = editado[editaveis].values
    if len(suspeitas) > len(exibidas):
        st.caption(f"Mostrando as {len(exibidas)} mais arriscadas de {len(suspeitas)}. O download traz todas.")
    note("O motivo indica em que a transação foge do comportamento comum da base de treino. "
         "É um indício para orientar a verificação, não uma prova de fraude.")


def _downloads(bundle, inferencia):
    tabela = inferencia["result"].table
    carimbo = inferencia["at"][:19].replace(":", "").replace("-", "")
    base = f"analise_{carimbo}"

    def para_excel(df):
        # Separador ';' e vírgula decimal: o arquivo abre direto no Excel em português
        return df.drop(columns=["Alerta"]).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")

    suspeitas = tabela[tabela["Alerta"]].sort_values("Escore de risco", ascending=False)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("Baixar as suspeitas (Excel/CSV)", para_excel(suspeitas), f"{base}_suspeitas.csv",
                           "text/csv", width="stretch", type="primary")
    with col2:
        st.download_button("Baixar todas as transações", para_excel(tabela), f"{base}_completo.csv", "text/csv",
                           width="stretch")
    with col3:
        azure = get_azure_client()
        if azure.available and st.button("Salvar o relatório na nuvem", width="stretch", key="salvar_relatorio"):
            ok = azure.upload_blob("audit-artifacts", f"{base}_suspeitas.csv", para_excel(suspeitas))
            status = suspeitas["Status da análise"].value_counts().to_dict()
            record_event("relatorio_triagem", {"arquivo": f"audit-artifacts/{base}_suspeitas.csv", "salvo": ok,
                                               "origem": inferencia["source"], "status": status,
                                               "model_id": bundle.metadata.get("model_id")})
            st.success("Relatório salvo na nuvem.") if ok else st.error(azure.error or "Falha ao salvar.")


def _step_results(bundle):
    section_header("3", "Resultados")
    inferencia = st.session_state.get("inference")
    if bundle is None or not inferencia:
        st.info("Os resultados aparecem aqui assim que você enviar as transações no passo 2.")
        return
    _summary(inferencia)
    _suspects_table(inferencia)
    _downloads(bundle, inferencia)


def view_analise():
    page_header("🔎", "Detecção de fraudes",
                "Envie o histórico de transações, depois as transações a verificar, e veja quais são suspeitas.")
    barra_passos = st.empty()  # preenchida no fim: o passo 2 pode concluir a análise nesta mesma execução
    bundle = st.session_state.get("model_bundle")
    _step_training(bundle)
    _step_analysis(bundle)
    _step_results(bundle)
    concluidos = 0 if bundle is None else 2 if st.session_state.get("inference") else 1
    barra_passos.markdown(steps_bar(PASSOS, current=min(concluidos + 1, 3), done=concluidos), unsafe_allow_html=True)
