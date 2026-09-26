"""Subvisão de gestão do Azure Blob Storage (Camada 1)."""

import pandas as pd
import streamlit as st

from cloud.azure_blob_client import sanitize_filename
from orchestration.audit_log import sha256_hex
from theme import note, page_header, section_header, status_badge
from views.common import get_azure_client, record_event

DIR_LABELS = {
    "raw-data": "Dados brutos",
    "processed-data": "Dados processados",
    "model-checkpoints": "Checkpoints do modelo",
    "audit-artifacts": "Artefatos de auditoria",
    "configuration": "Configuração",
}

ALLOWED_EXTENSIONS = {
    "raw-data": ["csv"],
    "processed-data": ["csv", "parquet"],
    "model-checkpoints": ["pt", "json"],
    "audit-artifacts": ["csv", "json", "pdf", "xlsx", "txt"],
    "configuration": ["json", "yaml", "yml", "toml", "txt"],
}


def _file_table(detalhes):
    return pd.DataFrame([{
        "Arquivo": d["name"],
        "Tamanho (KB)": round((d["size"] or 0) / 1024, 1),
        "Modificado em (UTC)": d["last_modified"].strftime("%Y-%m-%d %H:%M") if d["last_modified"] else "-",
    } for d in detalhes])


def view_azure_manager():
    page_header("☁️", "Azure Blob Storage", "Gestão de bases, checkpoints e artefatos do pipeline MPES.")
    azure = get_azure_client()

    if not azure.available:
        st.error(f"Conexão indisponível: {azure.error}")
        note("Configure AZURE_CONNECTION_STRING nas configurações do App Service (ou no arquivo .env em "
             "desenvolvimento). As demais subvisões funcionam com upload local.", "warning")
        return

    esquerda, direita = st.columns([1, 2], gap="large")
    with esquerda:
        section_header("1", "Diretório lógico")
        diretorio = st.radio(
            "Diretório", azure.logical_directories, format_func=lambda d: DIR_LABELS.get(d, d),
            key="azure_selected_directory", label_visibility="collapsed",
        )
        st.caption(f"{azure.container_name}/{diretorio}/")

    with direita:
        section_header("2", f"Conteúdo: {DIR_LABELS.get(diretorio, diretorio)}")
        detalhes = azure.list_blob_details(diretorio)
        if not detalhes:
            st.info("Nenhum artefato encontrado neste diretório.")
        else:
            st.markdown(status_badge(f"{len(detalhes)} arquivo(s)", "success"), unsafe_allow_html=True)
            st.dataframe(_file_table(detalhes), hide_index=True, width="stretch")
            escolhido = st.selectbox("Baixar arquivo", [d["name"] for d in detalhes], key="azure_download_sel")
            if st.button("Preparar download", key="azure_preparar"):
                dados = azure.download_blob_bytes(diretorio, escolhido)
                if dados is None:
                    st.error(azure.error)
                else:
                    st.session_state.azure_download = (escolhido, dados)
            pronto = st.session_state.get("azure_download")
            if pronto and pronto[0] == escolhido:
                st.download_button(f"Baixar {escolhido}", pronto[1], file_name=escolhido, key="azure_baixar")

    section_header("3", "Upload de novo artefato")
    extensoes = ALLOWED_EXTENSIONS.get(diretorio)
    arquivo = st.file_uploader(
        f"Arquivo para {DIR_LABELS.get(diretorio, diretorio)} ({', '.join(extensoes)})",
        type=extensoes, key=f"azure_upload_{diretorio}",
    )
    if arquivo is None:
        return
    nome = sanitize_filename(arquivo.name)
    if nome != arquivo.name:
        st.caption(f"O arquivo será salvo como: {nome}")
    sobrescrever = st.checkbox("Substituir se já existir", value=False, key="azure_sobrescrever",
                               help="Por padrão, artefatos existentes são preservados (evidência de auditoria).")
    if st.button("Enviar para a nuvem", width="stretch", key="azure_upload_button"):
        dados = arquivo.getvalue()
        with st.spinner("Enviando arquivo..."):
            ok = azure.upload_blob(diretorio, nome, dados, overwrite=sobrescrever)
        record_event("upload_azure", {"diretorio": diretorio, "arquivo": nome, "tamanho": len(dados),
                                      "sha256": sha256_hex(dados), "sobrescrever": sobrescrever, "sucesso": ok})
        if ok:
            st.success(f"Arquivo salvo em {diretorio}/{nome}.")
        else:
            st.error(azure.error or "Falha no upload.")
