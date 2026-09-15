import streamlit as st
from cloud.azure_blob_client import AzureBlobClient
from theme import apply_theme, page_header, section_header, status_badge

DIR_ICONS = {
    "raw-data": "🗂️",
    "processed-data": "🧪",
    "model-checkpoints": "💾",
    "audit-artifacts": "🧾",
    "configuration": "⚙️",
}

DIR_LABELS = {
    "raw-data": "Dados Brutos",
    "processed-data": "Dados Processados",
    "model-checkpoints": "Checkpoints do Modelo",
    "audit-artifacts": "Artefatos de Auditoria",
    "configuration": "Configuração",
}


def view_azure_manager():
    apply_theme()
    page_header(
        "☁️",
        "Azure Blob Storage",
        "Gestão centralizada de artefatos do pipeline MPES (dados, modelos e auditoria)",
    )

    azure_client = AzureBlobClient()

    if not azure_client.container_client:
        st.markdown(
            f"""<div class="mpes-card" style="border-left:3px solid #FF4D6D;">
                {status_badge("CONEXÃO INDISPONÍVEL", "danger")}
                <p style="margin-top:10px; color:#8A93B8; font-size:13px;">
                Não foi possível conectar ao Azure. Verifique a variável <code>AZURE_CONNECTION_STRING</code> no arquivo <code>.env</code>.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        section_header("1", "Diretório Lógico")
        selected_dir = st.radio(
            " ",
            options=azure_client.logical_directories,
            format_func=lambda d: f"{DIR_ICONS.get(d, '📁')}  {DIR_LABELS.get(d, d)}",
            label_visibility="collapsed",
        )
        st.markdown(
            f"""<div class="mpes-card-light" style="margin-top:8px;">
                <span style="font-size:12px; color:#8A93B8;">Container ativo</span><br>
                <code style="font-size:13px;">{azure_client.container_name}/{selected_dir}/</code>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        section_header("2", f"Conteúdo — {DIR_LABELS.get(selected_dir, selected_dir)}")
        files = azure_client.list_blobs(selected_dir)

        if files:
            st.markdown(status_badge(f"{len(files)} arquivo(s)", "ok"), unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            rows_html = ""
            for f in files:
                rows_html += f"""
                <div class="mpes-file-row">
                    <span class="fname">📄 {f}</span>
                    <span style="color:#8A93B8; font-size:11px;">{selected_dir}/</span>
                </div>"""
            st.markdown(rows_html, unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<div class="mpes-card" style="text-align:center; padding:32px;">
                    <div style="font-size:32px; margin-bottom:8px;">📭</div>
                    <span style="color:#8A93B8; font-size:13px;">Nenhum artefato encontrado neste diretório.</span>
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    section_header("3", "Upload de Novo Artefato")

    with st.container():
        st.markdown('<div class="mpes-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Selecione o arquivo para envio seguro",
            label_visibility="visible",
        )

        if uploaded_file is not None:
            st.markdown(
                f"""<div style="margin:10px 0;">
                    {status_badge(f"📎 {uploaded_file.name}", "neutral")}
                    {status_badge(f"{uploaded_file.size / 1024:.1f} KB", "neutral")}
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button("🚀 Enviar para Nuvem", use_container_width=False):
                with st.spinner("Enviando arquivo..."):
                    file_bytes = uploaded_file.getvalue()
                    if azure_client.upload_blob(selected_dir, uploaded_file.name, file_bytes):
                        st.success(f"✅ Arquivo salvo com sucesso em `{selected_dir}/`!")
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)