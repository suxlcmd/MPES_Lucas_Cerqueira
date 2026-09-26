import streamlit as st

from cloud.azure_blob_client import AzureBlobClient
from theme import (
    apply_theme,
    page_header,
    section_header,
    status_badge,
)


DIR_ICONS = {
    "raw-data": "bi-database",
    "processed-data": "bi-funnel",
    "model-checkpoints": "bi-cpu",
    "audit-artifacts": "bi-file-earmark-check",
    "configuration": "bi-gear",
}


DIR_LABELS = {
    "raw-data": "Dados brutos",
    "processed-data": "Dados processados",
    "model-checkpoints": "Checkpoints do modelo",
    "audit-artifacts": "Artefatos de auditoria",
    "configuration": "Configuração",
}


def _file_rows(files, selected_dir):
    """Monta a lista visual dos arquivos do Azure."""
    rows = []

    for filename in files:
        rows.append(
            f"""
            <div class="mpes-file-row">
                <span class="mpes-file-name">
                    <i class="bi bi-file-earmark-text"></i>
                    {filename}
                </span>
                <span class="mpes-file-directory">
                    {selected_dir}
                </span>
            </div>
            """
        )

    return "".join(rows)


def view_azure_manager():
    """Renderiza a subvisão corporativa do Azure Blob Storage."""
    apply_theme()

    page_header(
        "☁️",
        "Azure Blob Storage",
        "Gestão de bases, checkpoints e artefatos do pipeline MPES.",
    )

    azure_client = AzureBlobClient()

    if not azure_client.container_client:
        st.markdown(
            """
            <div class="alert alert-danger shadow-sm" role="alert">
                <i class="bi bi-exclamation-triangle-fill"></i>
                <strong>Conexão indisponível</strong>
                <p class="mb-0 mt-2">
                    Verifique a variável
                    <code>AZURE_CONNECTION_STRING</code>
                    no arquivo <code>.env</code>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        section_header("1", "Diretório lógico")

        selected_dir = st.radio(
            "Diretório",
            options=azure_client.logical_directories,
            format_func=lambda directory: (
                f"{DIR_LABELS.get(directory, directory)}"
            ),
            key="azure_selected_directory",
            label_visibility="collapsed",
        )

        icon_class = DIR_ICONS.get(
            selected_dir,
            "bi-folder",
        )

        st.markdown(
            f"""
            <div class="card shadow-sm border-0 mb-3">
                <div class="card-body">
                    <div class="text-muted small mb-2">
                        Diretório selecionado
                    </div>
                    <div class="fw-semibold">
                        <i class="bi {icon_class}"></i>
                        {DIR_LABELS.get(selected_dir, selected_dir)}
                    </div>
                    <div class="text-muted small mt-2">
                        <code>
                            {azure_client.container_name}/{selected_dir}/
                        </code>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        section_header(
            "2",
            f"Conteúdo: {DIR_LABELS.get(selected_dir, selected_dir)}",
        )

        files = azure_client.list_blobs(selected_dir)

        if files:
            st.markdown(
                status_badge(
                    f"{len(files)} arquivo(s)",
                    "success",
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div class='mt-3'></div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                _file_rows(files, selected_dir),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="card shadow-sm border-0 text-center py-5">
                    <div class="card-body">
                        <i class="bi bi-inbox fs-1 text-secondary"></i>
                        <p class="text-muted mt-3 mb-0">
                            Nenhum artefato encontrado neste diretório.
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_header("3", "Upload de novo artefato")

    st.markdown(
        '<div class="card shadow-sm border-0 mb-3">',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Selecione o arquivo para envio seguro",
        key="azure_upload_file",
    )

    if uploaded_file is not None:
        st.markdown(
            f"""
            <div class="d-flex gap-2 align-items-center mb-3">
                {status_badge(uploaded_file.name, "neutral")}
                {status_badge(
                    f"{uploaded_file.size / 1024:.1f} KB",
                    "neutral",
                )}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Enviar para a nuvem",
            width="stretch",
            key="azure_upload_button",
        ):
            with st.spinner("Enviando arquivo..."):
                success = azure_client.upload_blob(
                    selected_dir,
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )

            if success:
                st.success(
                    f"Arquivo salvo com sucesso em {selected_dir}."
                )
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    view_azure_manager()