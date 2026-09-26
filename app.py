import streamlit as st

from theme import apply_theme, brand_block, page_header
from views.azure_manager import view_azure_manager
from views.dashboard import view_dashboard
from views.training import view_training


st.set_page_config(
    page_title="MPES | Auditoria STEP-GAN",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


apply_theme()


SUBVIEWS = {
    "training": {
        "label": "Treinamento e Inferência",
        "icon": "⚙️",
    },
    "dashboard": {
        "label": "Dashboard de Auditoria",
        "icon": "📊",
    },
    "azure": {
        "label": "Azure Blob Storage",
        "icon": "☁️",
    },
}


if "auditor_section" not in st.session_state:
    st.session_state.auditor_section = "training"


def render_sidebar():
    """Renderiza a navegação corporativa lateral."""
    st.sidebar.markdown(
        brand_block(),
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div style="
            margin-bottom: 10px;
            color: #64748B;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .5px;
            text-transform: uppercase;
        ">
            Área do Auditor
        </div>
        """,
        unsafe_allow_html=True,
    )

    options = list(SUBVIEWS.keys())

    selected = st.sidebar.radio(
        "Subvisões",
        options=options,
        format_func=lambda key: (
            f"{SUBVIEWS[key]['icon']}  "
            f"{SUBVIEWS[key]['label']}"
        ),
        key="auditor_navigation",
        label_visibility="collapsed",
    )

    st.session_state.auditor_section = selected

    st.sidebar.markdown(
        """
        <div class="mpes-trust-strip">
            <i class="bi bi-shield-lock"></i>
            <br><br>
            Ambiente protegido<br>
            STEP-GAN · SMPC
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_training():
    page_header(
        "⚙️",
        "Treinamento e Inferência",
        "Treinamento federado, análise de transações e detecção de anomalias.",
    )

    view_training()


def render_dashboard():
    page_header(
        "📊",
        "Dashboard de Auditoria",
        "Indicadores, métricas de classificação, perdas e alertas.",
    )

    view_dashboard()


def render_azure():
    page_header(
        "☁️",
        "Azure Blob Storage",
        "Gestão de bases, checkpoints e artefatos do pipeline.",
    )

    view_azure_manager()


def main():
    render_sidebar()

    section = st.session_state.auditor_section

    if section == "training":
        render_training()
    elif section == "dashboard":
        render_dashboard()
    elif section == "azure":
        render_azure()
    else:
        st.session_state.auditor_section = "training"
        st.rerun()


if __name__ == "__main__":
    main()