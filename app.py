import hmac
import os

import streamlit as st

from theme import apply_theme, brand_block, esc
from views.azure_manager import view_azure_manager
from views.common import current_user
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
    "training": {"label": "Treinamento e Inferência", "icon": "⚙️", "view": view_training},
    "dashboard": {"label": "Dashboard de Auditoria", "icon": "📊", "view": view_dashboard},
    "azure": {"label": "Azure Blob Storage", "icon": "☁️", "view": view_azure_manager},
}


def access_granted():
    """
    Controle de acesso opcional: se MPES_ACCESS_PASSWORD estiver definida, exige a senha.
    Em produção, recomenda-se a autenticação do App Service (Easy Auth / Microsoft Entra ID).
    """
    senha = os.getenv("MPES_ACCESS_PASSWORD")
    if not senha or st.session_state.get("access_granted"):
        return True
    st.markdown(brand_block(), unsafe_allow_html=True)
    with st.form("login"):
        tentativa = st.text_input("Senha de acesso", type="password")
        if st.form_submit_button("Entrar"):
            if hmac.compare_digest(tentativa.encode(), senha.encode()):
                st.session_state.access_granted = True
                st.rerun()
            st.error("Senha incorreta.")
    return False


def render_sidebar():
    st.sidebar.markdown(brand_block(), unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div style="margin-bottom:10px;color:#64748B;font-size:11px;font-weight:700;'
        'letter-spacing:.5px;text-transform:uppercase;">Área do Auditor</div>',
        unsafe_allow_html=True,
    )
    selected = st.sidebar.radio(
        "Subvisões",
        options=list(SUBVIEWS),
        format_func=lambda key: f"{SUBVIEWS[key]['icon']}  {SUBVIEWS[key]['label']}",
        key="auditor_navigation",
        label_visibility="collapsed",
    )

    bundle = st.session_state.get("model_bundle")
    modelo = bundle.metadata.get("model_id", "-") if bundle else "nenhum"
    st.sidebar.markdown(
        f"""
        <div class="mpes-trust-strip">
            <i class="bi bi-shield-lock"></i><br><br>
            Ambiente protegido<br>STEP-GAN · SMPC<br><br>
            <span style="font-size:11px">Usuário: {esc(current_user())}<br>Modelo ativo: {esc(modelo)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return selected


def main():
    if not access_granted():
        return
    SUBVIEWS[render_sidebar()]["view"]()


if __name__ == "__main__":
    main()
