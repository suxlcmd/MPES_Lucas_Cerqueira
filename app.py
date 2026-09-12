import streamlit as st
from views.training import view_training
from views.dashboard import view_dashboard
from views.azure_manager import view_azure_manager

st.set_page_config(
    page_title="Framework STEP-GAN Auditoria",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Machine_Learning_AI_Icon.svg/512px-Machine_Learning_AI_Icon.svg.png", width=80)
    st.sidebar.title("Framework de Auditoria")
    
    menu = ["⚙️ Treinamento Federado", "📊 Dashboard Analítico", "☁️ Azure Storage Manager"]
    choice = st.sidebar.radio("Navegação", menu)
    
    if choice == "⚙️ Treinamento Federado": view_training()
    elif choice == "📊 Dashboard Analítico": view_dashboard()
    elif choice == "☁️ Azure Storage Manager": view_azure_manager()

if __name__ == "__main__":
    main()