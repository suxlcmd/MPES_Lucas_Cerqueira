import streamlit as st
from cloud.azure_blob_client import AzureBlobClient

def view_azure_manager():
    st.header("☁️ Azure Blob Storage - Gestão de Artefatos")
    azure_client = AzureBlobClient()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_dir = st.radio("Selecione o Diretório Lógico:", options=azure_client.logical_directories)
        
    with col2:
        st.subheader(f"Conteúdo de: {selected_dir}/")
        files = azure_client.list_blobs(selected_dir)
        
        if files:
            for f in files: st.markdown(f"📄 `{f}`")
        else:
            st.info("Nenhum artefato encontrado neste diretório.")
            
        st.write("---")
        st.subheader("Upload de Novo Artefato")
        uploaded_file = st.file_uploader("Selecione o arquivo para envio seguro")
        
        if uploaded_file is not None:
            if st.button("Enviar para Nuvem"):
                with st.spinner("Enviando arquivo..."):
                    file_bytes = uploaded_file.getvalue()
                    if azure_client.upload_blob(selected_dir, uploaded_file.name, file_bytes):
                        st.success(f"Arquivo salvo com sucesso em `{selected_dir}/`!")
                        st.rerun()