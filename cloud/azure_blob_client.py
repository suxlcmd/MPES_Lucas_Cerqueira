import os
import pandas as pd
import io
import streamlit as st
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env para o sistema
load_dotenv()

class AzureBlobClient:
    """Módulo azure_blob_client.py (RF01, Sec 3.5)."""
    
    def __init__(self):
        # Busca a string de conexão das variáveis de ambiente de forma segura
        self.connection_string = os.getenv("AZURE_CONNECTION_STRING")
        self.container_name = "mpes"
        self.logical_directories = ["raw-data", "processed-data", "model-checkpoints", "audit-artifacts", "configuration"]
        
        try:
            if not self.connection_string:
                raise ValueError("String de conexão do Azure não encontrada. Verifique o arquivo .env.")
                
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
        except Exception as e:
            st.error(f"Falha ao conectar no Azure: {e}")
            self.container_client = None

    def list_blobs(self, logical_dir):
        if not self.container_client: return []
        try:
            blobs = self.container_client.list_blobs(name_starts_with=f"{logical_dir}/")
            return [b.name.replace(f"{logical_dir}/", "") for b in blobs if b.name != f"{logical_dir}/"]
        except Exception:
            return []

    def upload_blob(self, logical_dir, file_name, file_data):
        if not self.container_client: return False
        blob_path = f"{logical_dir}/{file_name}"
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            blob_client.upload_blob(file_data, overwrite=True)
            return True
        except Exception as e:
            st.error(f"Erro no upload: {e}")
            return False

    def download_blob_as_dataframe(self, logical_dir, file_name):
        if not self.container_client: return None
        blob_path = f"{logical_dir}/{file_name}"
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            download_stream = blob_client.download_blob()
            conteudo = download_stream.readall()
            # Uso de engine=python e sep=None para auto-detectar delimitadores complexos
            df = pd.read_csv(io.BytesIO(conteudo), sep=None, engine='python')
            return df
        except Exception as e:
            st.error(f"Erro ao baixar {file_name}: {e}")
            return None