import pandas as pd
import io
import streamlit as st
from azure.storage.blob import BlobServiceClient

class AzureBlobClient:
    def __init__(self):
        self.connection_string = "DefaultEndpointsProtocol=https;AccountName=blobmpeslucas;AccountKey=hIgaU/Rx7lrDa+wkHaY/J+rKTQtwj4qV8x7/zOPTg6RuFyskbf7JUoamxWBJoDZ9b//C+VMj6iwJ+AStDGi4FQ==;EndpointSuffix=core.windows.net"
        self.container_name = "mpes"
        self.logical_directories = ["raw-data", "processed-data", "model-checkpoints", "audit-artifacts", "configuration"]
        
        try:
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
            df = pd.read_csv(io.BytesIO(conteudo), sep=None, engine='python')
            return df
        except Exception as e:
            st.error(f"Erro ao baixar {file_name}: {e}")
            return None