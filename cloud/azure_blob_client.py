"""
Cliente do Azure Blob Storage (Camada 5, RF01).

- Credenciais apenas por variável de ambiente (AZURE_CONNECTION_STRING; .env em desenvolvimento).
- Sem chamadas de interface aqui: erros ficam em `self.error` e a camada de apresentação decide como exibi-los.
- Nomes de arquivos são sanitizados e os diretórios lógicos são restritos à lista permitida.
"""

import csv
import io
import os
import re

import pandas as pd
from dotenv import load_dotenv

try:
    from azure.storage.blob import BlobServiceClient
except ImportError:  # pragma: no cover - dependência opcional em testes
    BlobServiceClient = None

load_dotenv()

LOGICAL_DIRECTORIES = ["raw-data", "processed-data", "model-checkpoints", "audit-artifacts", "configuration"]
_NOME_SEGURO = re.compile(r"[^A-Za-z0-9._\- ]+")


def sanitize_filename(name):
    """Remove caminhos e caracteres perigosos: '../x', 'a/b' ou '<script>' não saem do diretório lógico."""
    nome = os.path.basename(str(name).replace("\\", "/")).strip()
    nome = _NOME_SEGURO.sub("_", nome).lstrip(".")
    if not nome:
        raise ValueError("Nome de arquivo inválido.")
    return nome[:200]


class AzureBlobClient:
    def __init__(self, connection_string=None, container_name=None):
        self.connection_string = connection_string or os.getenv("AZURE_CONNECTION_STRING")
        self.container_name = container_name or os.getenv("AZURE_CONTAINER_NAME", "mpes")
        self.logical_directories = list(LOGICAL_DIRECTORIES)
        self.container_client = None
        self.error = None

        if BlobServiceClient is None:
            self.error = "Pacote azure-storage-blob não instalado."
            return
        if not self.connection_string:
            self.error = "AZURE_CONNECTION_STRING não configurada (variável de ambiente ou arquivo .env)."
            return
        try:
            service = BlobServiceClient.from_connection_string(self.connection_string)
            self.container_client = service.get_container_client(self.container_name)
        except Exception as exc:  # noqa: BLE001 - mensagem exibida ao usuário
            self.error = f"Falha ao conectar no Azure: {exc}"

    @property
    def available(self):
        return self.container_client is not None

    def _path(self, logical_dir, file_name):
        if logical_dir not in self.logical_directories:
            raise ValueError(f"Diretório lógico não permitido: {logical_dir}")
        return f"{logical_dir}/{sanitize_filename(file_name)}"

    def list_blob_details(self, logical_dir):
        if not self.available or logical_dir not in self.logical_directories:
            return []
        try:
            blobs = self.container_client.list_blobs(name_starts_with=f"{logical_dir}/")
            detalhes = [
                {
                    "name": b.name[len(logical_dir) + 1:],
                    "size": b.size,
                    "last_modified": b.last_modified,
                }
                for b in blobs
                if b.name != f"{logical_dir}/"
            ]
            return sorted(detalhes, key=lambda d: d["last_modified"] or 0, reverse=True)
        except Exception as exc:  # noqa: BLE001
            self.error = f"Erro ao listar {logical_dir}: {exc}"
            return []

    def list_blobs(self, logical_dir):
        return [d["name"] for d in self.list_blob_details(logical_dir)]

    def upload_blob(self, logical_dir, file_name, file_data, overwrite=False):
        """Envia o arquivo; por padrão não sobrescreve (preserva artefatos de auditoria)."""
        if not self.available:
            return False
        try:
            blob = self.container_client.get_blob_client(self._path(logical_dir, file_name))
            blob.upload_blob(file_data, overwrite=overwrite)
            return True
        except Exception as exc:  # noqa: BLE001
            self.error = f"Erro no upload de {file_name}: {exc}"
            return False

    def download_blob_bytes(self, logical_dir, file_name):
        if not self.available:
            return None
        try:
            blob = self.container_client.get_blob_client(self._path(logical_dir, file_name))
            return blob.download_blob().readall()
        except Exception as exc:  # noqa: BLE001
            self.error = f"Erro ao baixar {file_name}: {exc}"
            return None

    def download_blob_as_dataframe(self, logical_dir, file_name):
        conteudo = self.download_blob_bytes(logical_dir, file_name)
        if conteudo is None:
            return None
        return read_csv_bytes(conteudo)


def read_csv_bytes(data):
    """
    Lê CSV detectando o separador (vírgula, ponto e vírgula, tab ou barra vertical) pela amostra inicial
    e usa o motor C do pandas: `sep=None` forçaria o motor Python, muito lento em bases grandes.
    """
    amostra = data[:65536].decode("utf-8", errors="ignore")
    try:
        separador = csv.Sniffer().sniff(amostra, delimiters=",;\t|").delimiter
    except csv.Error:
        separador = ","
    return pd.read_csv(io.BytesIO(data), sep=separador, low_memory=False)
