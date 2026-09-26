"""Componentes compartilhados da camada de apresentação."""

import os

import numpy as np
import streamlit as st

from cloud.azure_blob_client import AzureBlobClient
from orchestration.audit_log import event_blob_name, event_to_bytes, make_event
from theme import metric_card

PRIMARY = "#1D4ED8"
ACCENT = "#0F766E"
DANGER = "#B91C1C"
WARNING = "#B45309"
MUTED = "#64748B"

RISK_COLORS = {"Crítico": "#7F1D1D", "Alto": DANGER, "Médio": WARNING, "Baixo": ACCENT}


@st.cache_resource(show_spinner=False)
def get_azure_client():
    """Uma conexão por processo, reaproveitada entre reruns e sessões."""
    return AzureBlobClient()


def current_user():
    """Usuário autenticado pelo Azure App Service (Easy Auth), quando habilitado."""
    try:
        headers = st.context.headers
        for nome in ("X-Ms-Client-Principal-Name", "X-MS-CLIENT-PRINCIPAL-NAME"):
            if headers.get(nome):
                return headers.get(nome)
    except Exception:  # noqa: BLE001 - fora de uma requisição HTTP (ex.: testes)
        pass
    return os.getenv("MPES_DEFAULT_USER", "usuário não identificado")


def record_event(event_type, details):
    """Registra o evento na trilha da sessão e, se possível, grava o JSON em audit-artifacts."""
    event = make_event(event_type, current_user(), details)
    client = get_azure_client()
    event["persisted"] = bool(
        client.available and client.upload_blob("audit-artifacts", event_blob_name(event), event_to_bytes(event))
    )
    st.session_state.setdefault("audit_events", []).append(event)
    return event


def fmt_metric(value):
    if isinstance(value, (float, np.floating)):
        return "-" if np.isnan(value) else f"{value:.4f}"
    return str(value)


def metric_grid(items, per_row=4):
    """items: lista de (rótulo, valor, legenda, cor)."""
    for inicio in range(0, len(items), per_row):
        linha = items[inicio:inicio + per_row]
        for coluna, (rotulo, valor, legenda, cor) in zip(st.columns(per_row), linha):
            with coluna:
                st.markdown(metric_card(rotulo, fmt_metric(valor), legenda, cor), unsafe_allow_html=True)


def model_metric_items(metrics):
    return [(nome, valor, "teste reservado (20%)", PRIMARY) for nome, valor in metrics.items()]
