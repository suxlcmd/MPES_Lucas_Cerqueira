"""
Trilha de auditoria (Camada 2): cada ação relevante gera um evento imutável com data/hora (UTC), usuário,
tipo e detalhes (dataset e seu SHA-256, modelo, parâmetros, resultados). Os eventos são gravados como
JSON em audit-artifacts/eventos/ quando o Azure está disponível.
"""

import datetime
import hashlib
import json
import uuid


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def make_event(event_type, user, details):
    agora = datetime.datetime.now(datetime.timezone.utc)
    return {
        "id": uuid.uuid4().hex,
        "timestamp": agora.isoformat(timespec="seconds"),
        "user": user,
        "type": event_type,
        "details": details,
    }


def event_blob_name(event):
    data = event["timestamp"][:10]
    carimbo = event["timestamp"].replace(":", "").replace("-", "")[:15]
    return f"eventos_{data}_{carimbo}_{event['type']}_{event['id'][:8]}.json"


def event_to_bytes(event):
    return json.dumps(event, ensure_ascii=False, indent=2, default=str).encode("utf-8")
