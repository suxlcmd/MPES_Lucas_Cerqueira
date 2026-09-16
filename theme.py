"""
theme.py
Módulo central de estilo visual do MPES (STEP-GAN + SMPC Fraud Detection Platform).
Importe `apply_theme()` no início de cada view/página para aplicar CSS consistente.
Também expõe pequenos helpers de UI (cards de métrica, badges de status, cabeçalhos de seção)
para reduzir duplicação de HTML/CSS entre as views.
"""

import streamlit as st

PRIMARY = "#4F7CFF"      # azul institucional
PRIMARY_DARK = "#2F4FCF"
ACCENT = "#00C2A8"       # verde-água para "sucesso"/normal
DANGER = "#FF4D6D"       # vermelho para alerta de fraude
WARNING = "#FFB020"
BG_CARD = "#12172B"
BG_CARD_LIGHT = "#1B2140"
TEXT_MUTED = "#8A93B8"


def apply_theme():
    """Injeta CSS global. Chame uma vez no topo de cada página/view."""
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            html, body, [class*="css"]  {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}

            /* ---------- Header da página ---------- */
            .mpes-page-header {{
                display: flex;
                align-items: center;
                gap: 14px;
                padding: 22px 28px;
                border-radius: 16px;
                background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
                box-shadow: 0 8px 24px rgba(47, 79, 207, 0.25);
                margin-bottom: 28px;
            }}
            .mpes-page-header .icon {{
                font-size: 34px;
                line-height: 1;
            }}
            .mpes-page-header .titles h1 {{
                color: #fff;
                font-size: 24px;
                font-weight: 800;
                margin: 0;
                letter-spacing: -0.3px;
            }}
            .mpes-page-header .titles p {{
                color: rgba(255,255,255,0.85);
                font-size: 14px;
                margin: 2px 0 0 0;
                font-weight: 400;
            }}

            /* ---------- Section header (subtítulos) ---------- */
            .mpes-section {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin: 28px 0 14px 0;
                padding-bottom: 8px;
                border-bottom: 1px solid rgba(138, 147, 184, 0.25);
            }}
            .mpes-section .step-badge {{
                background: {PRIMARY};
                color: #fff;
                font-size: 12px;
                font-weight: 700;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            }}
            .mpes-section h3 {{
                font-size: 17px;
                font-weight: 700;
                margin: 0;
            }}

            /* ---------- Cards genéricos ---------- */
            .mpes-card {{
                background: {BG_CARD};
                border: 1px solid rgba(138, 147, 184, 0.15);
                border-radius: 14px;
                padding: 18px 20px;
                margin-bottom: 12px;
            }}
            .mpes-card-light {{
                background: {BG_CARD_LIGHT};
                border-radius: 14px;
                padding: 16px 18px;
                margin-bottom: 10px;
                border-left: 3px solid {PRIMARY};
            }}

            /* ---------- Badges de status ---------- */
            .mpes-badge {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 4px 12px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.2px;
            }}
            .mpes-badge-ok {{ background: rgba(0, 194, 168, 0.15); color: {ACCENT}; }}
            .mpes-badge-danger {{ background: rgba(255, 77, 109, 0.15); color: {DANGER}; }}
            .mpes-badge-warning {{ background: rgba(255, 176, 32, 0.15); color: {WARNING}; }}
            .mpes-badge-neutral {{ background: rgba(138, 147, 184, 0.15); color: {TEXT_MUTED}; }}

            /* ---------- Métricas (substituem st.metric quando custom) ---------- */
            .mpes-metric {{
                background: {BG_CARD};
                border: 1px solid rgba(138, 147, 184, 0.15);
                border-radius: 14px;
                padding: 16px 18px;
                text-align: left;
            }}
            .mpes-metric .label {{
                color: {TEXT_MUTED};
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.4px;
                margin-bottom: 6px;
            }}
            .mpes-metric .value {{
                font-size: 26px;
                font-weight: 800;
                line-height: 1.1;
            }}
            .mpes-metric .delta {{
                font-size: 12px;
                font-weight: 600;
                margin-top: 4px;
            }}

            /* ---------- File list item (Azure manager) ---------- */
            .mpes-file-row {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 10px 14px;
                border-radius: 10px;
                background: {BG_CARD_LIGHT};
                margin-bottom: 6px;
                font-size: 13px;
            }}
            .mpes-file-row .fname {{
                font-family: 'Consolas', monospace;
                color: #E4E8FA;
            }}

            /* ---------- Botões primários ---------- */
            div.stButton > button, div.stFormSubmitButton > button {{
                background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
                color: #fff;
                border: none;
                border-radius: 10px;
                font-weight: 700;
                padding: 10px 22px;
                transition: all 0.15s ease-in-out;
                box-shadow: 0 4px 12px rgba(47, 79, 207, 0.25);
            }}
            div.stButton > button:hover, div.stFormSubmitButton > button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 6px 16px rgba(47, 79, 207, 0.35);
            }}

            /* ---------- Sidebar / radio / selectbox refinamento leve ---------- */
            [data-testid="stMetricValue"] {{
                font-weight: 800;
            }}
            hr {{
                border-color: rgba(138, 147, 184, 0.2) !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(icon: str, title: str, subtitle: str = ""):
    """Cabeçalho de página em gradiente, substitui st.header(...) simples."""
    st.markdown(
        f"""
        <div class="mpes-page-header">
            <div class="icon">{icon}</div>
            <div class="titles">
                <h1>{title}</h1>
                {f"<p>{subtitle}</p>" if subtitle else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(step_num, title: str):
    """Subtítulo de seção com numeração em badge circular."""
    st.markdown(
        f"""
        <div class="mpes-section">
            <div class="step-badge">{step_num}</div>
            <h3>{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(text: str, kind: str = "neutral"):
    """kind: ok | danger | warning | neutral"""
    return f'<span class="mpes-badge mpes-badge-{kind}">{text}</span>'


def metric_card(label: str, value: str, delta: str = "", color: str = "#E4E8FA"):
    delta_html = f'<div class="delta" style="color:{color}">{delta}</div>' if delta else ""
    return f"""
        <div class="mpes-metric">
            <div class="label">{label}</div>
            <div class="value" style="color:{color}">{value}</div>
            {delta_html}
        </div>
    """
