"""
Tema visual corporativo do MPES.

Este arquivo centraliza o estilo da aplicação Streamlit:
- paleta institucional;
- cabeçalhos e seções;
- cartões de métricas;
- botões;
- tabelas;
- formulários;
- alertas;
- navegação e componentes de upload.
"""

import streamlit as st


PRIMARY = "#2563EB"
PRIMARY_DARK = "#1E3A8A"
PRIMARY_LIGHT = "#DBEAFE"
ACCENT = "#0F766E"
SUCCESS = "#15803D"
DANGER = "#B91C1C"
WARNING = "#B45309"
BG_APP = "#F4F7FB"
BG_CARD = "#FFFFFF"
BG_MUTED = "#F8FAFC"
TEXT = "#172033"
TEXT_MUTED = "#64748B"
BORDER = "#E2E8F0"


def apply_theme():
    """Aplica o tema corporativo global da aplicação."""
    st.markdown(
        f"""
        <style>
        @import url(
            'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
        );

        :root {{
            --mpes-primary: {PRIMARY};
            --mpes-primary-dark: {PRIMARY_DARK};
            --mpes-primary-light: {PRIMARY_LIGHT};
            --mpes-accent: {ACCENT};
            --mpes-success: {SUCCESS};
            --mpes-danger: {DANGER};
            --mpes-warning: {WARNING};
            --mpes-bg: {BG_APP};
            --mpes-card: {BG_CARD};
            --mpes-muted: {BG_MUTED};
            --mpes-text: {TEXT};
            --mpes-text-muted: {TEXT_MUTED};
            --mpes-border: {BORDER};
        }}

        html, body, [class*="css"] {{
            font-family: Inter, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", sans-serif;
        }}

        .stApp {{
            background: {BG_APP};
            color: {TEXT};
        }}

        .main .block-container {{
            max-width: 1440px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid {BORDER};
        }}

        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1.5rem;
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: {PRIMARY_DARK};
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {{
            color: {TEXT};
        }}

        .mpes-page-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 22px 28px;
            margin-bottom: 26px;
            border-radius: 14px;
            background: linear-gradient(
                135deg,
                {PRIMARY_DARK} 0%,
                {PRIMARY} 100%
            );
            box-shadow: 0 8px 24px rgba(30, 58, 138, 0.18);
        }}

        .mpes-page-header .icon {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.16);
            font-size: 27px;
        }}

        .mpes-page-header .titles h1 {{
            margin: 0;
            color: #FFFFFF;
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.4px;
        }}

        .mpes-page-header .titles p {{
            margin: 4px 0 0 0;
            color: rgba(255, 255, 255, 0.84);
            font-size: 13px;
        }}

        .mpes-section {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 28px 0 14px 0;
            padding-bottom: 9px;
            border-bottom: 1px solid {BORDER};
        }}

        .mpes-section .step-badge {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 25px;
            height: 25px;
            flex-shrink: 0;
            border-radius: 50%;
            background: {PRIMARY};
            color: #FFFFFF;
            font-size: 12px;
            font-weight: 800;
        }}

        .mpes-section h3 {{
            margin: 0;
            color: {PRIMARY_DARK};
            font-size: 17px;
            font-weight: 750;
        }}

        .mpes-card {{
            margin-bottom: 14px;
            padding: 20px 22px;
            border: 1px solid {BORDER};
            border-radius: 12px;
            background: {BG_CARD};
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
        }}

        .mpes-card-light {{
            margin-bottom: 10px;
            padding: 16px 18px;
            border: 1px solid {BORDER};
            border-left: 3px solid {PRIMARY};
            border-radius: 10px;
            background: {BG_MUTED};
        }}

        .mpes-metric {{
            min-height: 106px;
            padding: 17px 18px;
            border: 1px solid {BORDER};
            border-radius: 12px;
            background: {BG_CARD};
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
        }}

        .mpes-metric .label {{
            margin-bottom: 8px;
            color: {TEXT_MUTED};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.45px;
            text-transform: uppercase;
        }}

        .mpes-metric .value {{
            font-size: 26px;
            font-weight: 800;
            line-height: 1.15;
        }}

        .mpes-metric .delta {{
            margin-top: 6px;
            color: {TEXT_MUTED};
            font-size: 12px;
            font-weight: 600;
        }}

        .mpes-badge {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 750;
            letter-spacing: 0.2px;
        }}

        .mpes-badge-ok {{
            background: #DCFCE7;
            color: {SUCCESS};
        }}

        .mpes-badge-danger {{
            background: #FEE2E2;
            color: {DANGER};
        }}

        .mpes-badge-warning {{
            background: #FEF3C7;
            color: {WARNING};
        }}

        .mpes-badge-neutral {{
            background: #E2E8F0;
            color: {TEXT_MUTED};
        }}

        .mpes-file-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 6px;
            padding: 11px 14px;
            border: 1px solid {BORDER};
            border-radius: 9px;
            background: {BG_MUTED};
            font-size: 13px;
        }}

        .mpes-file-row .fname {{
            overflow: hidden;
            color: {TEXT};
            font-family: Consolas, monospace;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        div.stButton > button,
        div.stFormSubmitButton > button {{
            min-height: 42px;
            border: 1px solid {PRIMARY};
            border-radius: 8px;
            background: {PRIMARY};
            color: #FFFFFF;
            font-size: 13px;
            font-weight: 700;
            box-shadow: 0 4px 10px rgba(37, 99, 235, 0.16);
            transition: all 0.15s ease-in-out;
        }}

        div.stButton > button:hover,
        div.stFormSubmitButton > button:hover {{
            border-color: {PRIMARY_DARK};
            background: {PRIMARY_DARK};
            color: #FFFFFF;
            transform: translateY(-1px);
            box-shadow: 0 6px 15px rgba(30, 58, 138, 0.22);
        }}

        div.stButton > button:focus,
        div.stFormSubmitButton > button:focus {{
            border-color: {PRIMARY};
            color: #FFFFFF;
        }}

        input, textarea, [data-baseweb="select"] > div {{
            border-radius: 8px !important;
        }}

        [data-testid="stDataFrame"] {{
            overflow: hidden;
            border: 1px solid {BORDER};
            border-radius: 10px;
        }}

        [data-testid="stFileUploader"] {{
            padding: 8px;
            border: 1px dashed #CBD5E1;
            border-radius: 10px;
            background: {BG_MUTED};
        }}

        [data-testid="stAlert"] {{
            border-radius: 9px;
        }}

        hr {{
            border-color: {BORDER};
        }}

        .stProgress > div > div > div > div {{
            background: {PRIMARY};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(icon: str, title: str, subtitle: str = ""):
    """Renderiza o cabeçalho corporativo da página."""
    subtitle_html = (
        f"<p>{subtitle}</p>"
        if subtitle
        else ""
    )

    st.markdown(
        f"""
        <div class="mpes-page-header">
            <div class="icon">{icon}</div>
            <div class="titles">
                <h1>{title}</h1>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(step_number, title: str):
    """Renderiza um título de seção numerado."""
    st.markdown(
        f"""
        <div class="mpes-section">
            <div class="step-badge">{step_number}</div>
            <h3>{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(text: str, kind: str = "neutral"):
    """Retorna um badge HTML de status."""
    allowed = {"ok", "danger", "warning", "neutral"}

    if kind not in allowed:
        kind = "neutral"

    return (
        f'<span class="mpes-badge mpes-badge-{kind}">'
        f"{text}</span>"
    )


def metric_card(
    label: str,
    value: str,
    delta: str = "",
    color: str = "#E4E8FA",
):
    """Retorna um cartão HTML de métrica."""
    delta_html = (
        f'<div class="delta">{delta}</div>'
        if delta
        else ""
    )

    return f"""
    <div class="mpes-metric">
        <div class="label">{label}</div>
        <div class="value" style="color: {color};">
            {value}
        </div>
        {delta_html}
    </div>
    """