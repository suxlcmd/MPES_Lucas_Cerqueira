"""
Tema corporativo MPES com Bootstrap 5.

O Bootstrap é utilizado para:
- grid responsivo;
- cards;
- badges;
- alertas;
- espaçamentos;
- identidade visual corporativa.

Os componentes interativos continuam sendo controlados pelo Streamlit.
"""

import streamlit as st


PRIMARY = "#1D4ED8"
PRIMARY_DARK = "#1E3A8A"
PRIMARY_LIGHT = "#DBEAFE"
ACCENT = "#0F766E"
SUCCESS = "#15803D"
DANGER = "#B91C1C"
WARNING = "#B45309"

BG_APP = "#F4F7FB"
BG_CARD = "#FFFFFF"
TEXT = "#172033"
TEXT_MUTED = "#64748B"
BORDER = "#E2E8F0"


def apply_theme():
    """
    Injeta Bootstrap e o tema corporativo global.
    """

    st.markdown(
        """
        <link
            href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css"
            rel="stylesheet"
            crossorigin="anonymous"
        >

        <link
            rel="stylesheet"
            href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
        >
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <style>
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
            --mpes-text: {TEXT};
            --mpes-text-muted: {TEXT_MUTED};
            --mpes-border: {BORDER};

            --bs-primary: {PRIMARY};
            --bs-primary-rgb: 29, 78, 216;
            --bs-body-bg: {BG_APP};
            --bs-body-color: {TEXT};
            --bs-border-color: {BORDER};
            --bs-border-radius: 0.75rem;
        }}

        html,
        body,
        [class*="css"] {{
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }}

        .stApp {{
            background: var(--mpes-bg);
            color: var(--mpes-text);
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
            border-right: 1px solid var(--mpes-border);
        }}

        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1.25rem;
        }}

        .mpes-brand {{
            padding: 8px 6px 22px 6px;
            border-bottom: 1px solid var(--mpes-border);
            margin-bottom: 20px;
        }}

        .mpes-brand-icon {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 42px;
            height: 42px;
            margin-bottom: 10px;
            border-radius: 10px;
            background: var(--mpes-primary-light);
            color: var(--mpes-primary-dark);
            font-size: 23px;
        }}

        .mpes-brand-title {{
            color: var(--mpes-primary-dark);
            font-size: 20px;
            font-weight: 800;
            letter-spacing: -0.4px;
        }}

        .mpes-brand-subtitle {{
            margin-top: 3px;
            color: var(--mpes-text-muted);
            font-size: 11px;
        }}

        .mpes-page-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 26px;
            padding: 22px 28px;
            border-radius: 14px;
            background:
                linear-gradient(
                    135deg,
                    var(--mpes-primary-dark),
                    var(--mpes-primary)
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
            color: #FFFFFF;
            font-size: 27px;
        }}

        .mpes-page-header h1 {{
            margin: 0;
            color: #FFFFFF;
            font-size: 24px;
            font-weight: 800;
        }}

        .mpes-page-header p {{
            margin: 4px 0 0 0;
            color: rgba(255, 255, 255, 0.85);
            font-size: 13px;
        }}

        .mpes-section {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 28px 0 14px 0;
            padding-bottom: 9px;
            border-bottom: 1px solid var(--mpes-border);
        }}

        .mpes-section .step-badge {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 25px;
            height: 25px;
            border-radius: 50%;
            background: var(--mpes-primary);
            color: #FFFFFF;
            font-size: 12px;
            font-weight: 800;
        }}

        .mpes-section h3 {{
            margin: 0;
            color: var(--mpes-primary-dark);
            font-size: 17px;
            font-weight: 750;
        }}

        .mpes-card {{
            margin-bottom: 14px;
            padding: 20px 22px;
            border: 1px solid var(--mpes-border);
            border-radius: 12px;
            background: var(--mpes-card);
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
        }}

        .mpes-card-title {{
            margin-bottom: 6px;
            color: var(--mpes-primary-dark);
            font-size: 16px;
            font-weight: 750;
        }}

        .mpes-card-description {{
            margin-bottom: 0;
            color: var(--mpes-text-muted);
            font-size: 13px;
            line-height: 1.6;
        }}

        .mpes-metric-card {{
            height: 100%;
            min-height: 112px;
            padding: 18px;
            border: 1px solid var(--mpes-border);
            border-radius: 12px;
            background: #FFFFFF;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
        }}

        .mpes-metric-label {{
            margin-bottom: 8px;
            color: var(--mpes-text-muted);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.45px;
            text-transform: uppercase;
        }}

        .mpes-metric-value {{
            color: var(--mpes-primary);
            font-size: 27px;
            font-weight: 800;
            line-height: 1.1;
        }}

        .mpes-metric-delta {{
            margin-top: 6px;
            color: var(--mpes-text-muted);
            font-size: 12px;
        }}

        .mpes-file-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 7px;
            padding: 12px 14px;
            border: 1px solid var(--mpes-border);
            border-radius: 9px;
            background: #F8FAFC;
        }}

        .mpes-file-name {{
            overflow: hidden;
            color: var(--mpes-text);
            font-family: Consolas, monospace;
            font-size: 13px;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .mpes-file-directory {{
            color: var(--mpes-text-muted);
            font-size: 11px;
        }}

        .mpes-status {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 750;
        }}

        .mpes-status-success {{
            background: #DCFCE7;
            color: var(--mpes-success);
        }}

        .mpes-status-danger {{
            background: #FEE2E2;
            color: var(--mpes-danger);
        }}

        .mpes-status-warning {{
            background: #FEF3C7;
            color: var(--mpes-warning);
        }}

        .mpes-status-neutral {{
            background: #E2E8F0;
            color: var(--mpes-text-muted);
        }}

        .mpes-trust-strip {{
            margin-top: 24px;
            padding: 15px 18px;
            border: 1px solid var(--mpes-border);
            border-radius: 10px;
            background: #FFFFFF;
            color: var(--mpes-text-muted);
            font-size: 12px;
            text-align: center;
        }}

        div.stButton > button,
        div.stFormSubmitButton > button {{
            min-height: 42px;
            border: 1px solid var(--mpes-primary);
            border-radius: 8px;
            background: var(--mpes-primary);
            color: #FFFFFF;
            font-size: 13px;
            font-weight: 700;
            box-shadow: 0 4px 10px rgba(37, 99, 235, 0.16);
            transition: all 0.15s ease-in-out;
        }}

        div.stButton > button:hover,
        div.stFormSubmitButton > button:hover {{
            border-color: var(--mpes-primary-dark);
            background: var(--mpes-primary-dark);
            color: #FFFFFF;
            transform: translateY(-1px);
        }}

        [data-testid="stDataFrame"] {{
            overflow: hidden;
            border: 1px solid var(--mpes-border);
            border-radius: 10px;
        }}

        [data-testid="stFileUploader"] {{
            padding: 8px;
            border: 1px dashed #CBD5E1;
            border-radius: 10px;
            background: #F8FAFC;
        }}

        [data-testid="stAlert"] {{
            border-radius: 9px;
        }}

        [data-testid="stMetricValue"] {{
            font-weight: 800;
        }}

        .stProgress > div > div > div > div {{
            background: var(--mpes-primary);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(icon: str, title: str, subtitle: str = ""):
    """Renderiza o cabeçalho principal da página."""
    subtitle_html = (
        f"<p>{subtitle}</p>"
        if subtitle
        else ""
    )

    st.markdown(
        f"""
        <div class="mpes-page-header">
            <div class="icon">{icon}</div>
            <div>
                <h1>{title}</h1>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(step_number, title: str):
    """Renderiza o título de uma seção."""
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
    """Retorna um badge corporativo em HTML."""
    kind_map = {
        "ok": "success",
        "success": "success",
        "danger": "danger",
        "warning": "warning",
        "neutral": "neutral",
    }

    css_kind = kind_map.get(kind, "neutral")

    return (
        f'<span class="mpes-status mpes-status-{css_kind}">'
        f"{text}</span>"
    )


def metric_card(
    label: str,
    value: str,
    delta: str = "",
    color: str = PRIMARY,
):
    """Retorna um cartão de métrica em HTML."""
    delta_html = (
        f'<div class="mpes-metric-delta">{delta}</div>'
        if delta
        else ""
    )

    return f"""
    <div class="mpes-metric-card">
        <div class="mpes-metric-label">{label}</div>
        <div class="mpes-metric-value" style="color: {color};">
            {value}
        </div>
        {delta_html}
    </div>
    """


def brand_block():
    """Retorna o bloco de identidade visual da barra lateral."""
    return """
    <div class="mpes-brand">
        <div class="mpes-brand-icon">
            <i class="bi bi-shield-check"></i>
        </div>
        <div class="mpes-brand-title">MPES</div>
        <div class="mpes-brand-subtitle">
            Fraud Detection Platform
        </div>
    </div>
    """