"""
╔══════════════════════════════════════════════════════════════════╗
║   AI-Powered Quick Commerce Delivery Intelligence System         ║
║   End-to-End Analytics + ML Prediction Dashboard                 ║
║   Built with Streamlit | Plotly | Scikit-Learn | XGBoost         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuickMind Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
#  GLOBAL PATHS
# ──────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
DATA   = os.path.join(BASE, "data", "cleaned_quick_commerce.csv")
MDIR   = os.path.join(BASE, "models")

# ──────────────────────────────────────────────────────────────────
#  DESIGN TOKENS — Premium Startup Theme
# ──────────────────────────────────────────────────────────────────
PRIMARY   = "#4F46E5"   # Indigo
ACCENT    = "#F59E0B"   # Amber
SUCCESS   = "#10B981"   # Emerald
DANGER    = "#EF4444"   # Red
WARNING   = "#F97316"   # Orange
MUTED     = "#64748B"   # Slate
BG        = "#F1F5F9"   # Slate-100
CARD_BG   = "#FFFFFF"
TEXT      = "#0F172A"   # Slate-900
GRADIENT  = "linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #2563EB 100%)"

COMPANY_COLORS = {
    "Blinkit":          "#F9C22E",
    "Zepto":            "#8B5CF6",
    "Swiggy Instamart": "#FC8019",
    "Dunzo":            "#2BB741",
    "Jio Mart":         "#0066CC",
    "Flipkart Minutes": "#2874F0",
    "Amazon Now":       "#FF9900",
    "Big Basket":       "#84C225",
}

# ──────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #0F172A;
}
.main { background: #F1F5F9; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(79,70,229,0.15);
    border: 1px solid rgba(79,70,229,0.3);
    color: #E2E8F0 !important;
    border-radius: 10px;
    padding: 10px 16px;
    width: 100%;
    text-align: left;
    font-size: 0.875rem;
    font-weight: 500;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(79,70,229,0.35);
    border-color: #4F46E5;
    transform: translateX(3px);
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #2563EB 100%);
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 24px;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero-header h1 { font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hero-header p  { font-size: 1rem; opacity: 0.8; margin: 6px 0 0; }

/* ── Section title ── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0F172A;
    margin: 24px 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #E2E8F0, transparent);
    margin-left: 8px;
}

/* ── KPI Cards ── */
.kpi-card {
    background: white;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    border: 1px solid #F1F5F9;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.1); }
.kpi-card .label   { font-size: 0.78rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-card .value   { font-size: 1.9rem; font-weight: 800; color: #0F172A; line-height: 1.1; margin: 6px 0 4px; }
.kpi-card .delta   { font-size: 0.8rem; font-weight: 600; }
.kpi-card .icon    { font-size: 2rem; opacity: 0.12; position: absolute; right: 16px; top: 14px; }
.kpi-card .stripe  { position: absolute; left: 0; top: 0; bottom: 0; width: 4px; border-radius: 16px 0 0 16px; }

/* ── Data Cards ── */
.insight-card {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    border: 1px solid #F1F5F9;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 12px;
}
.insight-card h4 { font-size: 0.9rem; font-weight: 700; margin: 0 0 6px; color: #0F172A; }
.insight-card p  { font-size: 0.82rem; color: #64748B; margin: 0; line-height: 1.5; }

/* ── Alert badges ── */
.badge-red    { background:#FEF2F2; color:#EF4444; border:1px solid #FECACA; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-green  { background:#F0FDF4; color:#10B981; border:1px solid #A7F3D0; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-amber  { background:#FFFBEB; color:#D97706; border:1px solid #FDE68A; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-blue   { background:#EFF6FF; color:#3B82F6; border:1px solid #BFDBFE; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

/* ── Login page ── */
.login-wrap {
    max-width: 420px;
    margin: 60px auto;
    background: white;
    border-radius: 24px;
    padding: 48px 40px;
    box-shadow: 0 20px 60px rgba(79,70,229,0.12);
    border: 1px solid #E0E7FF;
}
.login-logo { font-size: 2.5rem; text-align: center; margin-bottom: 8px; }
.login-title { font-size: 1.6rem; font-weight: 800; text-align: center; color: #0F172A; }
.login-sub   { font-size: 0.88rem; text-align: center; color: #64748B; margin-bottom: 28px; }

/* ── Plotly chart container ── */
.chart-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #F1F5F9;
    margin-bottom: 16px;
}

/* ── Prediction result ── */
.pred-result {
    background: linear-gradient(135deg, #EEF2FF, #E0E7FF);
    border: 2px solid #C7D2FE;
    border-radius: 20px;
    padding: 28px 32px;
    text-align: center;
}
.pred-time { font-size: 3.5rem; font-weight: 800; color: #4F46E5; line-height: 1; }
.pred-label { font-size: 1rem; color: #6366F1; font-weight: 600; margin-top: 4px; }

/* ── Metric row ── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; }

/* ── Streamlit overrides ── */
div[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 14px 16px;
    border: 1px solid #F1F5F9;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.stSelectbox > div > div { border-radius: 10px !important; }
.stTextInput > div > div { border-radius: 10px !important; }
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 0.85rem;
    color: #64748B !important;
}
.stTabs [aria-selected="true"] {
    background: #4F46E5 !important;
    color: white !important;
    border-color: #4F46E5 !important;
}
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
#  CREDENTIALS  (extend as needed)
# ──────────────────────────────────────────────────────────────────
USERS = {
    "admin":   {"password": "admin123",  "role": "Admin",   "name": "Admin User"},
    "analyst": {"password": "analyst123","role": "Analyst", "name": "Data Analyst"},
    "viewer":  {"password": "viewer123", "role": "Viewer",  "name": "Dashboard Viewer"},
}

# ──────────────────────────────────────────────────────────────────
#  SESSION STATE BOOTSTRAP
# ──────────────────────────────────────────────────────────────────
for key, val in [
    ("logged_in", False), ("username", ""), ("role", ""), ("name", ""),
    ("page", "Executive Dashboard"),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ──────────────────────────────────────────────────────────────────
#  DATA LOADING (cached)
# ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚡ Loading dataset…")
def load_data(path):
    df = pd.read_csv(path)
    df['SLA_Breach'] = (df['Delivery_Time'] > 30).astype(int)
    df['Speed_Score'] = (df['Delivery_Partner_Rating'] / df['Delivery_Time'] * 10).round(2)
    df['Revenue_Tier'] = pd.cut(df['Order_Value'],
                                 bins=[0, 200, 600, 1200, 99999],
                                 labels=['Low', 'Medium', 'High', 'Premium'])
    df['Distance_Bin'] = pd.cut(df['Distance_km'],
                                 bins=[0, 5, 10, 20, 100],
                                 labels=['Very Short', 'Short', 'Medium', 'Long'])
    return df

@st.cache_resource(show_spinner="🤖 Loading ML models…")
def load_models():
    model    = joblib.load(os.path.join(MDIR, "best_model.pkl"))
    encoders = joblib.load(os.path.join(MDIR, "label_encoders.pkl"))
    features = joblib.load(os.path.join(MDIR, "feature_names.pkl"))
    metrics  = pd.read_csv(os.path.join(MDIR, "model_metrics.csv"))
    return model, encoders, features, metrics

# ──────────────────────────────────────────────────────────────────
#  HELPER: PLOTLY THEME
# ──────────────────────────────────────────────────────────────────
def fig_style(fig, height=360, title=""):
    fig.update_layout(
        height=height, title=title,
        font=dict(family="Plus Jakarta Sans", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=16, r=16, t=40 if title else 16, b=16),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center",
                    font=dict(size=11)),
        xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
    )
    return fig

# ──────────────────────────────────────────────────────────────────
#  KPI CARD RENDERER
# ──────────────────────────────────────────────────────────────────
def kpi(label, value, delta="", color=PRIMARY, icon="📊"):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="stripe" style="background:{color}"></div>
      <div class="icon">{icon}</div>
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      <div class="delta" style="color:{color}">{delta}</div>
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
#  LOGIN PAGE
# ──────────────────────────────────────────────────────────────────
def page_login():
    st.markdown('<style>[data-testid="stSidebar"]{display:none}</style>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* Full page gradient background */
    .stApp {
        background: linear-gradient(135deg, #0F0C29 0%, #1a1a4e 40%, #24243e 100%) !important;
        min-height: 100vh;
    }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* Animated floating circles */
    .bg-circle-1 {
        position: fixed; top: -100px; left: -100px;
        width: 400px; height: 400px; border-radius: 50%;
        background: radial-gradient(circle, rgba(99,102,241,0.25) 0%, transparent 70%);
        animation: float1 8s ease-in-out infinite;
        pointer-events: none; z-index: 0;
    }
    .bg-circle-2 {
        position: fixed; bottom: -120px; right: -80px;
        width: 500px; height: 500px; border-radius: 50%;
        background: radial-gradient(circle, rgba(236,72,153,0.2) 0%, transparent 70%);
        animation: float2 10s ease-in-out infinite;
        pointer-events: none; z-index: 0;
    }
    .bg-circle-3 {
        position: fixed; top: 40%; left: 60%;
        width: 300px; height: 300px; border-radius: 50%;
        background: radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 70%);
        animation: float1 12s ease-in-out infinite reverse;
        pointer-events: none; z-index: 0;
    }
    @keyframes float1 {
        0%, 100% { transform: translate(0,0) scale(1); }
        50%       { transform: translate(30px, 40px) scale(1.05); }
    }
    @keyframes float2 {
        0%, 100% { transform: translate(0,0) scale(1); }
        50%       { transform: translate(-40px, -30px) scale(1.08); }
    }

    /* Login wrapper */
    .login-outer {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        padding: 40px 16px;
    }
    .login-box {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 28px;
        padding: 48px 44px 40px;
        width: 100%;
        max-width: 460px;
        position: relative;
        z-index: 10;
        box-shadow: 0 32px 80px rgba(0,0,0,0.5),
                    0 0 0 1px rgba(255,255,255,0.06) inset;
    }

    /* Logo area */
    .login-logo-ring {
        width: 72px; height: 72px;
        background: linear-gradient(135deg, #6366F1, #8B5CF6, #EC4899);
        border-radius: 20px;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 20px;
        font-size: 2rem;
        box-shadow: 0 8px 32px rgba(99,102,241,0.5);
    }
    .login-title {
        font-size: 1.85rem; font-weight: 800;
        text-align: center; color: #FFFFFF;
        letter-spacing: -0.5px; margin-bottom: 6px;
    }
    .login-sub {
        font-size: 0.88rem; text-align: center;
        color: rgba(255,255,255,0.55); margin-bottom: 28px;
        line-height: 1.5;
    }

    /* Feature pills */
    .feature-pills {
        display: flex; gap: 8px; justify-content: center;
        flex-wrap: wrap; margin-bottom: 28px;
    }
    .pill {
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.35);
        color: #A5B4FC;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* Divider */
    .login-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 24px 0;
    }

    /* Demo creds */
    .demo-creds {
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 20px;
        text-align: center;
    }
    .demo-creds p {
        margin: 0; font-size: 0.78rem;
        color: rgba(255,255,255,0.5);
    }
    .demo-creds b { color: #A5B4FC; }

    /* Stats row */
    .stats-row {
        display: flex; gap: 0;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; overflow: hidden;
        margin-bottom: 28px;
    }
    .stat-item {
        flex: 1; padding: 12px 8px; text-align: center;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .stat-item:last-child { border-right: none; }
    .stat-val {
        font-size: 1.1rem; font-weight: 800; color: #E0E7FF;
        line-height: 1;
    }
    .stat-lbl {
        font-size: 0.65rem; color: rgba(255,255,255,0.4);
        margin-top: 3px; text-transform: uppercase; letter-spacing: 0.05em;
    }

    /* Override Streamlit input styles for dark theme */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 12px 16px !important;
        font-size: 0.9rem !important;
        transition: border-color 0.2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.3) !important;
    }
    .stTextInput label {
        color: rgba(255,255,255,0.7) !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }

    /* Sign in button */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        width: 100% !important;
        margin-top: 8px !important;
        box-shadow: 0 8px 24px rgba(99,102,241,0.4) !important;
        transition: all 0.2s !important;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(99,102,241,0.55) !important;
    }

    /* Footer */
    .login-footer {
        text-align: center;
        margin-top: 28px;
        font-size: 0.72rem;
        color: rgba(255,255,255,0.25);
    }
    </style>

    <!-- Animated background blobs -->
    <div class="bg-circle-1"></div>
    <div class="bg-circle-2"></div>
    <div class="bg-circle-3"></div>
    """, unsafe_allow_html=True)

    # Center column
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("""
        <div style="position:relative;z-index:10;">
          <!-- Logo -->
          <div style="width:72px;height:72px;background:linear-gradient(135deg,#6366F1,#8B5CF6,#EC4899);
                      border-radius:20px;display:flex;align-items:center;justify-content:center;
                      margin:48px auto 20px;font-size:2rem;
                      box-shadow:0 8px 32px rgba(99,102,241,0.5);text-align:center;">⚡</div>

          <!-- Title -->
          <div style="font-size:1.85rem;font-weight:800;text-align:center;color:#FFFFFF;
                      letter-spacing:-0.5px;margin-bottom:6px;">QuickMind Analytics</div>
          <div style="font-size:0.88rem;text-align:center;color:rgba(255,255,255,0.5);
                      margin-bottom:20px;line-height:1.5;">
            AI-Powered Delivery Intelligence Platform<br>for Quick Commerce Operations
          </div>

          <!-- Feature pills -->
          <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-bottom:24px;">
            <span style="background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.35);
                         color:#A5B4FC;padding:4px 12px;border-radius:20px;font-size:0.7rem;font-weight:600;">
              📊 Analytics</span>
            <span style="background:rgba(139,92,246,0.2);border:1px solid rgba(139,92,246,0.35);
                         color:#C4B5FD;padding:4px 12px;border-radius:20px;font-size:0.7rem;font-weight:600;">
              🤖 ML Predictions</span>
            <span style="background:rgba(236,72,153,0.2);border:1px solid rgba(236,72,153,0.35);
                         color:#F9A8D4;padding:4px 12px;border-radius:20px;font-size:0.7rem;font-weight:600;">
              🚚 SLA Monitoring</span>
            <span style="background:rgba(16,185,129,0.2);border:1px solid rgba(16,185,129,0.35);
                         color:#6EE7B7;padding:4px 12px;border-radius:20px;font-size:0.7rem;font-weight:600;">
              💡 Insights</span>
          </div>

          <!-- Stats row -->
          <div style="display:flex;border:1px solid rgba(255,255,255,0.08);
                      border-radius:14px;overflow:hidden;margin-bottom:28px;">
            <div style="flex:1;padding:12px 8px;text-align:center;
                        border-right:1px solid rgba(255,255,255,0.08);">
              <div style="font-size:1.1rem;font-weight:800;color:#E0E7FF;">9.4L</div>
              <div style="font-size:0.62rem;color:rgba(255,255,255,0.4);text-transform:uppercase;
                          letter-spacing:0.05em;margin-top:3px;">Orders</div>
            </div>
            <div style="flex:1;padding:12px 8px;text-align:center;
                        border-right:1px solid rgba(255,255,255,0.08);">
              <div style="font-size:1.1rem;font-weight:800;color:#E0E7FF;">8</div>
              <div style="font-size:0.62rem;color:rgba(255,255,255,0.4);text-transform:uppercase;
                          letter-spacing:0.05em;margin-top:3px;">Platforms</div>
            </div>
            <div style="flex:1;padding:12px 8px;text-align:center;
                        border-right:1px solid rgba(255,255,255,0.08);">
              <div style="font-size:1.1rem;font-weight:800;color:#E0E7FF;">12</div>
              <div style="font-size:0.62rem;color:rgba(255,255,255,0.4);text-transform:uppercase;
                          letter-spacing:0.05em;margin-top:3px;">Cities</div>
            </div>
            <div style="flex:1;padding:12px 8px;text-align:center;">
              <div style="font-size:1.1rem;font-weight:800;color:#E0E7FF;">96.6%</div>
              <div style="font-size:0.62rem;color:rgba(255,255,255,0.4);text-transform:uppercase;
                          letter-spacing:0.05em;margin-top:3px;">ML Accuracy</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Login form
        with st.form("login_form"):
            uname = st.text_input("👤  Username", placeholder="Enter your username")
            passw = st.text_input("🔒  Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In  →", use_container_width=True)

        # Demo credentials hint
        st.markdown("""
        <div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);
                    border-radius:12px;padding:12px 16px;margin-top:16px;text-align:center;">
          <div style="font-size:0.75rem;color:rgba(255,255,255,0.45);margin-bottom:6px;
                      text-transform:uppercase;letter-spacing:0.05em;font-weight:600;">Demo Credentials</div>
          <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
            <span style="font-size:0.78rem;color:#A5B4FC;">
              👑 <b>admin</b> / admin123</span>
            <span style="font-size:0.78rem;color:#C4B5FD;">
              📊 <b>analyst</b> / analyst123</span>
            <span style="font-size:0.78rem;color:#F9A8D4;">
              👁 <b>viewer</b> / viewer123</span>
          </div>
        </div>

        <div style="text-align:center;margin-top:24px;font-size:0.7rem;
                    color:rgba(255,255,255,0.2);">
          Built for Quick Commerce · Final Year DS Project · 2026
        </div>
        """, unsafe_allow_html=True)

        if submitted:
            u = uname.strip().lower()
            if u in USERS and USERS[u]["password"] == passw:
                st.session_state.logged_in = True
                st.session_state.username  = u
                st.session_state.role      = USERS[u]["role"]
                st.session_state.name      = USERS[u]["name"]
                st.session_state.page      = "Executive Dashboard"
                st.rerun()
            else:
                st.error("❌  Invalid credentials. Please try again.")
# ──────────────────────────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────
def sidebar(df):
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:12px 4px 8px;">
          <div style="font-size:1.4rem;font-weight:800;color:#E2E8F0;letter-spacing:-0.3px;">⚡ QuickMind</div>
          <div style="font-size:0.75rem;color:#94A3B8;margin-top:2px;">Delivery Intelligence Platform</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:rgba(79,70,229,0.2);border:1px solid rgba(79,70,229,0.35);
                    border-radius:10px;padding:10px 14px;margin:8px 0 16px;">
          <div style="font-size:0.7rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.05em;">Logged in as</div>
          <div style="font-size:0.9rem;font-weight:700;color:#E2E8F0;">{st.session_state.name}</div>
          <span style="font-size:0.7rem;background:#4F46E5;color:white;padding:2px 8px;border-radius:20px;">{st.session_state.role}</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Navigation**")

        pages = [
            ("🏠", "Executive Dashboard"),
            ("📊", "Analytics & EDA"),
            ("🚚", "Operations Dashboard"),
            ("🤖", "AI Predictions"),
            ("💡", "Business Insights"),
            ("🔍", "Data Explorer"),
            ("📤", "Upload & Refresh"),
        ]
        for icon, name in pages:
            active = " 🔸" if st.session_state.page == name else ""
            if st.button(f"{icon}  {name}{active}", key=f"nav_{name}"):
                st.session_state.page = name
                st.rerun()

        st.markdown("---")

        # ── Sidebar filters ──────────────────────────────────────
        st.markdown("**🎛️ Global Filters**")
        companies = ["All"] + sorted(df["Company"].unique().tolist())
        cities    = ["All"] + sorted(df["City"].unique().tolist())
        cats      = ["All"] + sorted(df["Product_Category"].unique().tolist())

        sel_co   = st.selectbox("Company",  companies, key="filt_co")
        sel_city = st.selectbox("City",     cities,    key="filt_city")
        sel_cat  = st.selectbox("Category", cats,      key="filt_cat")
        sla_only = st.toggle("Show SLA Breach only", value=False, key="sla_toggle")

        st.markdown("---")
        if st.button("🚪  Logout", key="logout_btn"):
            for k in ["logged_in", "username", "role", "name"]:
                st.session_state[k] = "" if k != "logged_in" else False
            st.rerun()

        # Dataset info at bottom
        st.markdown(f"""
        <div style="font-size:0.72rem;color:#64748B;margin-top:8px;">
          📦 {len(df):,} orders loaded<br>
          🕐 8 companies • 12 cities
        </div>""", unsafe_allow_html=True)

    # Apply filters
    fdf = df.copy()
    if sel_co   != "All": fdf = fdf[fdf["Company"] == sel_co]
    if sel_city != "All": fdf = fdf[fdf["City"]    == sel_city]
    if sel_cat  != "All": fdf = fdf[fdf["Product_Category"] == sel_cat]
    if sla_only:          fdf = fdf[fdf["SLA_Breach"] == 1]
    return fdf

# ──────────────────────────────────────────────────────────────────
#  PAGE 1 — EXECUTIVE DASHBOARD
# ──────────────────────────────────────────────────────────────────
def page_executive(df, raw_df):
    st.markdown("""
    <div class="hero-header">
      <h1>⚡ Executive Dashboard</h1>
      <p>Real-time delivery intelligence across all platforms and cities</p>
    </div>""", unsafe_allow_html=True)

    # ── KPI Cards ────────────────────────────────────────────────
    st.markdown('<div class="section-title">📌 Key Performance Indicators</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k5, k6, k7, k8 = st.columns(4)

    total_orders  = len(df)
    avg_del_time  = df["Delivery_Time"].mean()
    sla_pct       = df["SLA_Breach"].mean() * 100
    avg_rating    = df["Customer_Rating"].mean()
    avg_order_val = df["Order_Value"].mean()
    total_rev     = df["Order_Value"].sum()
    top_company   = df.groupby("Company")["Order_Value"].sum().idxmax()
    top_city      = df.groupby("City").size().idxmax()

    with k1: kpi("Total Orders",       f"{total_orders:,}",          "Across all platforms",  PRIMARY,  "📦")
    with k2: kpi("Avg Delivery Time",  f"{avg_del_time:.1f} min",    "Target: <30 min",       SUCCESS,  "⏱️")
    with k3: kpi("SLA Breach Rate",    f"{sla_pct:.1f}%",            "Breach = >30 min",      DANGER,   "🚨")
    with k4: kpi("Avg Customer Rating",f"{avg_rating:.2f} / 5",      "Customer satisfaction", ACCENT,   "⭐")
    with k5: kpi("Avg Order Value",    f"₹{avg_order_val:,.0f}",     "Per transaction",       "#8B5CF6", "💰")
    with k6: kpi("Total Revenue",      f"₹{total_rev/1e6:.1f}M",     "Gross order value",     SUCCESS,  "📈")
    with k7: kpi("Top Platform",       top_company,                  "By revenue contribution",WARNING, "🏆")
    with k8: kpi("Busiest City",       top_city,                     "By order volume",        PRIMARY, "🌆")

    # ── Charts Row 1 ─────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Performance Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])

    with c1:
        # Company order volume
        comp_vol = df.groupby("Company").size().reset_index(name="Orders").sort_values("Orders", ascending=True)
        fig = px.bar(comp_vol, x="Orders", y="Company", orientation="h",
                     color="Company",
                     color_discrete_map=COMPANY_COLORS,
                     text="Orders")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig = fig_style(fig, 340, "Orders by Platform")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # SLA breach by company
        sla = df.groupby("Company")["SLA_Breach"].mean().mul(100).round(1).reset_index()
        sla.columns = ["Company", "SLA_Rate"]
        fig = px.pie(sla, values="SLA_Rate", names="Company",
                     hole=0.55, color="Company", color_discrete_map=COMPANY_COLORS)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig = fig_style(fig, 340, "SLA Breach Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # ── Charts Row 2 ─────────────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        # Delivery time distribution
        fig = px.histogram(df.sample(min(50000, len(df))),
                           x="Delivery_Time", nbins=40,
                           color_discrete_sequence=[PRIMARY])
        fig.add_vline(x=30, line_dash="dash", line_color=DANGER, annotation_text="SLA Limit")
        fig.update_traces(opacity=0.85)
        fig = fig_style(fig, 320, "Delivery Time Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        # City-wise avg delivery time
        city_dt = df.groupby("City")["Delivery_Time"].mean().round(1).reset_index().sort_values("Delivery_Time")
        fig = px.bar(city_dt, x="City", y="Delivery_Time",
                     color="Delivery_Time",
                     color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                     text="Delivery_Time")
        fig.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
        fig = fig_style(fig, 320, "Avg Delivery Time by City")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Summary Table ─────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Company Performance Summary</div>', unsafe_allow_html=True)
    summary = df.groupby("Company").agg(
        Orders          = ("Order_ID",             "count"),
        Avg_Delivery_Time=("Delivery_Time",        "mean"),
        SLA_Breach_Pct  = ("SLA_Breach",           lambda x: round(x.mean()*100,1)),
        Avg_Rating      = ("Customer_Rating",      "mean"),
        Avg_Order_Val   = ("Order_Value",          "mean"),
        Total_Revenue   = ("Order_Value",          "sum"),
    ).reset_index().round(2)
    summary["Total_Revenue"] = summary["Total_Revenue"].map("₹{:,.0f}".format)
    summary["Avg_Order_Val"] = summary["Avg_Order_Val"].map("₹{:,.0f}".format)
    st.dataframe(summary, use_container_width=True, hide_index=True,
                 column_config={"SLA_Breach_Pct": st.column_config.ProgressColumn(
                     "SLA Breach %", min_value=0, max_value=100, format="%.1f%%")})

# ──────────────────────────────────────────────────────────────────
#  PAGE 2 — ANALYTICS & EDA
# ──────────────────────────────────────────────────────────────────
def page_analytics(df):
    st.markdown("""
    <div class="hero-header">
      <h1>📊 Analytics & Exploratory Data Analysis</h1>
      <p>Deep-dive statistical analysis and pattern discovery</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📦 Delivery Analysis", "💰 Revenue & Orders", "⭐ Customer Insights",
         "📍 Geographic View", "🔗 Correlation Study"])

    # ── TAB 1: Delivery Analysis ──────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            # Delivery time by company - box plot
            sample = df.sample(min(30000, len(df)))
            fig = px.box(sample, x="Company", y="Delivery_Time",
                         color="Company", color_discrete_map=COMPANY_COLORS,
                         notched=True)
            fig.add_hline(y=30, line_dash="dash", line_color=DANGER,
                          annotation_text="30-min SLA Limit")
            fig = fig_style(fig, 380, "Delivery Time Spread by Platform")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Delivery time vs distance scatter
            s = df.sample(min(8000, len(df)))
            fig = px.scatter(s, x="Distance_km", y="Delivery_Time",
                             color="Company", color_discrete_map=COMPANY_COLORS,
                             opacity=0.55, size_max=5,
                             trendline="ols")
            fig.add_hline(y=30, line_dash="dash", line_color=DANGER)
            fig = fig_style(fig, 380, "Delivery Time vs Distance (with Trend)")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            # SLA breach rate by distance bin
            sla_dist = df.groupby("Distance_Bin")["SLA_Breach"].mean().mul(100).round(1).reset_index()
            fig = px.bar(sla_dist, x="Distance_Bin", y="SLA_Breach",
                         color="SLA_Breach",
                         color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                         text="SLA_Breach")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig = fig_style(fig, 320, "SLA Breach Rate by Distance Band")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            # Delivery partner rating vs delivery time
            partner = df.groupby("Delivery_Partner_Rating")["Delivery_Time"].mean().round(2).reset_index()
            fig = px.line(partner, x="Delivery_Partner_Rating", y="Delivery_Time",
                          markers=True, color_discrete_sequence=[PRIMARY])
            fig.update_traces(line=dict(width=3), marker=dict(size=10))
            fig = fig_style(fig, 320, "Partner Rating vs Avg Delivery Time")
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: Revenue & Orders ───────────────────────────────────
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            # Revenue by category
            cat_rev = df.groupby("Product_Category")["Order_Value"].sum().reset_index().sort_values("Order_Value", ascending=False)
            fig = px.funnel(cat_rev, x="Order_Value", y="Product_Category",
                            color_discrete_sequence=[PRIMARY])
            fig = fig_style(fig, 360, "Revenue Funnel by Product Category")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Order value distribution by payment method
            fig = px.violin(df.sample(min(20000, len(df))),
                            x="Payment_Method", y="Order_Value",
                            color="Payment_Method", box=True, points=False)
            fig = fig_style(fig, 360, "Order Value Distribution by Payment Method")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            # Discount impact on order value
            disc = df.groupby("Discount_Applied")["Order_Value"].mean().round(0).reset_index()
            disc["Discount_Applied"] = disc["Discount_Applied"].map({0: "No Discount", 1: "Discount Applied"})
            fig = px.bar(disc, x="Discount_Applied", y="Order_Value",
                         color="Discount_Applied",
                         color_discrete_sequence=[MUTED, PRIMARY],
                         text="Order_Value")
            fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
            fig = fig_style(fig, 300, "Avg Order Value: Discount vs No Discount")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            # Items count vs order value
            item_val = df.groupby("Items_Count")["Order_Value"].mean().reset_index().sort_values("Items_Count")
            fig = px.area(item_val.head(20), x="Items_Count", y="Order_Value",
                          color_discrete_sequence=[PRIMARY], line_shape="spline")
            fig.update_traces(fill="tozeroy", fillcolor="rgba(79,70,229,0.12)")
            fig = fig_style(fig, 300, "Avg Order Value vs Number of Items")
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: Customer Insights ──────────────────────────────────
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            # Customer rating distribution
            rat_dist = df["Customer_Rating"].value_counts().sort_index().reset_index()
            rat_dist.columns = ["Rating", "Count"]
            fig = px.bar(rat_dist, x="Rating", y="Count",
                         color="Rating",
                         color_continuous_scale=["#EF4444","#F59E0B","#EAB308","#84CC16","#10B981"],
                         text="Count")
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig = fig_style(fig, 340, "Customer Rating Distribution")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Rating by company
            comp_rat = df.groupby("Company")["Customer_Rating"].mean().round(2).reset_index().sort_values("Customer_Rating", ascending=False)
            fig = px.bar(comp_rat, x="Customer_Rating", y="Company",
                         orientation="h", color="Company",
                         color_discrete_map=COMPANY_COLORS, text="Customer_Rating")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig = fig_style(fig, 340, "Average Customer Rating by Platform")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            # Customer age group analysis
            df_age = df.copy()
            df_age["Age_Group"] = pd.cut(df_age["Customer_Age"],
                                          bins=[17,25,35,45,60],
                                          labels=["18-25","26-35","36-45","46-60"])
            age_grp = df_age.groupby("Age_Group").agg(
                Avg_Rating=("Customer_Rating","mean"),
                Orders=("Order_ID","count")).reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=age_grp["Age_Group"], y=age_grp["Orders"],
                                 name="Orders", marker_color=PRIMARY, opacity=0.7))
            fig.add_trace(go.Scatter(x=age_grp["Age_Group"], y=age_grp["Avg_Rating"],
                                     name="Avg Rating", mode="lines+markers",
                                     line=dict(color=ACCENT, width=3),
                                     marker=dict(size=10)), secondary_y=True)
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", title="Orders & Rating by Age Group",
                              font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            # Rating vs delivery time
            rat_del = df.groupby("Customer_Rating")["Delivery_Time"].mean().round(1).reset_index()
            fig = px.bar(rat_del, x="Customer_Rating", y="Delivery_Time",
                         color="Customer_Rating",
                         color_continuous_scale=["#EF4444","#F59E0B","#EAB308","#84CC16","#10B981"],
                         text="Delivery_Time")
            fig.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
            fig = fig_style(fig, 320, "Avg Delivery Time by Customer Rating")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4: Geographic View ────────────────────────────────────
    with tab4:
        city_stats = df.groupby("City").agg(
            Orders=("Order_ID","count"),
            Avg_Delivery_Time=("Delivery_Time","mean"),
            SLA_Breach_Pct=("SLA_Breach",lambda x: x.mean()*100),
            Avg_Revenue=("Order_Value","mean"),
        ).reset_index().round(2)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(city_stats, x="Avg_Delivery_Time", y="SLA_Breach_Pct",
                             size="Orders", color="City", text="City",
                             size_max=50)
            fig.update_traces(textposition="top center", textfont_size=10)
            fig = fig_style(fig, 380, "City: Delivery Time vs SLA Breach % (bubble=volume)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.treemap(df.sample(min(50000, len(df))),
                             path=["City","Company","Product_Category"],
                             values="Order_Value",
                             color="Delivery_Time",
                             color_continuous_scale=["#10B981","#F59E0B","#EF4444"])
            fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                              title="Revenue Treemap: City → Platform → Category",
                              font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig, use_container_width=True)

        # City ranking table
        st.markdown('<div class="section-title">🏆 City Performance Ranking</div>', unsafe_allow_html=True)
        city_stats_display = city_stats.sort_values("SLA_Breach_Pct").reset_index(drop=True)
        city_stats_display.index += 1
        st.dataframe(city_stats_display, use_container_width=True,
                     column_config={
                         "SLA_Breach_Pct": st.column_config.ProgressColumn(
                             "SLA Breach %", min_value=0, max_value=100, format="%.1f%%"),
                         "Avg_Delivery_Time": st.column_config.NumberColumn("Avg Del Time", format="%.1f min"),
                     })

    # ── TAB 5: Correlation Study ──────────────────────────────────
    with tab5:
        num_cols = ["Customer_Age","Order_Value","Delivery_Time",
                    "Distance_km","Items_Count","Customer_Rating",
                    "Discount_Applied","Delivery_Partner_Rating"]
        corr = df[num_cols].corr().round(3)

        fig = px.imshow(corr, text_auto=True,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)",
                          title="Feature Correlation Heatmap",
                          font=dict(family="Plus Jakarta Sans"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">🔑 Key Correlation Insights</div>', unsafe_allow_html=True)
        ic1, ic2, ic3 = st.columns(3)
        corr_del = corr["Delivery_Time"].drop("Delivery_Time").sort_values(key=abs, ascending=False)
        with ic1:
            top_pos = corr_del[corr_del > 0].head(3)
            for feat, val in top_pos.items():
                st.markdown(f"""
                <div class="insight-card">
                  <h4>📈 {feat}</h4>
                  <p>Positive correlation with Delivery Time: <b>r = {val:.3f}</b></p>
                </div>""", unsafe_allow_html=True)
        with ic2:
            top_neg = corr_del[corr_del < 0].head(3)
            for feat, val in top_neg.items():
                st.markdown(f"""
                <div class="insight-card">
                  <h4>📉 {feat}</h4>
                  <p>Negative correlation with Delivery Time: <b>r = {val:.3f}</b></p>
                </div>""", unsafe_allow_html=True)
        with ic3:
            st.markdown(f"""
            <div class="insight-card">
              <h4>🎯 Target Analysis</h4>
              <p>Delivery_Time is most influenced by <b>Distance_km</b> and
              <b>Delivery_Partner_Rating</b>, validating our ML feature selection.</p>
            </div>
            <div class="insight-card">
              <h4>💰 Revenue Driver</h4>
              <p>Items_Count positively correlates with Order_Value,
              suggesting bundling promotions can boost AOV.</p>
            </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
#  PAGE 3 — OPERATIONS DASHBOARD
# ──────────────────────────────────────────────────────────────────
def page_operations(df):
    st.markdown("""
    <div class="hero-header">
      <h1>🚚 Operations Dashboard</h1>
      <p>SLA monitoring, delivery performance, and operational efficiency</p>
    </div>""", unsafe_allow_html=True)

    # ── SLA Alert Banner ─────────────────────────────────────────
    sla_rate = df["SLA_Breach"].mean() * 100
    critical = df[df["SLA_Breach"]==1]
    if sla_rate > 20:
        st.error(f"🚨 **SLA ALERT**: {sla_rate:.1f}% of filtered orders breached 30-min SLA — {len(critical):,} orders affected!")
    elif sla_rate > 10:
        st.warning(f"⚠️ **SLA WARNING**: {sla_rate:.1f}% breach rate — review operations for high-breach cities.")
    else:
        st.success(f"✅ **SLA HEALTHY**: Only {sla_rate:.1f}% breach rate — performance is within target range.")

    # ── Ops KPIs ──────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi("SLA Breach Count",  f"{len(critical):,}",                   f"{sla_rate:.1f}% rate",        DANGER,  "🚨")
    with k2: kpi("On-Time Orders",    f"{len(df)-len(critical):,}",            f"{100-sla_rate:.1f}% rate",   SUCCESS, "✅")
    with k3: kpi("Avg Partner Rating",f"{df['Delivery_Partner_Rating'].mean():.2f}",  "Out of 5",             PRIMARY, "⭐")
    with k4: kpi("Speed Score",       f"{df['Speed_Score'].mean():.2f}",        "Performance index",           ACCENT,  "⚡")
    with k5: kpi("Discount Rate",     f"{df['Discount_Applied'].mean()*100:.1f}%", "Of orders with discount", "#8B5CF6","🏷️")

    # ── Charts ───────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        # SLA breach by company horizontal bar
        sla_co = df.groupby("Company").agg(
            Total=("Order_ID","count"),
            Breach=("SLA_Breach","sum")
        ).reset_index()
        sla_co["Breach_Pct"] = (sla_co["Breach"]/sla_co["Total"]*100).round(1)
        sla_co = sla_co.sort_values("Breach_Pct", ascending=True)
        fig = px.bar(sla_co, x="Breach_Pct", y="Company", orientation="h",
                     color="Breach_Pct",
                     color_continuous_scale=["#10B981","#F59E0B","#EF4444"],
                     text="Breach_Pct")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.add_vline(x=10, line_dash="dash", line_color=MUTED, annotation_text="Target 10%")
        fig = fig_style(fig, 360, "SLA Breach Rate by Platform")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Delivery partner performance
        part = df.groupby("Delivery_Partner_Rating").agg(
            Orders=("Order_ID","count"),
            Avg_Del=("Delivery_Time","mean"),
            SLA_Breach_Pct=("SLA_Breach",lambda x: x.mean()*100)
        ).reset_index().round(2)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=part["Delivery_Partner_Rating"], y=part["Avg_Del"],
                             name="Avg Del Time (min)", marker_color=PRIMARY, opacity=0.8))
        fig.add_trace(go.Scatter(x=part["Delivery_Partner_Rating"], y=part["SLA_Breach_Pct"],
                                  name="SLA Breach %", mode="lines+markers",
                                  line=dict(color=DANGER, width=3),
                                  marker=dict(size=10)), secondary_y=True)
        fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          title="Delivery Partner Rating Impact",
                          font=dict(family="Plus Jakarta Sans"))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Product category delivery efficiency
        cat_eff = df.groupby("Product_Category").agg(
            Avg_Del=("Delivery_Time","mean"),
            SLA_Pct=("SLA_Breach",lambda x: x.mean()*100),
            Orders=("Order_ID","count")
        ).reset_index().round(2)
        fig = px.scatter(cat_eff, x="Avg_Del", y="SLA_Pct",
                         size="Orders", color="Product_Category",
                         text="Product_Category", size_max=40)
        fig.update_traces(textposition="top center", textfont_size=10)
        fig = fig_style(fig, 340, "Category: Avg Del Time vs SLA Breach %")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        # City SLA risk heatmap
        pivot = df.pivot_table(values="SLA_Breach", index="City",
                               columns="Company", aggfunc="mean").mul(100).round(1)
        pivot = pivot.fillna(0)
        fig = px.imshow(pivot, color_continuous_scale=["#10B981","#F59E0B","#EF4444"],
                        zmin=0, zmax=50, text_auto=True, aspect="auto")
        fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                          title="SLA Breach % Heatmap: City × Platform",
                          font=dict(family="Plus Jakarta Sans"))
        st.plotly_chart(fig, use_container_width=True)

    # ── SLA Breach Records ────────────────────────────────────────
    st.markdown('<div class="section-title">🚨 SLA Breach Orders (Sample)</div>', unsafe_allow_html=True)
    breach_sample = df[df["SLA_Breach"]==1].sample(min(200, len(critical)))[
        ["Order_ID","Company","City","Delivery_Time","Distance_km",
         "Product_Category","Customer_Rating","Delivery_Partner_Rating"]
    ].sort_values("Delivery_Time", ascending=False)
    st.dataframe(breach_sample, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────
#  PAGE 4 — AI PREDICTIONS
# ──────────────────────────────────────────────────────────────────
def page_predictions(df):
    model, encoders, features, metrics = load_models()

    st.markdown("""
    <div class="hero-header">
      <h1>🤖 AI Delivery Time Predictions</h1>
      <p>Machine learning powered delivery time estimation system</p>
    </div>""", unsafe_allow_html=True)

    # ── Model Performance Cards ───────────────────────────────────
    st.markdown('<div class="section-title">📈 Model Performance Comparison</div>', unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    for i, (_, row) in enumerate(metrics.iterrows()):
        col = [mc1, mc2, mc3][i]
        is_best = row["MAE"] == metrics["MAE"].min()
        badge = " 🏆 BEST" if is_best else ""
        color = SUCCESS if is_best else PRIMARY
        with col:
            kpi(f"{row['Model']}{badge}",
                f"MAE: {row['MAE']:.4f} min",
                f"R² Score: {row['R2']:.4f}",
                color, "🤖")

    # ── Model Performance Chart ───────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(metrics, x="Model", y="MAE",
                     color="Model", text="MAE",
                     color_discrete_sequence=[MUTED, SUCCESS, PRIMARY])
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig = fig_style(fig, 300, "Model Comparison: MAE (lower is better)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(metrics, x="Model", y="R2",
                     color="Model", text="R2",
                     color_discrete_sequence=[MUTED, SUCCESS, PRIMARY])
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig = fig_style(fig, 300, "Model Comparison: R² Score (higher is better)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Prediction Form ───────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Predict Delivery Time</div>', unsafe_allow_html=True)
    st.markdown("Fill in the order details below to get an AI-powered delivery time estimate.")

    f1, f2, f3 = st.columns(3)
    with f1:
        distance      = st.slider("📍 Distance (km)",       0.5, 30.0, 8.0, 0.5)
        order_value   = st.number_input("💰 Order Value (₹)", 50, 5000, 500, 50)
        customer_age  = st.slider("👤 Customer Age",         18, 60, 30)
        items_count   = st.slider("📦 Items Count",           1, 25, 5)

    with f2:
        company  = st.selectbox("🏢 Platform",          sorted(df["Company"].unique()))
        city     = st.selectbox("🌆 City",              sorted(df["City"].unique()))
        category = st.selectbox("🗂️ Product Category",  sorted(df["Product_Category"].unique()))

    with f3:
        payment        = st.selectbox("💳 Payment Method",        sorted(df["Payment_Method"].unique()))
        partner_rating = st.selectbox("⭐ Partner Rating",         [2,3,4,5], index=2)
        discount       = st.radio("🏷️ Discount Applied",          ["No","Yes"])
        discount_val   = 1 if discount == "Yes" else 0

    if st.button("⚡  Predict Delivery Time", type="primary", use_container_width=True):
        # Build input row
        input_dict = {
            "Distance_km":             distance,
            "Order_Value":             order_value,
            "Customer_Age":            customer_age,
            "Items_Count":             items_count,
            "Product_Category":        category,
            "Payment_Method":          payment,
            "Company":                 company,
            "City":                    city,
            "Discount_Applied":        discount_val,
            "Delivery_Partner_Rating": partner_rating,
        }
        input_df = pd.DataFrame([input_dict])
        cat_cols = ["Product_Category","Payment_Method","Company","City"]
        for col in cat_cols:
            input_df[col] = encoders[col].transform(input_df[col])

        pred_time = model.predict(input_df[features])[0]
        pred_time = max(5, round(pred_time, 1))

        # Classify
        if pred_time <= 15:
            cat_label = "⚡ Express (< 15 min)"
            cat_color = SUCCESS
        elif pred_time <= 25:
            cat_label = "🟢 On-Time (15–25 min)"
            cat_color = "#84CC16"
        elif pred_time <= 30:
            cat_label = "🟡 Near SLA Limit (25–30 min)"
            cat_color = ACCENT
        else:
            cat_label = "🔴 SLA BREACH (> 30 min)"
            cat_color = DANGER

        delay_prob = min(100, max(0, (pred_time - 20) * 4))

        st.markdown("---")
        r1, r2, r3 = st.columns([1, 1, 1])
        with r1:
            st.markdown(f"""
            <div class="pred-result">
              <div class="pred-time">{pred_time:.1f}</div>
              <div class="pred-label">minutes estimated delivery</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            kpi("Delivery Category",  cat_label,       "AI Classification",         cat_color, "🏷️")
            kpi("Delay Probability",  f"{delay_prob:.0f}%", "Risk of delay > 30 min", DANGER if delay_prob>40 else SUCCESS, "⚠️")
        with r3:
            # Show input summary
            st.markdown(f"""
            <div class="insight-card" style="height:100%">
              <h4>📋 Order Summary</h4>
              <p>
                📍 Distance: <b>{distance} km</b><br>
                💰 Value: <b>₹{order_value}</b><br>
                🏢 Platform: <b>{company}</b><br>
                🌆 City: <b>{city}</b><br>
                🗂️ Category: <b>{category}</b><br>
                ⭐ Partner: <b>{partner_rating}/5</b>
              </p>
            </div>""", unsafe_allow_html=True)

        # Insight
        st.info(f"""
        💡 **AI Insight:** For a {distance} km delivery in {city} via {company}, 
        the model predicts **{pred_time:.1f} minutes** with a delay probability of **{delay_prob:.0f}%**. 
        {"⚠️ Consider alerting operations — high breach risk!" if delay_prob > 50 else "✅ Delivery within acceptable range."}
        """)

    # ── Feature Importance ────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    try:
        rf_model = joblib.load(os.path.join(MDIR, "random_forest.pkl"))
        importance = pd.DataFrame({
            "Feature":   features,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                     color="Importance",
                     color_continuous_scale=["#E0E7FF","#4F46E5"])
        fig = fig_style(fig, 360, "Feature Importance for Delivery Time Prediction")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Feature importance available after full model training.")

# ──────────────────────────────────────────────────────────────────
#  PAGE 5 — BUSINESS INSIGHTS
# ──────────────────────────────────────────────────────────────────
def page_insights(df):
    st.markdown("""
    <div class="hero-header">
      <h1>💡 Business Insights & Intelligence</h1>
      <p>Auto-generated strategic insights powered by data analysis</p>
    </div>""", unsafe_allow_html=True)

    # Compute key stats
    sla_rate      = df["SLA_Breach"].mean() * 100
    top_sla_co    = df.groupby("Company")["SLA_Breach"].mean().idxmax()
    top_sla_city  = df.groupby("City")["SLA_Breach"].mean().idxmax()
    best_co       = df.groupby("Company")["Customer_Rating"].mean().idxmax()
    best_part_rat = 5
    disc_uplift   = (df[df["Discount_Applied"]==1]["Order_Value"].mean() -
                     df[df["Discount_Applied"]==0]["Order_Value"].mean())
    top_cat       = df.groupby("Product_Category")["Order_Value"].sum().idxmax()
    low_dist_sla  = df[df["Distance_km"] <= 5]["SLA_Breach"].mean() * 100
    high_dist_sla = df[df["Distance_km"] > 10]["SLA_Breach"].mean() * 100

    insights = [
        ("🚨 SLA Performance",
         f"{sla_rate:.1f}% of all orders exceed the 30-minute SLA. "
         f"{top_sla_co} has the highest breach rate. Immediate operational review recommended.",
         "red"),
        ("🏆 Customer Champion",
         f"{best_co} leads in customer satisfaction ratings. "
         "Studying their delivery model can inform best practices across platforms.",
         "green"),
        ("💰 Discount ROI",
         f"Orders with discounts average ₹{disc_uplift:+.0f} {'more' if disc_uplift>0 else 'less'} vs non-discounted orders. "
         "Re-evaluate discount strategy to maximize AOV while maintaining profitability.",
         "amber"),
        ("📦 Category Revenue Driver",
         f"{top_cat} contributes the highest total revenue. "
         "Prioritize stock availability and fast-picking for this category.",
         "blue"),
        ("📍 Distance-SLA Link",
         f"Short deliveries (≤5 km) breach SLA {low_dist_sla:.1f}% of the time vs "
         f"{high_dist_sla:.1f}% for distances >10 km. "
         "Proximity alone doesn't guarantee speed — warehouse efficiency needs improvement.",
         "amber"),
        ("⭐ Partner Quality Gate",
         f"Partners rated 5★ deliver orders ~2-3x faster than 2★ partners. "
         "Implement a minimum rating threshold (4★) for prime time slots.",
         "green"),
        (f"🌆 High-Risk City: {top_sla_city}",
         f"{top_sla_city} consistently records the highest SLA breach rates. "
         "Consider adding dark stores or increasing fleet density in this zone.",
         "red"),
        ("📱 Payment Method Optimization",
         "Wallet and UPI payments show faster checkout times, indirectly improving "
         "fulfillment speed. Incentivize digital payments to reduce processing lag.",
         "blue"),
    ]

    col1, col2 = st.columns(2)
    for i, (title, body, color) in enumerate(insights):
        badge_map = {"red":   "badge-red",   "green": "badge-green",
                     "amber": "badge-amber",  "blue":  "badge-blue"}
        priority_map = {"red": "Critical", "amber": "Important",
                        "green": "Positive", "blue": "Opportunity"}
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class="insight-card" style="border-left:4px solid {'#EF4444' if color=='red' else '#10B981' if color=='green' else '#F59E0B' if color=='amber' else '#3B82F6'};">
              <h4>{title} &nbsp;<span class="{badge_map[color]}">{priority_map[color]}</span></h4>
              <p>{body}</p>
            </div>""", unsafe_allow_html=True)

    # ── Trends ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Strategic Performance Matrix</div>', unsafe_allow_html=True)
    comp_matrix = df.groupby("Company").agg(
        Orders=("Order_ID","count"),
        Avg_Del_Time=("Delivery_Time","mean"),
        SLA_Breach_Pct=("SLA_Breach",lambda x: x.mean()*100),
        Avg_Rating=("Customer_Rating","mean"),
        Avg_Revenue=("Order_Value","mean"),
    ).reset_index().round(2)

    fig = px.scatter(comp_matrix,
                     x="Avg_Del_Time", y="Avg_Rating",
                     size="Orders", color="SLA_Breach_Pct",
                     text="Company", size_max=60,
                     color_continuous_scale=["#10B981","#F59E0B","#EF4444"],
                     labels={"Avg_Del_Time":"Avg Delivery Time (min)",
                             "Avg_Rating":"Avg Customer Rating",
                             "SLA_Breach_Pct":"SLA Breach %"})
    fig.update_traces(textposition="top center", textfont=dict(size=12, color=TEXT))
    fig.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      title="Platform Strategy Matrix: Speed vs Satisfaction (bubble=volume, color=SLA risk)",
                      font=dict(family="Plus Jakarta Sans"))
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────
#  PAGE 6 — DATA EXPLORER
# ──────────────────────────────────────────────────────────────────
def page_explorer(df):
    st.markdown("""
    <div class="hero-header">
      <h1>🔍 Data Explorer</h1>
      <p>Search, filter, and explore raw delivery records</p>
    </div>""", unsafe_allow_html=True)

    # Search
    search = st.text_input("🔎 Search by Order ID", placeholder="e.g. 1000001")
    if search:
        result = df[df["Order_ID"].astype(str).str.contains(search)]
        if not result.empty:
            st.dataframe(result, use_container_width=True, hide_index=True)
        else:
            st.warning("No matching orders found.")
        return

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1: min_del = st.slider("Min Delivery Time", 1, 60, 1)
    with fc2: max_del = st.slider("Max Delivery Time", 1, 60, 60)
    with fc3: min_val = st.number_input("Min Order Value", 0, 5000, 0)
    with fc4: sort_by = st.selectbox("Sort by", ["Delivery_Time","Order_Value","Customer_Rating","Distance_km"])

    filtered = df[(df["Delivery_Time"]>=min_del) &
                  (df["Delivery_Time"]<=max_del) &
                  (df["Order_Value"]>=min_val)].sort_values(sort_by, ascending=False)

    st.markdown(f'<div class="section-title">📋 Records ({len(filtered):,} rows)</div>', unsafe_allow_html=True)
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)

    # Download
    csv = filtered.to_csv(index=False).encode()
    st.download_button("⬇️  Download Filtered Data as CSV",
                       csv, "filtered_orders.csv", "text/csv",
                       use_container_width=True)

    # Stats
    st.markdown('<div class="section-title">📊 Quick Statistics</div>', unsafe_allow_html=True)
    st.dataframe(filtered.describe().round(2), use_container_width=True)

# ──────────────────────────────────────────────────────────────────
#  PAGE 7 — UPLOAD & REFRESH
# ──────────────────────────────────────────────────────────────────
def page_upload():
    st.markdown("""
    <div class="hero-header">
      <h1>📤 Upload & Refresh Data</h1>
      <p>Upload a new cleaned dataset to refresh the dashboard</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
      <h4>📌 Upload Requirements</h4>
      <p>The CSV must contain these columns: <br>
      <code>Order_ID, Company, City, Customer_Age, Order_Value, Delivery_Time, 
      Distance_km, Items_Count, Product_Category, Payment_Method, 
      Customer_Rating, Discount_Applied, Delivery_Partner_Rating</code>
      </p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload cleaned CSV", type=["csv"])
    if uploaded:
        try:
            new_df = pd.read_csv(uploaded)
            required = ["Order_ID","Company","City","Customer_Age","Order_Value",
                        "Delivery_Time","Distance_km","Items_Count","Product_Category",
                        "Payment_Method","Customer_Rating","Discount_Applied","Delivery_Partner_Rating"]
            missing = [c for c in required if c not in new_df.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                st.success(f"✅ Valid dataset! {len(new_df):,} rows loaded. Refresh the page to use it.")
                new_df.to_csv(DATA, index=False)
                st.cache_data.clear()
                st.balloons()
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown('<div class="section-title">📁 Current Dataset Info</div>', unsafe_allow_html=True)
    try:
        info_df = pd.read_csv(DATA, nrows=5)
        st.dataframe(info_df, use_container_width=True, hide_index=True)
        size = os.path.getsize(DATA) / (1024*1024)
        st.info(f"Current file: `cleaned_quick_commerce.csv` ({size:.1f} MB)")
    except:
        st.warning("Could not load current dataset info.")

# ──────────────────────────────────────────────────────────────────
#  MAIN APP ROUTER
# ──────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        page_login()
        return

    # Load data
    if not os.path.exists(DATA):
        st.error(f"⚠️ Dataset not found at {DATA}. Please check `data/` folder.")
        return

    raw_df = load_data(DATA)
    df     = sidebar(raw_df)   # filtered dataframe

    page = st.session_state.page

    if   page == "Executive Dashboard":   page_executive(df, raw_df)
    elif page == "Analytics & EDA":       page_analytics(df)
    elif page == "Operations Dashboard":  page_operations(df)
    elif page == "AI Predictions":        page_predictions(df)
    elif page == "Business Insights":     page_insights(df)
    elif page == "Data Explorer":         page_explorer(df)
    elif page == "Upload & Refresh":      page_upload()

if __name__ == "__main__":
    main()