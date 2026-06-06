# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   H2 MAROC — DASHBOARD ULTRA-PRO v3                                        ║
║   Digital Twin + ML vs PyPSA + Front de Pareto + Waterfall LCOH           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="H2 Maroc Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API = "http://localhost:5000"

OUTPUT_ML = os.path.join(os.path.expanduser("~"), "Downloads",
                          "H2Morocco222_Outputs", "ml")

# ─────────────────────────────────────────────────────────────────────────────
# THEME CSS — Dark Tech avec accents verts
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500&display=swap');

  :root {
    --green:   #00E676;
    --blue:    #40C4FF;
    --orange:  #FFAB40;
    --red:     #FF5252;
    --bg:      #0D1117;
    --card:    #161B22;
    --border:  #21262D;
    --text:    #E6EDF3;
    --muted:   #8B949E;
  }

  html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
  }

  h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important; letter-spacing: 0.05em; }

  /* HEADER HERO */
  .hero {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 40%, #0D2137 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--green);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,230,118,0.06) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-title {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--green) !important;
    margin: 0;
    line-height: 1.1;
  }
  .hero-sub {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace !important;
  }

  /* METRIC CARDS */
  .metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: var(--green); }
  .metric-val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--green);
    line-height: 1;
  }
  .metric-val.blue  { color: var(--blue);   }
  .metric-val.orange{ color: var(--orange); }
  .metric-val.red   { color: var(--red);    }
  .metric-lbl {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }
  .metric-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    margin-top: 4px;
  }
  .delta-pos { color: #FF5252; }
  .delta-neg { color: #00E676; }

  /* SECTION TITLES */
  .section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text);
    border-left: 3px solid var(--green);
    padding-left: 12px;
    margin: 20px 0 12px 0;
  }

  /* COMPARISON BOX */
  .compare-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px;
  }
  .compare-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 12px;
  }
  .ml-color  { color: var(--blue);   }
  .psa-color { color: var(--orange); }

  /* SIDEBAR */
  [data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] label { color: var(--muted) !important; font-size: 0.8rem; }

  /* BUTTONS */
  .stButton > button {
    background: linear-gradient(135deg, #00E676, #00BCD4) !important;
    color: #0D1117 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    letter-spacing: 0.05em !important;
    transition: opacity 0.2s !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  /* TABS */
  [data-baseweb="tab-list"] { border-bottom: 1px solid var(--border) !important; }
  [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    background: transparent !important;
  }
  [aria-selected="true"] {
    color: var(--green) !important;
    border-bottom: 2px solid var(--green) !important;
  }

  /* BADGE */
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
  }
  .badge-ml   { background: rgba(64,196,255,0.15); color: var(--blue);   border: 1px solid rgba(64,196,255,0.3); }
  .badge-psa  { background: rgba(255,171,64,0.15); color: var(--orange); border: 1px solid rgba(255,171,64,0.3); }
  .badge-ok   { background: rgba(0,230,118,0.15);  color: var(--green);  border: 1px solid rgba(0,230,118,0.3); }
  .badge-warn { background: rgba(255,82,82,0.15);  color: var(--red);    border: 1px solid rgba(255,82,82,0.3); }

  /* ALERTS */
  .stAlert { border-radius: 8px !important; }

  /* MONO */
  .mono { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

  /* Hide default streamlit elements */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#0D1117',
    plot_bgcolor='#161B22',
    font=dict(family='Inter', color='#E6EDF3', size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=['#00E676','#40C4FF','#FFAB40','#FF5252','#B39DDB','#80DEEA'],
    legend=dict(bgcolor='#161B22', bordercolor='#21262D', borderwidth=1,
                font=dict(size=10)),
    xaxis=dict(gridcolor='#21262D', linecolor='#21262D', zeroline=False),
    yaxis=dict(gridcolor='#21262D', linecolor='#21262D', zeroline=False),
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS API
# ─────────────────────────────────────────────────────────────────────────────
#@st.cache_data(ttl=30)
def api_health():
    try:
        r = requests.get(f"{API}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def api_predict(ville, pv, eol, elec, bat, tech):
    try:
        r = requests.post(f"{API}/predict", json={
            "ville": ville, "PV_MW": pv, "EOL_MW": eol,
            "ELEC_MW": elec, "BAT_MWH": bat, "technologie": tech
        }, timeout=300)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        return {'erreur': str(e)}

def api_predict_comparison(ville, pv, eol, elec, bat, tech):
    try:
        r = requests.post(f"{API}/predict_comparison", json={
            "ville": ville, "PV_MW": pv, "EOL_MW": eol,
            "ELEC_MW": elec, "BAT_MWH": bat, "technologie": tech
        }, timeout=180)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        return {'erreur': str(e)}

# @st.cache_data(ttl=60)  # DÉSACTIVÉ
def api_toutes_villes(pv, eol, elec, bat, tech):
    try:
        r = requests.get(f"{API}/toutes_villes",
                         params={"PV_MW": pv, "EOL_MW": eol, "ELEC_MW": elec,
                                 "BAT_MWH": bat, "technologie": tech}, timeout=180)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        return None

# @st.cache_data(ttl=120)  # DÉSACTIVÉ
def api_pareto(ville, tech, n=300):
    try:
        r = requests.get(f"{API}/pareto",
                         params={"ville": ville, "technologie": tech, "n": n}, timeout=300)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

# @st.cache_data(ttl=60)  # DÉSACTIVÉ
def api_waterfall(ville, pv, eol, elec, bat, tech):
    try:
        r = requests.get(f"{API}/waterfall/{ville}",
                         params={"PV_MW": pv, "EOL_MW": eol, "ELEC_MW": elec,
                                 "BAT_MWH": bat, "technologie": tech}, timeout=300)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def load_validation_report():
    """Charge le rapport de validation ML depuis JSON."""
    path = os.path.join(OUTPUT_ML, "validation_report.json")
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Rajdhani,sans-serif;font-size:1.3rem;font-weight:700;color:#00E676;margin-bottom:16px;">⚡ PARAMÈTRES</div>', unsafe_allow_html=True)

    # Status API
    health = api_health()
    if health:
        ml_status = health.get('modele_ml', 'N/A')
        badge_col = '#00E676' if 'chargé' in ml_status else '#FFAB40'
        st.markdown(f"""
        <div style="background:#161B22;border:1px solid #21262D;border-radius:8px;padding:10px 14px;margin-bottom:16px;">
          <div style="font-size:0.7rem;color:#8B949E;margin-bottom:6px;">STATUT API</div>
          <div style="font-size:0.78rem;">
            <span style="color:#00E676;">●</span> API <span style="color:#00E676;font-weight:600;">online</span><br>
            <span style="color:{badge_col};">●</span> ML <span style="color:{badge_col};font-weight:600;">{ml_status}</span><br>
            <span style="color:#40C4FF;">●</span> PyPSA <span style="color:#40C4FF;font-weight:600;">simulé</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ API hors ligne — lancez api_flask_h2.py")

    st.markdown("---")

    VILLES_LISTE = [
        "Dakhla", "Laayoune", "Boujdour", "Ouarzazate", "Midelt",
        "Agadir", "Guelmim", "Marrakech", "Jorf_Lasfar",
        "Casablanca", "Tanger", "Nador"
    ]
    ville = st.selectbox("🌍 Ville", VILLES_LISTE)
    techno = st.selectbox("⚙️ Technologie", ["PEM", "AEL"])

    st.markdown('<div style="font-size:0.72rem;color:#8B949E;text-transform:uppercase;letter-spacing:.08em;margin:12px 0 6px 0;">CAPACITÉS</div>', unsafe_allow_html=True)
    pv   = st.slider("☀️ PV (MW)",           0, 500, 150, step=10)
    eol  = st.slider("🌬️ Éolien (MW)",       0, 300, 80,  step=10)
    elec = st.slider("⚡ Électrolyseur (MW)", 10, 200, 60,  step=5)
    bat  = st.slider("🔋 Batterie (MWh)",     0, 500, 100, step=10)

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;color:#8B949E;font-family:JetBrains Mono,monospace;">H2 Maroc Intelligence Platform<br>ML + PyPSA + Optimisation</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-title"> H2 MAROC — INTELLIGENCE PLATFORM</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS PRINCIPAUX
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Analyse & Prédiction",
    " Comparaison Villes",
    " ML vs PyPSA",
    " Front de Pareto",
    " Validation ML",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSE & PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run = st.button("🚀 LANCER L'ANALYSE")

    if run:
        with st.spinner("Calcul en cours (ML + PyPSA)..."):
            data = api_predict_comparison(ville, pv, eol, elec, bat, techno)
            # AFFICHER LA VALEUR BRUTE DE L'API DANS LE DASHBOARD
            st.write(f"**DEBUG - API retourne :** {data.get('LCOH_USD_kg') if data else 'None'} $/kg")
        if not data or 'erreur' in data:
            msg = data.get('erreur', 'inconnue') if data else 'API non joignable — vérifiez que api_flask_h2.py tourne sur le port 5000'
            st.error(f"Erreur API : {msg}")
            if data and 'trace' in data:
                with st.expander("Détail erreur serveur"):
                    st.code(data['trace'])
        else:
            lcoh  = data.get('LCOH_USD_kg', 0) or 0
            # FORCER la mise à jour depuis l'API
            if 'ml' in data and 'LCOH_USD_kg' in data['ml']:
                lcoh = data['ml']['LCOH_USD_kg']
            fiab  = data.get('fiabilite_pct', 0) or 0
            score = data.get('score_composite', 0) or 0
            ml_d  = data.get('ml', {}) or {}
            psa_d = data.get('pypsa', {}) or {}
            comp  = data.get('comparaison', {}) or {}
            res   = data.get('ressources', {}) or {}

            # ── KPI row ──────────────────────────────────────────────────────
            k1, k2, k3, k4, k5 = st.columns(5)
            kpis = [
                (k1, f"{lcoh:.3f}", "$/kg", "LCOH FINAL", ""),
                (k2, f"{fiab:.1f}%", "", "FIABILITÉ", "blue"),
                (k3, f"{score:.1f}", "", "SCORE COMPOSITE", "orange"),
                (k4, f"{psa_d.get('H2_prod_kt_an', 0) or 0:.3f}", "kt/an", "PRODUCTION H₂", ""),
                (k5, f"{psa_d.get('taux_curtailment_pct', 0) or 0:.1f}%", "", "CURTAILMENT", "red"),
            ]
            for col, val, unit, lbl, clr in kpis:
                with col:
                    clr_class = f' {clr}' if clr else ''
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-val{clr_class}">{val}</div>
                      <div class="metric-lbl" style="color:#8B949E;font-size:.6rem">{unit}</div>
                      <div class="metric-lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

                        # ── Comparaison ML vs PyPSA ───────────────────────────────────────
            st.markdown('<div class="section-title">🤖 Comparaison ML vs PyPSA Simplifié</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)

            with c1:
                # FORCER les valeurs depuis data (pas depuis ml_d)
                lcoh_ml   = data.get('LCOH_USD_kg', 'N/A')  # ← CHANGÉ : utilise data directement
                fiab_ml   = data.get('fiabilite_pct', 'N/A')  # ← CHANGÉ
                dispo_ml  = True
                st.markdown(f"""
                <div class="compare-box">
                  <div class="compare-title ml-color">🤖 Modèle ML <span class="badge badge-ml">RandomForest</span></div>
                  <div style="margin:8px 0">
                    <span class="mono" style="color:#40C4FF;font-size:1.4rem;font-weight:600">{lcoh_ml if lcoh_ml != 'N/A' else '--'}</span>
                    <span style="color:#8B949E;font-size:.75rem"> $/kg</span>
                  </div>
                  <div style="font-size:.8rem;color:#8B949E">Fiabilité : <span style="color:#40C4FF">{fiab_ml}%</span></div>
                  <div style="margin-top:8px">
                    <span class="badge badge-ok">✓ chargé</span>
                  </div>
                  <div style="font-size:.7rem;color:#8B949E;margin-top:6px">Prédiction instantanée<br>(&lt;1ms)</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                lcoh_psa  = data.get('pypsa', {}).get('LCOH_USD_kg', 'N/A') if data.get('pypsa') else 'N/A'
                fiab_psa  = data.get('pypsa', {}).get('fiabilite_pct', 'N/A') if data.get('pypsa') else 'N/A'
                h2_psa    = data.get('pypsa', {}).get('H2_prod_kg_an', 0) if data.get('pypsa') else 0
                st.markdown(f"""
                <div class="compare-box">
                  <div class="compare-title psa-color">⚙️ PyPSA Simplifié <span class="badge badge-psa">Bilan 8760h</span></div>
                  <div style="margin:8px 0">
                    <span class="mono" style="color:#FFAB40;font-size:1.4rem;font-weight:600">{lcoh_psa if lcoh_psa != 'N/A' else '--'}</span>
                    <span style="color:#8B949E;font-size:.75rem"> $/kg</span>
                  </div>
                  <div style="font-size:.8rem;color:#8B949E">Fiabilité : <span style="color:#FFAB40">{fiab_psa}%</span></div>
                  <div style="font-size:.8rem;color:#8B949E;margin-top:4px">H₂ : {h2_psa:,.0f} kg/an</div>
                  <div style="margin-top:8px"><span class="badge badge-ok">✓ simulation</span></div>
                  <div style="font-size:.7rem;color:#8B949E;margin-top:6px">Bilan horaire complet<br>(CF réels + batterie)</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                delta_lcoh = comp.get('delta_LCOH_pct', None)
                delta_fiab = comp.get('delta_fiabilite_pp', None)
                accord     = comp.get('accord', True)
                delta_lcoh_str = f"{delta_lcoh:+.1f}%" if delta_lcoh is not None else "N/A"
                delta_fiab_str = f"{delta_fiab:+.1f} pp" if delta_fiab is not None else "N/A"
                delta_class = 'badge-ok' if accord else 'badge-warn'
                # Couleur à 3 niveaux : vert <10%, orange 10-20%, rouge >20%
                lcoh_color = '#00E676' if (delta_lcoh is None or abs(delta_lcoh) <= 10) else \
                             '#FFA726' if abs(delta_lcoh) <= 20 else '#FF5252'
                # Note contextuelle si divergence
                ratio_bat_elec = bat / max(elec, 1) if elec > 0 else 0
                if not accord and delta_lcoh is not None:
                    if ratio_bat_elec > 4 or (eol > 60 and elec < 80):
                        note = "Cas limite : éolien fort ou batterie large → PyPSA détecte la saturation, le ML moyenne statistiquement."
                    else:
                        note = "PyPSA (simulation physique) est la référence. Le ML approche la valeur réelle."
                else:
                    note = ""
                st.markdown(f"""
                <div class="compare-box">
                  <div class="compare-title" style="color:#00E676">📊 Concordance</div>
                  <div style="margin:8px 0">
                    <div style="font-size:.8rem;color:#8B949E;margin-bottom:4px">Écart LCOH</div>
                    <span class="mono" style="color:{lcoh_color};font-size:1.2rem;font-weight:600">{delta_lcoh_str}</span>
                  </div>
                  <div style="font-size:.8rem;color:#8B949E">Écart fiabilité : <span style="color:#E6EDF3">{delta_fiab_str}</span></div>
                  <div style="margin-top:10px">
                    <span class="badge {delta_class}">{'✓ Accord' if accord else '⚠ Divergence'}</span>
                  </div>
                  <div style="font-size:.7rem;color:#8B949E;margin-top:6px">Seuil accord : &lt;20% d'écart relatif LCOH</div>
                  {f'<div style="font-size:.65rem;color:#FFA726;margin-top:5px;line-height:1.3">{note}</div>' if note else ''}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Waterfall LCOH + Sensibilité PV ─────────────────────────────
            col_wf, col_sens = st.columns(2)

            with col_wf:
                st.markdown('<div class="section-title">💧 Décomposition LCOH (Waterfall)</div>', unsafe_allow_html=True)
                wf = api_waterfall(ville, pv, eol, elec, bat, techno)
                if wf and 'decomposition' in wf:
                    items = wf['decomposition']
                    labels = [it['label'] for it in items] + ['LCOH Total']
                    values = [it['value'] for it in items]
                    colors = []
                    for it in items:
                        t = it['type']
                        colors.append('#40C4FF' if t == 'CAPEX' else '#FFAB40' if t == 'OPEX' else '#00E676')
                    colors.append('#00E676')

                    measure = ['relative'] * len(items) + ['total']
                    fig_wf = go.Figure(go.Waterfall(
                        name="LCOH", measure=measure,
                        x=labels, y=values + [None],
                        connector=dict(line=dict(color='#21262D', width=1)),
                        decreasing=dict(marker=dict(color='#00E676')),
                        increasing=dict(marker=dict(color='#FF5252')),
                        totals=dict(marker=dict(color='#40C4FF')),
                        texttemplate='%{y:.3f}',
                        textposition='outside',
                        textfont=dict(size=9, color='#E6EDF3'),
                    ))
                    fig_wf.update_layout(
                        title=f'Décomposition LCOH — {ville} ({techno})',
                        yaxis_title='$/kg H₂', height=320,
                        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != 'colorway'}
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)

            with col_sens:
                st.markdown('<div class="section-title">📈 Sensibilité PV — ML vs PyPSA</div>', unsafe_allow_html=True)
                sens = data.get('sensibilite_PV', [])
                if sens:
                    df_sens = pd.DataFrame(sens)
                    fig_sens = go.Figure()
                    fig_sens.add_trace(go.Scatter(
                        x=df_sens['PV_MW'], y=df_sens['LCOH_ml'],
                        name='ML', mode='lines+markers',
                        line=dict(color='#40C4FF', width=2),
                        marker=dict(size=6, color='#40C4FF'),
                    ))
                    fig_sens.add_trace(go.Scatter(
                        x=df_sens['PV_MW'], y=df_sens['LCOH_pypsa'],
                        name='PyPSA', mode='lines+markers',
                        line=dict(color='#FFAB40', width=2, dash='dot'),
                        marker=dict(size=6, color='#FFAB40'),
                    ))
                    fig_sens.add_vline(x=pv, line_dash='dash', line_color='#00E676',
                                       annotation_text=f'Config actuelle ({pv} MW)')
                    fig_sens.update_layout(
                        title=f'LCOH vs Capacité PV — {ville}',
                        xaxis_title='PV (MW)', yaxis_title='LCOH ($/kg)',
                        height=320, **{k: v for k, v in PLOTLY_LAYOUT.items() if k != 'colorway'}
                    )
                    st.plotly_chart(fig_sens, use_container_width=True)

            # ── Radar profil énergétique ──────────────────────────────────────
            st.markdown('<div class="section-title">🕸️ Profil Énergétique</div>', unsafe_allow_html=True)

            c_rad, c_info = st.columns([1, 1])
            with c_rad:
                fig_r = go.Figure()
                cf_pv  = res.get('CF_PV_pct', 0)
                cf_eol = res.get('CF_EOL_pct', 0)
                ghi    = res.get('GHI_kWh_m2', 0)
                v_vent = res.get('v_vent_moy', 0)

                fig_r.add_trace(go.Scatterpolar(
                    r=[pv/5, eol/3, elec/2, bat/5, cf_pv*5, cf_eol*5],
                    theta=['PV (MW)', 'Éolien (MW)', 'Électrolyseur (MW)',
                           'Batterie (MWh)', 'CF PV (%×5)', 'CF Éol (%×5)'],
                    fill='toself', name='Configuration',
                    line=dict(color='#00E676'), fillcolor='rgba(0,230,118,0.1)',
                ))
                fig_r.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, gridcolor='#21262D', color='#8B949E'),
                        angularaxis=dict(gridcolor='#21262D', color='#8B949E'),
                        bgcolor='#161B22',
                    ),
                    showlegend=False, height=300,
                    **{k: v for k, v in PLOTLY_LAYOUT.items()
                       if k not in ['xaxis', 'yaxis', 'colorway']}
                )
                st.plotly_chart(fig_r, use_container_width=True)

            with c_info:
                st.markdown(f"""
                <div class="compare-box" style="margin-top:8px">
                  <div style="font-family:Rajdhani,sans-serif;font-weight:600;margin-bottom:12px">
                    📍 Ressources — {ville}
                  </div>
                  <table style="width:100%;font-size:.82rem;border-collapse:collapse">
                    <tr><td style="color:#8B949E;padding:4px 0">CF Solaire</td>
                        <td style="color:#FFAB40;font-weight:600;text-align:right">{cf_pv}%</td></tr>
                    <tr><td style="color:#8B949E;padding:4px 0">CF Éolien</td>
                        <td style="color:#40C4FF;font-weight:600;text-align:right">{cf_eol}%</td></tr>
                    <tr><td style="color:#8B949E;padding:4px 0">GHI</td>
                        <td style="color:#E6EDF3;text-align:right">{ghi} kWh/m²</td></tr>
                    <tr><td style="color:#8B949E;padding:4px 0">Vent moyen</td>
                        <td style="color:#E6EDF3;text-align:right">{v_vent} m/s</td></tr>
                    <tr><td style="color:#8B949E;padding:4px 0">Technologie</td>
                        <td style="color:#00E676;font-weight:600;text-align:right">{techno}</td></tr>
                    <tr><td style="color:#8B949E;padding:4px 0">Batterie</td>
                        <td style="color:#E6EDF3;text-align:right">{'Oui — ' + str(bat) + ' MWh' if bat > 0 else 'Non'}</td></tr>
                    <tr><td style="color:#8B949E;padding:4px 0">Heures prod.</td>
                        <td style="color:#E6EDF3;text-align:right">{psa_d.get('heures_production', 'N/A')} h/an</td></tr>
                  </table>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARAISON VILLES
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🗺️ Analyse Comparative — 12 Villes Marocaines</div>', unsafe_allow_html=True)

    if st.button("📊 Analyser toutes les villes", key="btn_villes"):
        with st.spinner("Analyse des 12 villes en cours..."):
            result = api_toutes_villes(pv, eol, elec, bat, techno)

        if not result or 'erreur' in (result or {}):
            msg2 = result.get('erreur','timeout ou connexion refusée') if result else 'API non joignable'
            st.error(f"Erreur API toutes_villes : {msg2}")
            if result and 'trace' in result:
                with st.expander("Détail"):
                    st.code(result['trace'])
        else:
            rows = result.get('resultats', [])
            df_v = pd.DataFrame([{
                'Ville':      r['ville'],
                'Rang':       r.get('rang', 0),
                'LCOH_ML':    r.get('ml', {}).get('LCOH_USD_kg') if r.get('ml') else None,
                'LCOH_PyPSA': r.get('pypsa', {}).get('LCOH_USD_kg') if r.get('pypsa') else None,
                'LCOH_Final': r.get('LCOH_USD_kg'),
                'Fiabilité':  r.get('fiabilite_pct'),
                'Score':      r.get('score_composite'),
                'CF_PV':      r.get('ressources', {}).get('CF_PV_pct'),
                'CF_EOL':     r.get('ressources', {}).get('CF_EOL_pct'),
                'H2_kt_an':   r.get('pypsa', {}).get('H2_prod_kt_an') if r.get('pypsa') else None,
                'Curtailment':r.get('pypsa', {}).get('taux_curtailment_pct') if r.get('pypsa') else None,
                'Catégorie':  r.get('categorie_LCOH'),
                'Lat': r.get('latitude'), 'Lon': r.get('longitude'),
            } for r in rows])

            # ── KPIs top ─────────────────────────────────────────────────────
            best = df_v.sort_values('LCOH_Final').iloc[0]
            kc1, kc2, kc3, kc4 = st.columns(4)
            with kc1:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-val">{best['Ville']}</div>
                  <div class="metric-lbl">🏆 Meilleure ville</div>
                </div>""", unsafe_allow_html=True)
            with kc2:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-val">{best['LCOH_Final']:.3f}</div>
                  <div class="metric-lbl">LCOH MIN ($/kg)</div>
                </div>""", unsafe_allow_html=True)
            with kc3:
                worst = df_v.sort_values('LCOH_Final').iloc[-1]
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-val red">{worst['LCOH_Final']:.3f}</div>
                  <div class="metric-lbl">LCOH MAX ($/kg)</div>
                </div>""", unsafe_allow_html=True)
            with kc4:
                ex_villes = len(df_v[df_v['LCOH_Final'] < 7])
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-val orange">{ex_villes}</div>
                  <div class="metric-lbl">VILLES &lt;7 $/kg</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Graphique 1 : LCOH ML vs PyPSA grouped bar ───────────────────
            fig_comp = go.Figure()
            df_sorted = df_v.sort_values('LCOH_Final')
            fig_comp.add_trace(go.Bar(
                name='ML (RandomForest)',
                x=df_sorted['Ville'], y=df_sorted['LCOH_ML'],
                marker_color='#40C4FF', opacity=0.85,
            ))
            fig_comp.add_trace(go.Bar(
                name='PyPSA (Simulation)',
                x=df_sorted['Ville'], y=df_sorted['LCOH_PyPSA'],
                marker_color='#FFAB40', opacity=0.85,
            ))
            fig_comp.add_hline(y=7, line_dash='dot', line_color='#00E676',
                               annotation_text='Cible 7 $/kg (2030)',
                               annotation_font_color='#00E676')
            fig_comp.update_layout(
                title=f'LCOH par Ville — ML vs PyPSA | {techno} | PV={pv}MW Éol={eol}MW',
                xaxis_title='Ville', yaxis_title='LCOH ($/kg)',
                barmode='group', height=380, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # ── Graphique 2 : Scatter LCOH vs Fiabilité (bubble) ─────────────
            col_sc, col_cf = st.columns(2)
            with col_sc:
                fig_sc = px.scatter(
                    df_v, x='LCOH_Final', y='Fiabilité',
                    size='Score', color='Score',
                    text='Ville', hover_data=['CF_PV', 'CF_EOL', 'H2_kt_an'],
                    color_continuous_scale=['#FF5252', '#FFAB40', '#00E676'],
                    size_max=40,
                    title='LCOH vs Fiabilité — Espace décisionnel',
                )
                fig_sc.update_traces(textposition='top center', textfont_size=8)
                fig_sc.update_layout(height=360, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_sc, use_container_width=True)

            with col_cf:
                # Heatmap CF PV vs CF EOL
                fig_cf = go.Figure()
                colors_cf = ['#FF5252' if l > 10 else '#FFAB40' if l > 7 else
                             '#00E676' if l < 5 else '#40C4FF'
                             for l in df_v['LCOH_Final'].fillna(99)]
                fig_cf.add_trace(go.Scatter(
                    x=df_v['CF_PV'], y=df_v['CF_EOL'],
                    mode='markers+text',
                    marker=dict(size=14, color=colors_cf, line=dict(width=1, color='#0D1117')),
                    text=df_v['Ville'],
                    textposition='top center',
                    textfont=dict(size=8, color='#E6EDF3'),
                    customdata=df_v[['LCOH_Final']].values,
                    hovertemplate='%{text}<br>CF_PV=%{x}%<br>CF_EOL=%{y}%<br>LCOH=%{customdata[0]:.2f} $/kg',
                ))
                fig_cf.update_layout(
                    title='Ressources Solaire vs Éolien (couleur = LCOH)',
                    xaxis_title='CF Solaire (%)',
                    yaxis_title='CF Éolien (%)',
                    height=360, **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_cf, use_container_width=True)

            # ── Carte Maroc ───────────────────────────────────────────────────
            st.markdown('<div class="section-title">🗺️ Carte des Sites — LCOH</div>', unsafe_allow_html=True)
            df_map = df_v.dropna(subset=['Lat', 'Lon', 'LCOH_Final'])
            fig_map = px.scatter_mapbox(
                df_map, lat='Lat', lon='Lon',
                size='Score', color='LCOH_Final',
                color_continuous_scale=['#00E676', '#FFAB40', '#FF5252'],
                hover_name='Ville',
                hover_data={'LCOH_Final': ':.3f', 'Fiabilité': ':.1f',
                            'H2_kt_an': ':.3f', 'Lat': False, 'Lon': False},
                text='Ville',
                zoom=4.5, center=dict(lat=29, lon=-8),
                mapbox_style='carto-darkmatter',
                size_max=30,
                title='Maroc — LCOH par site ($/kg)',
            )
            fig_map.update_layout(height=420, **{k: v for k, v in PLOTLY_LAYOUT.items()
                                                  if k not in ['xaxis', 'yaxis', 'colorway']})
            st.plotly_chart(fig_map, use_container_width=True)

            # ── Tableau synthèse ──────────────────────────────────────────────
            st.markdown('<div class="section-title">📋 Tableau Synthèse</div>', unsafe_allow_html=True)
            df_display = df_v[['Rang', 'Ville', 'LCOH_ML', 'LCOH_PyPSA', 'LCOH_Final',
                                'Fiabilité', 'H2_kt_an', 'CF_PV', 'CF_EOL', 'Catégorie']].copy()
            df_display = df_display.sort_values('Rang')
            for col in ['LCOH_ML', 'LCOH_PyPSA', 'LCOH_Final']:
                df_display[col] = df_display[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
            df_display['Fiabilité'] = df_display['Fiabilité'].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
            df_display['H2_kt_an'] = df_display['H2_kt_an'].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
            st.dataframe(df_display.set_index('Rang'), use_container_width=True,
                         height=380)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML vs PyPSA (ANALYSE DÉTAILLÉE)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🔬 Analyse Détaillée ML vs PyPSA</div>', unsafe_allow_html=True)

    st.info("Cette section compare les prédictions du modèle RandomForest entraîné sur les simulations PyPSA, et la simulation physique directe.")

    if st.button("🔍 Analyser ML vs PyPSA", key="btn_mlvspsa"):
        with st.spinner("Calcul comparatif..."):
            data = api_predict_comparison(ville, pv, eol, elec, bat, techno)

        if not data or 'erreur' in (data or {}):
            msg3 = data.get('erreur','inconnue') if data else 'API non joignable'
            st.error(f"Erreur API ML vs PyPSA : {msg3}")
            if data and 'trace' in data:
                with st.expander("Détail"):
                    st.code(data['trace'])
        elif data:
            ml_d  = data.get('ml', {}) or {}
            psa_d = data.get('pypsa', {}) or {}
            comp  = data.get('comparaison', {}) or {}
            sens  = data.get('sensibilite_PV', [])

            # ── Métriques concordance ─────────────────────────────────────────
            mc1, mc2, mc3 = st.columns(3)
            delta_l = comp.get('delta_LCOH_pct', 0) or 0
            delta_f = comp.get('delta_fiabilite_pp', 0) or 0
            accord  = comp.get('accord', True)

            with mc1:
                color = '#FF5252' if abs(delta_l) > 20 else '#FFAB40' if abs(delta_l) > 10 else '#00E676'
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-val" style="color:{color}">{delta_l:+.1f}%</div>
                  <div class="metric-lbl">ÉCART LCOH ML/PyPSA</div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                color2 = '#FF5252' if abs(delta_f) > 10 else '#00E676'
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-val blue">{delta_f:+.1f} pp</div>
                  <div class="metric-lbl">ÉCART FIABILITÉ</div>
                </div>""", unsafe_allow_html=True)
            with mc3:
                ratio_bat = bat / max(elec, 1) if elec > 0 else 0
                note_mc3 = ""
                if not accord:
                    if ratio_bat > 4 or (eol > 60 and elec < 80):
                        note_mc3 = "Cas limite : saturation physique non captée par le RF"
                    else:
                        note_mc3 = "PyPSA est la référence physique"
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-val {'orange' if not accord else ''}">{'✓ Accord' if accord else '⚠ Divergence'}</div>
                  <div class="metric-lbl">STATUT CONCORDANCE</div>
                  <div style="font-size:.65rem;color:#FFA726;margin-top:4px;line-height:1.3">{note_mc3}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Décomposition LCOH côte à côte ────────────────────────────────
            if psa_d.get('decomposition_LCOH'):
                st.markdown('<div class="section-title">💧 Décomposition LCOH PyPSA</div>', unsafe_allow_html=True)
                decomp = psa_d['decomposition_LCOH']
                labels = [k.replace('_', ' ').upper() for k in decomp.keys()]
                values = list(decomp.values())
                fig_pie = make_subplots(rows=1, cols=2,
                                        specs=[[{"type": "pie"}, {"type": "bar"}]])
                fig_pie.add_trace(go.Pie(
                    labels=labels, values=values, name='Décomposition',
                    hole=0.45, textfont_size=9,
                    marker=dict(colors=['#40C4FF','#40C4FF','#40C4FF','#40C4FF',
                                        '#FFAB40','#FFAB40','#FFAB40','#FFAB40','#00E676'],
                                line=dict(color='#0D1117', width=2)),
                ), row=1, col=1)
                # Gauge LCOH
                lcoh_val = psa_d.get('LCOH_USD_kg') or 0
                fig_pie.add_trace(go.Bar(
                    x=labels, y=values,
                    marker_color=['#40C4FF']*4 + ['#FFAB40']*4 + ['#00E676'],
                    text=[f'{v:.4f}' for v in values],
                    textposition='outside', textfont_size=8,
                ), row=1, col=2)
                fig_pie.update_layout(height=350, title=f'Décomposition LCOH PyPSA — {ville} ({techno})',
                                      **PLOTLY_LAYOUT)
                st.plotly_chart(fig_pie, use_container_width=True)

            # ── Sensibilité PV ────────────────────────────────────────────────
            if sens:
                st.markdown('<div class="section-title">📈 Analyse de Sensibilité — Capacité PV</div>', unsafe_allow_html=True)
                df_s = pd.DataFrame(sens)
                fig_s = make_subplots(rows=1, cols=2,
                                      subplot_titles=('LCOH vs PV', 'Fiabilité vs PV'))
                fig_s.add_trace(go.Scatter(x=df_s['PV_MW'], y=df_s['LCOH_ml'],
                                           name='LCOH ML', line=dict(color='#40C4FF', width=2),
                                           mode='lines+markers'), row=1, col=1)
                fig_s.add_trace(go.Scatter(x=df_s['PV_MW'], y=df_s['LCOH_pypsa'],
                                           name='LCOH PyPSA', line=dict(color='#FFAB40', width=2, dash='dot'),
                                           mode='lines+markers'), row=1, col=1)
                fig_s.add_trace(go.Scatter(x=df_s['PV_MW'], y=df_s['fiab_ml'],
                                           name='Fiab ML', line=dict(color='#40C4FF', width=2),
                                           mode='lines+markers', showlegend=False), row=1, col=2)
                fig_s.add_trace(go.Scatter(x=df_s['PV_MW'], y=df_s['fiab_pypsa'],
                                           name='Fiab PyPSA', line=dict(color='#FFAB40', width=2, dash='dot'),
                                           mode='lines+markers', showlegend=False), row=1, col=2)
                for col_idx in [1, 2]:
                    fig_s.add_vline(x=pv, line_dash='dash', line_color='#00E676',
                                    row=1, col=col_idx)
                fig_s.update_layout(height=380,
                                     title=f'Sensibilité à la capacité PV — {ville}',
                                     **PLOTLY_LAYOUT)
                st.plotly_chart(fig_s, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FRONT DE PARETO
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">⚙️ Optimisation Multi-Objectifs — Front de Pareto</div>', unsafe_allow_html=True)
    st.info("Explore l'espace LCOH / Fiabilité : chaque point est une configuration testée. Le front de Pareto (non-dominé) représente les meilleures solutions possibles.")

    col_p1, col_p2, col_p3 = st.columns([1,1,1])
    with col_p1:
        ville_p = st.selectbox("Ville", list(["Dakhla","Laayoune","Ouarzazate","Boujdour",
                                               "Midelt","Agadir"]), key="ville_pareto")
    with col_p2:
        tech_p  = st.selectbox("Technologie", ["PEM","AEL"], key="tech_pareto")
    with col_p3:
        n_p = st.slider("Nb simulations", 100, 500, 200, step=50)

    if st.button("⚡ Calculer le front de Pareto", key="btn_pareto"):
        with st.spinner(f"Calcul {n_p} simulations..."):
            pareto_data = api_pareto(ville_p, tech_p, n_p)

        if not pareto_data:
            st.error("Erreur API Pareto")
        else:
            tous   = pareto_data.get('tous_points', [])
            front  = pareto_data.get('front_pareto', [])
            source = front[0].get('source', 'ML') if front else 'ML'

            df_tous  = pd.DataFrame(tous)
            df_front = pd.DataFrame(front)

            # KPIs
            kp1, kp2, kp3 = st.columns(3)
            with kp1:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-val">{len(front)}</div>
                  <div class="metric-lbl">POINTS PARETO</div>
                </div>""", unsafe_allow_html=True)
            with kp2:
                lcoh_min = df_front['LCOH'].min() if not df_front.empty else 0
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-val">{lcoh_min:.3f}</div>
                  <div class="metric-lbl">LCOH MIN PARETO ($/kg)</div>
                </div>""", unsafe_allow_html=True)
            with kp3:
                fiab_max = df_front['fiabilite'].max() if not df_front.empty else 0
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-val blue">{fiab_max:.1f}%</div>
                  <div class="metric-lbl">FIABILITÉ MAX PARETO</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Scatter Pareto ────────────────────────────────────────────────
            fig_par = go.Figure()

            # Tous les points (fond)
            if not df_tous.empty:
                fig_par.add_trace(go.Scatter(
                    x=df_tous['LCOH'], y=df_tous['fiabilite'],
                    mode='markers',
                    marker=dict(size=4, color='#21262D', opacity=0.6),
                    name='Toutes configs',
                    hovertemplate='LCOH=%{x:.3f}<br>Fiab=%{y:.1f}%<extra></extra>',
                ))

            # Front de Pareto
            if not df_front.empty:
                fig_par.add_trace(go.Scatter(
                    x=df_front['LCOH'], y=df_front['fiabilite'],
                    mode='markers+lines',
                    marker=dict(size=10, color='#00E676',
                                line=dict(width=2, color='#0D1117')),
                    line=dict(color='#00E676', width=1.5),
                    name='Front de Pareto',
                    customdata=df_front[['PV_MW','EOL_MW','ELEC_MW','BAT_MWH']].values,
                    hovertemplate=(
                        'LCOH=%{x:.3f} $/kg<br>Fiab=%{y:.1f}%<br>'
                        'PV=%{customdata[0]}MW<br>Éol=%{customdata[1]}MW<br>'
                        'Élec=%{customdata[2]}MW<br>Bat=%{customdata[3]}MWh'
                        '<extra>Pareto</extra>'
                    ),
                ))

            # Zone objectif
            fig_par.add_vrect(x0=0, x1=7, fillcolor='rgba(0,230,118,0.04)',
                               line_width=0, annotation_text='Cible LCOH <7$',
                               annotation_position='top left',
                               annotation_font_color='#00E676')
            fig_par.add_hline(y=80, line_dash='dot', line_color='#40C4FF',
                               annotation_text='Fiabilité 80%',
                               annotation_font_color='#40C4FF')

            fig_par.update_layout(
                title=f'Front de Pareto — {ville_p} | {tech_p} | {source}',
                xaxis_title='LCOH ($/kg) ← minimiser',
                yaxis_title='Fiabilité (%) ↑ maximiser',
                height=480, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_par, use_container_width=True)

            if not df_front.empty:
                # ── Tableau solutions Pareto ──────────────────────────────────
                st.markdown('<div class="section-title">🏆 Solutions Pareto (non-dominées)</div>', unsafe_allow_html=True)
                df_front_disp = df_front[['LCOH','fiabilite','PV_MW','EOL_MW',
                                           'ELEC_MW','BAT_MWH']].copy()
                df_front_disp.columns = ['LCOH ($/kg)','Fiabilité (%)','PV (MW)',
                                          'Éolien (MW)','Élec (MW)','Batterie (MWh)']
                df_front_disp = df_front_disp.sort_values('LCOH ($/kg)')
                for c in df_front_disp.columns:
                    if df_front_disp[c].dtype == float:
                        df_front_disp[c] = df_front_disp[c].round(2)
                st.dataframe(df_front_disp, use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — VALIDATION ML
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">🔬 Rapport de Validation du Modèle ML</div>', unsafe_allow_html=True)

    report = load_validation_report()

    if report is None:
        st.warning("Rapport de validation non trouvé. Lancez `train_ml_h2.py` pour générer `validation_report.json`.")
        st.code(f"# Chemin attendu :\n{os.path.join(OUTPUT_ML, 'validation_report.json')}")
    else:
        # ── KPIs ML ──────────────────────────────────────────────────────────
        vk1, vk2, vk3, vk4, vk5 = st.columns(5)
        r2_l = report.get('r2_lcoh', 0)
        mae  = report.get('mae_lcoh', 0)
        r2_f = report.get('r2_fiab', 0)
        cv   = report.get('cv_mean', 0)
        cv_s = report.get('cv_std', 0)

        for col, val, lbl, clr in [
            (vk1, f"{r2_l:.4f}", "R² LCOH (test)", '' if r2_l > 0.95 else 'orange'),
            (vk2, f"{mae:.4f}", "MAE LCOH ($/kg)", ''),
            (vk3, f"{r2_f:.4f}", "R² Fiabilité", '' if r2_f > 0.95 else 'orange'),
            (vk4, f"{cv:.4f}", "CV R² (5-folds)", ''),
            (vk5, f"±{cv_s:.4f}", "CV Std", ''),
        ]:
            with col:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-val {clr}">{val}</div>
                  <div class="metric-lbl">{lbl}</div>
                  <div class="metric-delta">{'✅ Excellent' if (r2_l>0.95 and lbl.startswith('R²')) else ''}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            # Prédictions vs Réalité
            pred_s = report.get('pred_lcoh_sample', [])
            real_s = report.get('real_lcoh_sample', [])
            if pred_s and real_s:
                fig_v1 = go.Figure()
                fig_v1.add_trace(go.Scatter(
                    x=real_s, y=pred_s, mode='markers',
                    marker=dict(size=4, color='#00E676', opacity=0.5),
                    name='Prédictions',
                ))
                lims = [min(min(real_s), min(pred_s)), max(max(real_s), max(pred_s))]
                fig_v1.add_trace(go.Scatter(
                    x=lims, y=lims, mode='lines',
                    line=dict(color='#FF5252', dash='dash', width=1.5),
                    name='Parfait (y=x)',
                ))
                fig_v1.update_layout(
                    title=f'Prédictions vs Réalité LCOH\nR²={r2_l:.4f} | MAE={mae:.4f}',
                    xaxis_title='LCOH PyPSA ($/kg)', yaxis_title='LCOH ML ($/kg)',
                    height=380, **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_v1, use_container_width=True)

        with col_right:
            # Distribution résidus
            residus = report.get('residus_lcoh', [])
            if residus:
                fig_v2 = go.Figure()
                fig_v2.add_trace(go.Histogram(
                    x=residus, nbinsx=40,
                    marker_color='#00E676', opacity=0.75,
                    name='Résidus',
                ))
                fig_v2.add_vline(x=0, line_dash='dash', line_color='#FF5252',
                                  annotation_text='Erreur=0')
                mean_res = np.mean(residus)
                fig_v2.add_vline(x=mean_res, line_dash='dot', line_color='#FFAB40',
                                  annotation_text=f'Moy={mean_res:.3f}')
                fig_v2.update_layout(
                    title='Distribution des Erreurs (ML − PyPSA)',
                    xaxis_title='Erreur ($/kg)', yaxis_title='Fréquence',
                    height=380, **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_v2, use_container_width=True)

        # ── Importance features ───────────────────────────────────────────────
        col_imp, col_cv = st.columns(2)

        with col_imp:
            importance = report.get('importance', {})
            if importance:
                imp_sorted = sorted(importance.items(), key=lambda x: x[1])
                fig_imp = go.Figure(go.Bar(
                    x=[v for _, v in imp_sorted],
                    y=[k for k, _ in imp_sorted],
                    orientation='h',
                    marker=dict(
                        color=[v for _, v in imp_sorted],
                        colorscale=[[0,'#21262D'],[0.5,'#40C4FF'],[1,'#00E676']],
                    ),
                    text=[f'{v*100:.1f}%' for _, v in imp_sorted],
                    textposition='outside',
                    textfont=dict(size=9, color='#E6EDF3'),
                ))
                fig_imp.update_layout(
                    title='Importance des Features (LCOH)',
                    xaxis_title='Importance', height=320,
                    **{k: v for k, v in PLOTLY_LAYOUT.items() if k != 'colorway'}
                )
                fig_imp.update_xaxes(tickformat='.0%')
                st.plotly_chart(fig_imp, use_container_width=True)

        with col_cv:
            cv_folds = report.get('cv_folds', [])
            if cv_folds:
                colors_cv = ['#00E676' if s > 0.95 else '#FFAB40' if s > 0.85 else '#FF5252'
                             for s in cv_folds]
                fig_cv = go.Figure(go.Bar(
                    x=[f'Fold {i+1}' for i in range(len(cv_folds))],
                    y=cv_folds,
                    marker_color=colors_cv,
                    text=[f'{s:.4f}' for s in cv_folds],
                    textposition='outside', textfont=dict(size=9, color='#E6EDF3'),
                ))
                fig_cv.add_hline(y=np.mean(cv_folds), line_dash='dash', line_color='#FF5252',
                                  annotation_text=f'Moy={np.mean(cv_folds):.4f}')
                fig_cv.add_hline(y=0.95, line_dash='dot', line_color='#00E676',
                                  annotation_text='Seuil excellent')
                fig_cv.update_layout(
                    title=f'Cross-Validation 5-Folds | R²={cv:.4f}±{cv_s:.4f}',
                    yaxis_title='R² Score', yaxis_range=[0.7, 1.02],
                    height=320, **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_cv, use_container_width=True)

        # ── Concordance par ville ─────────────────────────────────────────────
        conc = report.get('concordance_villes', [])
        if conc:
            st.markdown('<div class="section-title">🌍 Concordance ML vs PyPSA par Ville</div>', unsafe_allow_html=True)
            df_c = pd.DataFrame(conc)
            fig_conc = make_subplots(specs=[[{"secondary_y": True}]])
            fig_conc.add_trace(go.Bar(
                x=df_c['ville'], y=df_c['mae'],
                name='MAE ($/kg)', marker_color='#40C4FF', opacity=0.8,
            ), secondary_y=False)
            fig_conc.add_trace(go.Scatter(
                x=df_c['ville'], y=df_c['r2'],
                name='R²', mode='lines+markers',
                line=dict(color='#00E676', width=2),
                marker=dict(size=7),
            ), secondary_y=True)
            fig_conc.add_hline(y=0.95, line_dash='dot', line_color='#00E676',
                                secondary_y=True, opacity=0.5)
            fig_conc.update_yaxes(title_text="MAE LCOH ($/kg)", secondary_y=False,
                                   gridcolor='#21262D', color='#40C4FF')
            fig_conc.update_yaxes(title_text="R²", secondary_y=True,
                                   range=[0, 1.1], color='#00E676')
            fig_conc.update_xaxes(tickangle=30)
            fig_conc.update_layout(
                title='Concordance ML vs PyPSA — Performance par Ville',
                height=380, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_conc, use_container_width=True)

        # ── Footer avec infos modèle ──────────────────────────────────────────
        n_train = report.get('n_train', 'N/A')
        n_test  = report.get('n_test', 'N/A')
        feats   = report.get('features', [])
        st.markdown(f"""
        <div class="compare-box" style="margin-top:16px">
          <div style="font-family:JetBrains Mono,monospace;font-size:.75rem;color:#8B949E">
            <b style="color:#E6EDF3">Modèle :</b> RandomForestRegressor (300 arbres, min_samples_leaf=2)<br>
            <b style="color:#E6EDF3">Dataset :</b> {n_train} points train | {n_test} points test | {len(feats)} features<br>
            <b style="color:#E6EDF3">Features :</b> {', '.join(feats)}<br>
            <b style="color:#E6EDF3">Algorithme target :</b> PyPSA 8760h (simulation simplifiée + full PyPSA si disponible)
          </div>
        </div>
        """, unsafe_allow_html=True)
