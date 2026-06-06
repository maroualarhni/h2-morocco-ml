# -*- coding: utf-8 -*-
"""
API FLASK H2 MAROC  — Simulation vectorisée (numpy, sans boucle Python)
  - import logging ajouté (NameError corrigé sur /predict_comparison et /toutes_villes)
  - logger global 'h2api' initialisé une seule fois
  - Batterie accélérée via Numba @njit (×50-200 sur la boucle horaire)
  - Vérification explicite None pour delta_lcoh dans _construire_resultat
  - _predire_ml calcule ratio_bat_elec dynamiquement (compatible modèles v2 et v3 ML)
"""

import os, sys, json, logging
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, Response

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — initialisé une seule fois ici
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('h2api')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    from flask_cors import CORS
    CORS_OK = True
except ImportError:
    CORS_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CHEMIN_MODEL = os.path.join(
    os.path.expanduser("~"),
    "Downloads", "H2Morocco222_Outputs", "ml", "models_h2.pkl"
)

# Aligné avec ETAPE2.py → CALIB dict
CALIB_VILLES = {
    'Agadir':      {'CF_PV': 0.192, 'CF_EOL': 0.164, 'GHI': 2050, 'v_mean': 5.3,  'lat': 30.4,  'lon': -9.6},
    'Boujdour':    {'CF_PV': 0.200, 'CF_EOL': 0.384, 'GHI': 2160, 'v_mean': 9.1,  'lat': 26.1,  'lon': -14.5},
    'Casablanca':  {'CF_PV': 0.171, 'CF_EOL': 0.105, 'GHI': 1870, 'v_mean': 4.3,  'lat': 33.6,  'lon': -7.6},
    'Dakhla':      {'CF_PV': 0.197, 'CF_EOL': 0.415, 'GHI': 2180, 'v_mean': 9.8,  'lat': 23.7,  'lon': -15.9},
    'Guelmim':     {'CF_PV': 0.192, 'CF_EOL': 0.164, 'GHI': 2050, 'v_mean': 5.3,  'lat': 28.9,  'lon': -10.1},
    'Jorf_Lasfar': {'CF_PV': 0.173, 'CF_EOL': 0.128, 'GHI': 1900, 'v_mean': 4.8,  'lat': 32.7,  'lon': -8.6},
    'Laayoune':    {'CF_PV': 0.199, 'CF_EOL': 0.337, 'GHI': 2175, 'v_mean': 8.2,  'lat': 27.1,  'lon': -13.2},
    'Marrakech':   {'CF_PV': 0.190, 'CF_EOL': 0.060, 'GHI': 2080, 'v_mean': 3.8,  'lat': 31.6,  'lon': -8.0},
    'Midelt':      {'CF_PV': 0.201, 'CF_EOL': 0.150, 'GHI': 2200, 'v_mean': 5.0,  'lat': 32.7,  'lon': -4.7},
    'Nador':       {'CF_PV': 0.163, 'CF_EOL': 0.180, 'GHI': 1780, 'v_mean': 5.5,  'lat': 35.2,  'lon': -2.9},
    'Ouarzazate':  {'CF_PV': 0.198, 'CF_EOL': 0.225, 'GHI': 2172, 'v_mean': 5.8,  'lat': 30.9,  'lon': -6.9},
    'Tanger':      {'CF_PV': 0.168, 'CF_EOL': 0.156, 'GHI': 1850, 'v_mean': 5.1,  'lat': 35.8,  'lon': -5.8},
}

# Aligné avec ETAPE2.py → PARAMS_TECHNO
PARAMS_TECHNO = {
    'CAPEX_PV': 550,   'OPEX_PV': 12,   'LT_PV': 25,
    'CAPEX_EOL': 1100, 'OPEX_EOL': 35,  'LT_EOL': 20,
    'CAPEX_PEM': 900,  'OPEX_PEM': 0.03,'EFF_PEM': 55, 'LT_PEM': 20,
    'CAPEX_AEL': 650,  'OPEX_AEL': 0.02,'EFF_AEL': 52, 'LT_AEL': 25,
    'CAPEX_BAT': 150,  'OPEX_BAT': 0.01,'LT_BAT': 15,
    'EFF_BAT_CHG': 0.92, 'EFF_BAT_DCH': 0.92,
    'WATER_CONS': 21.1, 'WATER_COST': 0.72,
    'DR': 0.08,
    'MINLOAD_PEM': 0.10, 'MINLOAD_AEL': 0.20,
}

# ─────────────────────────────────────────────────────────────────────────────
# NUMBA — accélération boucle batterie (optionnel, fallback Python pur si absent)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from numba import njit as _njit
    NUMBA_OK = True
    logger.info("Numba disponible — boucle batterie JIT activée")
except ImportError:
    def _njit(fn):
        return fn
    NUMBA_OK = False
    logger.warning("Numba non installé — boucle batterie en Python pur (~200-500ms/appel avec batterie)")


@_njit
def _boucle_batterie_jit(P_dispo, P_max, P_min, BAT_kWh, eff, eta_c, eta_d):
    """
    Boucle horaire batterie compilée JIT (Numba) ou Python pur (fallback).
    Retourne (H2_kg, E_fournie_MWh, E_curtail_MWh, heures_ok).
    """
    SOC_min = BAT_kWh * 0.10
    SOC_max = BAT_kWh * 0.90
    soc = SOC_max * 0.5
    H2_kg = 0.0
    E_fournie = 0.0
    E_curtail = 0.0
    heures_ok = 0.0

    for i in range(len(P_dispo)):
        p = P_dispo[i]
        if p > P_max:
            surplus = p - P_max
            if soc < SOC_max:
                charge = min(surplus * eta_c, SOC_max - soc)
                soc += charge
                surplus -= charge / eta_c
            E_curtail += surplus / 1e3
            p = P_max
        elif p < P_min and soc > SOC_min:
            discharge = min((P_min - p) / eta_d, soc - SOC_min)
            soc -= discharge
            p += discharge * eta_d

        if p >= P_min:
            p_elec = min(p, P_max)
            H2_kg     += p_elec / eff
            E_fournie += p_elec / 1e3
            heures_ok += 1.0

    return H2_kg, E_fournie, E_curtail, heures_ok


# ─────────────────────────────────────────────────────────────────────────────
# PROFILS HORAIRES PRÉ-CALCULÉS (mis en cache au démarrage)
# ─────────────────────────────────────────────────────────────────────────────
# On génère 1 fois les profils CF_PV et CF_EOL pour chaque ville
# et on les garde en mémoire → plus de re-génération à chaque appel

_PROFILS_CACHE = {}

def _generer_profils(ville):
    """Génère les profils CF_PV(8760) et CF_EOL(8760) pour une ville (vectorisé)."""
    if ville in _PROFILS_CACHE:
        return _PROFILS_CACHE[ville]

    cal = CALIB_VILLES.get(ville, CALIB_VILLES['Dakhla'])
    CF_PV_cible  = cal['CF_PV']
    CF_EOL_cible = cal['CF_EOL']
    v_mean       = cal['v_mean']

    rng = np.random.default_rng(42)
    t   = np.arange(8760)

    # ── Profil PV vectorisé ──────────────────────────────────────────────────
    heure_j = t % 24
    jour_an = t // 24
    angle   = np.pi * (heure_j - 6) / 12
    angle   = np.clip(angle, 0, np.pi)
    saison  = 1 + 0.3 * np.cos(2 * np.pi * (jour_an - 172) / 365)
    pv_raw  = np.sin(angle) * saison
    bruit   = rng.normal(0, 0.05, 8760)
    cf_pv   = np.clip(pv_raw * (1 + bruit), 0, 0.95)
    cf_pv   = cf_pv * (CF_PV_cible / (cf_pv.mean() + 1e-9))
    cf_pv   = np.clip(cf_pv, 0, 0.95)

    # ── Profil éolien vectorisé (Weibull) ────────────────────────────────────
    k = 2.2
    scale = v_mean / (np.exp(np.log(k)/k + 0.5772/k))
    v_h   = rng.weibull(k, 8760) * scale
    # Courbe puissance : cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s
    cf_eol_raw = np.where(v_h < 3, 0.0,
                 np.where(v_h < 12, (v_h - 3) / 9.0,
                 np.where(v_h < 25, 1.0, 0.0))) * 0.85
    cf_eol = cf_eol_raw * (CF_EOL_cible / (cf_eol_raw.mean() + 1e-9))
    cf_eol = np.clip(cf_eol, 0, 1.0)

    _PROFILS_CACHE[ville] = (cf_pv, cf_eol)
    return cf_pv, cf_eol


def _precalculer_tous_profils():
    """Pré-calcule tous les profils au démarrage."""
    print("  Pré-calcul profils horaires...", end=" ", flush=True)
    for v in CALIB_VILLES:
        _generer_profils(v)
    print(f"✓ ({len(CALIB_VILLES)} villes)")


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION PHYSIQUE VECTORISÉE (numpy, ~1ms au lieu de ~500ms)
# ─────────────────────────────────────────────────────────────────────────────

def _simuler_bilan(ville, PV_MW, EOL_MW, ELEC_MW, BAT_MWH, technologie='PEM'):
    """
    Simulation bilan énergétique horaire VECTORISÉE — numpy pur, sans boucle Python.
    Alignée avec ETAPE2.py mode simplifié.
    ~1ms vs ~500ms pour la version boucle.
    """
    cf_pv, cf_eol = _generer_profils(ville)

    eff      = PARAMS_TECHNO[f'EFF_{technologie}']   # kWh/kgH2
    minload  = PARAMS_TECHNO[f'MINLOAD_{technologie}']
    P_dispo  = (cf_pv * PV_MW + cf_eol * EOL_MW) * 1e3  # kW, shape (8760,)
    P_max    = ELEC_MW * 1e3   # kW
    P_min    = P_max * minload  # kW

    E_enr_dispo = P_dispo.sum() / 1e3  # MWh

    if BAT_MWH > 0:
        # ── Avec batterie : boucle horaire via Numba JIT (×50-200 vs Python pur)
        BAT_kWh  = BAT_MWH * 1e3
        eta_c    = PARAMS_TECHNO['EFF_BAT_CHG']
        eta_d    = PARAMS_TECHNO['EFF_BAT_DCH']
        # S'assurer que P_dispo est un array C-contiguous float64 pour Numba
        P_dispo_c = np.ascontiguousarray(P_dispo, dtype=np.float64)

        H2_kg, E_fournie, E_curtail, heures_ok = _boucle_batterie_jit(
            P_dispo_c, float(P_max), float(P_min),
            float(BAT_kWh), float(eff), float(eta_c), float(eta_d)
        )
    else:
        # ── Sans batterie : 100% vectorisé ──────────────────────────────────
        # Puissance fournie à l'électrolyseur
        p_elec   = np.clip(P_dispo, 0, P_max)           # plafonner au max
        actif    = P_dispo >= P_min                      # heures où on produit
        p_elec   = np.where(actif, p_elec, 0.0)

        H2_kg     = (p_elec / eff).sum()
        E_fournie = (p_elec / 1e3).sum()
        heures_ok = int(actif.sum())

        # Curtailment = énergie au-dessus de P_max
        E_curtail = np.maximum(P_dispo - P_max, 0).sum() / 1e3

    fiabilite    = heures_ok / 8760
    taux_curtail = E_curtail / max(E_enr_dispo, 1)

    return {
        'H2_prod_kg_an':    round(float(H2_kg), 0),
        'H2_prod_kt_an':    round(float(H2_kg) / 1e6, 3),
        'E_fournie_MWh':    round(float(E_fournie), 0),
        'E_enr_dispo_MWh':  round(float(E_enr_dispo), 0),
        'fiabilite':        round(float(fiabilite), 4),
        'taux_curtailment': round(float(taux_curtail), 4),
        'heures_production': int(heures_ok),
    }


def _calculer_lcoh(PV_MW, EOL_MW, ELEC_MW, BAT_MWH, H2_prod_kg_an, technologie='PEM'):
    """Calcule LCOH + décomposition — aligné ETAPE2.calculer_LCOH()."""
    if H2_prod_kg_an < 1:
        return None, {}
    dr = PARAMS_TECHNO['DR']

    def ann(capex_total, lt):
        crf = dr * (1 + dr)**lt / ((1 + dr)**lt - 1)
        return capex_total * crf

    ann_PV   = ann(PV_MW   * 1e3 * PARAMS_TECHNO['CAPEX_PV'],  PARAMS_TECHNO['LT_PV'])
    ann_EOL  = ann(EOL_MW  * 1e3 * PARAMS_TECHNO['CAPEX_EOL'], PARAMS_TECHNO['LT_EOL'])
    ann_ELEC = ann(ELEC_MW * 1e3 * PARAMS_TECHNO[f'CAPEX_{technologie}'],
                   PARAMS_TECHNO[f'LT_{technologie}'])
    ann_BAT  = ann(BAT_MWH * 1e3 * PARAMS_TECHNO['CAPEX_BAT'], PARAMS_TECHNO['LT_BAT'])

    opex_PV   = PV_MW   * 1e3 * PARAMS_TECHNO['OPEX_PV']
    opex_EOL  = EOL_MW  * 1e3 * PARAMS_TECHNO['OPEX_EOL']
    opex_ELEC = ELEC_MW * 1e3 * PARAMS_TECHNO[f'CAPEX_{technologie}'] * PARAMS_TECHNO[f'OPEX_{technologie}']
    opex_BAT  = BAT_MWH * 1e3 * PARAMS_TECHNO['CAPEX_BAT'] * PARAMS_TECHNO['OPEX_BAT']
    C_eau     = H2_prod_kg_an * PARAMS_TECHNO['WATER_CONS'] / 1000 * PARAMS_TECHNO['WATER_COST']

    LCOH = (ann_PV + ann_EOL + ann_ELEC + ann_BAT +
            opex_PV + opex_EOL + opex_ELEC + opex_BAT + C_eau) / H2_prod_kg_an

    decomp = {
        'PV_capex':   ann_PV   / H2_prod_kg_an,
        'EOL_capex':  ann_EOL  / H2_prod_kg_an,
        'ELEC_capex': ann_ELEC / H2_prod_kg_an,
        'BAT_capex':  ann_BAT  / H2_prod_kg_an,
        'PV_opex':    opex_PV  / H2_prod_kg_an,
        'EOL_opex':   opex_EOL / H2_prod_kg_an,
        'ELEC_opex':  opex_ELEC/ H2_prod_kg_an,
        'BAT_opex':   opex_BAT / H2_prod_kg_an,
        'eau':        C_eau    / H2_prod_kg_an,
    }
    decomp = {k: round(v, 5) for k, v in decomp.items()}
    return round(float(LCOH), 4), decomp


def _predire_physique(ville, PV_MW, EOL_MW, ELEC_MW, BAT_MWH, technologie='PEM'):
    """Prédiction via simulation physique vectorisée."""
    sim  = _simuler_bilan(ville, PV_MW, EOL_MW, ELEC_MW, BAT_MWH, technologie)
    lcoh, decomp = _calculer_lcoh(PV_MW, EOL_MW, ELEC_MW, BAT_MWH,
                                   sim['H2_prod_kg_an'], technologie)
    if lcoh is None:
        return None
    sim['LCOH']              = lcoh
    sim['decomposition_LCOH'] = decomp
    return sim


# ─────────────────────────────────────────────────────────────────────────────
# MODÈLE ML
# ─────────────────────────────────────────────────────────────────────────────
MODEL_LCOH = MODEL_FIAB = FEATURES = None
ML_LOADED  = False
ML_META    = {}

def charger_modele():
    global MODEL_LCOH, MODEL_FIAB, FEATURES, ML_LOADED, ML_META
    if not os.path.exists(CHEMIN_MODEL):
        print(f"⚠️  Modèle ML non trouvé : {CHEMIN_MODEL}")
        return False
    try:
        art = joblib.load(CHEMIN_MODEL)
        MODEL_LCOH = art['model_lcoh']
        MODEL_FIAB = art['model_fiabilite']
        FEATURES   = art['features']
        ML_META    = {'n_train': art.get('n_train','N/A'), 'n_test': art.get('n_test','N/A'),
                      'features': FEATURES, 'villes': art.get('villes', list(CALIB_VILLES.keys()))}
        ML_LOADED  = True
        print(f"✓ ML chargé ({len(FEATURES)} features | {ML_META['n_train']} train pts)")
        return True
    except Exception as e:
        print(f"⚠️  Erreur ML : {e}")
        return False


def _predire_ml(ville, PV_MW, EOL_MW, ELEC_MW, BAT_MWH, technologie='PEM'):
    if not ML_LOADED:
        return None
    cal = CALIB_VILLES.get(ville, CALIB_VILLES['Dakhla'])
    ELEC_MW_f = max(float(ELEC_MW), 1)
    row = {
        'PV_MW':          float(PV_MW),
        'EOL_MW':         float(EOL_MW),
        'ELEC_MW':        float(ELEC_MW),
        'BAT_MWH':        float(BAT_MWH),
        'CF_PV_moy':      cal['CF_PV'],
        'CF_EOL_moy':     cal['CF_EOL'],
        'techno_PEM':     1 if technologie == 'PEM' else 0,
        'avec_bat':       1 if float(BAT_MWH) > 0 else 0,
        'ratio_bat_elec': float(BAT_MWH) / ELEC_MW_f,
        'ratio_eol_elec': float(EOL_MW) * cal['CF_EOL'] / ELEC_MW_f,
        'energie_dispo':  (cal['CF_PV'] * float(PV_MW) + cal['CF_EOL'] * float(EOL_MW)) / ELEC_MW_f,
    }
    X    = pd.DataFrame([row])[FEATURES]   # filtre selon features du modèle chargé
    lcoh = float(MODEL_LCOH.predict(X)[0])
    fiab = float(np.clip(MODEL_FIAB.predict(X)[0], 0, 1))
    return {'lcoh': round(lcoh, 3), 'fiabilite': round(fiab, 4)}


def _construire_resultat(ville, PV_MW, EOL_MW, ELEC_MW, BAT_MWH, technologie,
                          ml_res, phys_res):
    cal        = CALIB_VILLES.get(ville, CALIB_VILLES['Dakhla'])
    lcoh_ml    = ml_res['lcoh']      if ml_res   else None
    fiab_ml    = ml_res['fiabilite'] if ml_res   else None
    lcoh_phys  = phys_res['LCOH']    if phys_res else None
    fiab_phys  = phys_res['fiabilite'] if phys_res else None

    lcoh_final = lcoh_ml if lcoh_ml is not None else lcoh_phys
    fiab_final = fiab_ml if fiab_ml is not None else fiab_phys

    delta_lcoh = (round((lcoh_ml - lcoh_phys) / max(lcoh_phys, 0.1) * 100, 2)
                  if lcoh_ml is not None and lcoh_phys is not None else None)
    delta_fiab = (round((fiab_ml - fiab_phys) * 100, 2)
                  if fiab_ml is not None and fiab_phys is not None else None)

    score = (1 / max(lcoh_final or 1, 0.1)) * (fiab_final or 0) * 100

    return {
        'ville': ville,
        'PV_MW': round(float(PV_MW),1), 'EOL_MW': round(float(EOL_MW),1),
        'ELEC_MW': round(float(ELEC_MW),1), 'BAT_MWH': round(float(BAT_MWH),1),
        'technologie': technologie,
        'avec_batterie': 'Oui' if float(BAT_MWH) > 0 else 'Non',
        'LCOH_USD_kg':   round(lcoh_final, 3) if lcoh_final else None,
        'fiabilite_pct': round((fiab_final or 0) * 100, 1),
        'score_composite': round(score, 2),
        'ml': {
            'LCOH_USD_kg':   lcoh_ml,
            'fiabilite_pct': round(fiab_ml * 100, 1) if fiab_ml is not None else None,
            'disponible':    ML_LOADED,
        } if ml_res else {'disponible': False, 'LCOH_USD_kg': None, 'fiabilite_pct': None},
        'pypsa': {
            'LCOH_USD_kg':          lcoh_phys,
            'fiabilite_pct':        round((fiab_phys or 0) * 100, 1),
            'H2_prod_kg_an':        phys_res.get('H2_prod_kg_an') if phys_res else None,
            'H2_prod_kt_an':        phys_res.get('H2_prod_kt_an') if phys_res else None,
            'taux_curtailment_pct': round((phys_res.get('taux_curtailment', 0) or 0) * 100, 1) if phys_res else None,
            'heures_production':    phys_res.get('heures_production') if phys_res else None,
            'decomposition_LCOH':   phys_res.get('decomposition_LCOH') if phys_res else None,
        } if phys_res else {'LCOH_USD_kg': None, 'fiabilite_pct': None, 'H2_prod_kt_an': None,
                            'taux_curtailment_pct': None, 'heures_production': None, 'decomposition_LCOH': None},
        'comparaison': {
            'delta_LCOH_pct':     delta_lcoh,
            'delta_fiabilite_pp': delta_fiab,
        'accord':             (abs(delta_lcoh) < 20) if delta_lcoh is not None else None,
        },
        'ressources': {
            'CF_PV_pct':  round(cal['CF_PV'] * 100, 1),
            'CF_EOL_pct': round(cal['CF_EOL'] * 100, 1),
            'GHI_kWh_m2': cal['GHI'],
            'v_vent_moy': cal['v_mean'],
        },
        'latitude':  cal.get('lat', 0),
        'longitude': cal.get('lon', 0),
        'categorie_LCOH': (
            'Excellent (<5 $/kg)' if (lcoh_final or 99) < 5 else
            'Bon (5-7 $/kg)'      if (lcoh_final or 99) < 7 else
            'Acceptable (7-10 $/kg)' if (lcoh_final or 99) < 10 else
            'Élevé (>10 $/kg)'
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
if CORS_OK:
    CORS(app)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'modele_ml': 'chargé' if ML_LOADED else 'non disponible',
        'simulation_physique': 'disponible (vectorisée)',
        'features': FEATURES,
        'villes': list(CALIB_VILLES.keys()),
        'meta_ml': ML_META,
        'message': 'API H2 Maroc v3.1',
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        requis = ['ville','PV_MW','EOL_MW','ELEC_MW','BAT_MWH']
        manq = [k for k in requis if k not in data]
        if manq:
            return jsonify({'erreur': f'Manquants : {manq}'}), 400
        if data['ville'] not in CALIB_VILLES:
            return jsonify({'erreur': f"Ville inconnue : {data['ville']}",
                            'villes': list(CALIB_VILLES.keys())}), 400
        v, pv, eol = data['ville'], float(data['PV_MW']), float(data['EOL_MW'])
        elec, bat  = float(data['ELEC_MW']), float(data['BAT_MWH'])
        tech = data.get('technologie','PEM')
        ml   = _predire_ml(v, pv, eol, elec, bat, tech)
        phys = _predire_physique(v, pv, eol, elec, bat, tech)
        return jsonify(_construire_resultat(v, pv, eol, elec, bat, tech, ml, phys))
    except Exception as e:
        return jsonify({'erreur': str(e)}), 500


@app.route('/predict_comparison', methods=['POST'])
def predict_comparison():
    try:
        data = request.get_json()
        v    = data.get('ville','Dakhla')
        pv   = float(data.get('PV_MW',150))
        eol  = float(data.get('EOL_MW',80))
        elec = float(data.get('ELEC_MW',60))
        bat  = float(data.get('BAT_MWH',100))
        tech = data.get('technologie','PEM')

        if v not in CALIB_VILLES:
            return jsonify({'erreur': f'Ville inconnue : {v}'}), 400

        ml   = _predire_ml(v, pv, eol, elec, bat, tech)
        phys = _predire_physique(v, pv, eol, elec, bat, tech)
        result = _construire_resultat(v, pv, eol, elec, bat, tech, ml, phys)

        # Sensibilité PV (×5 appels vectorisés — rapide)
        sens = []
        for factor in [0.6, 0.8, 1.0, 1.2, 1.4]:
            pv_f = round(pv * factor, 1)
            ml_s = _predire_ml(v, pv_f, eol, elec, bat, tech)
            ph_s = _predire_physique(v, pv_f, eol, elec, bat, tech)
            if ml_s and ph_s:
                sens.append({
                    'PV_MW':      pv_f,
                    'LCOH_ml':    ml_s['lcoh'],
                    'LCOH_pypsa': ph_s['LCOH'],
                    'fiab_ml':    round(ml_s['fiabilite']*100,1),
                    'fiab_pypsa': round(ph_s['fiabilite']*100,1),
                })
        result['sensibilite_PV'] = sens
        return jsonify(result)
    except Exception as e:
        import traceback as _tb
        logger.error(_tb.format_exc())
        return jsonify({'erreur': str(e)}), 500


@app.route('/toutes_villes', methods=['GET'])
def toutes_villes():
    try:
        pv   = float(request.args.get('PV_MW',150))
        eol  = float(request.args.get('EOL_MW',80))
        elec = float(request.args.get('ELEC_MW',70))
        bat  = float(request.args.get('BAT_MWH',100))
        tech = request.args.get('technologie','PEM')

        resultats = []
        for ville in CALIB_VILLES:
            ml   = _predire_ml(ville, pv, eol, elec, bat, tech)
            phys = _predire_physique(ville, pv, eol, elec, bat, tech)
            r = _construire_resultat(ville, pv, eol, elec, bat, tech, ml, phys)
            resultats.append(r)

        resultats.sort(key=lambda x: x['LCOH_USD_kg'] or 99)
        for i, r in enumerate(resultats):
            r['rang'] = i + 1

        return jsonify({
            'config': {'PV_MW':pv,'EOL_MW':eol,'ELEC_MW':elec,'BAT_MWH':bat,'technologie':tech},
            'nb_villes': len(resultats),
            'meilleure': resultats[0]['ville'],
            'LCOH_min':  resultats[0]['LCOH_USD_kg'],
            'LCOH_max':  resultats[-1]['LCOH_USD_kg'],
            'resultats': resultats,
        })
    except Exception as e:
        import traceback as _tb
        logger.error(_tb.format_exc())
        return jsonify({'erreur': str(e)}), 500


@app.route('/waterfall/<ville>', methods=['GET'])
def waterfall(ville):
    try:
        if ville not in CALIB_VILLES:
            return jsonify({'erreur': f'Ville inconnue : {ville}'}), 400
        pv   = float(request.args.get('PV_MW',150))
        eol  = float(request.args.get('EOL_MW',80))
        elec = float(request.args.get('ELEC_MW',70))
        bat  = float(request.args.get('BAT_MWH',100))
        tech = request.args.get('technologie','PEM')

        sim  = _simuler_bilan(ville, pv, eol, elec, bat, tech)
        lcoh, decomp = _calculer_lcoh(pv, eol, elec, bat, sim['H2_prod_kg_an'], tech)
        if lcoh is None:
            return jsonify({'erreur': 'Production H2 nulle'}), 400

        items = [
            {'label': 'PV CAPEX',      'value': decomp['PV_capex'],   'type': 'CAPEX'},
            {'label': 'Éolien CAPEX',  'value': decomp['EOL_capex'],  'type': 'CAPEX'},
            {'label': f'{tech} CAPEX', 'value': decomp['ELEC_capex'], 'type': 'CAPEX'},
            {'label': 'Batt. CAPEX',   'value': decomp['BAT_capex'],  'type': 'CAPEX'},
            {'label': 'PV OPEX',       'value': decomp['PV_opex'],    'type': 'OPEX'},
            {'label': 'Éolien OPEX',   'value': decomp['EOL_opex'],   'type': 'OPEX'},
            {'label': f'{tech} OPEX',  'value': decomp['ELEC_opex'],  'type': 'OPEX'},
            {'label': 'Batt. OPEX',    'value': decomp['BAT_opex'],   'type': 'OPEX'},
            {'label': 'Eau',           'value': decomp['eau'],        'type': 'Variable'},
        ]
        return jsonify({
            'ville': ville, 'LCOH_total': lcoh, 'technologie': tech,
            'decomposition': items,
            'H2_prod_kg_an': sim['H2_prod_kg_an'],
            'fiabilite_pct': round(sim['fiabilite']*100,1),
        })
    except Exception as e:
        return jsonify({'erreur': str(e)}), 500


@app.route('/pareto', methods=['GET'])
def pareto():
    try:
        ville = request.args.get('ville','Dakhla')
        tech  = request.args.get('technologie','PEM')
        n     = int(request.args.get('n', 200))
        if ville not in CALIB_VILLES:
            return jsonify({'erreur': f'Ville inconnue : {ville}'}), 400

        rng    = np.random.default_rng(42)
        points = []
        for _ in range(n):
            pv   = float(rng.uniform(10, 400))
            eol  = float(rng.uniform(0, 250))
            elec = float(rng.uniform(10, 180))
            bat  = float(rng.uniform(0, 400))
            if ML_LOADED:
                ml = _predire_ml(ville, pv, eol, elec, bat, tech)
                if ml:
                    points.append({'PV_MW':round(pv,1),'EOL_MW':round(eol,1),
                                   'ELEC_MW':round(elec,1),'BAT_MWH':round(bat,1),
                                   'LCOH':ml['lcoh'],'fiabilite':round(ml['fiabilite']*100,1),
                                   'source':'ML'})
            else:
                ph = _predire_physique(ville, pv, eol, elec, bat, tech)
                if ph:
                    points.append({'PV_MW':round(pv,1),'EOL_MW':round(eol,1),
                                   'ELEC_MW':round(elec,1),'BAT_MWH':round(bat,1),
                                   'LCOH':ph['LCOH'],'fiabilite':round(ph['fiabilite']*100,1),
                                   'source':'PyPSA'})

        # Filtre Pareto
        pareto_pts = [p for p in points
                      if not any(q['LCOH'] <= p['LCOH'] and q['fiabilite'] >= p['fiabilite']
                                 and (q['LCOH'] < p['LCOH'] or q['fiabilite'] > p['fiabilite'])
                                 for q in points)]
        pareto_pts.sort(key=lambda x: x['LCOH'])

        return jsonify({'ville':ville,'technologie':tech,'n_simulations':n,
                        'n_pareto':len(pareto_pts),
                        'tous_points':points,'front_pareto':pareto_pts})
    except Exception as e:
        return jsonify({'erreur': str(e)}), 500


@app.route('/optimiser', methods=['GET'])
def optimiser():
    try:
        ville      = request.args.get('ville','Dakhla')
        lcoh_cible = float(request.args.get('lcoh_cible',5.0))
        tech       = request.args.get('technologie','PEM')
        n_essais   = int(request.args.get('n_essais',500))
        if ville not in CALIB_VILLES:
            return jsonify({'erreur': f'Ville inconnue : {ville}'}), 400

        rng = np.random.default_rng(42)
        candidats = []
        for _ in range(n_essais):
            pv   = float(rng.uniform(10,500))
            eol  = float(rng.uniform(0,300))
            elec = float(rng.uniform(10,200))
            bat  = float(rng.uniform(0,500))
            ml   = _predire_ml(ville,pv,eol,elec,bat,tech)
            phys = _predire_physique(ville,pv,eol,elec,bat,tech)
            lcoh = (ml['lcoh'] if ml else None) or (phys['LCOH'] if phys else None)
            fiab = (ml['fiabilite'] if ml else None) or (phys['fiabilite'] if phys else None)
            if lcoh and abs(lcoh-lcoh_cible) < 2.0:
                candidats.append({
                    'PV_MW':round(pv,1),'EOL_MW':round(eol,1),
                    'ELEC_MW':round(elec,1),'BAT_MWH':round(bat,1),
                    'LCOH_USD_kg':round(lcoh,3),
                    'fiabilite_pct':round((fiab or 0)*100,1),
                    'LCOH_ml':ml['lcoh'] if ml else None,
                    'LCOH_pypsa':phys['LCOH'] if phys else None,
                })
        candidats.sort(key=lambda x:(abs(x['LCOH_USD_kg']-lcoh_cible),-x['fiabilite_pct']))
        return jsonify({'ville':ville,'lcoh_cible':lcoh_cible,'technologie':tech,
                        'n_essais':n_essais,'n_candidats':len(candidats),'top5':candidats[:5]})
    except Exception as e:
        return jsonify({'erreur': str(e)}), 500


@app.route('/export_csv', methods=['GET'])
def export_csv():
    try:
        CONFIGS = [
            (100, 50,  50,   0, 'PEM','Petite_PEM_sans_bat'),
            (150, 80,  70, 100, 'PEM','Moyenne_PEM_avec_bat'),
            (200,100,  90, 200, 'PEM','Grande_PEM'),
            (150,100,  70,   0, 'AEL','Moyenne_AEL_sans_bat'),
            (200,150, 100, 300, 'AEL','Grande_AEL'),
        ]
        rows = []
        for pv,eol,elec,bat,tech,label in CONFIGS:
            for ville in CALIB_VILLES:
                ml   = _predire_ml(ville,pv,eol,elec,bat,tech)
                phys = _predire_physique(ville,pv,eol,elec,bat,tech)
                r    = _construire_resultat(ville,pv,eol,elec,bat,tech,ml,phys)
                rows.append({
                    'configuration':label,'ville':ville,
                    'PV_MW':pv,'EOL_MW':eol,'ELEC_MW':elec,'BAT_MWH':bat,'technologie':tech,
                    'LCOH_USD_kg':r['LCOH_USD_kg'],'fiabilite_pct':r['fiabilite_pct'],
                    'score_composite':r['score_composite'],
                    'LCOH_ml':r['ml']['LCOH_USD_kg'],'LCOH_pypsa':r['pypsa']['LCOH_USD_kg'],
                    'delta_LCOH_pct':r['comparaison']['delta_LCOH_pct'],
                    'H2_prod_kt_an':r['pypsa']['H2_prod_kt_an'],
                    'curtailment_pct':r['pypsa']['taux_curtailment_pct'],
                    'CF_PV_pct':r['ressources']['CF_PV_pct'],'CF_EOL_pct':r['ressources']['CF_EOL_pct'],
                    'latitude':r['latitude'],'longitude':r['longitude'],
                    'categorie_LCOH':r['categorie_LCOH'],
                })
        df = pd.DataFrame(rows)
        df['rang'] = df.groupby('configuration')['LCOH_USD_kg'].rank().astype(int)
        return Response(df.to_csv(index=False,encoding='utf-8-sig'), mimetype='text/csv',
                        headers={'Content-Disposition':'attachment;filename=h2_v3.csv'})
    except Exception as e:
        return jsonify({'erreur': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# DÉMARRAGE
# ─────────────────────────────────────────────────────────────────────────────
charger_modele()
_precalculer_tous_profils()   # ← génère les profils 1 seule fois au démarrage

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  API H2 MAROC v3.3 — Simulation vectorisée + Numba")
    print("="*60)
    print(f"  ML       : {'✓ chargé' if ML_LOADED else '⚠ absent (simulation physique seule)'}")
    print(f"  Numba    : {'✓ JIT activé' if NUMBA_OK else '⚠ non installé (pip install numba)'}")
    print(f"  Villes   : {len(CALIB_VILLES)}")
    print(f"  Profils  : pré-calculés (1 fois au démarrage)")
    print("\n  Routes :")
    for r in ['/health','/predict [POST]','/predict_comparison [POST]',
              '/toutes_villes','/waterfall/<ville>','/pareto','/optimiser','/export_csv']:
        print(f"    {r}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
