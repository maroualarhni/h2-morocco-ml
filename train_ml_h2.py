# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ML H2 MAROC v3 — Random Forest + Comparaison PyPSA vs ML                ║
║                                                                              ║
║   Nouveautés v3 :                                                            ║
║     - Sur-échantillonnage des configs "batterie large + haute fiabilité"    ║
║       (30% des simulations forcées sur BAT_MWH ∈ [300,500] MWh)            ║
║     - Nouvelle feature ratio_bat_elec = BAT_MWH / ELEC_MW                 ║
║       (capture l'effet batterie relatif → réduit divergence ML/PyPSA)      ║
║     - N_SIMULATIONS réduit à 300 (÷1.7 temps, dataset plus équilibré)     ║
║     - Rapport visuel mis à jour (feature importance avec ratio_bat_elec)   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys, os, time, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DOSSIER_PFE = os.path.join(os.path.expanduser("~"), "Downloads", "pfe")
OUTPUT_ML   = os.path.join(os.path.expanduser("~"), "Downloads",
                            "H2Morocco222_Outputs", "ml")
os.makedirs(OUTPUT_ML, exist_ok=True)

N_SIMULATIONS = 400   # v4 : augmenté à 400 avec distribution équilibrée (4 zones BAT)

VILLES = [
    'Agadir', 'Boujdour', 'Casablanca', 'Dakhla',
    'Guelmim', 'Jorf_Lasfar', 'Laayoune', 'Marrakech',
    'Midelt', 'Nador', 'Ouarzazate', 'Tanger'
]
TECHNOS   = ['PEM', 'AEL']
SCENARIOS = ['avec', 'sans']

RF_PARAMS = {
    'n_estimators':     300,
    'max_depth':        None,
    'min_samples_leaf': 2,
    'n_jobs':           -1,
    'random_state':     42,
}

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT ETAPE2
# ─────────────────────────────────────────────────────────────────────────────
def importer_etape2():
    chemin = os.path.join(DOSSIER_PFE, "ETAPE2.py")
    if not os.path.exists(chemin):
        print(f"❌ ETAPE2.py non trouvé dans : {DOSSIER_PFE}")
        sys.exit(1)
    if DOSSIER_PFE not in sys.path:
        sys.path.insert(0, DOSSIER_PFE)
    import importlib.util
    spec   = importlib.util.spec_from_file_location("etape2", chemin)
    module = importlib.util.module_from_spec(spec)
    print("  Chargement ETAPE2.py...", end=" ", flush=True)
    spec.loader.exec_module(module)
    print("✓")
    return module

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — GÉNÉRATION DATASET
# ─────────────────────────────────────────────────────────────────────────────
def generer_dataset(etape2):
    fichier_cache = os.path.join(OUTPUT_ML, "dataset_h2.csv")
    if os.path.exists(fichier_cache):
        print(f"\n  ✓ Dataset cache : {fichier_cache}")
        df = pd.read_csv(fichier_cache)
        print(f"    → {len(df)} lignes chargées")
        return df

    print(f"\n{'='*65}")
    print(f"  ÉTAPE 1 — GÉNÉRATION DATASET ({N_SIMULATIONS} sims × {len(VILLES)} villes)")
    print(f"{'='*65}")

    records = []
    n_erreurs = 0
    t_debut = time.time()

    for ville in VILLES:
        print(f"\n  [{ville}] Chargement profils...", end=" ", flush=True)
        try:
            profils = etape2.charger_profils_T10(ville, annee=2024, force_synthetic=False)
            CF_PV_moy  = float(profils['CF_PV_h'].mean())
            CF_EOL_moy = float(profils['CF_eol_h'].mean())
            print(f"✓  CF_PV={CF_PV_moy*100:.1f}% | CF_EOL={CF_EOL_moy*100:.1f}%")
        except Exception as e:
            print(f"⚠️  fallback synthétique ({e})")
            profils = etape2.charger_profils_T10(ville, annee=2024, force_synthetic=True)
            CF_PV_moy  = float(profils['CF_PV_h'].mean())
            CF_EOL_moy = float(profils['CF_eol_h'].mean())

        etape2.clear_cache()

        for i in range(N_SIMULATIONS):
            techno  = np.random.choice(TECHNOS)
            PV_MW   = np.random.uniform(10,  500)
            EOL_MW  = np.random.uniform(0,   300)
            ELEC_MW = np.random.uniform(10,  200)

            # ── v4 : distribution équilibrée sur 4 scénarios égaux ──────────
            # On abandonne le sur-échantillonnage "batterie large" qui biaisait
            # la fiabilité vers le haut et empirait la concordance ML/PyPSA.
            # À la place : 4 zones équiprobables couvrant tout l'espace BAT.
            #
            #   25% : sans batterie         (BAT=0)
            #   25% : petite batterie       (BAT=10-100 MWh)
            #   25% : batterie moyenne      (BAT=100-300 MWh)
            #   25% : grande batterie       (BAT=300-500 MWh)
            #
            # → fiabilité répartie uniformément de 50% à 100% dans le dataset
            r = np.random.random()
            if r < 0.25:
                BAT_MWH = 0.0                              # sans batterie
            elif r < 0.50:
                BAT_MWH = np.random.uniform(10,  100)     # petite
            elif r < 0.75:
                BAT_MWH = np.random.uniform(100, 300)     # moyenne
            else:
                BAT_MWH = np.random.uniform(300, 500)     # grande

            try:
                res = etape2.simuler_pypsa(
                    profils_df=profils, PV_MW=PV_MW, EOL_MW=EOL_MW,
                    ELEC_MW=ELEC_MW, BAT_MWH=BAT_MWH,
                    technologie=techno, region=ville, verbose=False, use_cache=True,
                )
                H2_prod = res['H2_prod_kg_an']
                lcoh = etape2.calculer_LCOH(
                    PV_MW=PV_MW, EOL_MW=EOL_MW, ELEC_MW=ELEC_MW,
                    BAT_MWH=BAT_MWH, H2_prod_kg_an=H2_prod,
                    technologie=techno, detail=False,
                )
                fiabilite = res['fiabilite']
                if lcoh > 30 or lcoh < 0.5 or H2_prod < 1:
                    continue

                # ── Calcul LCOH PyPSA détaillé pour stocker en référence ──
                try:
                    # calculer_LCOH(detail=True) retourne un tuple (float, dict)
                    _lcoh_val, lcoh_detail = etape2.calculer_LCOH(
                        PV_MW=PV_MW, EOL_MW=EOL_MW, ELEC_MW=ELEC_MW,
                        BAT_MWH=BAT_MWH, H2_prod_kg_an=H2_prod,
                        technologie=techno, detail=True,
                    )
                    curtail_pct = res['taux_curtailment'] * 100
                    pct_slack   = res.get('pct_slack', 0)
                except Exception:
                    lcoh_detail = {}
                    curtail_pct = 0
                    pct_slack   = 0

                records.append({
                    'PV_MW': round(PV_MW, 2), 'EOL_MW': round(EOL_MW, 2),
                    'ELEC_MW': round(ELEC_MW, 2), 'BAT_MWH': round(BAT_MWH, 2),
                    'CF_PV_moy': round(CF_PV_moy, 4), 'CF_EOL_moy': round(CF_EOL_moy, 4),
                    'techno_PEM': 1 if techno == 'PEM' else 0,
                    'avec_bat':   1 if BAT_MWH > 0 else 0,
                    'ratio_bat_elec': round(BAT_MWH / max(ELEC_MW, 1), 4),
                    'ratio_eol_elec': round(EOL_MW * CF_EOL_moy / max(ELEC_MW, 1), 4),
                    'energie_dispo':  round((CF_PV_moy * PV_MW + CF_EOL_moy * EOL_MW) / max(ELEC_MW, 1), 4),
                    'H2_prod_kg_an':    round(H2_prod, 0),
                    'fiabilite_pct':    round(fiabilite * 100, 2),
                    'taux_curtail_pct': round(curtail_pct, 2),
                    'pct_slack':        round(pct_slack, 2),
                    'LCOH': round(lcoh, 4),
                    'fiabilite': round(fiabilite, 4),
                    'LCOH_PV_capex':   round(lcoh_detail.get('LCOH_PV_capex', 0), 5),
                    'LCOH_EOL_capex':  round(lcoh_detail.get('LCOH_EOL_capex', 0), 5),
                    'LCOH_ELEC_capex': round(lcoh_detail.get('LCOH_ELEC_capex', 0), 5),
                    'LCOH_BAT_capex':  round(lcoh_detail.get('LCOH_BAT_capex', 0), 5),
                    'LCOH_eau':        round(lcoh_detail.get('LCOH_eau', 0), 5),
                    'ville': ville, 'technologie': techno,
                    'scenario': 'avec' if BAT_MWH > 0 else 'sans',
                })
            except Exception as e:
                n_erreurs += 1
                if n_erreurs <= 3:
                    print(f"\n     Sim {i} : {e}")

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t_debut
                print(f"    → {i+1}/{N_SIMULATIONS} | {len(records)} pts | {elapsed:.0f}s")

    df = pd.DataFrame(records)
    print(f"\n  ✓ Dataset : {len(df)} points | {n_erreurs} erreurs ignorées")
    print(f"    LCOH : {df['LCOH'].min():.2f} — {df['LCOH'].max():.2f} $/kg")
    df.to_csv(fichier_cache, index=False)
    print(f"    → {fichier_cache}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────
def entrainer_modele(df):
    print(f"\n{'='*65}\n  ÉTAPE 2 — ENTRAÎNEMENT RANDOM FOREST\n{'='*65}")
    FEATURES = [
        'PV_MW', 'EOL_MW', 'ELEC_MW', 'BAT_MWH',
        'CF_PV_moy', 'CF_EOL_moy', 'techno_PEM', 'avec_bat',
        'ratio_bat_elec',    # BAT_MWH / ELEC_MW
        'ratio_eol_elec',    # EOL_MW × CF_EOL / ELEC_MW — effet ville éolienne
        'energie_dispo',     # (CF_PV×PV + CF_EOL×EOL) / ELEC — énergie relative
    ]
    cols_manq = [c for c in FEATURES if c not in df.columns]
    if cols_manq:
        print(f"  ❌ Colonnes manquantes : {cols_manq}"); return None

    X = df[FEATURES].values
    y_lcoh = df['LCOH'].values
    y_fiab = df['fiabilite'].values

    X_train, X_test, y_lcoh_train, y_lcoh_test, y_fiab_train, y_fiab_test = \
        train_test_split(X, y_lcoh, y_fiab, test_size=0.2, random_state=42)

    print(f"  Split : {len(X_train)} train | {len(X_test)} test")

    print(f"  [1/2] RF LCOH...", end=" ", flush=True)
    t0 = time.time()
    model_lcoh = RandomForestRegressor(**RF_PARAMS)
    model_lcoh.fit(X_train, y_lcoh_train)
    print(f"✓ ({time.time()-t0:.1f}s)")

    print(f"  [2/2] RF Fiabilité...", end=" ", flush=True)
    t0 = time.time()
    model_fiab = RandomForestRegressor(**RF_PARAMS)
    model_fiab.fit(X_train, y_fiab_train)
    print(f"✓ ({time.time()-t0:.1f}s)")

    artifact = {
        'model_lcoh': model_lcoh, 'model_fiabilite': model_fiab,
        'features': FEATURES, 'n_train': len(X_train), 'n_test': len(X_test),
        'villes': VILLES,
    }
    chemin = os.path.join(OUTPUT_ML, "models_h2.pkl")
    joblib.dump(artifact, chemin)
    print(f"\n  ✓ Modèles → {chemin}")

    return {
        'model_lcoh': model_lcoh, 'model_fiab': model_fiab, 'features': FEATURES,
        'X_train': X_train, 'X_test': X_test,
        'y_lcoh_train': y_lcoh_train, 'y_lcoh_test': y_lcoh_test,
        'y_fiab_train': y_fiab_train, 'y_fiab_test': y_fiab_test,
    }

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — VALIDATION + COMPARAISON ML vs PyPSA
# ─────────────────────────────────────────────────────────────────────────────
def valider_modele(resultats, df):
    print(f"\n{'='*65}\n  ÉTAPE 3 — VALIDATION + COMPARAISON ML vs PyPSA\n{'='*65}")

    model_lcoh = resultats['model_lcoh']
    model_fiab = resultats['model_fiab']
    features   = resultats['features']
    X_test     = resultats['X_test']
    X_train    = resultats['X_train']
    y_lcoh_test  = resultats['y_lcoh_test']
    y_fiab_test  = resultats['y_fiab_test']
    y_lcoh_train = resultats['y_lcoh_train']

    pred_lcoh_test  = model_lcoh.predict(X_test)
    pred_fiab_test  = model_fiab.predict(X_test)
    pred_lcoh_train = model_lcoh.predict(X_train)

    r2_lcoh   = r2_score(y_lcoh_test, pred_lcoh_test)
    mae_lcoh  = mean_absolute_error(y_lcoh_test, pred_lcoh_test)
    rmse_lcoh = np.sqrt(mean_squared_error(y_lcoh_test, pred_lcoh_test))
    r2_fiab   = r2_score(y_fiab_test, pred_fiab_test)
    mae_fiab  = mean_absolute_error(y_fiab_test, pred_fiab_test)
    r2_train  = r2_score(y_lcoh_train, pred_lcoh_train)

    print(f"  Cross-Validation 5-folds...", end=" ", flush=True)
    X_full = df[features].values
    cv = cross_val_score(RandomForestRegressor(**RF_PARAMS), X_full, df['LCOH'].values,
                         cv=5, scoring='r2', n_jobs=-1)
    print("✓")

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  RÉSULTATS MODÈLE LCOH                              │
  ├─────────────────────────────────────────────────────┤
  │  R² test    : {r2_lcoh:.4f}  {' EXCELLENT' if r2_lcoh>0.95 else '  ACCEPTABLE'}  │
  │  R² train   : {r2_train:.4f}                            │
  │  MAE        : {mae_lcoh:.4f} $/kg                       │
  │  RMSE       : {rmse_lcoh:.4f} $/kg                       │
  │  CV R²      : {cv.mean():.4f} ± {cv.std():.4f}              │
  ├─────────────────────────────────────────────────────┤
  │  RÉSULTATS MODÈLE FIABILITÉ                         │
  │  R² test    : {r2_fiab:.4f}  {' EXCELLENT' if r2_fiab>0.95 else ' ACCEPTABLE'}  │
  │  MAE        : {mae_fiab:.4f}                            │
  └─────────────────────────────────────────────────────┘""")

    # ── Concordance ML vs PyPSA par ville ────────────────────────────────────
    print("\n  Concordance ML vs PyPSA par ville :")
    print(f"  {'Ville':<15} {'N pts':>6} {'MAE ML/PyPSA':>14} {'R²':>8} {'Biais moy':>12}")
    print(f"  {'-'*15} {'-'*6} {'-'*14} {'-'*8} {'-'*12}")
    concordance_villes = []
    for ville in df['ville'].unique():
        sub = df[df['ville'] == ville].copy()
        X_v = sub[features].values
        pred_v = model_lcoh.predict(X_v)
        real_v = sub['LCOH'].values
        mae_v = mean_absolute_error(real_v, pred_v)
        r2_v  = r2_score(real_v, pred_v) if len(real_v) > 2 else 0
        biais = (pred_v - real_v).mean()
        print(f"  {ville:<15} {len(sub):>6} {mae_v:>14.3f} {r2_v:>8.4f} {biais:>12.3f}")
        concordance_villes.append({'ville': ville, 'n': len(sub), 'mae': mae_v,
                                    'r2': r2_v, 'biais': biais})

    # ── Sauvegarde rapport JSON (pour le dashboard) ───────────────────────────
    metriques = {
        'r2_lcoh': r2_lcoh, 'mae_lcoh': mae_lcoh, 'rmse_lcoh': rmse_lcoh,
        'r2_train': r2_train, 'r2_fiab': r2_fiab, 'mae_fiab': mae_fiab,
        'cv_mean': float(cv.mean()), 'cv_std': float(cv.std()),
        'cv_folds': cv.tolist(),
        'n_test': len(y_lcoh_test), 'n_train': len(y_lcoh_train),
        'features': features,
        'importance': dict(zip(features, model_lcoh.feature_importances_.tolist())),
        'concordance_villes': concordance_villes,
        'pred_lcoh_sample': pred_lcoh_test[:100].tolist(),
        'real_lcoh_sample': y_lcoh_test[:100].tolist(),
        'pred_fiab_sample': pred_fiab_test[:100].tolist(),
        'real_fiab_sample': y_fiab_test[:100].tolist(),
        'residus_lcoh': (pred_lcoh_test - y_lcoh_test).tolist(),
    }
    chemin_json = os.path.join(OUTPUT_ML, "validation_report.json")
    with open(chemin_json, 'w', encoding='utf-8') as f:
        json.dump(metriques, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Rapport JSON → {chemin_json}")

    # ── Rapport visuel (7 graphiques) ─────────────────────────────────────────
    _rapport_visuel(y_lcoh_test, pred_lcoh_test, y_fiab_test, pred_fiab_test,
                    dict(zip(features, model_lcoh.feature_importances_)),
                    cv, concordance_villes, r2_lcoh, mae_lcoh, r2_fiab)

    return metriques

def _rapport_visuel(y_lcoh, pred_lcoh, y_fiab, pred_fiab,
                    importance, cv_scores, concordance_villes,
                    r2_lcoh, mae_lcoh, r2_fiab):
    """Rapport 7-panneaux : validation ML + concordance ML vs PyPSA."""
    residus = pred_lcoh - y_lcoh

    fig = plt.figure(figsize=(21, 14))
    fig.patch.set_facecolor('#0D1117')
    fig.suptitle("Rapport Validation — ML H2 Maroc v2\n"
                 "Random Forest × Concordance ML vs PyPSA",
                 fontsize=15, fontweight='bold', color='white', y=0.98)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.92, bottom=0.08)

    COLORS = {'vert': '#00E676', 'rouge': '#FF5252', 'bleu': '#40C4FF',
              'orange': '#FFAB40', 'fond': '#161B22', 'texte': '#E6EDF3',
              'grille': '#21262D'}
    STYLE = {'facecolor': COLORS['fond'], 'labelcolor': COLORS['texte'],
             'tickcolor': COLORS['texte']}

    def style_ax(ax, title):
        ax.set_facecolor(COLORS['fond'])
        ax.set_title(title, color=COLORS['texte'], fontsize=9, pad=6)
        ax.tick_params(colors=COLORS['texte'], labelsize=7)
        ax.xaxis.label.set_color(COLORS['texte'])
        ax.yaxis.label.set_color(COLORS['texte'])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS['grille'])
        ax.grid(color=COLORS['grille'], linewidth=0.5, alpha=0.5)

    # 1. Prédictions vs Réalité — LCOH
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_lcoh, pred_lcoh, alpha=0.35, s=8, color=COLORS['vert'], rasterized=True)
    lims = [min(y_lcoh.min(), pred_lcoh.min()), max(y_lcoh.max(), pred_lcoh.max())]
    ax1.plot(lims, lims, '--', color=COLORS['rouge'], lw=1.5, label='Parfait')
    ax1.set_xlabel('LCOH PyPSA ($/kg)'); ax1.set_ylabel('LCOH ML ($/kg)')
    style_ax(ax1, f'Pred vs Réalité — LCOH\nR²={r2_lcoh:.4f} | MAE={mae_lcoh:.3f}')
    ax1.legend(fontsize=7, facecolor=COLORS['fond'], labelcolor=COLORS['texte'])

    # 2. Résidus LCOH
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_lcoh, residus, alpha=0.3, s=8, color=COLORS['bleu'], rasterized=True)
    ax2.axhline(0, color=COLORS['rouge'], lw=1.5, ls='--')
    ax2.axhline(mae_lcoh,  color=COLORS['orange'], lw=1, ls=':', label=f'+MAE')
    ax2.axhline(-mae_lcoh, color=COLORS['orange'], lw=1, ls=':')
    ax2.set_xlabel('LCOH réel'); ax2.set_ylabel('Résidu (ML−PyPSA)')
    style_ax(ax2, 'Résidus LCOH\n(centré=OK)')
    ax2.legend(fontsize=7, facecolor=COLORS['fond'], labelcolor=COLORS['texte'])

    # 3. Pred vs Réalité — Fiabilité
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(y_fiab*100, pred_fiab*100, alpha=0.35, s=8,
                color=COLORS['orange'], rasterized=True)
    ax3.plot([0,100],[0,100],'--',color=COLORS['rouge'],lw=1.5)
    ax3.set_xlabel('Fiabilité PyPSA (%)'); ax3.set_ylabel('Fiabilité ML (%)')
    r2_f = r2_lcoh if r2_fiab is None else r2_fiab  # fallback
    style_ax(ax3, f'Pred vs Réalité — Fiabilité\nR²={r2_fiab:.4f}')

    # 4. Importance features
    ax4 = fig.add_subplot(gs[0, 3])
    labels = list(importance.keys())
    vals   = list(importance.values())
    sorted_idx = np.argsort(vals)
    bars = ax4.barh([labels[i] for i in sorted_idx], [vals[i] for i in sorted_idx],
                    color=[COLORS['vert'] if vals[i] > 0.15 else COLORS['bleu']
                           if vals[i] > 0.05 else COLORS['grille'] for i in sorted_idx])
    ax4.set_xlabel('Importance')
    style_ax(ax4, 'Importance Features\n(modèle LCOH)')
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x*100:.0f}%'))

    # 5. Cross-Validation
    ax5 = fig.add_subplot(gs[1, 0])
    folds = [f'F{i+1}' for i in range(len(cv_scores))]
    cs    = [COLORS['vert'] if s > 0.95 else COLORS['orange'] if s > 0.85 else COLORS['rouge']
             for s in cv_scores]
    bars5 = ax5.bar(folds, cv_scores, color=cs, edgecolor=COLORS['grille'])
    ax5.axhline(cv_scores.mean(), color=COLORS['rouge'], ls='--', lw=1.5,
                label=f'Moy={cv_scores.mean():.4f}')
    ax5.axhline(0.95, color=COLORS['vert'], ls=':', lw=1)
    ax5.set_ylim(0.7, 1.0); ax5.set_ylabel('R²')
    style_ax(ax5, f'Cross-Validation 5-folds\nR²={cv_scores.mean():.4f}±{cv_scores.std():.4f}')
    for bar, s in zip(bars5, cv_scores):
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                 f'{s:.3f}', ha='center', fontsize=7, color=COLORS['texte'])
    ax5.legend(fontsize=7, facecolor=COLORS['fond'], labelcolor=COLORS['texte'])

    # 6. Distribution résidus
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(residus, bins=40, color=COLORS['vert'], alpha=0.75, edgecolor=COLORS['fond'])
    ax6.axvline(0, color=COLORS['rouge'], lw=2, ls='--')
    ax6.axvline(residus.mean(), color=COLORS['orange'], lw=1.5,
                label=f'Moy={residus.mean():.3f}')
    ax6.set_xlabel('Erreur ($/kg)'); ax6.set_ylabel('Fréquence')
    style_ax(ax6, 'Distribution Erreurs LCOH')
    ax6.legend(fontsize=7, facecolor=COLORS['fond'], labelcolor=COLORS['texte'])

    # 7. Concordance ML vs PyPSA par ville
    ax7 = fig.add_subplot(gs[1, 2:])
    villes_c = [c['ville'] for c in concordance_villes]
    maes_c   = [c['mae']   for c in concordance_villes]
    r2s_c    = [c['r2']    for c in concordance_villes]
    x = np.arange(len(villes_c))
    bars_mae = ax7.bar(x - 0.2, maes_c, 0.35, label='MAE ($/kg)',
                        color=COLORS['bleu'], alpha=0.8)
    ax7_r = ax7.twinx()
    ax7_r.plot(x, r2s_c, 'o-', color=COLORS['vert'], lw=2,
               markersize=5, label='R²')
    ax7_r.axhline(0.95, color=COLORS['vert'], ls=':', lw=1, alpha=0.5)
    ax7_r.set_ylim(0, 1.1)
    ax7_r.set_ylabel('R²', color=COLORS['vert'])
    ax7_r.tick_params(colors=COLORS['vert'])
    ax7.set_xticks(x); ax7.set_xticklabels(villes_c, rotation=35, ha='right', fontsize=7)
    ax7.set_ylabel('MAE LCOH ($/kg)', color=COLORS['bleu'])
    ax7.tick_params(colors=COLORS['bleu'])
    style_ax(ax7, 'Concordance ML vs PyPSA par Ville\n(MAE = écart moyen ML/PyPSA)')
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7_r.get_legend_handles_labels()
    ax7.legend(lines1+lines2, labels1+labels2, fontsize=7,
               facecolor=COLORS['fond'], labelcolor=COLORS['texte'])

    chemin = os.path.join(OUTPUT_ML, "rapport_validation_v2.png")
    plt.savefig(chemin, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Rapport visuel → {chemin}")

# ─────────────────────────────────────────────────────────────────────────────
# PRÉDICTION INSTANTANÉE
# ─────────────────────────────────────────────────────────────────────────────
CALIB_VILLES_CF = {
    'Agadir': {'CF_PV': 0.192, 'CF_EOL': 0.164},
    'Boujdour': {'CF_PV': 0.200, 'CF_EOL': 0.384},
    'Casablanca': {'CF_PV': 0.171, 'CF_EOL': 0.105},
    'Dakhla': {'CF_PV': 0.197, 'CF_EOL': 0.415},
    'Guelmim': {'CF_PV': 0.192, 'CF_EOL': 0.164},
    'Jorf_Lasfar': {'CF_PV': 0.173, 'CF_EOL': 0.128},
    'Laayoune': {'CF_PV': 0.199, 'CF_EOL': 0.337},
    'Marrakech': {'CF_PV': 0.190, 'CF_EOL': 0.060},
    'Midelt': {'CF_PV': 0.201, 'CF_EOL': 0.150},
    'Nador': {'CF_PV': 0.163, 'CF_EOL': 0.180},
    'Ouarzazate': {'CF_PV': 0.198, 'CF_EOL': 0.225},
    'Tanger': {'CF_PV': 0.168, 'CF_EOL': 0.156},
}

def predire_lcoh(PV_MW, EOL_MW, ELEC_MW, BAT_MWH, ville, technologie='PEM',
                  chemin_model=None):
    if chemin_model is None:
        chemin_model = os.path.join(OUTPUT_ML, "models_h2.pkl")
    artifact   = joblib.load(chemin_model)
    model_lcoh = artifact['model_lcoh']
    model_fiab = artifact['model_fiabilite']
    features   = artifact['features']
    cal = CALIB_VILLES_CF.get(ville, {'CF_PV': 0.185, 'CF_EOL': 0.200})
    X = pd.DataFrame([{
        'PV_MW': PV_MW, 'EOL_MW': EOL_MW, 'ELEC_MW': ELEC_MW, 'BAT_MWH': BAT_MWH,
        'CF_PV_moy': cal['CF_PV'], 'CF_EOL_moy': cal['CF_EOL'],
        'techno_PEM': 1 if technologie == 'PEM' else 0,
        'avec_bat': 1 if BAT_MWH > 0 else 0,
        'ratio_bat_elec': float(BAT_MWH) / max(float(ELEC_MW), 1),
        'ratio_eol_elec': float(EOL_MW) * cal['CF_EOL'] / max(float(ELEC_MW), 1),
        'energie_dispo':  (cal['CF_PV'] * float(PV_MW) + cal['CF_EOL'] * float(EOL_MW)) / max(float(ELEC_MW), 1),
    }])[features]
    return {
        'lcoh': round(float(model_lcoh.predict(X)[0]), 3),
        'fiabilite': round(float(np.clip(model_fiab.predict(X)[0], 0, 1)), 4),
        'ville': ville,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print("  ML H2 MAROC v2 — Pipeline complet")
    print("  Random Forest + Comparaison PyPSA vs ML")
    print("="*65)

    etape2 = importer_etape2()
    df     = generer_dataset(etape2)

    if len(df) < 50:
        print(f"\n❌ Dataset trop petit ({len(df)} lignes).")
        sys.exit(1)

    resultats = entrainer_modele(df)
    if resultats is None:
        print("\n❌ Entraînement échoué."); sys.exit(1)

    metriques = valider_modele(resultats, df)

    print(f"\n{'='*65}\n  DÉMONSTRATION — Prédiction instantanée\n{'='*65}")
    exemples = [
        ('Dakhla', 150, 100, 80, 200, 'AEL'),
        ('Ouarzazate', 200, 30, 90, 0, 'PEM'),
        ('Casablanca', 80, 20, 40, 50, 'PEM'),
        ('Laayoune', 180, 120, 70, 150, 'AEL'),
    ]
    print(f"\n  {'Ville':<14} {'PV':>5} {'EOL':>5} {'ELEC':>5} {'BAT':>5} {'Tech':<5} │ {'LCOH':>8} {'Fiab':>8}")
    print(f"  {'-'*14} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} ┼ {'-'*8} {'-'*8}")
    for ville, pv, eol, elec, bat, tech in exemples:
        t0 = time.time()
        res = predire_lcoh(pv, eol, elec, bat, ville, tech)
        dt = (time.time()-t0)*1000
        print(f"  {ville:<14} {pv:>5.0f} {eol:>5.0f} {elec:>5.0f} {bat:>5.0f} "
              f"{tech:<5} │ {res['lcoh']:>7.3f}$ {res['fiabilite']*100:>7.1f}%  ({dt:.1f}ms)")

    print(f"\n  Fichiers générés :")
    print(f"    → dataset_h2.csv")
    print(f"    → models_h2.pkl")
    print(f"    → validation_report.json  ← lu par le dashboard")
    print(f"    → rapport_validation_v2.png")
    print(f"\n  R² LCOH : {metriques['r2_lcoh']:.4f} | MAE : {metriques['mae_lcoh']:.4f} $/kg")
    print(f"  R² Fiabilité : {metriques['r2_fiab']:.4f}")
    print(f"  CV R² : {metriques['cv_mean']:.4f} ± {metriques['cv_std']:.4f}")
    print("\n  Terminé.\n")
