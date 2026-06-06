#  H2 Morocco — ML Intelligence Layer

> **Module Machine Learning** pour la prédiction rapide du LCOH et de la fiabilité des systèmes hydrogène vert au Maroc.  
> Construit sur un pipeline **Random Forest + Flask API + Streamlit Dashboard**, ce module fonctionne en complément du simulateur physique PyPSA.

<p align="center">
  <img src="https://img.shields.io/badge/Random_Forest-sklearn-F7931E?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Flask_API-REST-000000?logo=flask" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit" />
  <img src="https://img.shields.io/badge/Numba-JIT_Acceleration-00A3E0" />
  <img src="https://img.shields.io/badge/R²_LCOH-≥_0.97-brightgreen" />
  <img src="https://img.shields.io/badge/R²_Fiabilité-≥_0.95-brightgreen" />
</p>

---

##  Fichiers

| Fichier | Rôle |
|---------|------|
| `train_ml_h2.py` | Pipeline d'entraînement — génération dataset, Random Forest, validation |
| `api_flask_h2.py` | API REST Flask — exposition des modèles ML + simulation physique vectorisée |
| `dashboard.py` | Dashboard Streamlit "Digital Twin" — interface utilisateur avancée |

---

##  Architecture ML

```
ETAPE2.py (PyPSA)
    ↓  400 simulations × 12 villes
train_ml_h2.py  ──→  dataset_h2.csv
                ──→  models_h2.pkl  (Random Forest × 2)
                ──→  validation_report.json

models_h2.pkl
    ↓
api_flask_h2.py  (Flask :5000)
    ↓  REST /predict, /toutes_villes, /pareto, /waterfall
dashboard.py  (Streamlit :8501)
```

---

##  Modèle Machine Learning

### Features d'entrée (11 variables)

| Feature | Description | Unité |
|---------|-------------|-------|
| `PV_MW` | Capacité solaire PV | MW |
| `EOL_MW` | Capacité éolienne | MW |
| `ELEC_MW` | Capacité électrolyseur | MW |
| `BAT_MWH` | Capacité batterie | MWh |
| `CF_PV_moy` | Facteur de charge PV moyen annuel | — |
| `CF_EOL_moy` | Facteur de charge éolien moyen annuel | — |
| `techno_PEM` | Technologie électrolyseur (1=PEM, 0=AEL) | binaire |
| `avec_bat` | Présence batterie (1=oui, 0=non) | binaire |
| `ratio_bat_elec` | BAT_MWH / ELEC_MW — effet batterie relatif | — |
| `ratio_eol_elec` | EOL_MW × CF_EOL / ELEC_MW — potentiel éolien | — |
| `energie_dispo` | (CF_PV×PV + CF_EOL×EOL) / ELEC — énergie relative | — |

### Cibles (2 modèles indépendants)

- **LCOH** (Levelized Cost of Hydrogen) — en $/kgH₂
- **Fiabilité** — taux de couverture de la demande en H₂ (0–1)

### Hyperparamètres Random Forest

```python
RF_PARAMS = {
    'n_estimators':     300,
    'max_depth':        None,   # arbres complets
    'min_samples_leaf': 2,
    'n_jobs':           -1,     # parallélisation totale
    'random_state':     42,
}
```

### Performances obtenues

| Métrique | LCOH | Fiabilité |
|----------|------|-----------|
| R² (test) | **≥ 0.97** | **≥ 0.95** |
| MAE | < 0.15 $/kg | < 0.02 |
| Validation croisée (5-fold) | R² ≥ 0.96 ± 0.01 | — |

### Distribution du dataset (v4)

Le dataset est généré avec une distribution équilibrée sur 4 zones de capacité batterie :

| Zone | Distribution | Plage BAT |
|------|-------------|-----------|
| Sans batterie | 25% | 0 MWh |
| Petite batterie | 25% | 10–100 MWh |
| Batterie moyenne | 25% | 100–300 MWh |
| Grande batterie | 25% | 300–500 MWh |

Cette distribution assure une fiabilité prédite uniformément répartie de 50% à 100%, évitant le biais vers les configurations haute-fiabilité des versions précédentes.

---

##  API Flask — Endpoints

Démarrer l'API :

```bash
python api_flask_h2.py
# → http://localhost:5000
```

### `GET /health`

Statut de l'API et des modèles chargés.

```json
{
  "status": "ok",
  "modele_ml": "chargé",
  "simulation_physique": "disponible (vectorisée)",
  "villes": ["Agadir", "Dakhla", "Tanger", ...],
  "message": "API H2 Maroc v3.1"
}
```

### `POST /predict`

Prédiction unique (ML + simulation physique vectorisée).

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ville": "Dakhla",
    "PV_MW": 150,
    "EOL_MW": 80,
    "ELEC_MW": 60,
    "BAT_MWH": 100,
    "technologie": "PEM"
  }'
```

Réponse :

```json
{
  "ville": "Dakhla",
  "LCOH_USD_kg": 4.82,
  "fiabilite_pct": 91.3,
  "LCOH_ml": 4.79,
  "LCOH_pypsa": 4.82,
  "delta_lcoh": -0.03,
  "categorie_LCOH": "Excellent (<5 $/kg)"
}
```

### `POST /predict_comparison`

Prédiction + analyse de sensibilité PV (5 points ×0.6 → ×1.4).

### `GET /toutes_villes`

Compare les 12 villes pour une configuration donnée, triées par LCOH croissant.

```bash
curl "http://localhost:5000/toutes_villes?PV_MW=150&EOL_MW=80&ELEC_MW=60&BAT_MWH=100&technologie=PEM"
```

### `GET /pareto`

Génère le front de Pareto LCOH / Fiabilité pour une ville et une technologie.

### `GET /waterfall/{ville}`

Décomposition waterfall du LCOH par composante (PV CAPEX, EOL CAPEX, ELEC CAPEX, BAT CAPEX, eau).

---

## 🖥 Dashboard Streamlit

```bash
# L'API Flask doit tourner en parallèle sur le port 5000
python api_flask_h2.py &
streamlit run dashboard.py
```

### Onglets disponibles

1. ** Simulation** — Prédiction ML vs PyPSA en temps réel, métriques comparatives
2. **Toutes les villes** — Classement LCOH des 12 sites, carte interactive
3. ** Front de Pareto** — Optimum LCOH/fiabilité, exploration de l'espace de conception
4. ** Décomposition LCOH** — Waterfall par composante de coût
5. ** Validation ML** — Métriques R², MAE, résidus, importance des features, concordance ML/PyPSA par ville

### Design

Thème "Dark Tech" avec accents verts (#00E676), police Rajdhani/JetBrains Mono.  
Toutes les visualisations sont Plotly interactives.

---

##  Installation

```bash
pip install -r requirements_ml.txt
```

**Dépendances spécifiques à ce module :**

```
flask>=3.0
flask-cors>=4.0
scikit-learn>=1.4
joblib>=1.3
numba>=0.59        # optionnel — accélère ×50-200 la boucle batterie
streamlit>=1.32
plotly>=5.20
requests>=2.31
```

---

##  Utilisation complète

### 1. Entraîner le modèle

```bash
# Nécessite ETAPE2.py dans ~/Downloads/pfe/
python train_ml_h2.py
```

Outputs générés dans `~/Downloads/H2Morocco222_Outputs/ml/` :

```
ml/
├── dataset_h2.csv          # Dataset d'entraînement (400 sim × 12 villes)
├── models_h2.pkl           # Modèles sérialisés (Random Forest × 2)
└── validation_report.json  # Métriques de performance + échantillons de prédiction
```

### 2. Démarrer l'API

```bash
python api_flask_h2.py
```

L'API charge automatiquement `models_h2.pkl`. Si le fichier est absent, elle tourne en mode **simulation physique uniquement** (pas de prédiction ML).

**Accélération Numba :** si `numba` est installé, la boucle batterie horaire est compilée JIT (×50–200 sur la vitesse de simulation).

### 3. Lancer le Dashboard

```bash
streamlit run dashboard.py
```

---

## Interprétation des résultats

### Catégories LCOH

| Catégorie | Plage | Interprétation |
|-----------|-------|----------------|
| Excellent | < 5 $/kg | Compétitif à l'export |
| Bon | 5–7 $/kg | Viable avec prime RFNBO |
| Acceptable | 7–10 $/kg | Marché domestique ou subvention |
| Élevé | > 10 $/kg | Non compétitif aujourd'hui |

### Certifications CO₂

- **RFNBO (UE)** : < 3.38 kgCO₂/kgH₂ → éligible contrats UE
- **H₂ vert premium** : < 1.0 kgCO₂/kgH₂ → marchés japonais/coréen

---

## Structure des fichiers de sortie

```
H2Morocco222_Outputs/
└── ml/
    ├── dataset_h2.csv           # Dataset brut (entrée RF)
    ├── models_h2.pkl            # Modèles sauvegardés (joblib)
    ├── validation_report.json   # Métriques + résidus + importance features
    └── rapport_ml_*.png         # Figures de validation (7 graphiques)
```

---

##  Intégration avec le projet principal

Ce module ML est **complémentaire** au projet H2 Morocco principal :

- `ETAPE2.py` génère les simulations PyPSA qui servent de données d'entraînement
- `engine.py` fournit les formules LCOH utilisées dans la simulation physique de l'API
- Les résultats ML viennent enrichir le dashboard principal `app.py`
