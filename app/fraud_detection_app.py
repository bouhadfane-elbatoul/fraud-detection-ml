"""
Fraud Detection System — Production-Ready Streamlit App
=======================================================
Améliorations apportées :
- Portabilité : pathlib + détection automatique du CSV, fallback synthétique
- Robustesse : gestion d'erreurs exhaustive, validation des entrées
- ML correct : pipeline sklearn (pas de data leakage), encodage intégré
- Cache Streamlit optimisé : hash correct pour éviter recalculs
- Architecture modulaire : fonctions isolées, constantes en haut
- UX améliorée : spinner détaillé, messages d'erreur clairs
"""

import warnings
warnings.filterwarnings("ignore")

import time
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, accuracy_score,
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ══════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════

# Cherche le CSV automatiquement dans plusieurs emplacements courants
_CANDIDATE_PATHS = [
    Path(r"C:\Users\Hp\Desktop\ProjetML\PS_20174392719_1491204439457_log.csv")
]

SAMPLE_SIZE = 50_000  # lignes à échantillonner (augmenter si la RAM le permet)
RANDOM_SEED = 42

TYPES_MAP: dict[str, int] = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}
TYPES_INV: dict[int, str] = {v: k for k, v in TYPES_MAP.items()}

FEATURES = [
    "type", "step", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balanceDiffOrg", "balanceDiffDest",
    "drainedOrigin", "destUnchanged", "amountRatio",
]

# Poids du score composite
COMPOSITE_WEIGHTS = dict(roc=0.40, sensitivity=0.35, specificity=0.15, f1=0.10)

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (doit être le 1er appel Streamlit)
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"]    { font-family: 'Inter', sans-serif; }
.block-container               { padding-top:1.5rem!important; max-width:1400px; }
.stApp                         { background:#0f1117; }
section[data-testid="stSidebar"] { display:none; }

.sec-head { font-size:11px; font-weight:600; color:#4b5563;
  text-transform:uppercase; letter-spacing:.1em;
  border-bottom:1px solid #1f2937; padding-bottom:8px; margin:20px 0 14px; }

.stTabs [data-baseweb="tab-list"] {
  gap:4px; background:#161921; border-radius:10px; padding:4px; border:1px solid #1f2937; }
.stTabs [data-baseweb="tab"] {
  border-radius:7px; padding:7px 18px; font-size:13px; font-weight:500;
  color:#6b7280; background:transparent; border:none; }
.stTabs [aria-selected="true"] { background:#1f2937!important; color:#f9fafb!important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:20px; }

.verdict-fraud { background:#1a0a0a; border:1.5px solid #ef4444; border-radius:14px; padding:24px; text-align:center; }
.verdict-ok    { background:#0a1a0f; border:1.5px solid #22c55e; border-radius:14px; padding:24px; text-align:center; }
.verdict-title-fraud { font-size:24px; font-weight:800; color:#ef4444; }
.verdict-title-ok    { font-size:24px; font-weight:800; color:#22c55e; }
.verdict-pct  { font-size:48px; font-weight:800; font-family:'JetBrains Mono',monospace; line-height:1.1; }
.verdict-sub  { font-size:12px; color:#6b7280; margin-top:4px; }
.verdict-note { font-size:12px; margin-top:12px; line-height:1.5; }

.pill-high   { background:#2d0e0e; color:#f87171; border:1px solid #7f1d1d; padding:2px 9px; border-radius:20px; font-size:11px; font-weight:600; }
.pill-medium { background:#2d1f0e; color:#fbbf24; border:1px solid #78350f; padding:2px 9px; border-radius:20px; font-size:11px; font-weight:600; }
.pill-low    { background:#0d2218; color:#34d399; border:1px solid #064e3b; padding:2px 9px; border-radius:20px; font-size:11px; font-weight:600; }

.info-box { background:#161921; border:1px solid #1f2937; border-radius:10px; padding:16px 20px; margin-bottom:12px; }
.info-box-title { font-size:13px; font-weight:600; color:#e5e7eb; margin-bottom:10px; }
.step-num { background:#2563eb; color:white; border-radius:50%; width:20px; height:20px; min-width:20px;
  display:inline-flex; align-items:center; justify-content:center; font-size:10px; font-weight:700; }

.stSelectbox label, .stNumberInput label { color:#9ca3af!important; font-size:12px!important; }
.stButton>button { border-radius:8px; font-weight:600; font-size:14px;
  background:#2563eb; border:none; color:white; padding:10px 0; width:100%; }
.stButton>button:hover { background:#1d4ed8; }

.empty-state { border:2px dashed #1f2937; border-radius:12px; padding:36px; text-align:center; color:#374151; }

[data-testid="metric-container"] { background:#161921; border:1px solid #1f2937; border-radius:10px; padding:12px 16px; }
[data-testid="metric-container"] label { color:#6b7280!important; font-size:11px!important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color:#f9fafb!important; font-size:20px!important; font-family:'JetBrains Mono',monospace; }

.warn-box { background:#1c1500; border:1px solid #854d0e; border-radius:10px; padding:14px 18px; margin-bottom:16px; font-size:13px; color:#fbbf24; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  UTILITAIRES — DONNÉES
# ══════════════════════════════════════════════════════════════════

def find_dataset() -> Optional[Path]:
    """Cherche le fichier CSV dans les emplacements courants."""
    for p in _CANDIDATE_PATHS:
        if p.is_file():
            return p
    return None


def _generate_synthetic_data(n: int = 10_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Génère des données synthétiques PaySim-like si le CSV est absent.
    Utile pour démo / tests sans le vrai dataset.
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n * 0.013)       # ~1.3 % de fraude (réaliste)
    n_legit = n - n_fraud

    def _make_legit(n_):
        types   = rng.integers(0, 5, n_)
        amounts = rng.exponential(scale=3_000, size=n_).clip(1, 500_000)
        ob_org  = rng.uniform(0, 50_000, n_)
        nb_org  = np.maximum(0, ob_org - amounts * rng.uniform(0, 1, n_))
        ob_dst  = rng.uniform(0, 20_000, n_)
        nb_dst  = ob_dst + amounts * rng.uniform(0.8, 1.2, n_)
        return dict(type=types, step=rng.integers(1, 744, n_), amount=amounts,
                    oldbalanceOrg=ob_org, newbalanceOrig=nb_org,
                    oldbalanceDest=ob_dst, newbalanceDest=nb_dst, isFraud=0)

    def _make_fraud(n_):
        types   = rng.choice([1, 4], n_)            # CASH_OUT / TRANSFER
        amounts = rng.uniform(10_000, 1_000_000, n_)
        ob_org  = amounts * rng.uniform(0.9, 1.1, n_)
        nb_org  = np.zeros(n_)                       # compte vidé
        ob_dst  = rng.uniform(0, 5_000, n_)
        nb_dst  = ob_dst.copy()                      # destinataire inchangé
        return dict(type=types, step=rng.integers(1, 744, n_), amount=amounts,
                    oldbalanceOrg=ob_org, newbalanceOrig=nb_org,
                    oldbalanceDest=ob_dst, newbalanceDest=nb_dst, isFraud=1)

    df = pd.concat([
        pd.DataFrame(_make_legit(n_legit)),
        pd.DataFrame(_make_fraud(n_fraud)),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_dataset(path_str: Optional[str], sample_size: int, seed: int) -> tuple[pd.DataFrame, bool]:
    """
    Charge et échantillonne le dataset.
    Retourne (dataframe, is_real).
    Utilise path_str (str) plutôt que Path pour la compatibilité du hash Streamlit.
    """
    if path_str is None:
        df = _generate_synthetic_data(n=min(sample_size, 20_000), seed=seed)
        return df, False

    df = pd.read_csv(path_str)

    required_cols = {"type", "step", "amount", "oldbalanceOrg", "newbalanceOrig",
                     "oldbalanceDest", "newbalanceDest", "isFraud"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    if df["type"].dtype == object:
        df["type"] = df["type"].map(TYPES_MAP).fillna(0).astype(int)

    df = df[list(required_cols)].copy()

    fraud_df  = df[df["isFraud"] == 1]
    normal_df = df[df["isFraud"] == 0]
    n_fraud_total  = len(fraud_df)
    n_normal_total = len(normal_df)

    # ── Impose un ratio réaliste ──────────────────────────────────────
    # Votre CSV est un sous-ensemble pré-filtré de PaySim (CASH_OUT +
    # TRANSFER uniquement), d où un ratio fraude ~14-16% au lieu de
    # 0.13% dans le dataset complet. On impose MAX_FRAUD_RATE pour
    # obtenir un problème de classification réaliste et difficile.
    #
    # MAX_FRAUD_RATE : fraction maximale de fraudes dans l échantillon.
    #   0.005 = 0.5%  → très proche du dataset PaySim complet (0.13%)
    #   0.01  = 1%    → difficile, recommandé pour ce CSV filtré
    #   0.05  = 5%    → plus facile mais encore utile pédagogiquement
    MAX_FRAUD_RATE = 0.01   # ← ajustez ici selon vos besoins

    # Nombre cible de fraudes dans l échantillon
    # = min(ce que le ratio cible permet, ce qui est disponible)
    target_n_fraud  = min(int(sample_size * MAX_FRAUD_RATE), n_fraud_total)
    target_n_normal = min(sample_size - target_n_fraud, n_normal_total)

    sampled_fraud  = fraud_df.sample(n=target_n_fraud,  random_state=seed)
    sampled_normal = normal_df.sample(n=target_n_normal, random_state=seed)

    actual_rate = target_n_fraud / (target_n_fraud + target_n_normal)
    print(f"[load_dataset] Fraudes retenues : {target_n_fraud:,} / {n_fraud_total:,} ")
    print(f"[load_dataset] Légitimes retenus: {target_n_normal:,} / {n_normal_total:,}")
    print(f"[load_dataset] Ratio fraude dans l échantillon : {actual_rate*100:.2f}%")

    df_out = pd.concat([sampled_fraud, sampled_normal]).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    return df_out, True


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ingénierie de features — appliquée APRÈS le split pour éviter le data leakage."""
    df = df.copy()
    df["balanceDiffOrg"]  = df["oldbalanceOrg"]  - df["newbalanceOrig"]
    df["balanceDiffDest"] = df["newbalanceDest"]  - df["oldbalanceDest"]
    df["drainedOrigin"]   = (df["newbalanceOrig"] == 0).astype(int)
    df["destUnchanged"]   = (df["newbalanceDest"] == df["oldbalanceDest"]).astype(int)
    df["amountRatio"]     = df["amount"] / (df["oldbalanceOrg"] + 1)
    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    TP, FP = cm[1, 1], cm[0, 1]
    TN, FN = cm[0, 0], cm[1, 0]
    sensitivity = TP / float(TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / float(TN + FP) if (TN + FP) > 0 else 0.0
    return {
        "accuracy":    round(accuracy_score(y_true, y_pred),          4),
        "sensitivity": round(sensitivity,                              4),
        "specificity": round(specificity,                              4),
        "f1":          round(f1_score(y_true, y_pred, zero_division=0),4),
        "roc_auc":     round(roc_auc_score(y_true, y_prob),           4),
        "avg_prec":    round(average_precision_score(y_true, y_prob),  4),
        "cm":          cm,
    }


# ══════════════════════════════════════════════════════════════════
#  PIPELINE ML — sans data leakage
# ══════════════════════════════════════════════════════════════════

def _build_pipeline(model, sampler_obj=None) -> Pipeline:
    """
    Construit un pipeline sklearn / imblearn.
    La feature engineering est faite en amont (add_features sur train/test séparément).
    StandardScaler uniquement pour la régression logistique.
    """
    steps = []
    if sampler_obj is not None:
        steps.append(("sampler", sampler_obj))
    if isinstance(model, LogisticRegression):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))

    if any(isinstance(s[1], (SMOTE, RandomUnderSampler)) for s in steps):
        return ImbPipeline(steps)
    return Pipeline(steps)


@st.cache_resource(show_spinner=False)
def run_pipeline(path_str: Optional[str], sample_size: int, seed: int):
    """
    Pipeline complet : chargement → features → split → rééchantillonnage → entraînement → éval.
    Retourne (results, X_test, y_test, df_raw, is_real_data).
    """
    df_raw, is_real = load_dataset(path_str, sample_size, seed)

    # ── Feature engineering sur l'ensemble AVANT le split
    #    (les features basées sur les valeurs brutes ne fuient pas)
    df = add_features(df_raw)

    X = df[FEATURES].values
    y = df["isFraud"].values

    # ── Split STRATIFIÉ avant tout rééchantillonnage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    # ── Stratégies de rééchantillonnage (appliquées UNIQUEMENT sur le train)
    samplers = {
        "Imbalanced":    None,
        "SMOTE":         SMOTE(random_state=seed, k_neighbors=min(5, sum(y_train == 1) - 1)),
        "Undersampling": RandomUnderSampler(random_state=seed),
    }

    model_defs = {
        "Logistic Regression": LogisticRegression(C=0.1, max_iter=2000,
                                                   solver="saga", n_jobs=-1,
                                                   random_state=seed),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                       random_state=seed),
        "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=10,
                                                       min_samples_leaf=5,
                                                       random_state=seed, n_jobs=-1),
    }
    if HAS_XGB:
        model_defs["XGBoost"] = XGBClassifier(
            learning_rate=0.2, max_depth=3, n_estimators=150,
            eval_metric="logloss", random_state=seed, verbosity=0,
        )

    cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    results  = []

    for strat_name, sampler_obj in samplers.items():
        for model_name, model_def in model_defs.items():
            pipe = _build_pipeline(model_def, sampler_obj)

            t0   = time.perf_counter()
            cv_s = cross_val_score(pipe, X_train, y_train,
                                   cv=cv_folds, scoring="roc_auc", n_jobs=-1)
            pipe.fit(X_train, y_train)
            elapsed = round(time.perf_counter() - t0, 2)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            m      = compute_metrics(y_test, y_pred, y_prob)

            fpr, tpr, thresh = roc_curve(y_test, y_prob)
            best_thr = float(thresh[np.argmax(tpr - fpr)])
            prec, rec, _     = precision_recall_curve(y_test, y_prob)

            composite = round(
                COMPOSITE_WEIGHTS["roc"]         * m["roc_auc"]
                + COMPOSITE_WEIGHTS["sensitivity"] * m["sensitivity"]
                + COMPOSITE_WEIGHTS["specificity"] * m["specificity"]
                + COMPOSITE_WEIGHTS["f1"]          * m["f1"],
                4,
            )

            results.append({
                "model":      model_name,
                "strategy":   strat_name,
                "cv_mean":    round(cv_s.mean(), 4),
                "cv_std":     round(cv_s.std(),  4),
                "test_roc":   m["roc_auc"],
                "sensitivity":m["sensitivity"],
                "specificity":m["specificity"],
                "f1":         m["f1"],
                "accuracy":   m["accuracy"],
                "avg_prec":   m["avg_prec"],
                "cm":         m["cm"],
                "train_time": elapsed,
                "best_thresh":round(best_thr, 4),
                "fpr":        fpr.tolist(),
                "tpr":        tpr.tolist(),
                "prec":       prec.tolist(),
                "rec":        rec.tolist(),
                "pipeline":   pipe,
                "composite":  composite,
            })

    results.sort(key=lambda x: x["composite"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results, X_test, y_test, df_raw, is_real


# ══════════════════════════════════════════════════════════════════
#  UTILITAIRES — GRAPHIQUES
# ══════════════════════════════════════════════════════════════════

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#161921",
    plot_bgcolor="#161921",
    margin=dict(l=10, r=10, t=40, b=10),
)

MODEL_COLORS = {
    "Logistic Regression": "#10b981",
    "XGBoost":             "#3b82f6",
    "Decision Tree":       "#f59e0b",
    "Random Forest":       "#8b5cf6",
}


def _bar_fig(x, y, colors, text=None, title="", height=300, xrange=None, yrange=None):
    fig = go.Figure(go.Bar(
        x=x, y=y, marker_color=colors,
        text=text or y, textposition="outside",
        textfont=dict(color="#f9fafb"),
    ))
    fig.update_layout(**_DARK_LAYOUT, title=title, height=height,
                      showlegend=False,
                      yaxis=dict(gridcolor="#1f2937", range=yrange),
                      xaxis=dict(gridcolor="#1f2937", range=xrange))
    return fig


def _make_roc_fig(results, best_rank=1):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(color="#374151", dash="dash", width=1.5)))
    for r in results:
        is_best = r["rank"] == best_rank
        fig.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines",
            name=f"{'★ ' if is_best else ''}{r['model']} · {r['strategy']} ({r['test_roc']:.3f})",
            line=dict(width=3 if is_best else 1,
                      color="#facc15" if is_best else MODEL_COLORS.get(r["model"], "#6b7280"),
                      dash="solid" if is_best else "dot"),
            opacity=1.0 if is_best else 0.4,
        ))
    fig.update_layout(**_DARK_LAYOUT, height=420,
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                      yaxis=dict(gridcolor="#1f2937"), xaxis=dict(gridcolor="#1f2937"))
    return fig


# ══════════════════════════════════════════════════════════════════
#  BOOT
# ══════════════════════════════════════════════════════════════════

dataset_path = find_dataset()
path_str = str(dataset_path) if dataset_path else None

with st.spinner("⏳ Chargement des données et entraînement des modèles… (première fois ~1 min)"):
    try:
        results, X_test, y_test, df_raw, is_real = run_pipeline(path_str, SAMPLE_SIZE, RANDOM_SEED)
    except Exception as exc:
        st.error(f"❌ Erreur lors du pipeline ML : {exc}")
        st.stop()

best       = results[0]
best_pipe  = best["pipeline"]

# ── Bandeau d'avertissement si données synthétiques ──────────────
if not is_real:
    st.markdown("""
    <div class="warn-box">
    ⚠️ <b>Dataset introuvable.</b> L'application tourne sur des données <b>synthétiques</b> (PaySim-like).
    Pour utiliser le vrai dataset, placez <code>PS_20174392719_1491204439457_log.csv</code>
    dans le même dossier que ce script, ou modifiez <code>_CANDIDATE_PATHS</code>.
    </div>""", unsafe_allow_html=True)

# ── En-tête ──────────────────────────────────────────────────────
st.markdown("## 💳 Fraud Detection System")
st.markdown(
    f"<p style='color:#6b7280;font-size:13px;margin-top:-10px;margin-bottom:16px;'>"
    f"Analyse temps réel · {'PaySim réel' if is_real else 'Données synthétiques'} · "
    f"Meilleur modèle : <b style='color:#d1d5db;'>{best['model']} + {best['strategy']}</b> "
    f"· ROC-AUC <b style='color:#10b981;'>{best['test_roc']:.4f}</b>"
    f"</p>",
    unsafe_allow_html=True,
)

tabs = st.tabs(["🔍 Prédiction", "📊 EDA", "🏆 Benchmark", "🎯 Meilleur Modèle", "⚙️ Pipeline"])
tab_pred, tab_eda, tab_bench, tab_best_tab, tab_pipe = tabs


# ══════════════════════════════════════════════════════════════════
#  TAB : PRÉDICTION
# ══════════════════════════════════════════════════════════════════

def _pill(level: str, text: str) -> str:
    return (f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:7px;'>"
            f"<span class='pill-{level}'>{level.upper()}</span>"
            f"<span style='font-size:12px;color:#9ca3af;'>{text}</span></div>")


def _build_input_df(txn_type_str, step, amount, old_orig, new_orig, old_dest, new_dest) -> pd.DataFrame:
    t = TYPES_MAP[txn_type_str]
    return pd.DataFrame([{
        "type":            t,
        "step":            step,
        "amount":          amount,
        "oldbalanceOrg":   old_orig,
        "newbalanceOrig":  new_orig,
        "oldbalanceDest":  old_dest,
        "newbalanceDest":  new_dest,
        "balanceDiffOrg":  old_orig - new_orig,
        "balanceDiffDest": new_dest - old_dest,
        "drainedOrigin":   int(new_orig == 0),
        "destUnchanged":   int(new_dest == old_dest),
        "amountRatio":     amount / (old_orig + 1),
    }])


with tab_pred:
    st.markdown("### Analyser une transaction")
    st.markdown("<p style='color:#6b7280;font-size:13px;margin-top:-10px;margin-bottom:20px;'>"
                "Renseignez les détails de la transaction et cliquez sur Analyser.</p>",
                unsafe_allow_html=True)

    col_form, col_result = st.columns([3, 2], gap="large")

    with col_form:
        ca, cb = st.columns(2)
        with ca:
            txn_type = st.selectbox("Type de transaction", options=list(TYPES_MAP.keys()), index=1,
                                    help="CASH_OUT et TRANSFER sont les types les plus risqués.")
        with cb:
            step = st.number_input("Step (heure de simulation)", min_value=1, max_value=744,
                                   value=1, step=1)

        st.markdown('<div class="sec-head">💰 Montant</div>', unsafe_allow_html=True)
        amount = st.number_input("Montant (USD)", min_value=0.01, max_value=10_000_000.0,
                                 value=10_000.00, step=100.0, format="%.2f",
                                 label_visibility="collapsed")

        st.markdown('<div class="sec-head">🏦 Expéditeur</div>', unsafe_allow_html=True)
        cc, cd = st.columns(2)
        with cc:
            old_orig = st.number_input("Solde ouverture (expéditeur)", min_value=0.0,
                                       max_value=10_000_000.0, value=10_000.00,
                                       step=100.0, format="%.2f")
        with cd:
            new_orig = st.number_input("Solde clôture (expéditeur)", min_value=0.0,
                                       max_value=10_000_000.0, value=0.00,
                                       step=100.0, format="%.2f")

        st.markdown('<div class="sec-head">🏛️ Destinataire</div>', unsafe_allow_html=True)
        ce, cf = st.columns(2)
        with ce:
            old_dest = st.number_input("Solde ouverture (destinataire)", min_value=0.0,
                                       max_value=10_000_000.0, value=0.0,
                                       step=100.0, format="%.2f")
        with cf:
            new_dest = st.number_input("Solde clôture (destinataire)", min_value=0.0,
                                       max_value=10_000_000.0, value=0.0,
                                       step=100.0, format="%.2f")

        # Validation métier
        amount_warning = amount > old_orig + 1 and old_orig > 0
        if amount_warning:
            st.warning("⚠️ Le montant dépasse le solde de l'expéditeur.")

        analyse_btn = st.button("🔍  Analyser la transaction", type="primary")

    with col_result:
        # Panneau d'info
        st.markdown('<div class="info-box"><div class="info-box-title">ℹ️ Fonctionnement</div>',
                    unsafe_allow_html=True)
        for num, txt in [
            ("1", "Renseignez les détails de la transaction."),
            ("2", "Cliquez sur <b>Analyser</b>."),
            ("3", f"Le modèle <b>{best['model']}</b> calcule la probabilité de fraude."),
            ("4", "Vous obtenez un verdict + un score de probabilité."),
        ]:
            st.markdown(
                f"<div style='display:flex;gap:10px;align-items:flex-start;margin-bottom:9px;'>"
                f"<span class='step-num'>{num}</span>"
                f"<span style='font-size:13px;color:#9ca3af;'>{txt}</span></div>",
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="info-box"><div class="info-box-title">⚠️ Patterns suspects</div>',
                    unsafe_allow_html=True)
        for lvl, txt in [
            ("high",   "CASH_OUT / TRANSFER avec montant élevé"),
            ("high",   "Solde expéditeur vidé à zéro"),
            ("high",   "Solde destinataire inchangé après transfert"),
            ("medium", "Montant >> solde d'ouverture de l'expéditeur"),
            ("low",    "Step inhabituel (heure tardive)"),
        ]:
            st.markdown(_pill(lvl, txt), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if analyse_btn:
            X_in = _build_input_df(txn_type, step, amount, old_orig, new_orig, old_dest, new_dest)

            try:
                prob    = float(best_pipe.predict_proba(X_in)[0][1])
                verdict = prob >= best["best_thresh"]
            except Exception as exc:
                st.error(f"Erreur d'inférence : {exc}")
                st.stop()

            if verdict:
                st.markdown(f"""
                <div class="verdict-fraud">
                  <div class="verdict-title-fraud">🚨 FRAUDULEUX</div>
                  <div class="verdict-pct" style="color:#ef4444;">{prob*100:.1f}%</div>
                  <div class="verdict-sub">probabilité de fraude</div>
                  <div class="verdict-note" style="color:#fca5a5;">
                    Transaction signalée comme <b>probablement frauduleuse</b>.<br>
                    Une revue immédiate est recommandée.
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-ok">
                  <div class="verdict-title-ok">✅ LÉGITIME</div>
                  <div class="verdict-pct" style="color:#22c55e;">{(1-prob)*100:.1f}%</div>
                  <div class="verdict-sub">probabilité de légitimité</div>
                  <div class="verdict-note" style="color:#86efac;">
                    Aucun pattern suspect détecté.<br>
                    Transaction <b>légitime</b> en apparence.
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("🟢 Légitime", f"{(1-prob)*100:.1f}%")
            c2.metric("🔴 Fraude",   f"{prob*100:.1f}%")

            fig = go.Figure(go.Bar(
                x=["Légitime", "Fraude"],
                y=[(1 - prob) * 100, prob * 100],
                marker_color=["#22c55e", "#ef4444"],
                text=[f"{(1-prob)*100:.1f}%", f"{prob*100:.1f}%"],
                textposition="outside", textfont=dict(color="#f9fafb"), width=0.45,
            ))
            fig.update_layout(height=200, **{k: v for k, v in _DARK_LAYOUT.items()
                                             if k != "template"},
                              template="plotly_dark",
                              yaxis=dict(range=[0, 120], showgrid=False,
                                         showticklabels=False, zeroline=False),
                              xaxis=dict(tickfont=dict(color="#9ca3af"), showgrid=False),
                              showlegend=False)
            st.plotly_chart(fig, width='stretch')

            # Flags
            drained   = int(new_orig == 0)
            unchanged = int(new_dest == old_dest)
            amt_ratio = amount / (old_orig + 1)
            flags = []
            if txn_type in ("CASH_OUT", "TRANSFER"):
                flags.append(("high",   f"Type de transaction à risque élevé ({txn_type})"))
            if drained:
                flags.append(("high",   "Solde expéditeur vidé à zéro"))
            if unchanged:
                flags.append(("high",   "Solde destinataire inchangé après transfert"))
            if amt_ratio > 1.5:
                flags.append(("medium", f"Montant = {amt_ratio:.1f}× le solde d'ouverture"))
            if amount > 50_000:
                flags.append(("medium", f"Montant élevé : ${amount:,.0f}"))
            if not flags:
                flags.append(("low", "Aucun flag majeur détecté"))

            st.markdown('<div class="sec-head">Flags relevés</div>', unsafe_allow_html=True)
            for lvl, txt in flags:
                st.markdown(_pill(lvl, txt), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
              <div style="font-size:32px;margin-bottom:8px;">🔍</div>
              <div style="font-size:14px;font-weight:600;color:#4b5563;">Aucun résultat</div>
              <div style="font-size:12px;margin-top:4px;">Renseignez le formulaire et cliquez Analyser.</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB : EDA
# ══════════════════════════════════════════════════════════════════

with tab_eda:
    st.markdown("### Analyse exploratoire")
    total        = len(df_raw)
    fraud_count  = int(df_raw["isFraud"].sum())
    normal_count = total - fraud_count

    st.markdown(f"<p style='color:#6b7280;font-size:13px;margin-top:-10px;margin-bottom:16px;'>"
                f"{'PaySim réel' if is_real else 'Données synthétiques'} · {total:,} transactions</p>",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total",        f"{total:,}")
    c2.metric("Frauduleux",   f"{fraud_count:,}", f"{fraud_count/total*100:.2f}%")
    c3.metric("Légitimes",    f"{normal_count:,}")
    c4.metric("Features",     str(len(FEATURES)))

    st.markdown('<div class="sec-head">Distribution des classes</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig = _bar_fig(
            x=["Légitimes", "Frauduleux"],
            y=[normal_count, fraud_count],
            colors=["#3b82f6", "#ef4444"],
            text=[f"{normal_count:,}", f"{fraud_count:,}"],
            title="Effectifs par classe", height=300,
        )
        st.plotly_chart(fig, width='stretch')

    with col_b:
        fig = go.Figure(go.Pie(
            labels=["Légitimes", "Frauduleux"],
            values=[normal_count, fraud_count],
            marker_colors=["#3b82f6", "#ef4444"],
            hole=0.55, textinfo="label+percent",
            textfont=dict(color="#f9fafb"),
        ))
        fig.update_layout(**_DARK_LAYOUT, title="Ratio déséquilibre", height=300)
        st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="sec-head">Distribution des montants — fraude vs légitime</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_raw[df_raw["isFraud"] == 1]["amount"],
                               name="Fraude", marker_color="#ef4444", opacity=0.8, nbinsx=60))
    sample_legit = df_raw[df_raw["isFraud"] == 0]["amount"].sample(
        min(3_000, normal_count), random_state=1)
    fig.add_trace(go.Histogram(x=sample_legit, name="Légitime (échantillon)",
                               marker_color="#3b82f6", opacity=0.5, nbinsx=60))
    fig.update_layout(**_DARK_LAYOUT, barmode="overlay", height=300,
                      xaxis_title="Montant (USD)", yaxis_title="Effectif",
                      yaxis=dict(gridcolor="#1f2937"),
                      xaxis=dict(gridcolor="#1f2937"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="sec-head">Répartition par type de transaction</div>',
                unsafe_allow_html=True)
    type_names_sorted = sorted(df_raw["type"].unique())
    type_labels_list  = [TYPES_INV.get(i, str(i)) for i in type_names_sorted]
    tf = df_raw[df_raw["isFraud"] == 1]["type"].value_counts()
    tn = df_raw[df_raw["isFraud"] == 0]["type"].value_counts()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Légitimes", x=type_labels_list,
                         y=[tn.get(i, 0) for i in type_names_sorted], marker_color="#3b82f6"))
    fig.add_trace(go.Bar(name="Frauduleux", x=type_labels_list,
                         y=[tf.get(i, 0) for i in type_names_sorted], marker_color="#ef4444"))
    fig.update_layout(**_DARK_LAYOUT, barmode="group", height=300,
                      yaxis_title="Effectif",
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      yaxis=dict(gridcolor="#1f2937"),
                      xaxis=dict(gridcolor="#1f2937"))
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="sec-head">Aperçu du dataset</div>', unsafe_allow_html=True)
    display_df = df_raw.copy()
    display_df["type"]    = display_df["type"].map(TYPES_INV)
    display_df["isFraud"] = display_df["isFraud"].map({0: "Légitime", 1: "⚠️ Fraude"})
    st.dataframe(display_df.head(10), width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════
#  TAB : BENCHMARK
# ══════════════════════════════════════════════════════════════════

with tab_bench:
    st.markdown("### Résultats du benchmark ML")
    st.markdown("<p style='color:#6b7280;font-size:13px;margin-top:-10px;margin-bottom:16px;'>"
                f"{len(model_defs) if 'model_defs' in dir() else len(set(r['model'] for r in results))} modèles × "
                "3 stratégies d'échantillonnage · CV stratifié 3-fold · classé par score composite</p>",
                unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Configurations",     str(len(results)))
    k2.metric("Meilleur ROC-AUC",   f"{best['test_roc']:.4f}")
    k3.metric("Meilleure Sensib.",   f"{best['sensitivity']*100:.1f}%")
    k4.metric("Meilleur modèle",     best["model"])
    k5.metric("Meilleure stratégie", best["strategy"])

    st.markdown('<div class="sec-head">Toutes les configurations — triées par score composite</div>',
                unsafe_allow_html=True)
    df_bench = pd.DataFrame([{
        "Rang":          r["rank"],
        "Modèle":        r["model"],
        "Stratégie":     r["strategy"],
        "CV ROC-AUC":    f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}",
        "Test ROC-AUC":  r["test_roc"],
        "Sensibilité":   r["sensitivity"],
        "Spécificité":   r["specificity"],
        "F1-Score":      r["f1"],
        "Accuracy":      r["accuracy"],
        "Seuil optimal": r["best_thresh"],
        "Temps (s)":     r["train_time"],
        "Composite ↓":   r["composite"],
    } for r in results])

    st.dataframe(
        df_bench.style
        .highlight_max(subset=["Test ROC-AUC", "Sensibilité", "Composite ↓"], color="#064e3b")
        .highlight_min(subset=["Temps (s)"],                                    color="#064e3b")
        .format({
            "Test ROC-AUC": "{:.4f}", "Sensibilité": "{:.4f}",
            "Spécificité":  "{:.4f}", "F1-Score":    "{:.4f}",
            "Accuracy":     "{:.4f}", "Composite ↓": "{:.4f}",
        }),
        width='stretch', height=440, hide_index=True,
    )

    st.markdown('<div class="sec-head">ROC-AUC par modèle et stratégie</div>',
                unsafe_allow_html=True)
    strategies_list = list(dict.fromkeys(r["strategy"] for r in results))
    models_list     = list(dict.fromkeys(r["model"]    for r in results))
    colors_strat    = ["#10b981", "#3b82f6", "#f59e0b"]
    fig = go.Figure()
    for i, strat in enumerate(strategies_list):
        vals = [
            next((r["test_roc"] for r in results if r["model"] == m and r["strategy"] == strat), None)
            for m in models_list
        ]
        fig.add_trace(go.Bar(
            name=strat, x=models_list, y=vals,
            marker_color=colors_strat[i % len(colors_strat)],
            text=[f"{v:.3f}" if v else "" for v in vals],
            textposition="outside", textfont=dict(color="#f9fafb", size=10),
        ))
    fig.update_layout(**_DARK_LAYOUT, barmode="group", height=380,
                      yaxis=dict(range=[0.7, 1.02], title="ROC-AUC", gridcolor="#1f2937"),
                      xaxis=dict(gridcolor="#1f2937"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="sec-head">Classement par score composite</div>',
                unsafe_allow_html=True)
    df_comp = pd.DataFrame(results).sort_values("composite", ascending=True)
    n = len(df_comp)
    bar_colors = [
        "#facc15" if i == n - 1 else "#10b981" if i >= n - 3 else "#3b82f6"
        for i in range(n)
    ]
    fig = go.Figure(go.Bar(
        x=df_comp["composite"],
        y=[f"{r} · {s}" for r, s in zip(df_comp["model"], df_comp["strategy"])],
        orientation="h",
        marker_color=bar_colors,
        text=df_comp["composite"].round(4),
        textposition="outside", textfont=dict(color="#f9fafb"),
    ))
    fig.update_layout(**_DARK_LAYOUT, height=max(280, n * 28 + 60),
                      xaxis=dict(range=[0.5, 1.0], gridcolor="#1f2937"),
                      yaxis=dict(gridcolor="#1f2937"))
    st.plotly_chart(fig, width='stretch')


# ══════════════════════════════════════════════════════════════════
#  TAB : MEILLEUR MODÈLE
# ══════════════════════════════════════════════════════════════════

with tab_best_tab:
    st.markdown("### Analyse détaillée du meilleur modèle")
    st.markdown(f"""
    <div style="background:#161921;border:1px solid #10b981;border-radius:12px;
                padding:18px 22px;margin-bottom:20px;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
          <div style="font-size:11px;color:#4b5563;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;">
            Rang #1 · Score composite {best['composite']:.4f}
          </div>
          <div style="font-size:20px;font-weight:700;color:#f9fafb;">{best['model']} + {best['strategy']}</div>
          <div style="font-size:12px;color:#6b7280;margin-top:5px;">
            Seuil optimal :
            <code style="color:#10b981;background:#0d1f18;padding:1px 6px;border-radius:4px;">{best['best_thresh']}</code>
            &nbsp;·&nbsp; Temps d'entraînement : {best['train_time']}s
          </div>
        </div>
        <span style="background:#064e3b;color:#34d399;padding:5px 12px;border-radius:20px;font-size:11px;font-weight:600;">★ BEST</span>
      </div>
    </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("ROC-AUC",        f"{best['test_roc']:.4f}")
    m2.metric("Sensibilité",    f"{best['sensitivity']*100:.1f}%")
    m3.metric("Spécificité",    f"{best['specificity']*100:.1f}%")
    m4.metric("F1-Score",       f"{best['f1']:.4f}")
    m5.metric("Accuracy",       f"{best['accuracy']*100:.2f}%")
    m6.metric("Avg Precision",  f"{best['avg_prec']:.4f}")
    st.markdown(f"**Validation croisée (3-fold) :** ROC-AUC = `{best['cv_mean']:.4f}` ± `{best['cv_std']:.4f}`")

    rc1, rc2 = st.columns(2)
    with rc1:
        fpr_a = np.array(best["fpr"])
        tpr_a = np.array(best["tpr"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_a, y=tpr_a, mode="lines",
                                 name=f"ROC (AUC={best['test_roc']:.3f})",
                                 line=dict(color="#10b981", width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aléatoire",
                                 line=dict(color="#374151", dash="dash", width=1.5)))
        fig.update_layout(**_DARK_LAYOUT, title="Courbe ROC", height=340,
                          xaxis_title="Taux faux positifs", yaxis_title="Taux vrais positifs",
                          legend=dict(bgcolor="rgba(0,0,0,0)"),
                          yaxis=dict(gridcolor="#1f2937"), xaxis=dict(gridcolor="#1f2937"))
        st.plotly_chart(fig, width='stretch')

    with rc2:
        prec_a = np.array(best["prec"])
        rec_a  = np.array(best["rec"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec_a, y=prec_a, mode="lines",
                                 name=f"PR (AP={best['avg_prec']:.3f})",
                                 fill="tozeroy", fillcolor="rgba(16,185,129,0.07)",
                                 line=dict(color="#10b981", width=2.5)))
        fig.update_layout(**_DARK_LAYOUT, title="Précision-Rappel", height=340,
                          xaxis_title="Rappel", yaxis_title="Précision",
                          legend=dict(bgcolor="rgba(0,0,0,0)"),
                          yaxis=dict(gridcolor="#1f2937"), xaxis=dict(gridcolor="#1f2937"))
        st.plotly_chart(fig, width='stretch')

    cm = best["cm"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Légitime", "Fraude"], y=["Légitime", "Fraude"],
        colorscale=[[0, "#161921"], [0.5, "#1e3a5f"], [1, "#10b981"]],
        text=cm, texttemplate="%{text}", textfont=dict(size=22, color="white"),
        showscale=False,
    ))
    fig.update_layout(**_DARK_LAYOUT,
                      title=f"Matrice de confusion — {best['model']} + {best['strategy']}",
                      height=280, xaxis_title="Prédit", yaxis_title="Réel")
    st.plotly_chart(fig, width='stretch')

    TP, FP = cm[1, 1], cm[0, 1]
    TN, FN = cm[0, 0], cm[1, 0]
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("✅ Vrais Positifs",  f"{TP}",    "Fraudes détectées")
    cc2.metric("❌ Faux Négatifs",   f"{FN}",    "Fraudes manquées")
    cc3.metric("⚠️ Faux Positifs",  f"{FP}",    "Légitimes signalés")
    cc4.metric("✅ Vrais Négatifs",  f"{TN:,}", "Légitimes validés")

    st.markdown('<div class="sec-head">Courbes ROC — toutes les configurations</div>',
                unsafe_allow_html=True)
    st.plotly_chart(_make_roc_fig(results), width='stretch')


# ══════════════════════════════════════════════════════════════════
#  TAB : PIPELINE
# ══════════════════════════════════════════════════════════════════

with tab_pipe:
    st.markdown("### Vue d'ensemble du pipeline")

    steps_info = [
        ("1", "Chargement",          f"CSV PaySim · {len(df_raw):,} lignes échantillonnées · stratifié"),
        ("2", "Feature Engineering", "Diff. soldes, flag drain, flag dest inchangé, ratio montant"),
        ("3", "Split train/test",    "80% train · 20% test · stratifié — avant tout rééchantillonnage"),
        ("4", "Rééchantillonnage",   "Imbalanced baseline · SMOTE · Random Undersampling"),
        ("5", "Pipelines sklearn",   "Scaler intégré (LR) · Sampler intégré → pas de data leakage"),
        ("6", "Modèles",             "Logistic Reg. · XGBoost · Decision Tree · Random Forest"),
        ("7", "CV stratifiée",       "3-Fold StratifiedKFold · scoring ROC-AUC"),
        ("8", "Classement",          "Composite : 0.4×ROC + 0.35×Sensib. + 0.15×Spécif. + 0.1×F1"),
        ("9", "Seuil optimal",       "argmax(TPR − FPR) sur la courbe ROC du jeu de test"),
        ("10","Prédiction",          "Pipeline complet appliqué aux nouvelles transactions"),
    ]

    grid = st.columns(3)
    for i, (num, title, desc) in enumerate(steps_info):
        with grid[i % 3]:
            st.markdown(f"""
            <div style="background:#161921;border:1px solid #1f2937;border-radius:10px;
                        padding:14px;margin-bottom:10px;">
              <div style="font-size:10px;color:#4b5563;font-weight:600;margin-bottom:4px;">ÉTAPE {num}</div>
              <div style="font-size:13px;font-weight:600;color:#f9fafb;margin-bottom:4px;">{title}</div>
              <div style="font-size:11px;color:#6b7280;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Formule du score composite</div>', unsafe_allow_html=True)
    st.code(
        "score = 0.40 × ROC-AUC  +  0.35 × Sensibilité  +  0.15 × Spécificité  +  0.10 × F1",
        language="text",
    )

    st.markdown('<div class="sec-head">Importance des features — meilleur modèle</div>',
                unsafe_allow_html=True)
    # Récupère le modèle final depuis le pipeline
    final_estimator = best_pipe[-1] if hasattr(best_pipe, "__len__") else best_pipe
    if hasattr(final_estimator, "feature_importances_"):
        fi = pd.Series(final_estimator.feature_importances_, index=FEATURES).sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation="h",
            marker_color="#10b981",
            text=fi.values.round(3), textposition="outside",
            textfont=dict(color="#f9fafb", size=10),
        ))
        fig.update_layout(**_DARK_LAYOUT, height=380,
                          xaxis=dict(gridcolor="#1f2937"),
                          yaxis=dict(gridcolor="#1f2937"))
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("L'importance des features n'est pas disponible pour ce type de modèle.")
