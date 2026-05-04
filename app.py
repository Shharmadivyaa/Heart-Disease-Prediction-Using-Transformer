"""
Heart Disease Prediction - Streamlit Web App
============================================
Run with: streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import os

from model import HeartDiseaseTransformer
from data import prepare_data, generate_synthetic_dataset, FEATURE_NAMES, FEATURE_DESCRIPTIONS
from train import train, evaluate
from predict import predict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction | Transformer",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a0f1a 100%); }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .risk-high { color: #ff4b4b; font-size: 2rem; font-weight: bold; }
    .risk-medium { color: #ffa500; font-size: 2rem; font-weight: bold; }
    .risk-low { color: #00cc88; font-size: 2rem; font-weight: bold; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a78bfa;
        border-bottom: 1px solid rgba(167,139,250,0.3);
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "history" not in st.session_state:
    st.session_state.history = None
if "trained" not in st.session_state:
    st.session_state.trained = False


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ Heart Disease\n### Transformer")
    st.markdown("---")

    st.markdown("**Model Architecture**")
    st.caption("Tabular Transformer with CLS token")

    col1, col2 = st.columns(2)
    with col1:
        d_model = st.selectbox("d_model", [32, 64, 128], index=1)
        nhead = st.selectbox("Heads", [2, 4, 8], index=1)
    with col2:
        num_layers = st.selectbox("Layers", [1, 2, 3, 4], index=2)
        epochs = st.slider("Epochs", 10, 100, 60, step=10)

    st.markdown("---")
    train_btn = st.button("🚀 Train Model", use_container_width=True, type="primary")

    if st.session_state.trained:
        st.success("✅ Model trained & ready")

    st.markdown("---")
    st.markdown("**About**")
    st.caption("Built with PyTorch Transformer.\n13 clinical features → binary classification.\nAUC-ROC ~0.91")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# ❤️ Heart Disease Prediction using Transformer")
st.markdown("A deep learning system using **Multi-Head Self-Attention** over tabular clinical features.")
st.markdown("---")

# ── Train ──────────────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner("Training Transformer model... (~1 min)"):
        progress_bar = st.progress(0)
        status = st.empty()

        # Monkey-patch to show progress
        original_train = train

        model, history, scaler = train(
            epochs=epochs,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        )

        progress_bar.progress(100)
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.history = history
        st.session_state.trained = True

    st.success("✅ Training complete! Scroll down to make predictions.")
    st.balloons()


# ── Training Results ───────────────────────────────────────────────────────────
if st.session_state.trained and st.session_state.history:
    st.markdown("## 📊 Training Results")
    history = st.session_state.history

    col1, col2, col3, col4 = st.columns(4)

    best_auc = max(history["val_auc"])
    best_acc = max(history["val_acc"])
    final_loss = history["val_loss"][-1]
    n_epochs = len(history["train_loss"])

    with col1:
        st.metric("Best Val AUC-ROC", f"{best_auc:.4f}", delta="+vs random (0.5)")
    with col2:
        st.metric("Best Val Accuracy", f"{best_acc:.4f}")
    with col3:
        st.metric("Final Val Loss", f"{final_loss:.4f}")
    with col4:
        st.metric("Epochs Trained", n_epochs)

    # Training curves
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.plot(history["train_loss"], color='#a78bfa', label='Train Loss', linewidth=2)
        ax.plot(history["val_loss"], color='#f472b6', label='Val Loss', linewidth=2, linestyle='--')
        ax.set_title("Loss Curves", color='white', fontsize=13)
        ax.set_xlabel("Epoch", color='#aaa')
        ax.set_ylabel("Loss", color='#aaa')
        ax.tick_params(colors='#aaa')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        for spine in ax.spines.values(): spine.set_color('#333')
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.plot(history["val_acc"], color='#34d399', label='Val Accuracy', linewidth=2)
        ax.plot(history["val_auc"], color='#60a5fa', label='Val AUC-ROC', linewidth=2, linestyle='--')
        ax.set_title("Accuracy & AUC", color='white', fontsize=13)
        ax.set_xlabel("Epoch", color='#aaa')
        ax.set_ylabel("Score", color='#aaa')
        ax.tick_params(colors='#aaa')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        ax.set_ylim(0, 1)
        for spine in ax.spines.values(): spine.set_color('#333')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")


# ── Prediction Panel ───────────────────────────────────────────────────────────
st.markdown("## 🩺 Patient Risk Prediction")

if not st.session_state.trained:
    st.info("👈 Click **Train Model** in the sidebar first to enable predictions.")
else:
    # Preset buttons
    st.markdown("**Quick load a patient profile:**")
    p1, p2, p3, p4 = st.columns(4)

    PRESETS = {
        "low":  dict(age=35, sex=0, cp=0, trestbps=115, chol=180, fbs=0, restecg=0, thalach=175, exang=0, oldpeak=0.1, slope=0, ca=0, thal=2),
        "moderate": dict(age=52, sex=1, cp=1, trestbps=140, chol=260, fbs=0, restecg=1, thalach=140, exang=0, oldpeak=1.8, slope=1, ca=1, thal=2),
        "high": dict(age=65, sex=1, cp=3, trestbps=160, chol=310, fbs=1, restecg=2, thalach=100, exang=1, oldpeak=4.0, slope=2, ca=3, thal=3),
    }

    if "preset" not in st.session_state:
        st.session_state.preset = None

    with p1:
        if st.button("🟢 Low Risk", use_container_width=True):
            st.session_state.preset = "low"
    with p2:
        if st.button("🟡 Moderate Risk", use_container_width=True):
            st.session_state.preset = "moderate"
    with p3:
        if st.button("🔴 High Risk", use_container_width=True):
            st.session_state.preset = "high"
    with p4:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.preset = None

    defaults = PRESETS.get(st.session_state.preset, PRESETS["moderate"])

    st.markdown("### Enter Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
        age      = st.slider("Age (years)", 29, 77, defaults["age"])
        sex      = st.radio("Sex", ["Female", "Male"], index=defaults["sex"], horizontal=True)
        sex_val  = 1 if sex == "Male" else 0

    with col2:
        st.markdown('<div class="section-header">Cardiac Measurements</div>', unsafe_allow_html=True)
        cp       = st.selectbox("Chest Pain Type", ["0 – Typical angina", "1 – Atypical angina", "2 – Non-anginal", "3 – Asymptomatic"], index=defaults["cp"])
        cp_val   = int(cp[0])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, defaults["trestbps"])
        chol     = st.slider("Cholesterol (mg/dL)", 126, 564, defaults["chol"])
        thalach  = st.slider("Max Heart Rate Achieved", 71, 202, defaults["thalach"])

    with col3:
        st.markdown('<div class="section-header">Clinical Tests</div>', unsafe_allow_html=True)
        fbs      = st.radio("Fasting Blood Sugar > 120", ["No", "Yes"], index=defaults["fbs"], horizontal=True)
        fbs_val  = 1 if fbs == "Yes" else 0
        restecg  = st.selectbox("Resting ECG", ["0 – Normal", "1 – ST-T Abnormality", "2 – LV Hypertrophy"], index=defaults["restecg"])
        restecg_val = int(restecg[0])
        exang    = st.radio("Exercise Induced Angina", ["No", "Yes"], index=defaults["exang"], horizontal=True)
        exang_val = 1 if exang == "Yes" else 0
        oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.2, float(defaults["oldpeak"]), step=0.1)
        slope    = st.selectbox("ST Slope", ["0 – Upsloping", "1 – Flat", "2 – Downsloping"], index=defaults["slope"])
        slope_val = int(slope[0])
        ca       = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3], index=defaults["ca"])
        thal     = st.selectbox("Thalassemia", ["0 – Normal", "1 – Fixed defect", "2 – Normal (reversible)", "3 – Reversible defect"], index=defaults["thal"])
        thal_val = int(thal[0])

    patient = dict(age=age, sex=sex_val, cp=cp_val, trestbps=trestbps, chol=chol,
                   fbs=fbs_val, restecg=restecg_val, thalach=thalach, exang=exang_val,
                   oldpeak=oldpeak, slope=slope_val, ca=ca, thal=thal_val)

    st.markdown("---")

    if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
        with st.spinner("Running transformer inference..."):
            time.sleep(0.4)
            result = predict(st.session_state.model, st.session_state.scaler, patient)

        prob = result["probability_disease"]
        pct  = int(prob * 100)

        # Result display
        c1, c2, c3 = st.columns([1, 1, 2])

        with c1:
            if result["risk_level"] == "High":
                st.markdown(f'<p class="risk-high">🔴 {pct}%</p>', unsafe_allow_html=True)
                st.error(f"**{result['prediction']}** — High Risk")
            elif result["risk_level"] == "Medium":
                st.markdown(f'<p class="risk-medium">🟡 {pct}%</p>', unsafe_allow_html=True)
                st.warning(f"**{result['prediction']}** — Moderate Risk")
            else:
                st.markdown(f'<p class="risk-low">🟢 {pct}%</p>', unsafe_allow_html=True)
                st.success(f"**{result['prediction']}** — Low Risk")

        with c2:
            st.metric("Disease Probability", f"{prob:.3f}")
            st.metric("Healthy Probability", f"{result['probability_healthy']:.3f}")

        with c3:
            # Attention bar chart
            feat_imp = result["feature_importance"]
            sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
            names  = [FEATURE_DESCRIPTIONS.get(f, f) for f, _ in sorted_feats]
            values = [v for _, v in sorted_feats]

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#16213e')
            colors = ['#ff4b4b' if v > 0.12 else '#a78bfa' if v > 0.07 else '#60a5fa' for v in values]
            bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.6)
            ax.set_title("Attention-based Feature Importance", color='white', fontsize=11)
            ax.set_xlabel("Attention Weight", color='#aaa')
            ax.tick_params(colors='#aaa', labelsize=9)
            for spine in ax.spines.values(): spine.set_color('#333')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown(f"**Top contributing features:** {', '.join(result['top_features'])}")


# ── Architecture Explainer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🏗️ Model Architecture")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ```
    Input: 13 clinical features
           ↓
    Feature Embedding Layer
    (each scalar → 64-dim vector)
           ↓
    Prepend [CLS] Token
           ↓
    ┌─────────────────────────────┐
    │  Transformer Block × 3     │
    │  ┌─────────────────────┐   │
    │  │ Multi-Head Attention │   │
    │  │  (4 heads)          │   │
    │  ├─────────────────────┤   │
    │  │ Feed-Forward (GELU) │   │
    │  │ LayerNorm + Residual│   │
    │  └─────────────────────┘   │
    └─────────────────────────────┘
           ↓
    [CLS] token output
           ↓
    Classification Head
           ↓
    Output: [Healthy | Disease]
    ```
    """)

with col2:
    st.markdown("""
    **Why Transformer for tabular data?**

    - Self-attention learns **feature interactions** (e.g. high ST depression + exercise angina together raise risk more than individually)
    - The **[CLS] token** acts as a global patient summary
    - Attention weights give **natural explainability** — you can see which features the model focuses on
    - **154,082 parameters** — lightweight enough for CPU

    **Training details:**
    - Optimizer: AdamW (lr=3e-4, weight decay=1e-4)
    - Schedule: Cosine Annealing
    - Regularization: Dropout 0.2 + Label Smoothing 0.05
    - Early stopping on validation AUC-ROC
    """)

st.markdown("---")
st.caption("Heart Disease Prediction using Transformer | Built with PyTorch + Streamlit")
