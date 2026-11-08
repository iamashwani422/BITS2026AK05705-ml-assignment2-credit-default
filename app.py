"""
Credit Card Default Prediction - Streamlit app (ML Assignment 2)

Loads the 5 pre-trained models from model/, lets the user upload a test CSV
(features X1..X23 + target Y), and shows evaluation metrics, confusion matrix
and classification report for the selected model. No retraining happens here.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

MODEL_DIR = "model"
TARGET = "Y"
FEATURES = [f"X{i}" for i in range(1, 24)]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

FEATURE_INFO = {
    "X1": "Credit limit (NT$)",
    "X2": "Sex (1 = male, 2 = female)",
    "X3": "Education (1 = grad school, 2 = university, 3 = high school, 4 = other)",
    "X4": "Marital status (1 = married, 2 = single, 3 = other)",
    "X5": "Age (years)",
    "X6 - X11": "Repayment status, Sep -> Apr (-1 = paid on time, 1..9 = months late)",
    "X12 - X17": "Bill amount, Sep -> Apr (NT$)",
    "X18 - X23": "Amount paid, Sep -> Apr (NT$)",
    "Y": "Default next month (1 = yes, 0 = no)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    loaded = {}
    for name, file in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, file)
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
    return loaded


def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    return y_pred, compute_metrics(y, y_pred, y_score)


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["No default (0)", "Default (1)"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Credit Card Default Predictor", page_icon="💳", layout="wide")

st.title("💳 Credit Card Default Prediction")
st.caption("ML Assignment 2 · BITS WILP M.Tech AIML · UCI Default of Credit Card Clients dataset")

models = load_models()
if not models:
    st.error("No trained models found in the `model/` folder. Run the training notebook first.")
    st.stop()

# ---- Sidebar: inputs ------------------------------------------------------
with st.sidebar:
    st.header("1. Upload test data")
    uploaded = st.file_uploader("Test CSV (X1..X23 + Y)", type=["csv"])
    use_sample = st.checkbox("Use bundled test_data.csv", value=uploaded is None)

    st.header("2. Choose a model")
    model_name = st.selectbox("Model", list(models.keys()))

    with st.expander("Feature dictionary"):
        for k, v in FEATURE_INFO.items():
            st.markdown(f"**{k}** – {v}")

# ---- Load data -------------------------------------------------------------
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample and os.path.exists("test_data.csv"):
    df = pd.read_csv("test_data.csv")
    source = "test_data.csv (bundled)"
else:
    st.info("👈 Upload a test CSV or tick **Use bundled test_data.csv** in the sidebar to begin.")
    st.stop()

df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()

X = df[FEATURES]
y = df[TARGET].astype(int)

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Defaulters (Y = 1)", f"{int(y.sum()):,}")
c3.metric("Default rate", f"{y.mean():.1%}")
st.caption(f"Data source: {source}")

with st.expander("Preview data"):
    st.dataframe(df.head(20), use_container_width=True)

tab_single, tab_compare = st.tabs([f"🔍 {model_name}", "📊 Compare all models"])

# ---- Single model ----------------------------------------------------------
with tab_single:
    y_pred, m = evaluate(models[model_name], X, y)

    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        st.pyplot(plot_confusion_matrix(cm))
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"Caught {tp:,} of {tp + fn:,} defaulters · {fp:,} false alarms")
    with right:
        st.subheader("Classification report")
        report = classification_report(
            y, y_pred, target_names=["No default", "Default"],
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    out = df.copy()
    out["Predicted"] = y_pred
    st.download_button(
        "⬇️ Download predictions", out.to_csv(index=False).encode(),
        file_name=f"predictions_{MODEL_FILES[model_name].replace('.joblib', '')}.csv",
        mime="text/csv",
    )

# ---- All models ------------------------------------------------------------
with tab_compare:
    rows = []
    for name, mdl in models.items():
        _, mm = evaluate(mdl, X, y)
        rows.append({"ML Model Name": name, **mm})
    table = pd.DataFrame(rows).set_index("ML Model Name").round(4)

    st.dataframe(
        table.style.highlight_max(axis=0, color="#c8e6c9").format("{:.4f}"),
        use_container_width=True,
    )
    metric = st.selectbox("Chart metric", list(table.columns), index=5)
    st.bar_chart(table[metric])
    best = table["MCC"].idxmax()
    st.success(f"Best model by MCC on this data: **{best}** ({table.loc[best, 'MCC']:.4f})")
