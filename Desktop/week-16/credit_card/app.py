import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from sklearn.neural_network import MLPRegressor
from imblearn.over_sampling import SMOTE

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Fraud Detection Analytics", layout="wide")

st.title("💳 Enterprise Fraud Detection Analytics Dashboard")
st.markdown("ML + Autoencoder + Business Cost Simulation + Executive Insights")
st.markdown("---")

# ================= DATA =================
st.sidebar.header("📁 Dataset")

dataset_option = st.sidebar.radio(
    "Choose Dataset",
    ["Default Synthetic Dataset", "Upload CSV"]
)

if dataset_option == "Default Synthetic Dataset":
    X, y = make_classification(
        n_samples=6000,
        n_features=20,
        n_informative=12,
        weights=[0.94, 0.06],
        random_state=42
    )
    df = pd.DataFrame(X)
    df["Class"] = y
else:
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
    else:
        st.stop()

# ================= PREPROCESS =================
X = df.drop("Class", axis=1)
y = df["Class"]

X = X.select_dtypes(include=[np.number])
X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Handle imbalance
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# ================= RANDOM FOREST =================
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train_sm, y_train_sm)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

# ================= AUTOENCODER =================
X_train_normal = X_train[y_train == 0]

autoencoder = MLPRegressor(
    hidden_layer_sizes=(16, 8, 16),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42
)

autoencoder.fit(X_train_normal, X_train_normal)

reconstructed = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)

# ================= THRESHOLD SLIDER =================
st.subheader("🎚 Autoencoder Threshold Tuning")

threshold = st.slider(
    "Adjust Reconstruction Error Threshold",
    float(mse.min()),
    float(mse.max()),
    float(np.percentile(mse, 95))
)

ae_pred = (mse > threshold).astype(int)

# ================= KPI =================
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("RF Accuracy", f"{accuracy_score(y_test, rf_pred):.3f}")
    st.metric("RF Recall", f"{recall_score(y_test, rf_pred):.3f}")
    st.metric("RF Precision", f"{precision_score(y_test, rf_pred):.3f}")

with col2:
    st.metric("AE Accuracy", f"{accuracy_score(y_test, ae_pred):.3f}")
    st.metric("AE Recall", f"{recall_score(y_test, ae_pred):.3f}")
    st.metric("AE Precision", f"{precision_score(y_test, ae_pred):.3f}")

st.divider()

# ================= ROC CURVE =================
st.subheader("📉 ROC Curve Comparison")

roc_fig = go.Figure()

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
rf_auc = auc(rf_fpr, rf_tpr)

ae_fpr, ae_tpr, _ = roc_curve(y_test, mse)
ae_auc = auc(ae_fpr, ae_tpr)

roc_fig.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr,
                             mode="lines",
                             name=f"Random Forest (AUC={rf_auc:.2f})"))

roc_fig.add_trace(go.Scatter(x=ae_fpr, y=ae_tpr,
                             mode="lines",
                             name=f"Autoencoder (AUC={ae_auc:.2f})"))

roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash"))

st.plotly_chart(roc_fig, use_container_width=True)

# ================= PRECISION-RECALL =================
st.subheader("📈 Precision-Recall Curve (Important for Fraud)")

precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_prob)
ap_rf = average_precision_score(y_test, rf_prob)

precision_ae, recall_ae, _ = precision_recall_curve(y_test, mse)
ap_ae = average_precision_score(y_test, mse)

pr_fig = go.Figure()

pr_fig.add_trace(go.Scatter(x=recall_rf, y=precision_rf,
                            mode='lines',
                            name=f"RF (AP={ap_rf:.2f})"))

pr_fig.add_trace(go.Scatter(x=recall_ae, y=precision_ae,
                            mode='lines',
                            name=f"AE (AP={ap_ae:.2f})"))

st.plotly_chart(pr_fig, use_container_width=True)

# ================= BUSINESS COST SIMULATION =================
st.subheader("💰 Business Cost Simulation")

col1, col2 = st.columns(2)

cost_fn = col1.number_input("Cost of Missing Fraud (False Negative)", value=500)
cost_fp = col2.number_input("Cost of False Alarm (False Positive)", value=50)

cm = confusion_matrix(y_test, rf_pred)
tn, fp, fn, tp = cm.ravel()

total_cost = (fn * cost_fn) + (fp * cost_fp)

st.metric("Estimated Business Loss (RF)", f"${total_cost:,.2f}")

# ================= FEATURE CORRELATION =================
st.subheader("📊 Feature Correlation Heatmap")

corr = df.corr(numeric_only=True)

heatmap = px.imshow(
    corr,
    color_continuous_scale="RdBu_r",
    aspect="auto"
)

st.plotly_chart(heatmap, use_container_width=True)

# ================= MONTHLY FRAUD TREND =================
st.subheader("📅 Monthly Fraud Trend")

date_range = pd.date_range(start="2024-01-01", periods=len(df), freq="H")
df["Date"] = date_range
df["Month"] = df["Date"].dt.to_period("M").astype(str)

monthly = df.groupby("Month")["Class"].sum().reset_index()

trend_fig = px.line(
    monthly,
    x="Month",
    y="Class",
    markers=True,
    title="Monthly Fraud Cases"
)

st.plotly_chart(trend_fig, use_container_width=True)

# ================= EXECUTIVE SUMMARY =================
st.subheader("🏢 Executive Summary")

fraud_rate = (df["Class"].sum() / len(df)) * 100

st.markdown(f"""
### Key Insights:
- Total Transactions: **{len(df):,}**
- Fraud Rate: **{fraud_rate:.2f}%**
- Random Forest Recall: **{recall_score(y_test, rf_pred):.2f}**
- Autoencoder Recall: **{recall_score(y_test, ae_pred):.2f}**
- Estimated Business Loss: **${total_cost:,.2f}**

### Business Recommendation:
- Optimize detection threshold based on business risk tolerance.
- Monitor monthly fraud patterns for trend escalation.
- Use Precision-Recall curve for decision-making under class imbalance.
""")

st.divider()
st.caption("🚀 Fraud Detection Analytics System | Data Analyst Edition")