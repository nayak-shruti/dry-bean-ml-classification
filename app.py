from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model" / "model"
st.set_page_config(page_title="Dry Bean Classifier | ML Assignment 2",
                   page_icon="🫘", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

@st.cache_resource
def load_artifacts():
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return None, None
    with open(metrics_path, encoding="utf-8") as f:
        metadata = json.load(f)
    models = {}
        for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)

    # Compatibility fix for older saved SimpleImputer models
    for model in models.values():
        preprocess = model.named_steps.get("preprocess")
        if preprocess is not None:
            imputer = preprocess.named_steps.get("imputer")
            if isinstance(imputer, SimpleImputer):
                if not hasattr(imputer, "_fill_dtype"):
                    imputer._fill_dtype = imputer.statistics_.dtype

    return metadata, models

st.title("🫘 Dry Bean Variety Classification")
st.caption("Machine Learning Assignment 2 | Five classifiers | Seven bean varieties")
st.markdown("""
This application compares five pre-trained classification pipelines. Upload a
**held-out test CSV** containing the 16 input features and the `Class` target
column. The application evaluates the selected saved model; it does not retrain.
""")

metadata, models = load_artifacts()
if metadata is None or not models:
    st.error("Trained model files are not present. Run `python train_models.py` first, then commit the generated `model/` files.")
    st.stop()

with st.sidebar:
    st.header("Evaluation settings")
    selected_model = st.selectbox("Choose classifier", list(MODEL_FILES.keys()))
    st.write("Expected target column: `Class`")
    st.write(f"Expected input features: {len(metadata['feature_names'])}")

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded is None:
    st.info("Use the included `test_data.csv` to try the application.")
    sample_path = ROOT / "test_data.csv"
    if sample_path.exists():
        st.download_button("Download included test data",
                           sample_path.read_bytes(), "test_data.csv", "text/csv")
    st.stop()

try:
    test_df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

feature_names = metadata["feature_names"]
required = feature_names + ["Class"]
missing = [col for col in required if col not in test_df.columns]
if missing:
    st.error("The uploaded CSV is missing required columns: " + ", ".join(missing))
    st.stop()
if test_df.empty:
    st.warning("The uploaded CSV has no rows.")
    st.stop()

X_test = test_df[feature_names].apply(pd.to_numeric, errors="coerce")
y_test = test_df["Class"].astype(str).str.strip()
if X_test.isna().any().any():
    st.error("Input feature columns contain missing or non-numeric values. Please correct the CSV.")
    st.stop()

model = models.get(selected_model)
if model is None:
    st.error(f"Saved model not found for {selected_model}.")
    st.stop()

try:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    model_classes = model.classes_
    auc = roc_auc_score(y_test, probabilities, labels=model_classes,
                        multi_class="ovr", average="weighted")
except Exception as exc:
    st.error(f"Evaluation failed. Check that target labels match the trained dataset. Details: {exc}")
    st.stop()

metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "AUC (weighted OVR)": auc,
    "Precision (weighted)": precision_score(y_test, predictions, average="weighted", zero_division=0),
    "Recall (weighted)": recall_score(y_test, predictions, average="weighted", zero_division=0),
    "F1 (weighted)": f1_score(y_test, predictions, average="weighted", zero_division=0),
    "MCC": matthews_corrcoef(y_test, predictions),
}
st.subheader(f"Results: {selected_model}")
cols = st.columns(3)
for i, (label, value) in enumerate(metrics.items()):
    cols[i % 3].metric(label, f"{value:.4f}")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, predictions, labels=model_classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=model_classes).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.subheader("Prediction preview")
preview = test_df.copy()
preview["Predicted_Class"] = predictions
st.dataframe(preview.head(25), use_container_width=True)
st.download_button("Download predictions", preview.to_csv(index=False),
                   "dry_bean_predictions.csv", "text/csv")

st.subheader("Saved test-set comparison")
comparison = []
for name, result in metadata["metrics"].items():
    comparison.append({
        "Model": name, "Accuracy": result["accuracy"], "AUC": result["auc"],
        "Precision": result["precision"], "Recall": result["recall"],
        "F1": result["f1"], "MCC": result["mcc"]
    })
st.dataframe(pd.DataFrame(comparison).set_index("Model").round(4),
             use_container_width=True)
