"""
Train five classifiers on the UCI Dry Bean dataset.
Run: python train_models.py
This script downloads the official dataset through ucimlrepo, creates a
stratified 80/20 split, evaluates all models, and saves deployment artifacts.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

RANDOM_STATE = 42
TEST_SIZE = 0.20
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    print("Downloading the UCI Dry Bean dataset (dataset ID 602)...")
    dataset = fetch_ucirepo(id=602)
    X = dataset.data.features.copy()
    y_frame = dataset.data.targets.copy()
    y = y_frame.iloc[:, 0].astype(str).str.strip()
    X = X.apply(pd.to_numeric, errors="coerce")
    feature_names = X.columns.tolist()

    # Remove exact duplicate records consistently before splitting.
    combined = X.copy()
    combined["__target__"] = y.values
    before = len(combined)
    combined = combined.drop_duplicates().reset_index(drop=True)
    removed_duplicates = before - len(combined)
    y = combined.pop("__target__")
    X = combined

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scaling is included only where distance/margin/linear optimization needs it.
    scaled_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    plain_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    models = {
        "Logistic Regression": Pipeline([
            ("preprocess", scaled_preprocessor),
            ("classifier", LogisticRegression(max_iter=3000, solver="lbfgs",
                                               class_weight=None, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": Pipeline([
            ("preprocess", plain_preprocessor),
            ("classifier", DecisionTreeClassifier(max_depth=None, min_samples_leaf=2,
                                                   random_state=RANDOM_STATE))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("preprocess", scaled_preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("preprocess", scaled_preprocessor),
            ("classifier", GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ("preprocess", scaled_preprocessor),
            ("classifier", SVC(C=3.0, kernel="rbf", probability=True,
                               decision_function_shape="ovr", random_state=RANDOM_STATE))
        ]),
    }

    class_names = sorted(y.unique().tolist())
    results = {}
    for display_name, estimator in models.items():
        print(f"Training {display_name}...")
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "auc": float(roc_auc_score(y_test, probabilities,
                                       labels=estimator.classes_,
                                       multi_class="ovr", average="weighted")),
            "precision": float(precision_score(y_test, predictions, average="weighted",
                                               zero_division=0)),
            "recall": float(recall_score(y_test, predictions, average="weighted",
                                         zero_division=0)),
            "f1": float(f1_score(y_test, predictions, average="weighted",
                                 zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, predictions)),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=estimator.classes_
            ).tolist(),
            "class_names": estimator.classes_.tolist()
        }
        results[display_name] = metrics
        slug = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "K-Nearest Neighbors": "knn",
            "Gaussian Naive Bayes": "naive_bayes",
            "Support Vector Machine": "svm"
        }[display_name]
        joblib.dump(estimator, MODEL_DIR / f"{slug}.joblib")
        print("  Accuracy={:.4f} | AUC={:.4f} | F1={:.4f} | MCC={:.4f}".format(
            metrics["accuracy"], metrics["auc"], metrics["f1"], metrics["mcc"]
        ))

    # Keep the held-out target in this CSV: the Streamlit app evaluates rather
    # than retrains on the uploaded test data.
    test_output = X_test.copy()
    test_output["Class"] = y_test.values
    test_output.to_csv(ROOT / "test_data.csv", index=False)

    summary = {
        "dataset": "UCI Dry Bean",
        "uci_dataset_id": 602,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "original_instances": int(before),
        "duplicates_removed": int(removed_duplicates),
        "instances_after_duplicate_removal": int(len(X)),
        "training_instances": int(len(X_train)),
        "test_instances": int(len(X_test)),
        "feature_names": feature_names,
        "target_name": "Class",
        "classes": class_names,
        "metrics": results
    }
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(MODEL_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print("\nCompleted. Artifacts saved in ./model and test_data.csv.")
    print("Review model/metrics.json before writing your final observations.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
