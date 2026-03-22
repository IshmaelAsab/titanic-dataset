from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Titanic.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
ROC_CURVE_PATH = OUTPUT_DIR / "roc_curve.png"
TARGET_COLUMN = "Survived"
IDENTIFIER_COLUMNS = ["PassengerId"]
HIGH_CARDINALITY_COLUMNS = ["Name", "Ticket", "Cabin"]


def log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def save_roc_curve(y_true: pd.Series, y_score: pd.Series) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="Logistic Regression", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Titanic Survival ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close()


def main() -> None:
    log("1/7", f"Loading dataset from {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    log("1/7", f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

    log("2/7", "Inspecting columns, dtypes, and missing values")
    inspection_df = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_values": df.isna().sum(),
        }
    )
    print(inspection_df.to_string(), flush=True)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' was not found in the dataset.")

    drop_columns = [
        column
        for column in [TARGET_COLUMN, *IDENTIFIER_COLUMNS, *HIGH_CARDINALITY_COLUMNS]
        if column in df.columns
    ]
    feature_df = df.drop(columns=drop_columns)
    target = df[TARGET_COLUMN]

    categorical_features = [
        column for column in ["Pclass", "Sex", "Embarked"] if column in feature_df.columns
    ]
    numeric_features = [column for column in feature_df.columns if column not in categorical_features]

    log("3/7", f"Target column selected: {TARGET_COLUMN}")
    log("3/7", f"Using features: {feature_df.columns.tolist()}")
    log("3/7", f"Dropped columns: {[column for column in drop_columns if column != TARGET_COLUMN]}")
    log("3/7", f"Numeric features: {numeric_features}")
    log("3/7", f"Categorical features: {categorical_features}")

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    log("4/7", f"Train/test split complete: train={len(X_train)}, test={len(X_test)}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    log("5/7", "Training baseline logistic regression model")
    model.fit(X_train, y_train)

    log("6/7", "Evaluating model performance")
    predictions = model.predict(X_test)
    prediction_scores = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, prediction_scores)

    save_roc_curve(y_test, prediction_scores)
    log("6/7", f"Accuracy: {accuracy:.4f}")
    log("6/7", f"ROC-AUC: {roc_auc:.4f}")
    log("6/7", f"Saved ROC curve to {ROC_CURVE_PATH}")

    metrics = {
        "dataset_path": str(DATA_PATH),
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "target_column": TARGET_COLUMN,
        "features_used": feature_df.columns.tolist(),
        "dropped_columns": [column for column in drop_columns if column != TARGET_COLUMN],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "column_dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {column: int(count) for column, count in df.isna().sum().items()},
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log("7/7", f"Saved metrics summary to {METRICS_PATH}")
    log("7/7", "Workflow completed successfully")


if __name__ == "__main__":
    main()
