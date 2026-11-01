
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "bank_marketing_preprocessing")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

y_train = pd.read_csv(
    os.path.join(DATA_DIR, "y_train.csv")
).squeeze("columns")

y_test = pd.read_csv(
    os.path.join(DATA_DIR, "y_test.csv")
).squeeze("columns")


# ==========================================
# 2. HYPERPARAMETER TUNING
# ==========================================

model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

param_dist = {
    "n_estimators": [100, 150, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    scoring="f1",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


# ==========================================
# 3. MLflow MANUAL LOGGING
# ==========================================

mlflow.set_experiment("Bank Marketing - Hyperparameter Tuning")

with mlflow.start_run(run_name="Random Forest Tuning"):

    # Training dan tuning
    random_search.fit(X_train, y_train)

    # Model terbaik
    best_model = random_search.best_estimator_

    # Prediksi
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Evaluasi
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    # Manual logging parameters
    mlflow.log_params(random_search.best_params_)

    mlflow.log_param("model_name", "RandomForestClassifier")
    mlflow.log_param("cv", 3)
    mlflow.log_param("scoring", "f1")
    mlflow.log_param("n_iter", 10)

    # Manual logging metrics
    mlflow.log_metrics(metrics)

    # Simpan model terbaik ke MLflow
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

    # Simpan confusion matrix sebagai artifact
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    )

    cm_path = os.path.join(BASE_DIR, "confusion_matrix.csv")
    cm_df.to_csv(cm_path)

    mlflow.log_artifact(cm_path)

    # Tampilkan hasil
    print("\n=== BEST PARAMETERS ===")
    print(random_search.best_params_)

    print("\n=== HASIL EVALUASI MODEL TUNING ===")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nHyperparameter tuning berhasil!")
