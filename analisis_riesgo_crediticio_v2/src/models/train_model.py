import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Configura MLflow local
mlflow.set_tracking_uri("file:mlruns")
mlflow.set_experiment("model_comparison")

# Cargar datos preprocesados
df = pd.read_csv("src/data/processed/train.csv")
X = df.drop(columns=["target_bad"])
y = df["target_bad"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

def evaluate_and_log(model, name, params=None):
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_name", name)
        if params:
            for k,v in params.items():
                mlflow.log_param(k,v)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        pr_auc = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print(f"{name}: PR-AUC={pr_auc:.3f}, ROC-AUC={roc_auc:.3f}, F1={f1:.3f}")

# ---- Entrenar modelos ----
logit = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=500))
])
evaluate_and_log(logit, "LogisticRegression_Balanced", {"type":"logit"})

rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
)
evaluate_and_log(rf, "RandomForest_Tuned", {"n_estimators":300,"max_depth":8})
