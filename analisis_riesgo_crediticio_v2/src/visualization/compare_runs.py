# src/visualization/compare_runs.py
# (La ruta en el comentario es solo una guía, el nombre del archivo es lo importante)
import mlflow
import pandas as pd
from pathlib import Path

# NOTA: Esta variable no se usa, pero la dejamos por consistencia
PACKAGE_NAME = "riesgo_crediticio" 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()
REPORTS_DIR = PROJECT_ROOT / "reports"

def fetch_and_compare_runs(experiment_name="model_comparison"):
    print(f"🔍 Conectando a MLflow en: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"❌ Experimento '{experiment_name}' no encontrado.")
        return
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        print("⚠️ No se encontraron runs.")
        return

    metric_cols = ["metrics.pr_auc", "metrics.roc_auc", "metrics.f1_score"]
    param_cols = ["params.model_name","params.clf__C","params.clf__n_estimators","params.clf__max_depth"]
    cols = ["run_id","tags.mlflow.runName"] + [c for c in metric_cols + param_cols if c in runs.columns]
    df = runs[cols].copy()
    df.columns = df.columns.str.replace("metrics.","").str.replace("params.","")
    df_sorted = df.sort_values("pr_auc", ascending=False)
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "compare_runs.csv"
    df_sorted.to_csv(out, index=False)
    
    print("\n🏁 Top 5 modelos por PR-AUC:\n")
    print(df_sorted.head(5).to_markdown(index=False))
    print(f"\n✅ Reporte guardado en: {out}")

if __name__ == "__main__":
    fetch_and_compare_runs(experiment_name="model_comparison")