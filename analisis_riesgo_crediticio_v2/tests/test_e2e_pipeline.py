import subprocess, sys
from pathlib import Path
import pandas as pd
import joblib
import numpy as np

# Artefactos guardados durante entrenamiento
FEATURE_INFO_PATH = Path("models/transformers/feature_info.joblib")
OHE_PATH = Path("models/transformers/onehot_encoder.joblib")
SEL_PATH = Path("models/transformers/feature_selector.joblib")

def _transform_with_saved_artifacts(df: pd.DataFrame):
    """
    Reconstruye EXACTAMENTE el diseño usado en entrenamiento:
      - num_cols y cat_cols (en el mismo orden).
      - One-Hot con el mismo OHE (mismos nombres/salida).
      - features ingenierizadas con los mismos nombres.
      - reordena/padea a 'design_cols_before_selector'.
      - aplica SelectKBest (con máscara guardada o transform()).
    Devuelve un np.ndarray listo para model.predict_proba().
    """
    assert OHE_PATH.exists(), "❌ Falta models/transformers/onehot_encoder.joblib"
    ohe = joblib.load(OHE_PATH)

    info = {}
    if FEATURE_INFO_PATH.exists():
        info = joblib.load(FEATURE_INFO_PATH)

    # Columnas base
    num_cols = info.get("num_cols", [])
    # Preferimos las cat_cols que guardaste; si no, usamos las que el OHE "vio".
    cat_cols = info.get("cat_cols", None)
    if cat_cols is None:
        assert hasattr(ohe, "feature_names_in_"), "❌ OHE no tiene feature_names_in_. Reentrenar."
        cat_cols = list(ohe.feature_names_in_)

    design_cols = info.get("design_cols_before_selector")  # lista final ANTES del selector (
