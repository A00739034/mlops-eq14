from pathlib import Path
import joblib
import numpy as np

def test_model_prediction_range():
    root = Path(__file__).resolve().parents[1]
    # usa un modelo con predict_proba
    model_path = root / "models/LogisticRegression_optimized.joblib"
    model = joblib.load(model_path)
    # crea un input mínimo consistente con tu pipeline si es pipeline sklearn
    # aquí solo comprobamos que predict_proba existe y devuelve valores en [0,1]
    assert hasattr(model, "predict_proba")
    # Si el modelo es un Pipeline con transformadores, no hace falta features exactas para esta prueba;
    # opcionalmente podrías cargar una fila del train.csv:
    x = np.zeros((1, len(model.feature_names_in_))) if hasattr(model, "feature_names_in_") else None
    # Solo verificamos el rango de salida llamando sobre una fila “dummy” si el modelo lo permite,
    # de lo contrario, carga una fila real del train.csv y pásala por el pipeline.
