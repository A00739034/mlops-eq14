# Fase 3: Procesamiento final, pruebas automatizadas y setup para FastAPI

**Rol:** Data Scientist (Lucero)  
**Fecha:** 2025-11-06

## 1) Objetivo
Dejar el modelo y artefactos listos para despliegue, con procesamiento determinista, pruebas automatizadas y tracking en MLflow.

## 2) Procesamiento de datos
- Script: `src/features/build_features.py`
- Dataset final: `src/data/processed/train.csv`
- Sin nulos, dominios validados, tipos corregidos.

## 3) Modelo entrenado y artefactos
- Mejor modelo serializado: `models/best_model.joblib`
- Artefactos:  
  `models/transformers/onehot_encoder.joblib`,  
  `models/transformers/feature_selector.joblib`,  
  `models/transformers/feature_info.joblib`
- Helper de inferencia coherente: `_transform_with_saved_artifacts()`.

## 4) MLflow
- Tracking local: `mlruns/` — experimento `german_credit_risk`
- Se registró modelo con `signature` + `input_example`.

## 5) Pruebas automatizadas
- Framework: `pytest`
- **Resultado:** `2 passed`  
- E2E verifica: build_features → carga de artefactos → predict_proba en [0,1] y no-constante.

## 6) Reproducibilidad
- Dependencias fijadas (clave): `scikit-learn==1.6.1`
- Seeds: `random_state=42`
- Métricas reproducibles en entorno limpio.

## 7) Resultados (referencia)
| Modelo | PR-AUC | ROC-AUC | F1 |
|---|---:|---:|---:|
| LogisticRegression (balanced) | 0.689 | 0.826 | 0.633 |
| RandomForest (tuned) | 0.658 | 0.842 | 0.667 |

## 8) Entrega y siguientes pasos
**Entregado:** dataset final, artefactos, modelo, MLflow, pruebas, reporte.  
**Siguiente (Alberto):** servicio FastAPI 

— **Fase 3 completada** 
