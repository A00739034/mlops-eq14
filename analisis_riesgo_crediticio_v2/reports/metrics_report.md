# Reporte de Métrica y Selección de Modelo
**Proyecto:** Predicción de Riesgo Crediticio
**Fecha:** 30 de octubre de 2025
**Autor(a):** Lucero Aponte Pérez

---

## 1. Resumen Ejecutivo

Se recomienda seleccionar el modelo **`LogisticRegression_Balanced`** como el candidato principal para el despliegue en producción.

Este modelo demostró el mejor rendimiento en la métrica de negocio clave, **PR-AUC (0.728)**, superando al modelo `RandomForest_Tuned` (0.706).

---

## 2. Metodología

Se entrenaron y evaluaron dos modelos candidatos utilizando el script `src/models/train_model.py`. Los experimentos fueron registrados en MLflow bajo el experimento `"model_comparison"`.

* **Métrica Principal:** Se seleccionó el **Área Bajo la Curva de Precisión-Recall (PR-AUC)** como la métrica primaria. Esta métrica es la más indicada para este problema debido al desbalance de clases (más clientes "buenos" que "malos") y al objetivo de negocio de identificar correctamente a los clientes de alto riesgo (maximizando la precisión).
* **Métricas Secundarias:** Se utilizaron ROC-AUC y F1-Score como métricas de soporte.

---

## 3. Resultados de la Comparación

Los resultados de la ejecución del script `src/visualization/compare_runs.py` (guardados en `compare_runs.csv`) son los siguientes:

| tags.mlflow.runName | pr_auc | roc_auc | f1_score |
| :--- | ---: | ---: | ---: |
| **`LogisticRegression_Balanced`** | **0.728** | **0.864** | **0.662** |
| `RandomForest_Tuned` | 0.706 | 0.862 | 0.624 |

---

## 4. Análisis y Conclusión

El modelo `LogisticRegression_Balanced` es el claro ganador por las siguientes razones:

1.  **Rendimiento (PR-AUC):** Supera al Random Forest en la métrica principal. Esto significa que ofrece un mejor balance entre encontrar clientes de riesgo (Recall) y estar en lo correcto cuando lo hace (Precisión).
2.  **Rendimiento General:** También supera al Random Forest en *todas* las métricas secundarias (ROC-AUC y F1-Score), lo que indica que es un modelo más robusto en general.
3.  **Interpretabilidad:** Como beneficio adicional, un modelo de Regresión Logística es inherentemente más simple e interpretable que un Random Forest. En un contexto de riesgo financiero, poder explicar *por qué* un préstamo es denegado es una ventaja regulatoria y de negocio crucial.

## 5. Recomendación

Se aprueba el modelo **`LogisticRegression_Balanced`** para ser registrado y promovido a la siguiente etapa del pipeline de MLOps (staging/producción).