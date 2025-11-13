# Reporte Consolidado de Detección de Drift

Este reporte consolida los resultados de diferentes tipos de drift simulado.

## Resumen Ejecutivo

| Tipo de Drift | Drift Detectado | Severidad | Alertas | Degradación de Desempeño |
|---------------|-----------------|-----------|---------|---------------------------|
| mean_shift | No | none | 0 | No |
| missing_values | No | none | 1 | Sí |
| combined | No | none | 1 | Sí |

## Detalles por Tipo de Drift

### MEAN_SHIFT

#### Métricas de Desempeño

| Métrica | Referencia | Con Drift | Cambio Relativo (%) |
|---------|------------|-----------|---------------------|
| accuracy | 0.8272 | 0.8476 | +2.46% |
| precision | 0.8291 | 0.8459 | +2.03% |
| recall | 0.8272 | 0.8476 | +2.46% |
| f1 | 0.8141 | 0.8407 | +3.26% |
| roc_auc | 0.9049 | 0.9136 | +0.96% |
| average_precision | 0.8303 | 0.8424 | +1.46% |

### MISSING_VALUES

#### Métricas de Desempeño

| Métrica | Referencia | Con Drift | Cambio Relativo (%) |
|---------|------------|-----------|---------------------|
| accuracy | 0.8272 | 0.8191 | -0.98% |
| precision | 0.8291 | 0.8174 | -1.40% |
| recall | 0.8272 | 0.8191 | -0.98% |
| f1 | 0.8141 | 0.8067 | -0.91% |
| roc_auc | 0.9049 | 0.8309 | -8.18% |
| average_precision | 0.8303 | 0.7512 | -9.53% |

#### Alertas

- **roc_auc** (medium): ROC-AUC cambió -8.18% (umbral: 5.0%)

### COMBINED

#### Métricas de Desempeño

| Métrica | Referencia | Con Drift | Cambio Relativo (%) |
|---------|------------|-----------|---------------------|
| accuracy | 0.8272 | 0.8252 | -0.25% |
| precision | 0.8291 | 0.8218 | -0.87% |
| recall | 0.8272 | 0.8252 | -0.25% |
| f1 | 0.8141 | 0.8159 | +0.22% |
| roc_auc | 0.9049 | 0.8503 | -6.04% |
| average_precision | 0.8303 | 0.7805 | -6.00% |

#### Alertas

- **roc_auc** (medium): ROC-AUC cambió -6.04% (umbral: 5.0%)


## Recomendaciones Generales

### ⚠️ Se detectaron problemas significativos

**Acciones recomendadas:**

1. **Revisión inmediata del feature pipeline**
   - Verificar que el procesamiento de datos sigue siendo correcto
   - Validar que no hay cambios en las transformaciones aplicadas

2. **Análisis de causas del drift**
   - Investigar cambios en la distribución de datos de entrada
   - Revisar si hay cambios en el proceso de recolección de datos

3. **Considerar retraining del modelo**
   - Si la degradación es significativa (>5% en métricas clave)
   - Reentrenar con datos más recientes que reflejen la nueva distribución
   - Validar el nuevo modelo antes de desplegar

4. **Implementar monitoreo continuo**
   - Configurar alertas automáticas para drift en producción
   - Establecer umbrales de alerta basados en estos resultados
   - Programar evaluaciones periódicas del modelo

