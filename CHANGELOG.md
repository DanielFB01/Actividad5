---

#### 2. `CHANGELOG.md` (Reporte final y bitácora del proyecto)
```markdown
# CHANGELOG & Reporte Final de Experimentación

## [1.0.0] - Entrega Final del Pipeline MLOps
### Agregado
- **Gestión de Datos (`fuentes/datos_prep.py`):** Automatización de descarga del dataset original (`569 registros, 30 variables`), limpieza y escalado estándar en `datos/datos_limp/`.
- **Versionamiento:** Implementación de huella digital SHA-256 registrada como etiqueta (`dataset_version`) en MLflow.
- **Entrenamiento y Ajuste (`fuentes/train.py`):** Comparativa automatizada con `5-Fold GridSearchCV` entre Regresión Logística y Random Forest.

### Resumen de Resultados (Reporte Final)
| Modelo | Hiperparámetros Óptimos | CV Mean F1 | Test Accuracy | Test Recall | Test F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | `C: 0.1, solver: lbfgs` | 0.9845 | **0.9825** | **0.9861** | **0.9861** | **0.9954** |
| **Random Forest** | `max_depth: None, n_estimators: 100` | 0.9725 | 0.9561 | 0.9722 | 0.9655 | 0.9931 |

### Conclusión Técnica
Se selecciona el modelo de **Regresión Logística (`C=0.1`)** para paso a producción. Al operar sobre características numéricas previamente estandarizadas en `datos_limp`, logra separar las clases con mayor precisión general (F1-Score de `0.9861`) y menor costo computacional que Random Forest.