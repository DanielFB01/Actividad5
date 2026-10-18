# Actividad 5: Pipeline de Entrenamiento, Ajuste y Registro con MLflow

## Descripción del Proyecto
Implementación de un flujo de trabajo reproducible de Machine Learning (MLOps) para clasificación clínica (*Breast Cancer Wisconsin Dataset*). El proyecto abarca la ingesta y limpieza modular de datos, versionamiento mediante hashes criptográficos (SHA-256), ajuste de hiperparámetros con validación cruzada (`5-Fold GridSearchCV`) y registro de experimentos en **MLflow**.

## Estructura del Repositorio
```text
Actividad5/
│── datos/
│   ├── datos_ini/        # Datos originales sin procesar (raw)
│   └── datos_limp/       # Datos estandarizados y libres de nulos/duplicados
│── fuentes/
│   ├── entrena.ipynb     # Notebook interactivo (Colab) de experimentación
│   ├── datos_prep.py     # Módulo de ingesta, limpieza y versionamiento de datos
│   └── train.py          # Script de entrenamiento, GridSearch y logging en MLflow
│── README.md             # Guía de reproducción y decisiones arquitectónicas
└── CHANGELOG.md          # Reporte evolutivo y resultados finales del experimento