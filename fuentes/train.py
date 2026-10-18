import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, ConfusionMatrixDisplay
)

# Importar funciones del módulo de preparación de datos
from datos_prep import cargar_datos_originales, limpiar_y_preparar_datos

# Configuración global de MLflow
TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "Actividad5_Clasificacion_MLOps"

def configurar_mlflow():
    """Configura el seguimiento local de MLflow y retorna el objeto experimento."""
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    print("=================================================================")
    print("CONFIGURACIÓN DE MLFLOW (SEGUIMIENTO LOCAL)")
    print("=================================================================")
    print(f"Tracking URI : {mlflow.get_tracking_uri()}")
    print(f"Experimento  : {experiment.name} (ID: {experiment.experiment_id})")
    print(f"Artefactos   : {experiment.artifact_location}\n")
    return experiment

def entrenar_y_registrar(model, param_grid, run_name, X_train, X_test, y_train, y_test, data_version):
    """
    Ejecuta validación cruzada (5-Fold GridSearchCV), evalúa el mejor modelo
    y registra etiquetas, hiperparámetros, métricas y artefactos en MLflow.
    """
    with mlflow.start_run(run_name=run_name) as run:
        print(f"--- Entrenando y ajustando: {run_name} (5-Fold GridSearchCV) ---")
        
        # 1. Validación cruzada y búsqueda de hiperparámetros
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_best_score = grid_search.best_score_

        # 2. Predicciones sobre el conjunto de prueba (Test Set)
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # 3. Cálculo de métricas estandarizadas
        metrics = {
            "cv_mean_f1": round(cv_best_score, 4),
            "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
            "test_precision": round(precision_score(y_test, y_pred), 4),
            "test_recall": round(recall_score(y_test, y_pred), 4),
            "test_f1_score": round(f1_score(y_test, y_pred), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_prob), 4)
        }

        # 4. Registro de Tags y Parámetros en MLflow
        mlflow.set_tag("dataset_version", data_version)
        mlflow.set_tag("validation_strategy", "5-Fold GridSearchCV")
        mlflow.log_params(best_params)

        # 5. Registro de Métricas en MLflow
        mlflow.log_metrics(metrics)

        # 6. Registro de Artefacto 1: Gráfico de Matriz de Confusión (.png)
        os.makedirs("temp_artifacts", exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
        ax.set_title(f"Matriz de Confusión - {run_name}")
        plot_path = f"temp_artifacts/confusion_matrix_{run_name}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(plot_path, artifact_path="evaluacion_plots")

        # 7. Registro de Artefacto 2: Modelo Serializado (.pkl) con firma de entrada
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_train.iloc[:3]
        )

        print(f" -> Run ID registrado      : {run.info.run_id}")
        print(f" -> Mejores Hiperparámetros: {best_params}")
        print(f" -> Métricas registradas   : {metrics}\n")
        
        return {"Modelo": run_name, **best_params, **metrics}

def validar_registros_mlflow(experiment_id):
    """
    Verifica y audita programáticamente que cada Run en MLflow tenga
    registrados sus parámetros, métricas y archivos de artefactos.
    """
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    print("=================================================================")
    print(f"VALIDACIÓN DE REGISTROS EN MLFLOW ({len(runs)} ejecuciones auditadas)")
    print("=================================================================")

    for r in runs:
        run_id = r.info.run_id
        run_name = r.data.tags.get("mlflow.runName", "Sin_Nombre")
        artefactos_raiz = [art.path for art in client.list_artifacts(run_id)]
        artefactos_modelo = [art.path for art in client.list_artifacts(run_id, "model")]
        artefactos_plots = [art.path for art in client.list_artifacts(run_id, "evaluacion_plots")]

        print(f"\n▶ RUN: {run_name} (ID: {run_id})")
        print(f"  1. Tag Versión Datos : {r.data.tags.get('dataset_version')}")
        print(f"  2. Parámetros        : {r.data.params}")
        print(f"  3. Métricas          : {r.data.metrics}")
        print(f"  4. Carpetas Artefacto: {artefactos_raiz}")
        print(f"     ├── Archivos Modelo: {artefactos_modelo}")
        print(f"     └── Archivos Plots : {artefactos_plots}")
        print("-" * 65)

def main():
    # Paso 1: Ingesta, limpieza y versionamiento de datos
    df_raw = cargar_datos_originales()
    df_clean, data_version = limpiar_y_preparar_datos(df_raw)

    # División estratificada 80% Train / 20% Test
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Paso 2: Configuración de MLflow
    experiment = configurar_mlflow()

    # Paso 3: Entrenamiento y ajuste de Algoritmo 1 (Regresión Logística)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }
    entrenar_y_registrar(
        lr_model, lr_grid, "Logistic_Regression_Tuned",
        X_train, X_test, y_train, y_test, data_version
    )

    # Paso 4: Entrenamiento y ajuste de Algoritmo 2 (Random Forest)
    rf_model = RandomForestClassifier(random_state=42)
    rf_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    entrenar_y_registrar(
        rf_model, rf_grid, "Random_Forest_Tuned",
        X_train, X_test, y_train, y_test, data_version
    )

    # Paso 5: Auditoría y validación de parámetros, métricas y artefactos
    validar_registros_mlflow(experiment.experiment_id)

if __name__ == "__main__":
    main()
