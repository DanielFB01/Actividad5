import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datos_prep import cargar_datos_originales, limpiar_y_preparar_datos

def evaluar_y_registrar(model, param_grid, run_name, X_train, X_test, y_train, y_test, data_version):
    """Ejecuta GridSearchCV (5-Fold) y registra parámetros, métricas y artefactos en MLflow."""
    with mlflow.start_run(run_name=run_name):
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "cv_mean_f1": round(grid.best_score_, 4),
            "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
            "test_precision": round(precision_score(y_test, y_pred), 4),
            "test_recall": round(recall_score(y_test, y_pred), 4),
            "test_f1_score": round(f1_score(y_test, y_pred), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_prob), 4)
        }

        # Registro de trazabilidad en MLflow
        mlflow.set_tag("dataset_version", data_version)
        mlflow.set_tag("cv_strategy", "5-Fold GridSearchCV")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        print(f"\n[MLflow] Run '{run_name}' registrado exitosamente.")
        print(f" -> Mejores Hiperparámetros: {grid.best_params_}")
        print(f" -> Métricas en Test: {metrics}")

def main():
    # 1. Preparar datos y obtener versión
    df_raw = cargar_datos_originales()
    df_clean, data_version = limpiar_y_preparar_datos(df_raw)

    X = df_clean.drop('target', axis=1)
    y = df_clean['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2. Configurar experimento en MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Actividad5_Clasificacion_MLOps")

    # 3. Modelo 1: Regresión Logística
    lr_grid = {'C': [0.01, 0.1, 1.0, 10.0], 'solver': ['lbfgs', 'liblinear']}
    evaluar_y_registrar(
        LogisticRegression(max_iter=1000, random_state=42),
        lr_grid, "Logistic_Regression", X_train, X_test, y_train, y_test, data_version
    )

    # 4. Modelo 2: Random Forest
    rf_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
    evaluar_y_registrar(
        RandomForestClassifier(random_state=42),
        rf_grid, "Random_Forest", X_train, X_test, y_train, y_test, data_version
    )

if __name__ == "__main__":
    main()