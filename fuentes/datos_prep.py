import os
import hashlib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

def cargar_datos_originales(ruta_ini="datos/datos_ini/breast_cancer_raw.csv"):
    """Descarga el dataset público y lo guarda como datos originales (raw)."""
    os.makedirs(os.path.dirname(ruta_ini), exist_ok=True)
    data = load_breast_cancer()
    df_raw = pd.DataFrame(data.data, columns=data.feature_names)
    df_raw['target'] = data.target
    df_raw.to_csv(ruta_ini, index=False)
    print(f"[INFO] Datos originales guardados en: {ruta_ini} ({df_raw.shape[0]} filas, {df_raw.shape[1]} columnas)")
    return df_raw

def limpiar_y_preparar_datos(df_raw, ruta_limp="datos/datos_limp/breast_cancer_clean.csv"):
    """Aplica limpieza de nulos, duplicados, escalado de características y versionamiento SHA-256."""
    os.makedirs(os.path.dirname(ruta_limp), exist_ok=True)
    df = df_raw.copy()

    # 1. Eliminación de duplicados y manejo de valores nulos
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))

    # 2. Separar características (X) y etiqueta (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # 3. Estandarización (Media = 0, Desviación Estándar = 1)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 4. Reconstruir dataframe limpio y exportar
    df_clean = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    df_clean.to_csv(ruta_limp, index=False)

    # 5. Generar Hash criptográfico para versionamiento de datos
    with open(ruta_limp, "rb") as f:
        data_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    print(f"[INFO] Datos limpios guardados en: {ruta_limp}")
    print(f"[INFO] Versión del Dataset generada: v1.0_{data_hash}")
    return df_clean, f"v1.0_{data_hash}"

if __name__ == "__main__":
    df_ini = cargar_datos_originales()
    limpiar_y_preparar_datos(df_ini)