%%writefile fuentes/datos_prep.py
import os
import hashlib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

def cargar_datos_originales(ruta_ini="datos/datos_ini/breast_cancer_raw.csv"):
    os.makedirs(os.path.dirname(ruta_ini), exist_ok=True)
    data = load_breast_cancer()
    df_raw = pd.DataFrame(data.data, columns=data.feature_names)
    df_raw['target'] = data.target
    df_raw.to_csv(ruta_ini, index=False)
    print(f"[DATOS_INI] Datos originales guardados en: {ruta_ini} ({df_raw.shape[0]}x{df_raw.shape[1]})")
    return df_raw

def limpiar_y_preparar_datos(df_raw, ruta_limp="datos/datos_limp/breast_cancer_clean.csv"):
    os.makedirs(os.path.dirname(ruta_limp), exist_ok=True)
    df = df_raw.copy()
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))

    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    df_clean = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    df_clean.to_csv(ruta_limp, index=False)

    with open(ruta_limp, "rb") as f:
        data_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    
    version_tag = f"v1.0_{data_hash}"
    print(f"[DATOS_LIMP] Datos limpios guardados en: {ruta_limp} | Versión: {version_tag}\n")
    return df_clean, version_tag

if __name__ == "__main__":
    df_original = cargar_datos_originales()
    limpiar_y_preparar_datos(df_original)
