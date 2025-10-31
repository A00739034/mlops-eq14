import pandas as pd
import numpy as np
import json
from pathlib import Path

# --- Configuración de Paths (Adaptado para tu script local) ---
# Asume que corres el script desde la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_FILE = PROJECT_ROOT / "src/data/raw/german_credit_modified.csv"
OUT_DIR = PROJECT_ROOT / "src/data/processed"
OUT_FILE = OUT_DIR / "train.csv" # El archivo que train_model.py necesita

# --- Definiciones del Dominio (Copiadas de tu Colab) ---
VALID = {
    "laufkont":{1,2,3,4}, "moral":{0,1,2,3,4}, "verw":set(range(0,11)), "sparkont":{1,2,3,4,5},
    "beszeit":{1,2,3,4,5}, "rate":{1,2,3,4}, "famges":{1,2,3,4}, "buerge":{1,2,3},
    "wohnzeit":{1,2,3,4}, "verm":{1,2,3,4}, "weitkred":{1,2,3}, "wohn":{1,2,3},
    "bishkred":{1,2,3,4}, "beruf":{1,2,3,4}, "pers":{1,2}, "telef":{1,2}, "gastarb":{1,2},
    "kredit":{0,1}
}
CONT = ["hoehe","laufzeit","alter"]
RANGE = {"alter":(18,75), "laufzeit":(4,72), "hoehe":(250,None)}

def brief(df, title="DF"):
    print(f"— {title} — shape={df.shape}, nans={int(df.isna().sum().sum())}, dups={df.duplicated().sum()}")

def process_data():
    print(f"Iniciando procesamiento de datos...")
    print(f"Leyendo datos crudos de: {IN_FILE}")
    
    if not IN_FILE.exists():
        print(f"❌ ERROR: No se encontró el archivo de datos crudos en:")
        # (Corregí un error aquí: IN_File -> IN_FILE)
        print(IN_FILE) 
        print("Por favor, asegúrate de que 'german_credit_modified.csv' esté en 'src/data/raw/'")
        return

    df = pd.read_csv(IN_FILE)

    if "mixed_type_col" in df.columns:
        df = df.drop(columns=["mixed_type_col"])
    brief(df, "Cargado")

    # --- Forzar numérico + validación de dominios/códigos ---
    for c in df.columns:
        if df[c].dtype == "O":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for col, ok in VALID.items():
        if col in df.columns:
            m = df[col].notna() & ~df[col].isin(ok)
            df.loc[m, col] = np.nan
    brief(df, "Tras coerción + dominios")

    # --- Reglas de rango + outliers en ‘hoehe’ ---
    for col,(lo,hi) in RANGE.items():
        if col in df.columns:
            if lo is not None: df.loc[df[col] < lo, col] = np.nan
            if hi is not None: df.loc[df[col] > hi, col] = np.nan

    if "hoehe" in df.columns and df["hoehe"].notna().any():
        s = df["hoehe"].dropna()
        q1,q3 = s.quantile([.25,.75]); iqr = q3-q1
        low, high = max(q1-1.5*iqr, 250), q3+1.5*iqr
        m = df["hoehe"].between(low, high)
        df = df[m].copy()
    brief(df, "Tras rangos + IQR")

    # --- Target binario consistente + imputaciones simples ---
    if "kredit" not in df.columns:
        raise ValueError("Falta columna 'kredit' (target)")
    df["target_bad"] = df["kredit"].map({1:0, 0:1}).astype("Int64")

    df_imp = df.copy()
    for c in df_imp.columns:
        if c == "target_bad": continue
        if c in CONT:
            df_imp[c] = df_imp[c].fillna(df_imp[c].median())
        else:
            moda = df_imp[c].mode(dropna=True)
            if not moda.empty:
                df_imp[c] = df_imp[c].fillna(moda.iloc[0])

    if "kredit" in df_imp.columns:
        df_imp = df_imp.drop(columns=["kredit"])
    brief(df_imp, "Tras imputación")

    # --- Duplicados (exactos y conflictivos) ---
    df_imp = df_imp.drop_duplicates().reset_index(drop=True)
    feats = [c for c in df_imp.columns if c != "target_bad"]
    g = df_imp.groupby(feats, dropna=False)["target_bad"].nunique()
    conflict_keys = set(g[g > 1].index)
    if conflict_keys:
        df_imp["__key__"] = list(map(tuple, df_imp[feats].values))
        keep_mask = ~df_imp["__key__"].isin(conflict_keys)
        df_imp = df_imp.loc[keep_mask].drop(columns="__key__").reset_index(drop=True)
    brief(df_imp, "Final sin duplicados")
    
    # --- Limpieza final: Eliminar filas sin target ---
    # (Este bloque ahora está INDENTADO correctamente)
    initial_rows = len(df_imp)
    df_imp = df_imp.dropna(subset=["target_bad"]).copy()
    removed_rows = initial_rows - len(df_imp)
    if removed_rows > 0:
        print(f"Eliminando {removed_rows} filas donde 'target_bad' era NaN.")
    brief(df_imp, "Final limpio (sin NaNs en target)")

    # --- Guardar el archivo final ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_imp.to_csv(OUT_FILE, index=False)
    print(f"\n✅ ¡Éxito! Datos procesados guardados en: {OUT_FILE}")

if __name__ == "__main__":
    process_data()