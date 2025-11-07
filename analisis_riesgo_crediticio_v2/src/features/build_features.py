import pandas as pd
import numpy as np
import json
from pathlib import Path

# --- Configuración de Paths ---
# orre el script desde la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_FILE = PROJECT_ROOT / "src/data/raw/german_credit_modified.csv"
OUT_DIR = PROJECT_ROOT / "src/data/processed"
OUT_FILE = OUT_DIR / "train.csv"  # El archivo que train_model.py necesita

# --- Definiciones del Dominio (Copiadas de tu Colab) ---
VALID = {
    "laufkont": {1, 2, 3, 4},
    "moral": {0, 1, 2, 3, 4},
    "verw": set(range(0, 11)),
    "sparkont": {1, 2, 3, 4, 5},
    "beszeit": {1, 2, 3, 4, 5},
    "rate": {1, 2, 3, 4},
    "famges": {1, 2, 3, 4},
    "buerge": {1, 2, 3},
    "wohnzeit": {1, 2, 3, 4},
    "verm": {1, 2, 3, 4},
    "weitkred": {1, 2, 3},
    "wohn": {1, 2, 3},
    "bishkred": {1, 2, 3, 4},
    "beruf": {1, 2, 3, 4},
    "pers": {1, 2},
    "telef": {1, 2},
    "gastarb": {1, 2},
    "kredit": {0, 1},
}
CONT = ["hoehe", "laufzeit", "alter"]
RANGE = {"alter": (18, 75), "laufzeit": (4, 72), "hoehe": (250, None)}


def brief(df, title="DF"):
    print(
        f"— {title} — shape={df.shape}, "
        f"nans={int(df.isna().sum().sum())}, dups={df.duplicated().sum()}"
    )


# ==============================
#   FUNCIÓN PURA PARA TESTS ✅
# ==============================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma un DataFrame crudo en uno listo para modelar:
    - Fuerza numérico y valida dominios
    - Aplica reglas de rango + IQR en 'hoehe'
    - Crea 'target_bad' y hace imputaciones (mediana/moda)
    - Quita duplicados y elimina filas sin target
    - Elimina 'kredit' si existe

    NO realiza I/O (no lee ni escribe archivos).
    """
    out = df.copy()

    # Forzar numérico donde haya strings de números
    for c in out.columns:
        if out[c].dtype == "O":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Validación de dominios: códigos fuera del set -> NaN
    for col, ok in VALID.items():
        if col in out.columns:
            m = out[col].notna() & ~out[col].isin(ok)
            out.loc[m, col] = np.nan

    # Reglas de rango (se dejan como NaN lo fuera de rango)
    for col, (lo, hi) in RANGE.items():
        if col in out.columns:
            if lo is not None:
                out.loc[out[col] < lo, col] = np.nan
            if hi is not None:
                out.loc[out[col] > hi, col] = np.nan

    # Filtrado IQR en 'hoehe' con piso HOEHE>=250
    if "hoehe" in out.columns and out["hoehe"].notna().any():
        s = out["hoehe"].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = max(q1 - 1.5 * iqr, 250), q3 + 1.5 * iqr
        m = out["hoehe"].between(low, high)
        out = out[m].copy()

    # Target binario consistente
    if "kredit" not in out.columns:
        raise ValueError("Falta columna 'kredit' (target)")
    out["target_bad"] = out["kredit"].map({1: 0, 0: 1}).astype("Int64")

    # Imputaciones: medianas para continuas, moda para categóricas
    for c in out.columns:
        if c == "target_bad":
            continue
        if c in CONT:
            out[c] = out[c].fillna(out[c].median())
        else:
            moda = out[c].mode(dropna=True)
            if not moda.empty:
                out[c] = out[c].fillna(moda.iloc[0])

    # Quitar la columna original 'kredit' (ya tenemos target_bad)
    if "kredit" in out.columns:
        out = out.drop(columns=["kredit"])

    # Duplicados exactos y conflictivos
    out = out.drop_duplicates().reset_index(drop=True)
    feats = [c for c in out.columns if c != "target_bad"]
    if feats:  # evitar error si solo hay target
        g = out.groupby(feats, dropna=False)["target_bad"].nunique()
        conflict_keys = set(g[g > 1].index)
        if conflict_keys:
            out["__key__"] = list(map(tuple, out[feats].values))
            keep_mask = ~out["__key__"].isin(conflict_keys)
            out = out.loc[keep_mask].drop(columns="__key__").reset_index(drop=True)

    # Eliminar filas sin target
    out = out.dropna(subset=["target_bad"]).copy()

    return out


# ==================================
#   SCRIPT I/O QUE REUTILIZA LO DE ARRIBA
# ==================================
def process_data():
    print("Iniciando procesamiento de datos...")
    print(f"Leyendo datos crudos de: {IN_FILE}")

    if not IN_FILE.exists():
        print("❌ ERROR: No se encontró el archivo de datos crudos en:")
        print(IN_FILE)
        print("Por favor, coloca 'german_credit_modified.csv' en 'src/data/raw/'")
        return

    df = pd.read_csv(IN_FILE)
    brief(df, "Cargado")

    # Reusar la función pura (apta para tests)
    df_imp = build_features(df)
    brief(df_imp, "Final limpio (build_features)")

    # Guardar
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_imp.to_csv(OUT_FILE, index=False)
    print(f"\n✅ ¡Éxito! Datos procesados guardados en: {OUT_FILE}")


__all__ = ["build_features", "process_data"]


if __name__ == "__main__":
    process_data()
