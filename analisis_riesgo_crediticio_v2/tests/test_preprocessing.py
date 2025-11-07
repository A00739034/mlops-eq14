import pandas as pd

def test_no_null_values():
    # Usa el dataset que realmente genera tu pipeline
    df = pd.read_csv("src/data/processed/train.csv")

    # No debe haber NaNs
    assert int(df.isna().sum().sum()) == 0

    # Debe existir la columna objetivo y ser binaria {0,1}
    assert "target_bad" in df.columns
    vals = set(df["target_bad"].dropna().unique().tolist())
    assert vals.issubset({0, 1})
