"""Comprobación de esquema y auditoría de calidad de E-Commerce Shipping."""
import pandas as pd

COLUMNAS_REQUERIDAS = (
    "ID", "Warehouse_block", "Mode_of_Shipment", "Customer_care_calls",
    "Customer_rating", "Cost_of_the_Product", "Prior_purchases",
    "Product_importance", "Gender", "Discount_offered", "Weight_in_gms",
    "Reached.on.Time_Y.N",
)

VARIABLES_NUMERICAS = (
    "Customer_care_calls", "Customer_rating", "Cost_of_the_Product",
    "Prior_purchases", "Discount_offered", "Weight_in_gms",
)


def validar_esquema(df: pd.DataFrame) -> None:
    """Verifica columnas, identidad única y dominios básicos; no corrige datos."""
    faltantes = sorted(set(COLUMNAS_REQUERIDAS) - set(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas esperadas: {faltantes}")
    if df.empty:
        raise ValueError("El dataset está vacío.")
    if df["ID"].isna().any() or df["ID"].duplicated().any():
        raise ValueError("La columna ID debe estar completa y sin duplicados.")
    objetivo = df["Reached.on.Time_Y.N"]
    if objetivo.isna().any() or not objetivo.isin([0, 1]).all():
        raise ValueError("La variable objetivo debe contener únicamente 0 y 1, sin nulos.")
    for columna in VARIABLES_NUMERICAS:
        if not pd.api.types.is_numeric_dtype(df[columna]):
            raise TypeError(f"Se esperaba una variable numérica en '{columna}'.")


def informe_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Resume tipo, nulos y cardinalidad de todas las columnas."""
    return pd.DataFrame({
        "columna": df.columns,
        "tipo": [str(df[col].dtype) for col in df.columns],
        "nulos": [int(df[col].isna().sum()) for col in df.columns],
        "porcentaje_nulos": [round(df[col].isna().mean() * 100, 2) for col in df.columns],
        "valores_unicos": [int(df[col].nunique(dropna=True)) for col in df.columns],
    })
