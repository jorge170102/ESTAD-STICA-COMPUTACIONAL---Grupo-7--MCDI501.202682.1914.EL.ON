"""Transformaciones reproducibles para el dataset E-Commerce Shipping."""
import numpy as np
import pandas as pd


def generar_faltantes_mcar(
    df: pd.DataFrame,
    columnas: tuple[str, ...] = ("Weight_in_gms", "Cost_of_the_Product"),
    proporcion: float = 0.10,
    semilla: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera faltantes MCAR y registra los valores verdaderos para Fase/Sumativa 3.

    El resultado no modifica `df`. Se elimina la misma proporción aleatoria
    de valores en cada variable elegida, independientemente de sus magnitudes.
    """
    if not 0.10 <= proporcion <= 0.15:
        raise ValueError("La proporción debe estar entre 0.10 y 0.15 según la guía.")
    if len(columnas) not in (1, 2) or len(set(columnas)) != len(columnas):
        raise ValueError("Selecciona una o dos variables distintas.")
    if not df.index.is_unique:
        raise ValueError("El índice del DataFrame debe ser único.")
    if df[list(columnas)].isna().any().any():
        raise ValueError("Ya hay faltantes en columnas seleccionadas; revisa antes de generar.")
    if df["ID"].isna().any() or df["ID"].duplicated().any():
        raise ValueError("Se requiere un ID único y sin faltantes.")
    if any(not pd.api.types.is_numeric_dtype(df[col]) for col in columnas):
        raise TypeError("Las columnas MCAR deben ser numéricas.")

    rng = np.random.default_rng(semilla)
    resultado = df.copy(deep=True)
    n = max(1, int(round(len(df) * proporcion)))
    registros = []

    for columna in columnas:
        indices = rng.choice(df.index.to_numpy(), size=n, replace=False)
        copia = df.loc[indices, ["ID", columna]].copy()
        copia = copia.rename(columns={columna: "valor_original"})
        copia["variable"] = columna
        registros.append(copia[["ID", "variable", "valor_original"]])
        resultado.loc[indices, columna] = np.nan

    auditoria = pd.concat(registros, ignore_index=True)
    return resultado, auditoria
