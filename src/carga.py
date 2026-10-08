"""Lectura y persistencia del conjunto de datos."""
from pathlib import Path
import pandas as pd


def cargar_csv(ruta: str | Path) -> pd.DataFrame:
    """Carga un archivo CSV sin modificar sus datos originales."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo CSV: {ruta}")
    return pd.read_csv(ruta)


def guardar_csv(df: pd.DataFrame, ruta: str | Path) -> None:
    """Guarda datos derivados sin modificar el archivo raw."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)
