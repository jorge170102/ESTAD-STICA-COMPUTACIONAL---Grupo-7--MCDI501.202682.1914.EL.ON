"""Descripción univariada de E-Commerce Shipping sobre datos preparados MCAR."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

NUMERICAS = ["Customer_care_calls", "Cost_of_the_Product", "Prior_purchases", "Discount_offered", "Weight_in_gms"]
CATEGORICAS = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender", "Customer_rating", "Estado_entrega"]


def resumen_numerico(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos muestrales con exclusión de nulos por variable."""
    datos = []
    for nombre in NUMERICAS:
        x = pd.to_numeric(df[nombre], errors="raise").dropna()
        if len(x) < 2:
            raise ValueError(f"No hay suficientes observaciones para {nombre}")
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        datos.append({
            "variable": nombre, "n válido": len(x), "faltantes": int(df[nombre].isna().sum()),
            "media": x.mean(), "mediana": x.median(), "modas": ", ".join(map(str, x.mode().tolist())),
            "DE": x.std(ddof=1), "varianza": x.var(ddof=1), "mínimo": x.min(),
            "Q1": q1, "Q3": q3, "máximo": x.max(), "rango": x.max()-x.min(), "IQR": iqr,
            "fuera 1,5 IQR": int(((x < q1-1.5*iqr) | (x > q3+1.5*iqr)).sum())
        })
    return pd.DataFrame(datos).set_index("variable")


def graficar_numericas(df: pd.DataFrame):
    """Histograma y caja por variable, sin reemplazar faltantes."""
    for nombre in NUMERICAS:
        x = df[nombre].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        if nombre in ("Customer_care_calls", "Prior_purchases"):
            bins = np.arange(x.min()-0.5, x.max()+1.5)
            axes[0].hist(x, bins=bins, color="#267FA3", edgecolor="white")
        else:
            axes[0].hist(x, bins="auto", color="#267FA3", edgecolor="white")
        axes[1].boxplot(x, vert=True); axes[0].set_title("Histograma")
        axes[1].set_title("Diagrama de caja")
        axes[0].set_xlabel(nombre); axes[1].set_ylabel(nombre)
        fig.suptitle(f"Distribución: {nombre} (n={len(x):,})")
        fig.tight_layout(); plt.show(); plt.close(fig)


def frecuencias(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Tabla de frecuencias y porcentajes incluyendo nulos explícitamente."""
    serie = df[columna].astype("object").where(df[columna].notna(), "Sin información")
    n = serie.value_counts(dropna=False)
    return pd.DataFrame({"frecuencia": n, "porcentaje": n*100/len(df)})


def graficar_categoricas(df: pd.DataFrame):
    for col in CATEGORICAS:
        tabla = frecuencias(df, col)
        print(f"\n{col}:")
        print(tabla.round(2).to_string())
        fig, ax = plt.subplots(figsize=(8, 3.5))
        tabla["frecuencia"].plot.bar(ax=ax, color="#267FA3")
        ax.set_title(f"Distribución: {col}"); ax.set_ylabel("Frecuencia absoluta")
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout(); plt.show(); plt.close(fig)
