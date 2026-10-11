"""Visualización y resumen bivariado. Solo opera sobre el df preparado recibido."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OBJETIVO = "Reached.on.Time_Y.N"
NUMERICAS = ["Customer_care_calls", "Prior_purchases", "Cost_of_the_Product", "Discount_offered", "Weight_in_gms"]
ETIQUETAS = {0:"Entrega a tiempo", 1:"Entrega fuera de plazo"}


def _preparar(df):
    d = df.copy(deep=True)
    if not d[OBJETIVO].isin([0, 1]).all():
        raise ValueError("La variable objetivo debe contener solo 0 y 1")
    d["Estado_entrega"] = d[OBJETIVO].map(ETIQUETAS)
    return d


def mostrar_correlaciones(df):
    x = df[NUMERICAS + ["Customer_rating"]]
    matriz = x.corr(method="spearman")  # Exclusión por pares para cada correlación.
    from IPython.display import display
    display(matriz.round(3))
    fig, ax=plt.subplots(figsize=(10,7))
    sns.heatmap(matriz, vmin=-1,vmax=1,center=0,cmap="vlag",annot=True,fmt=".2f",ax=ax)
    ax.set_title("Correlaciones de Spearman (valores disponibles por pares)")
    fig.tight_layout();plt.show();plt.close(fig)
    return matriz


def mostrar_resumen_por_entrega(df):
    d=_preparar(df)
    out=d.groupby("Estado_entrega")[NUMERICAS].agg(["count","mean","median","std"])
    from IPython.display import display
    display(out.round(2));return out


def mostrar_distribuciones_por_entrega(df):
    d=_preparar(df)
    for columna in ("Weight_in_gms","Cost_of_the_Product","Discount_offered"):
        fig,ax=plt.subplots(figsize=(8,4))
        ax.boxplot([d.loc[d.Estado_entrega.eq(label), columna].dropna() for label in ETIQUETAS.values()], tick_labels=list(ETIQUETAS.values()))
        ax.set_title(f"{columna} por puntualidad (sin imputación)")
        fig.tight_layout();plt.show();plt.close(fig)


def mostrar_tasas_por_categoria(df):
    d=_preparar(df)
    from IPython.display import display
    for columna in ("Mode_of_Shipment","Warehouse_block","Product_importance"):
        tab=d.groupby(columna,dropna=False)[OBJETIVO].agg(total="count",retrasados="sum",tasa="mean")
        tab["tasa (%)"]=100*tab.pop("tasa")
        print("\n",columna);display(tab.round(2))
        fig,ax=plt.subplots(figsize=(8,3.5))
        tab["tasa (%)"].plot.bar(ax=ax,color="#267FA3")
        ax.set_ylim(0,100);ax.set_ylabel("Fuera de plazo (%)")
        ax.set_title(f"Tasa de retraso según {columna}")
        fig.tight_layout();plt.show();plt.close(fig)


def mostrar_densidad_peso_descuento(df):
    d=df[["Weight_in_gms","Discount_offered"]].dropna()
    rho=d.corr(method="spearman").loc["Weight_in_gms","Discount_offered"]
    fig,ax=plt.subplots(figsize=(9,4))
    h=ax.hexbin(d["Weight_in_gms"],d["Discount_offered"],gridsize=35,mincnt=1,bins="log",cmap="viridis")
    fig.colorbar(h,ax=ax,label="N por celda (escala logarítmica)")
    ax.set_xlabel("Peso (g)");ax.set_ylabel("Descuento (unidad no documentada)")
    ax.set_title(f"Peso y descuento (n={len(d):,}, Spearman={rho:.3f})")
    fig.tight_layout();plt.show();plt.close(fig)
    return rho
