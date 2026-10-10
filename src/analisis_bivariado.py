"""Análisis bivariado de las variables numéricas y categóricas frente al estado de la entrega."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown


# ============================================================
# CONFIGURACIÓN DE VARIABLES Y ETIQUETAS
# ============================================================

NUMERICAS = [
    "Customer_care_calls",
    "Prior_purchases",
    "Cost_of_the_Product",
    "Discount_offered",
    "Weight_in_gms",
]

NOMBRES = {
    "Customer_care_calls": "Cantidad de llamadas del cliente",
    "Prior_purchases": "Compras anteriores",
    "Cost_of_the_Product": "Costo del producto (USD)",
    "Discount_offered": "Descuento ofrecido",
    "Weight_in_gms": "Peso del producto (gramos)",
    "Customer_rating": "Valoración del cliente",
    "Mode_of_Shipment": "Modo de envío",
    "Warehouse_block": "Bloque del almacén",
    "Product_importance": "Importancia del producto",
    "Estado_entrega": "Estado de la entrega",
}

ETIQUETAS = {
    "Mode_of_Shipment": {
        "Ship": "Barco",
        "Flight": "Avión",
        "Road": "Transporte terrestre",
    },
    "Warehouse_block": {
        letra: f"Bloque {letra}" for letra in "ABCDEF"
    },
    "Product_importance": {
        "low": "Baja",
        "medium": "Media",
        "high": "Alta",
    },
}

ORDENES = {
    "Mode_of_Shipment": ["Ship", "Flight", "Road"],
    "Warehouse_block": ["A", "B", "C", "D", "E", "F"],
    "Product_importance": ["low", "medium", "high"],
}

OBJETIVO = "Reached.on.Time_Y.N"

ESTADOS = {
    0: "Entrega a tiempo",
    1: "Entrega fuera de plazo",
}

ORDEN_ESTADOS = list(ESTADOS.values())


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def _preparar_entregas(df):
    """Crea una copia y agrega etiquetas al resultado de entrega."""
    objetivo = df[OBJETIVO]

    if objetivo.isna().any() or not objetivo.isin([0, 1]).all():
        raise ValueError(
            f"{OBJETIVO} debe contener solo 0 y 1, sin faltantes."
        )

    datos = df.copy(deep=True)
    datos["Estado_entrega"] = datos[OBJETIVO].map(ESTADOS)
    return datos


def _mostrar_figura(fig):
    """Ajusta, muestra y cierra una figura."""
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def _ordenar_categorias(tabla, columna):
    """Ordena categorías observadas y conserva las no previstas."""
    previstas = [
        categoria
        for categoria in ORDENES[columna]
        if categoria in tabla.index
    ]
    adicionales = [
        categoria
        for categoria in tabla.index
        if categoria not in previstas
    ]
    return tabla.reindex(previstas + adicionales)


# ============================================================
# 1. CORRELACIONES
# ============================================================

def mostrar_correlaciones(df):
    """Muestra tabla y gráfico de correlaciones de Spearman."""
    display(Markdown("### Correlaciones de Spearman"))

    columnas = NUMERICAS + ["Customer_rating"]

    matriz = (
        df[columnas]
        .corr(method="spearman")
        .rename(index=NOMBRES, columns=NOMBRES)
    )

    display(matriz.round(3))

    fig, ax = plt.subplots(figsize=(11, 8))

    sns.heatmap(
        matriz,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        cbar_kws={"label": "Correlación de Spearman"},
        ax=ax,
    )

    ax.set_title("Correlaciones de Spearman")
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
    )

    _mostrar_figura(fig)

    display(Markdown(
        "Los valores positivos indican que ambas variables tienden "
        "a aumentar juntas; los negativos, que una tiende a disminuir "
        "cuando aumenta la otra. Los valores cercanos a cero indican "
        "poca asociación monotónica. La valoración del cliente se "
        "incluye por su naturaleza ordinal."
        "**Estas asociaciones no demuestran causalidad.**"
    ))


# ============================================================
# 2. ESTADÍSTICOS POR ESTADO DE ENTREGA
# ============================================================

def mostrar_resumen_por_entrega(df):
    """Muestra estadísticos numéricos según el estado de entrega."""
    datos = _preparar_entregas(df)

    display(Markdown(
        "### Estadísticos según el estado de la entrega"
    ))

    tabla = (
        datos.groupby("Estado_entrega")[NUMERICAS]
        .agg(["count", "mean", "median", "std"])
        .reindex(ORDEN_ESTADOS)
        .rename(columns=NOMBRES, level=0)
        .rename(
            columns={
                "count": "N válido",
                "mean": "Media",
                "median": "Mediana",
                "std": "Desviación estándar",
            },
            level=1,
        )
    )

    tabla.index.name = "Estado de la entrega"
    tabla.columns.names = ["Variable", "Estadístico"]

    display(tabla.round(2))


# ============================================================
# 3. DISTRIBUCIONES POR ESTADO DE ENTREGA
# ============================================================

def mostrar_distribuciones_por_entrega(df):
    """Muestra boxplots verticales y comparaciones de promedios."""
    datos = _preparar_entregas(df)

    columnas = [
        "Weight_in_gms",
        "Cost_of_the_Product",
        "Discount_offered",
    ]

    for columna in columnas:
        nombre = NOMBRES[columna]

        display(Markdown(
            f"### {nombre} según estado de la entrega"
        ))

        fig, ax = plt.subplots(figsize=(9, 5))

        sns.boxplot(
            data=datos,
            x="Estado_entrega",
            y=columna,
            order=ORDEN_ESTADOS,
            ax=ax,
        )

        ax.set_title(f"{nombre} según estado de la entrega")
        ax.set_xlabel("Estado de la entrega")
        ax.set_ylabel(nombre)

        _mostrar_figura(fig)

        medias = (
            datos.groupby("Estado_entrega")[columna]
            .mean()
            .reindex(ORDEN_ESTADOS)
        )

        a_tiempo = medias[ORDEN_ESTADOS[0]]
        fuera_plazo = medias[ORDEN_ESTADOS[1]]

        if pd.isna(a_tiempo) or pd.isna(fuera_plazo):
            display(Markdown(
                "No hay valores válidos suficientes en ambos "
                "grupos para comparar sus promedios."
            ))
            continue

        diferencia = fuera_plazo - a_tiempo

        display(Markdown(
            f"**{nombre}:** el promedio en las entregas a tiempo "
            f"fue **{a_tiempo:.2f}**, y en las entregas fuera de "
            f"plazo fue **{fuera_plazo:.2f}**. La diferencia "
            f"(fuera de plazo menos a tiempo) fue "
            f"**{diferencia:.2f}**. Esta comparación es descriptiva "
            f"y no está ajustada por otras características de "
            f"los envíos."
        ))


# ============================================================
# 4. TASAS FUERA DE PLAZO POR CATEGORÍA
# ============================================================

def mostrar_tasas_por_categoria(df):
    """Muestra tabla, barras y comparación de tasas por categoría."""
    datos = _preparar_entregas(df)

    columnas = [
        "Mode_of_Shipment",
        "Warehouse_block",
        "Product_importance",
    ]

    for columna in columnas:
        nombre = NOMBRES[columna]

        display(Markdown(
            f"### Entregas fuera de plazo según {nombre.lower()}"
        ))

        categorias = (
            datos[columna]
            .astype("object")
            .fillna("Sin información")
        )

        tabla = (
            datos.groupby(categorias)[OBJETIVO]
            .agg(
                total="count",
                fuera_plazo="sum",
                proporcion="mean",
            )
        )

        tabla = _ordenar_categorias(tabla, columna)

        tabla["Porcentaje fuera de plazo (%)"] = (
            100 * tabla["proporcion"]
        )

        tabla = (
            tabla.drop(columns="proporcion")
            .rename(columns={
                "total": "Total de entregas",
                "fuera_plazo": "Entregas fuera de plazo",
            })
        )

        tabla.index = [
            ETIQUETAS[columna].get(categoria, str(categoria))
            for categoria in tabla.index
        ]
        tabla.index.name = nombre

        display(tabla.round(2))

        fig, ax = plt.subplots(figsize=(9, 5))

        barras = ax.bar(
            tabla.index,
            tabla["Porcentaje fuera de plazo (%)"],
            color="#4472C4",
        )

        ax.bar_label(
            barras,
            labels=[
                f"{valor:.2f} %"
                for valor in tabla["Porcentaje fuera de plazo (%)"]
            ],
            padding=3,
        )

        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_title(
            f"Porcentaje de entregas fuera de plazo según "
            f"{nombre.lower()}"
        )
        ax.set_xlabel(nombre)
        ax.set_ylabel("Entregas fuera de plazo (%)")

        _mostrar_figura(fig)

        tasas = tabla["Porcentaje fuera de plazo (%)"]

        if tasas.empty:
            continue

        mayor = tasas.max()
        menor = tasas.min()

        categorias_mayor = ", ".join(
            tasas.index[tasas.eq(mayor)].tolist()
        )
        categorias_menor = ", ".join(
            tasas.index[tasas.eq(menor)].tolist()
        )

        display(Markdown(
            f"**{nombre}:** la mayor tasa observada corresponde "
            f"a **{categorias_mayor}**, con **{mayor:.2f} %**; "
            f"la menor corresponde a **{categorias_menor}**, con "
            f"**{menor:.2f} %**. Los porcentajes se calculan dentro "
            f"de cada categoría, considerando que sus tamaños "
            f"pueden ser distintos. Estas diferencias no establecen "
            f"por sí solas significancia estadística ni causalidad."
        ))


# ============================================================
# 5. PESO, DESCUENTO Y ESTADO DE ENTREGA
# ============================================================

def mostrar_peso_descuento(df, n_muestra=2500, semilla=42):
    """Muestra correlación y dispersión de peso frente a descuento."""
    datos = _preparar_entregas(df)

    display(Markdown(
        "### Peso del producto y descuento ofrecido"
    ))

    pares = datos[
        ["Weight_in_gms", "Discount_offered"]
    ].dropna()

    if len(pares) < 2:
        display(Markdown(
            "No hay suficientes pares válidos para este análisis."
        ))
        return

    correlacion = (
        pares.corr(method="spearman")
        .loc["Weight_in_gms", "Discount_offered"]
    )

    display(Markdown(
        f"La correlación de Spearman entre **Peso del producto "
        f"(gramos)** y **Descuento ofrecido** es "
        f"**{correlacion:.3f}**, calculada con "
        f"**{len(pares):,} pares válidos**. Si alguna variable "
        f"es constante, la correlación no está definida y se "
        f"muestra como `nan`. Esta asociación debe considerarse "
        f"al estudiar ambas variables frente al estado de la entrega."
    ))

    if n_muestra < 1:
        raise ValueError("n_muestra debe ser mayor que cero.")

    # Seleccionar la muestra entre los registros graficables.
    muestra = (
        datos.dropna(
            subset=["Weight_in_gms", "Discount_offered"]
        )
        .sample(
            n=min(n_muestra, len(pares)),
            random_state=semilla,
        )
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=muestra,
        x="Weight_in_gms",
        y="Discount_offered",
        hue="Estado_entrega",
        hue_order=ORDEN_ESTADOS,
        alpha=0.35,
        s=20,
        ax=ax,
    )

    ax.set_title(
        "Peso del producto y descuento ofrecido "
        "según estado de la entrega"
    )
    ax.set_xlabel(NOMBRES["Weight_in_gms"])
    ax.set_ylabel(NOMBRES["Discount_offered"])
    ax.legend(title="Estado de la entrega")

    _mostrar_figura(fig)

    display(Markdown(
        f"El gráfico representa una muestra reproducible de "
        f"**{len(muestra):,} registros con valores disponibles** "
        f"en ambas variables. La correlación utiliza todos los "
        f"pares válidos, no solo la muestra visual. La unidad "
        f"del descuento no está especificada por la fuente. "
        f"**Estos resultados no demuestran causalidad.**"
    ))
    

def mostrar_densidad_peso_descuento(df, corte_descuento=10):
    """Muestra densidad conjunta y un acercamiento a descuentos bajos."""
    datos = df[
        ["Weight_in_gms", "Discount_offered"]
    ].dropna()

    if datos.empty:
        display(Markdown(
            "No hay registros con peso y descuento disponibles."
        ))
        return

    acercamiento = datos[
        datos["Discount_offered"].le(corte_descuento)
    ]

    display(Markdown(
        "### Concentración del peso y el descuento"
    ))

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        sharex=True,
    )

    paneles = [
        (datos, "Todos los valores disponibles"),
        (
            acercamiento,
            f"Acercamiento: descuento ≤ {corte_descuento:g}",
        ),
    ]

    for ax, (subconjunto, titulo) in zip(axes, paneles):
        if subconjunto.empty:
            ax.set_title(f"{titulo}: sin registros")
            continue

        grafico = ax.hexbin(
            subconjunto["Weight_in_gms"],
            subconjunto["Discount_offered"],
            gridsize=35,
            mincnt=1,
            bins="log",
            cmap="viridis",
        )

        ax.set_title(f"{titulo}\nN = {len(subconjunto):,}")
        ax.set_xlabel("Peso del producto (gramos)")
        ax.set_ylabel("Descuento ofrecido")
        ax.grid(False)

        barra = fig.colorbar(grafico, ax=ax)
        barra.set_label(
            "Registros por celda (escala logarítmica)"
        )

    _mostrar_figura(fig)

    porcentaje = 100 * len(acercamiento) / len(datos)

    display(Markdown(
        f"Cada hexágono agrupa registros con pesos y descuentos "
        f"cercanos. Los colores más claros indican más registros. "
        f"La escala logarítmica del color permite distinguir "
        f"concentraciones de diferentes tamaños; los ejes mantienen "
        f"sus unidades originales. "
        f"El panel derecho representa **{len(acercamiento):,} "
        f"registros ({porcentaje:.2f} % de los pares disponibles)** "
        f"con descuento ≤ **{corte_descuento:g}**. "
        f"Los colores se interpretan usando la barra de cada panel. "
        f"La unidad del descuento no está especificada por la fuente."
    ))