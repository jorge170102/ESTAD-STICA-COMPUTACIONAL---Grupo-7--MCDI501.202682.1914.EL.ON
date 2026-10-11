"""Estimaciones e hipótesis para el dataset de entregas."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display, Markdown


# ============================================================
# CONFIGURACIÓN
# ============================================================

OBJETIVO = "Reached.on.Time_Y.N"

NOMBRES = {
    "Weight_in_gms": "Peso del producto",
    "Cost_of_the_Product": "Costo del producto",
    "Discount_offered": "Descuento ofrecido",
}

UNIDADES = {
    "Weight_in_gms": "gramos",
    "Cost_of_the_Product": "USD",
    "Discount_offered": "sin información",
}

MODOS = {
    "Ship": "Barco",
    "Flight": "Avión",
    "Road": "Transporte terrestre",
}

ESTADOS = {
    0: "Entrega a tiempo",
    1: "Entrega fuera de plazo",
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def _numerica_valida(serie):
    """Obtiene valores observados y comprueba que sean finitos."""
    x = serie.dropna()

    if not pd.api.types.is_numeric_dtype(x):
        raise TypeError(f"La variable {serie.name} debe ser numérica.")

    x = x.astype(float)

    if not np.isfinite(x).all():
        raise ValueError(
            f"La variable {serie.name} contiene valores infinitos."
        )

    if len(x) < 2:
        raise ValueError(
            f"La variable {serie.name} necesita al menos dos valores."
        )

    return x


def _validar_objetivo(df):
    """Comprueba la codificación de la variable de entrega."""
    y = df[OBJETIVO]

    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError(
            f"{OBJETIVO} debe contener solo 0 y 1, sin faltantes."
        )


def _formato_p(p):
    """Presenta p sin redondearlo a cero."""
    return "< 0.0001" if p < 0.0001 else f"= {p:.4f}"


def _mostrar_figura(fig):
    fig.tight_layout()
    plt.show()
    plt.close(fig)


# ============================================================
# 1. ESTIMACIÓN DE MEDIAS E INTERVALOS DE CONFIANZA
# ============================================================

def mostrar_estimaciones(df, confianza=0.95):
    """Muestra estimaciones puntuales e intervalos de confianza."""
    if not 0 < confianza < 1:
        raise ValueError("La confianza debe estar entre 0 y 1.")

    columnas = [
        "Weight_in_gms",
        "Cost_of_the_Product",
        "Discount_offered",
    ]

    resultados = []

    for columna in columnas:
        x = _numerica_valida(df[columna])

        n = len(x)
        media = x.mean()
        de = x.std(ddof=1)
        error_estandar = de / np.sqrt(n)

        t_critico = stats.t.ppf(
            (1 + confianza) / 2,
            df=n - 1,
        )

        margen = t_critico * error_estandar
        inferior = media - margen
        superior = media + margen

        resultados.append({
            "Variable": NOMBRES[columna],
            "N válido": n,
            "Faltantes excluidos": int(df[columna].isna().sum()),
            "Límite inferior": inferior,
            "Media estimada": media,
            "Límite superior": superior,
            "Desviación estándar": de,
            "Error estándar": error_estandar,
            "Margen de error": margen,
        })

    tabla = pd.DataFrame(resultados).set_index("Variable")

    display(Markdown(
        f"### Estimaciones puntuales e intervalos del "
        f"{confianza:.0%}"
    ))
    display(tabla.round(3))


    display(Markdown(
        "**Interpretación de la confianza:** si repitiéramos "
        "el muestreo y este procedimiento muchas veces, "
        f"aproximadamente el {confianza:.0%} de los intervalos "
        "contendría la verdadera media poblacional para cada una de las variables. "
        "La cobertura depende de los supuestos y de la "
        "forma en que se obtuvieron los datos."
    ))

    return tabla


# ============================================================
# 2. PRUEBA T DE WELCH: PESO SEGÚN ESTADO DE ENTREGA
# ============================================================

def prueba_peso_por_entrega(df, alpha=0.05):
    """Contrasta dos medias independientes con Welch."""
    if not 0 < alpha < 1:
        raise ValueError("alpha debe estar entre 0 y 1.")

    _validar_objetivo(df)

    # Diferencia definida como:
    # fuera de plazo menos a tiempo.
    a_tiempo = _numerica_valida(
        df.loc[df[OBJETIVO].eq(0), "Weight_in_gms"]
    )
    fuera_plazo = _numerica_valida(
        df.loc[df[OBJETIVO].eq(1), "Weight_in_gms"]
    )

    grupos = {
        "Entrega a tiempo": a_tiempo,
        "Entrega fuera de plazo": fuera_plazo,
    }

    descriptivos = pd.DataFrame({
        nombre: {
            "N válido": len(x),
            "Media (gramos)": x.mean(),
            "Mediana (gramos)": x.median(),
            "Desviación estándar (gramos)": x.std(ddof=1),
        }
        for nombre, x in grupos.items()
    }).T

    descriptivos.index.name = "Estado de la entrega"

    display(Markdown(
        "### Prueba de hipótesis: peso promedio y estado de entrega"
    ))
    display(descriptivos.round(2))

    # Diagnóstico visual de la distribución en cada grupo.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for fila, (nombre, x) in enumerate(grupos.items()):
        axes[fila, 0].hist(
            x,
            bins="auto",
            color="#4472C4",
            edgecolor="white",
        )
        axes[fila, 0].set_title(f"Distribución: {nombre}")
        axes[fila, 0].set_xlabel("Peso del producto (gramos)")
        axes[fila, 0].set_ylabel("Frecuencia absoluta")

        stats.probplot(
            x.to_numpy(),
            dist="norm",
            plot=axes[fila, 1],
        )
        axes[fila, 1].set_title(f"Gráfico Q–Q: {nombre}")
        axes[fila, 1].set_xlabel("Cuantiles normales teóricos")
        axes[fila, 1].set_ylabel("Peso observado (gramos)")

    _mostrar_figura(fig)

    display(Markdown(
        "**Revisión de supuestos:** los histogramas y los "
        "gráficos Q–Q permiten examinar asimetría, colas y "
        "desviaciones respecto de la normalidad. Welch no "
        "requiere varianzas iguales. Con grupos grandes, "
        "la inferencia de medias puede apoyarse en una "
        "aproximación asintótica, siempre que exista "
        "independencia y varianza finita. La independencia "
        "no se comprueba con estos gráficos: debe justificarse "
        "con el diseño de recolección de los datos."
    ))

    n0, n1 = len(a_tiempo), len(fuera_plazo)
    v0, v1 = a_tiempo.var(ddof=1), fuera_plazo.var(ddof=1)

    componente0 = v0 / n0
    componente1 = v1 / n1

    error_estandar = np.sqrt(componente0 + componente1)

    if error_estandar == 0:
        raise ValueError(
            "No hay variabilidad suficiente para aplicar Welch."
        )

    # Grados de libertad de Welch–Satterthwaite.
    grados_libertad = (
        (componente0 + componente1) ** 2
        / (
            componente0 ** 2 / (n0 - 1)
            + componente1 ** 2 / (n1 - 1)
        )
    )

    prueba = stats.ttest_ind(
        fuera_plazo,
        a_tiempo,
        equal_var=False,
        alternative="two-sided",
    )

    diferencia = fuera_plazo.mean() - a_tiempo.mean()
    confianza = 1 - alpha

    t_critico = stats.t.ppf(
        1 - alpha / 2,
        df=grados_libertad,
    )

    inferior = diferencia - t_critico * error_estandar
    superior = diferencia + t_critico * error_estandar

    resultado = pd.DataFrame([{
        "Diferencia de medias (gramos)": diferencia,
        "Estadístico t": prueba.statistic,
        "Grados de libertad": grados_libertad,
        "Valor p": prueba.pvalue,
        "Nivel de significancia": alpha,
        "Límite inferior de la diferencia": inferior,
        "Límite superior de la diferencia": superior,
    }])

    display(resultado)

    if prueba.pvalue < alpha:
        conclusion = (
            "**Se rechaza H₀:** existe evidencia estadística "
            "de una diferencia entre los pesos promedio "
            "de ambos grupos."
        )
    else:
        conclusion = (
            "**No se rechaza H₀:** no se encontró evidencia "
            "estadística suficiente de una diferencia entre "
            "los pesos promedio. Esto no demuestra que "
            "las medias sean iguales."
        )

    return resultado


# ============================================================
# 3. CHI-CUADRADO: MODO DE ENVÍO Y ESTADO DE ENTREGA
# ============================================================

def prueba_modo_por_entrega(df, alpha=0.05):
    """Contrasta independencia entre transporte y entrega."""
    if not 0 < alpha < 1:
        raise ValueError("alpha debe estar entre 0 y 1.")

    _validar_objetivo(df)

    datos = df[
        ["Mode_of_Shipment", OBJETIVO]
    ].dropna()

    faltantes = len(df) - len(datos)

    tabla = pd.crosstab(
        datos["Mode_of_Shipment"],
        datos[OBJETIVO],
    )

    # Conservar categorías observadas, sin agregar filas de ceros.
    orden = [
        modo for modo in ["Ship", "Flight", "Road"]
        if modo in tabla.index
    ]
    adicionales = [
        modo for modo in tabla.index if modo not in orden
    ]
    tabla = tabla.reindex(orden + adicionales)

    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        raise ValueError(
            "Se necesitan al menos dos modos de envío "
            "y ambos estados de entrega."
        )

    observados = tabla.rename(
        index=MODOS,
        columns=ESTADOS,
    )
    observados.index.name = "Modo de envío"
    observados.columns.name = "Estado de la entrega"

    display(Markdown(
        "### Prueba de hipótesis: modo de envío y estado de entrega"
    ))
    display(Markdown("**Frecuencias observadas**"))
    display(observados)

    chi2, p, gl, esperados = stats.chi2_contingency(
        tabla,
        correction=False,
    )

    tabla_esperados = pd.DataFrame(
        esperados,
        index=observados.index,
        columns=observados.columns,
    )

    display(Markdown(
        "**Frecuencias esperadas si las variables fueran independientes**"
    ))
    display(tabla_esperados.round(2))

    minimo_esperado = esperados.min()
    aproximacion_adecuada = bool((esperados >= 5).all())

    display(Markdown(
        f"Se analizaron **{len(datos):,} registros** y se "
        f"excluyeron **{faltantes:,}** por faltantes en el modo "
        f"de envío. La frecuencia esperada mínima es "
        f"**{minimo_esperado:.2f}**. "
        f"El control conservador de frecuencias esperadas "
        f"≥ 5 en todas las celdas "
        f"**{'se cumple' if aproximacion_adecuada else 'no se cumple'}**. "
        f"Además, se requiere que cada registro contribuya "
        f"a una sola celda y que las observaciones sean independientes."
    ))

    if not aproximacion_adecuada:
        display(Markdown(
            "**No se interpreta el p de la aproximación "
            "chi-cuadrado con este código**, porque no se cumple "
            "el control de frecuencias esperadas. Corresponde "
            "considerar un procedimiento exacto o por simulación."
        ))
        return None

    n = tabla.to_numpy().sum()

    # Tamaño de asociación: V de Cramér.
    v_cramer = np.sqrt(
        chi2 / (n * min(tabla.shape[0] - 1, tabla.shape[1] - 1))
    )

    resultado = pd.DataFrame([{
        "Chi-cuadrado": chi2,
        "Grados de libertad": gl,
        "Valor p": p,
        "Nivel de significancia": alpha,
        "V de Cramér": v_cramer,
        "Frecuencia esperada mínima": minimo_esperado,
    }])

    display(resultado)

    tasas = pd.DataFrame({
        "Total de entregas": tabla.sum(axis=1),
        "Entregas fuera de plazo": tabla[1],
        "Porcentaje fuera de plazo (%)": (
            100 * tabla[1] / tabla.sum(axis=1)
        ),
    }).rename(index=MODOS)

    tasas.index.name = "Modo de envío"
    display(tasas.round(2))

    fig, ax = plt.subplots(figsize=(9, 4))

    barras = ax.bar(
        tasas.index,
        tasas["Porcentaje fuera de plazo (%)"],
        color="#4472C4",
    )

    ax.bar_label(
        barras,
        labels=[
            f"{valor:.2f} %"
            for valor in tasas["Porcentaje fuera de plazo (%)"]
        ],
        padding=3,
    )

    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("Modo de envío")
    ax.set_ylabel("Entregas fuera de plazo (%)")
    ax.set_title("Porcentaje fuera de plazo según modo de envío")

    _mostrar_figura(fig)

    if p < alpha:
        conclusion = (
            "**Se rechaza H₀:** existe evidencia estadística "
            "de asociación entre el modo de envío y el estado "
            "de entrega."
        )
    else:
        conclusion = (
            "**No se rechaza H₀:** no se encontró evidencia "
            "estadística suficiente de asociación entre el "
            "modo de envío y el estado de entrega. Esto no "
            "demuestra independencia."
        )

    display(Markdown(
        f"Con **α = {alpha:.2f}** y **p {_formato_p(p)}**, "
        f"{conclusion} El **V de Cramér es {v_cramer:.3f}**: "
        f"valores cercanos a cero indican una asociación débil "
        f"y valores mayores indican más asociación. **"
    ))

    return resultado