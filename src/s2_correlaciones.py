"""Estadística computacional Grupo 7: módulo de Sumativa 2 (no modifica S1)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def spearman_por_lote(x, y, axis=-1):
    """Spearman = Pearson de rangos recalculados; admite lotes y empates."""
    rx = stats.rankdata(x, axis=axis, method="average")
    ry = stats.rankdata(y, axis=axis, method="average")
    rx = rx - rx.mean(axis=axis, keepdims=True)
    ry = ry - ry.mean(axis=axis, keepdims=True)
    return np.sum(rx * ry, axis=axis) / np.sqrt(
        np.sum(rx**2, axis=axis) * np.sum(ry**2, axis=axis))

def _rangos_por_frecuencias(codigos: np.ndarray, niveles: int) -> np.ndarray:
    """Rangos medios exactos, por fila, con la corrección de empates de Spearman.

    En lugar de ordenar de nuevo cada remuestra, cuenta sus categorías y calcula
    los rangos acumulados: inicio del bloque + (frecuencia + 1) / 2.
    Los códigos se construyen UNA vez desde los valores originales ordenados, pero
    las frecuencias y los rangos se vuelven a calcular en CADA remuestra.
    """
    n_filas, _ = codigos.shape
    desplazamiento = np.arange(n_filas, dtype=np.int64)[:, None] * niveles
    conteos = np.bincount(
        (desplazamiento + codigos).ravel(),
        minlength=n_filas * niveles,
    ).reshape(n_filas, niveles)
    # El promedio de los rangos empatados es el acumulado menos (conteo - 1)/2.
    rangos_medios = np.cumsum(conteos, axis=1) - (conteos - 1) * 0.5
    return rangos_medios[np.arange(n_filas)[:, None], codigos]


def bootstrap_correlacion(datos_pares, semilla, B=10_000, LOTE=50):
    """Bootstrap de Spearman acelerado; respeta empates y pares completos.

    Los índices reproducen los muestreos de scipy.stats.bootstrap con rng=Generator
    en SciPy 1.17, pero se evita ordenar x e y 10.000 veces por relación.
    No se cambian el número de remuestras, los valores p ni el estimador.
    """
    x, y = datos_pares.to_numpy(dtype=float).T
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Se requieren pares completos, finitos y al menos 2 observaciones")
    if B < 1 or LOTE < 1:
        raise ValueError("B y LOTE deben ser enteros positivos")

    # Mapeo monotónico de los valores, conservando exactamente sus empates.
    _, cod_x = np.unique(x, return_inverse=True)
    _, cod_y = np.unique(y, return_inverse=True)
    n_niveles_x = int(cod_x.max()) + 1
    n_niveles_y = int(cod_y.max()) + 1
    if n_niveles_x < 2 or n_niveles_y < 2:
        raise ValueError("Spearman no está definido para una variable constante")

    # Comprobar que el estimador coincide con scipy antes de simular.
    np.testing.assert_allclose(
        spearman_por_lote(x, y), stats.spearmanr(x, y).statistic, rtol=1e-12, atol=1e-12
    )

    rng = np.random.default_rng(semilla)
    n = len(x)
    media_rango = (n + 1) / 2
    valores = np.empty(B, dtype=float)
    for ini in range(0, B, LOTE):
        lote_actual = min(LOTE, B - ini)
        # La misma secuencia de índices del bootstrap emparejado de SciPy.
        indices = rng.integers(0, n, size=(lote_actual, n))
        rangos_x = _rangos_por_frecuencias(cod_x[indices], n_niveles_x) - media_rango
        rangos_y = _rangos_por_frecuencias(cod_y[indices], n_niveles_y) - media_rango
        producto = np.sum(rangos_x * rangos_y, axis=1)
        norma = np.sqrt(np.sum(rangos_x**2, axis=1) * np.sum(rangos_y**2, axis=1))
        valores[ini:ini+lote_actual] = producto / norma
    assert np.isfinite(valores).all(), "Alguna remuestra tiene varianza nula"
    intervalo = np.quantile(valores, [0.025, 0.975])
    return valores, intervalo


def analizar_correlaciones(df, pares, semillas, B=10_000, LOTE=50):
    distribuciones_rho, filas = {}, []
    for (a, b, nombre), semilla in zip(pares, semillas):
        muestra_pares = df[[a, b]].dropna()
        valores, intervalo = bootstrap_correlacion(muestra_pares, semilla, B=B, LOTE=LOTE)
        distribuciones_rho[nombre] = valores
        contiene_cero = bool(intervalo[0] <= 0 <= intervalo[1])
        signo = "Signo no resuelto" if contiene_cero else ("Negativo estable" if intervalo[1] < 0 else "Positivo estable")
        filas.append({"relacion": nombre, "n": len(muestra_pares),
            "rho_s1": stats.spearmanr(muestra_pares[a], muestra_pares[b]).statistic,
            "ic_inf": intervalo[0], "ic_sup": intervalo[1],
            "ancho_ic": intervalo[1] - intervalo[0], "ee_bootstrap": valores.std(ddof=1),
            "incluye_cero": contiene_cero, "signo": signo})
    correlaciones = pd.DataFrame(filas).set_index("relacion")
    # Tabla de presentación de resultados
    tabla_correlaciones = correlaciones.copy()
    tabla_correlaciones["IC bootstrap 95 %"] = tabla_correlaciones.apply(lambda r: f"[{r['ic_inf']:.4f}; {r['ic_sup']:.4f}]", axis=1)
    tabla_correlaciones = tabla_correlaciones[[
        "n",
        "rho_s1",
        "IC bootstrap 95 %",
        "ee_bootstrap",
        "ancho_ic",
        "incluye_cero",
        "signo"
    ]]
    return distribuciones_rho, correlaciones, tabla_correlaciones

