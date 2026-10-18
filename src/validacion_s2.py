"""Utilidades auditables para el notebook S2 del Grupo 7.

No altera el dataset de S1 ni efectúa imputaciones. Se entregan pruebas unitarias
con datos sintéticos para validar mecánica de permutación, diagnóstico y recorte.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def permutacion_diferencia_medias(df: pd.DataFrame, *, variable="Weight_in_gms",
                                objetivo="Reached.on.Time_Y.N", iteraciones=10_000,
                                semilla=202, alfa=0.05):
    """Test bilateral de etiquetas para delta=fuera(1)-tiempo(0).

    La permutación simple requiere intercambiabilidad bajo H0. Los dos tamaños
    originales se conservan y p se corrige por (b+1)/(B+1).
    """
    datos = df[[variable, objetivo]].dropna()
    valores = datos[variable].to_numpy(dtype=float)
    codigos = datos[objetivo].to_numpy(dtype=int)
    if len(valores) < 4 or set(np.unique(codigos)) != {0, 1}:
        raise ValueError("Se requieren al menos cuatro observaciones válidas y ambos grupos 0/1")
    fuera = valores[codigos == 1]
    tiempo = valores[codigos == 0]
    n1 = len(fuera)
    n0 = len(tiempo)
    if min(n1, n0) < 2:
        raise ValueError("Cada grupo requiere al menos dos observaciones")
    diferencia = float(fuera.mean() - tiempo.mean())
    total = valores.sum()
    rng = np.random.default_rng(semilla)
    distribucion = np.empty(iteraciones, dtype=float)
    for i in range(iteraciones):
        indices = rng.choice(len(valores), size=n1, replace=False)
        suma_fuera = valores[indices].sum()
        distribucion[i] = suma_fuera/n1 - (total-suma_fuera)/n0
    extremos = int((np.abs(distribucion) >= abs(diferencia)-1e-12).sum())
    p = (extremos+1)/(iteraciones+1)
    welch = stats.ttest_ind(fuera, tiempo, equal_var=False)
    return {
        "variable": variable, "n": int(len(valores)), "n_a_tiempo": int(n0),
        "n_fuera_plazo": int(n1), "diferencia": diferencia,
        "p_permutacion": float(p), "p_welch": float(welch.pvalue),
        "intervalo_central_nulo_95": np.quantile(distribucion,[.025,.975]).astype(float).tolist(),
        "rango_nulo_simulado": [float(distribucion.min()), float(distribucion.max())],
        "extremos": extremos, "replicas": distribucion,
        "rechaza_welch_s1": bool(welch.pvalue < alfa),
        "rechaza_permutacion": bool(p < alfa),
    }


def diferencia_recortada(fuera, tiempo, proporcion=0.10, axis=-1):
    """Diferencia de medias con un recorte de proporcion en cada cola y grupo."""
    if not 0 <= proporcion < .5:
        raise ValueError("proporcion debe pertenecer a [0, 0.5)")
    return stats.trim_mean(fuera, proportiontocut=proporcion, axis=axis) - \
           stats.trim_mean(tiempo, proportiontocut=proporcion, axis=axis)


def diagnostico_cuantiles_peso(peso_tiempo, peso_fuera, grupos_s1,
                                probabilidades=(.05,.25,.50,.75,.95)):
    """Compara cuantiles S1 con familias paramétricas calibradas SOLO por momentos S1.

    No estima parámetros extra ni interpreta un error en cuantiles como test formal
    de bondad de ajuste. El cuantíl empírico es diagnóstico de validación.
    """
    filas=[]
    for grupo, x in [("A tiempo",peso_tiempo),("Fuera de plazo",peso_fuera)]:
        x = np.asarray(x,dtype=float)
        x = x[np.isfinite(x)]
        r=grupos_s1.loc[grupo]
        m,s=float(r['media']),float(r['de'])
        if m <= 0 or s <= 0 or (x<=0).any():
            raise ValueError("Los pesos, medias y desviaciones deben ser positivos")
        sigma2 = np.log1p((s/m)**2)
        p_emp = np.quantile(x,probabilidades)
        p_log = stats.lognorm.ppf(probabilidades,s=np.sqrt(sigma2),scale=np.exp(np.log(m)-sigma2/2))
        p_gamma = stats.gamma.ppf(probabilidades,a=(m/s)**2,scale=s**2/m)
        for p,e,l,g in zip(probabilidades,p_emp,p_log,p_gamma):
            filas.append({'grupo':grupo,'probabilidad':float(p),'cuantil_empirico_g':float(e),
                          'cuantil_lognormal_g':float(l),'cuantil_gamma_g':float(g),
                          'error_abs_lognormal_g':float(abs(l-e)),
                          'error_abs_gamma_g':float(abs(g-e))})
    return pd.DataFrame(filas)
