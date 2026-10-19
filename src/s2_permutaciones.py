"""Estadística computacional Grupo 7: módulo de Sumativa 2 (no modifica S1)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def permutar_transporte(df, tabla_contingencia, chi_s1, p_chi_s1, gl_chi, esperados, v_cramer, *, objetivo, B=10_000, semilla=201, alfa=0.05):
    rng_perm = np.random.default_rng(semilla)
    codigos = pd.Categorical(df.Mode_of_Shipment, categories=tabla_contingencia.index).codes
    etiquetas = df[objetivo].to_numpy(dtype=int)
    chi_nulos = np.empty(B)
    for i in range(B):
        mezcladas = rng_perm.permutation(etiquetas)
        tabla = np.bincount(2 * codigos + mezcladas, minlength=6).reshape(3, 2)
        chi_nulos[i] = np.sum((tabla - esperados)**2 / esperados)

    extremos = int(np.count_nonzero(chi_nulos >= chi_s1 - 1e-12))
    p_perm = (extremos + 1) / (B + 1)
    ee_p_perm = np.sqrt(p_perm * (1 - p_perm) / B)
    permutacion = {"chi2_observado": chi_s1, "gl": int(gl_chi),
        "p_s1": p_chi_s1, "p_permutacion": p_perm, "ee_monte_carlo_p": ee_p_perm,
        "diferencia_p": p_perm - p_chi_s1, "permutaciones": B,
        "esperado_minimo": float(esperados.min()), "v_cramer": v_cramer,
        "rechaza_s1": bool(p_chi_s1 < alfa), "rechaza_permutacion": bool(p_perm < alfa)}
    return chi_nulos, permutacion

