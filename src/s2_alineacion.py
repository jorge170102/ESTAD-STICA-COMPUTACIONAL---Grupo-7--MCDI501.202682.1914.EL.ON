"""Reconstrucción de estadísticos S1 a partir del CSV MCAR, sin modificarlo."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class LineaBaseS1:
    medias: pd.DataFrame
    grupos: pd.DataFrame
    peso_tiempo: np.ndarray
    peso_fuera: np.ndarray
    delta: float
    ee_delta: float
    ic_delta: np.ndarray
    welch: object
    tabla_contingencia: pd.DataFrame
    chi: float
    p_chi: float
    gl_chi: int
    esperados: np.ndarray
    v_cramer: float
    pares: list[tuple[str, str, str]]
    correlaciones: pd.DataFrame


def resumen_media(x, alfa=0.05):
    x = np.asarray(x, dtype=float)
    n, media, de = len(x), x.mean(), x.std(ddof=1)
    ee = de / np.sqrt(n)
    margen = stats.t.ppf(1-alfa/2, n-1)*ee
    return {"n": n, "media": media, "de": de, "ee": ee,
            "ic_t_inf": media-margen, "ic_t_sup": media+margen}


def reconstruir_s1(df, variables, objetivo, *, alfa=0.05):
    medias = pd.DataFrame({v:resumen_media(df[v].dropna(), alfa) for v in variables}).T
    medias.index.name = "variable"
    peso_tiempo = df.loc[df[objetivo].eq(0), "Weight_in_gms"].dropna().to_numpy()
    peso_fuera = df.loc[df[objetivo].eq(1), "Weight_in_gms"].dropna().to_numpy()
    grupos = pd.DataFrame([resumen_media(peso_tiempo, alfa),resumen_media(peso_fuera, alfa)],
                          index=["A tiempo", "Fuera de plazo"])
    delta = float(peso_fuera.mean()-peso_tiempo.mean())
    v_t = peso_tiempo.var(ddof=1)/len(peso_tiempo)
    v_f = peso_fuera.var(ddof=1)/len(peso_fuera)
    ee_delta = float(np.sqrt(v_t+v_f))
    gl_welch = (v_t+v_f)**2/(v_t**2/(len(peso_tiempo)-1)+v_f**2/(len(peso_fuera)-1))
    ic_delta = delta + np.array([-1,1])*stats.t.ppf(1-alfa/2,gl_welch)*ee_delta
    welch = stats.ttest_ind(peso_fuera,peso_tiempo,equal_var=False)
    tabla = pd.crosstab(df.Mode_of_Shipment,df[objetivo])
    chi, p_chi, gl_chi, esperados = stats.chi2_contingency(tabla,correction=False)
    v_cramer = np.sqrt(chi/len(df))
    pares = [("Weight_in_gms","Discount_offered","Peso y descuento"),
             ("Customer_care_calls","Cost_of_the_Product","Llamadas y costo"),
             ("Customer_rating","Weight_in_gms","Valoración y peso")]
    correlaciones = []
    for a,b,nombre in pares:
        observados = df[[a,b]].dropna()
        correlaciones.append({"relacion":nombre,"n":len(observados),
                               "rho_s1":stats.spearmanr(observados[a],observados[b]).statistic})
    return LineaBaseS1(medias,grupos,peso_tiempo,peso_fuera,delta,ee_delta,ic_delta,
                       welch,tabla,float(chi),float(p_chi),int(gl_chi),esperados,
                       float(v_cramer),pares,pd.DataFrame(correlaciones))
