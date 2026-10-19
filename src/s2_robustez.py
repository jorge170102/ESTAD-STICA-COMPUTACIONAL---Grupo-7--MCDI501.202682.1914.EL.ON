"""Estadística computacional Grupo 7: módulo de Sumativa 2 (no modifica S1)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def analizar_jackknife(df, variables, base_medias):
    filas_influencia, resumen_influencia = [], []
    for variable in variables:
        datos = df[["ID", variable]].dropna()
        x = datos[variable].to_numpy()
        base = base_medias.loc[variable]
        medias_sin_uno = (x.sum() - x) / (len(x) - 1)
        cambios = medias_sin_uno - x.mean()
        ee_jackknife = np.sqrt((len(x)-1)/len(x) * np.sum((medias_sin_uno-medias_sin_uno.mean())**2))
        casos = pd.DataFrame({"variable": variable, "ID": datos.ID.to_numpy(),
            "valor": x, "media_sin_caso": medias_sin_uno, "cambio_media": cambios,
            "cambio_abs_en_ee": np.abs(cambios)/base.ee})
        filas_influencia.append(casos.sort_values(["cambio_abs_en_ee", "ID"], ascending=[False, True]).head(3))
        resumen_influencia.append({"variable": variable, "max_cambio_absoluto": np.abs(cambios).max(),
            "max_cambio_en_ee": np.abs(cambios).max()/base.ee,
            "ee_s1": base.ee, "ee_jackknife": ee_jackknife})
    influencia = pd.concat(filas_influencia, ignore_index=True)
    resumen_influencia = pd.DataFrame(resumen_influencia).set_index("variable")
    return influencia, resumen_influencia


from src.s2_bootstrap import bootstrap_media
from src.s2_correlaciones import bootstrap_correlacion

def analizar_extremos(df, variables, bootstrap_medias, correlaciones, semillas, *, B=10_000, LOTE=50):
    filas = []
    for variable in variables:
        x = df[variable].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        inferior, superior = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
        fuera = (x < inferior) | (x > superior)
        filas.append({"variable": variable, "limite_inf": inferior, "limite_sup": superior,
            "n_validos": len(x), "marcados": int(fuera.sum()), "porcentaje": 100*fuera.mean()})
    auditoria_extremos = pd.DataFrame(filas).set_index("variable")


    limites = auditoria_extremos.loc["Discount_offered"]
    conservar = df.Discount_offered.between(limites.limite_inf, limites.limite_sup)
    df_sensible = df.loc[conservar].copy()
    n_excluidos = int((~conservar).sum())
    dist_desc_filtrado, pct_desc_filtrado, bca_desc_filtrado = bootstrap_media(
        df_sensible.Discount_offered, semillas["descuento_filtrado"], B=B, LOTE=LOTE)
    pares_filtrados = df_sensible[["Weight_in_gms", "Discount_offered"]].dropna()
    dist_rho_filtrado, ic_rho_filtrado = bootstrap_correlacion(pares_filtrados, semillas["correlacion_filtrada"], B=B, LOTE=LOTE)
    media_desc_filtrado = df_sensible.Discount_offered.mean()
    rho_filtrado = stats.spearmanr(pares_filtrados.iloc[:, 0], pares_filtrados.iloc[:, 1]).statistic
    desc_base = bootstrap_medias.loc["Discount_offered"]
    rho_base = correlaciones.loc["Peso y descuento"]
    sensibilidad_extremos = pd.DataFrame([
        {"estimador": "Media descuento (IC BCa)", "n_base": len(df), "n_escenario": len(df_sensible),
         "base": desc_base.media, "escenario": media_desc_filtrado,
         "ic_base_inf": desc_base.bca_inf, "ic_base_sup": desc_base.bca_sup,
         "ic_escenario_inf": bca_desc_filtrado[0], "ic_escenario_sup": bca_desc_filtrado[1]},
        {"estimador": "Spearman peso–descuento (IC percentil)", "n_base": int(rho_base.n), "n_escenario": len(pares_filtrados),
         "base": rho_base.rho_s1, "escenario": rho_filtrado,
         "ic_base_inf": rho_base.ic_inf, "ic_base_sup": rho_base.ic_sup,
         "ic_escenario_inf": ic_rho_filtrado[0], "ic_escenario_sup": ic_rho_filtrado[1]},
    ])
    sensibilidad_extremos["cambio"] = sensibilidad_extremos.escenario - sensibilidad_extremos.base
    return auditoria_extremos, sensibilidad_extremos, n_excluidos, pct_desc_filtrado, media_desc_filtrado, rho_filtrado

def comparar_familias(peso_fuera, peso_tiempo, mc_lognormal, mc_gamma, delta_s1, ee_delta, *, semilla=403, B=10_000, LOTE=50):
    # mc_gamma ya se calculó en la sección 6.2, junto con su cobertura de Welch.

    def diferencia_medias(fuera, tiempo, axis=-1):
        return fuera.mean(axis=axis) - tiempo.mean(axis=axis)

    boot_delta = stats.bootstrap((peso_fuera, peso_tiempo), diferencia_medias,
        paired=False, vectorized=True, n_resamples=B, batch=LOTE,
        method="percentile", confidence_level=0.95,
        rng=np.random.default_rng(semilla))
    dist_delta_boot = boot_delta.bootstrap_distribution
    comparacion_modelos = pd.DataFrame([
        {"metodo": "Monte Carlo lognormal", **resumir_distribucion(mc_lognormal)},
        {"metodo": "Monte Carlo Gamma", **resumir_distribucion(mc_gamma)},
        {"metodo": "Bootstrap por estado", **resumir_distribucion(dist_delta_boot)},
    ]).set_index("metodo")
    comparacion_modelos["ancho_95"] = comparacion_modelos.p97_5 - comparacion_modelos.p2_5
    comparacion_modelos["cambio_media_vs_s1"] = comparacion_modelos.media - delta_s1
    comparacion_modelos["cambio_de_vs_s1_pct"] = 100*(comparacion_modelos.de/ee_delta-1)
    return dist_delta_boot, comparacion_modelos

from src.s2_montecarlo import resumir_distribucion

from src.validacion_s2 import diferencia_recortada

def analizar_medias_recortadas(peso_fuera, peso_tiempo, delta_s1, *, semilla=603, B=10_000, LOTE=50):

    def delta_media_recortada(fuera, tiempo, axis=-1):
        return diferencia_recortada(fuera, tiempo, proporcion=.10, axis=axis)

    delta_recortada = float(delta_media_recortada(peso_fuera, peso_tiempo))
    boot_recortada = stats.bootstrap(
        (peso_fuera, peso_tiempo), delta_media_recortada,
        paired=False, vectorized=True, n_resamples=B, batch=LOTE,
        method="percentile", confidence_level=.95,
        rng=np.random.default_rng(semilla),
    )
    ic_recortada = np.array(boot_recortada.confidence_interval, dtype=float)
    resumen_recortadas = {
        "proporcion_recortada_en_cada_cola": .10,
        "diferencia_media_s1_g": float(delta_s1),
        "diferencia_recortada_g": delta_recortada,
        "cambio_respecto_s1_g": delta_recortada - delta_s1,
        "ic95_bootstrap_percentil_g": [float(ic_recortada[0]), float(ic_recortada[1])],
        "iteraciones": B,
        "semilla": semilla,
        "mismo_signo": bool(np.sign(delta_recortada) == np.sign(delta_s1)),
        "ic_excluye_cero": bool(ic_recortada[0] * ic_recortada[1] > 0),
    }
    return delta_recortada, boot_recortada, ic_recortada, resumen_recortadas
