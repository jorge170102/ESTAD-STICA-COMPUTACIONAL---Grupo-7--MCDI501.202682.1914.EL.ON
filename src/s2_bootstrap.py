"""Estadística computacional Grupo 7: módulo de Sumativa 2 (no modifica S1)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def bootstrap_media(x, semilla, B=10_000, LOTE=50):
    """IC BCa de SciPy e IC percentil a partir de las mismas remuestras."""
    x = np.asarray(x, dtype=float)
    resultado = stats.bootstrap((x,), np.mean, vectorized=True,
        n_resamples=B, batch=LOTE, confidence_level=0.95, method="BCa",
        rng=np.random.default_rng(semilla))
    distribucion = resultado.bootstrap_distribution
    percentil = np.quantile(distribucion, [0.025, 0.975])
    bca = np.array(resultado.confidence_interval)
    assert len(distribucion) == B and np.isfinite(bca).all()
    return distribucion, percentil, bca


def analizar_medias(df, variables, base_medias, semillas, B=10_000, LOTE=50):
    distribuciones_medias, filas = {}, []
    for variable, semilla in zip(variables, semillas):
        x = df[variable].dropna().to_numpy()
        distribucion, percentil, bca = bootstrap_media(x, semilla, B=B, LOTE=LOTE)
        distribuciones_medias[variable] = distribucion
        base = base_medias.loc[variable]
        ancho_t = base.ic_t_sup - base.ic_t_inf
        filas.append({"variable": variable, **base.to_dict(),
            "percentil_inf": percentil[0], "percentil_sup": percentil[1],
            "bca_inf": bca[0], "bca_sup": bca[1],
            "ancho_t": ancho_t, "ancho_percentil": np.diff(percentil)[0],
            "ancho_bca": np.diff(bca)[0],
            "cambio_ancho_bca_pct": 100 * (np.diff(bca)[0] / ancho_t - 1),
            "desplazamiento_limites_en_ee": np.max(np.abs(bca - [base.ic_t_inf, base.ic_t_sup])) / base.ee,
            "sesgo_bootstrap": distribucion.mean() - x.mean(),
            "ee_bootstrap": distribucion.std(ddof=1),
            "asimetria_datos": stats.skew(x, bias=False),
            "asimetria_medias_bootstrap": stats.skew(distribucion, bias=False)})
    bootstrap_medias = pd.DataFrame(filas).set_index("variable")

    tabla_ic = pd.DataFrame(index=variables)
    tabla_ic["n"] = base_medias.n.astype(int)
    tabla_ic["Media"] = base_medias.media.round(3)
    for titulo, inferior, superior in [("IC clásico S1", "ic_t_inf", "ic_t_sup"),
            ("IC percentil", "percentil_inf", "percentil_sup"), ("IC BCa", "bca_inf", "bca_sup")]:
        tabla_ic[titulo] = bootstrap_medias.apply(lambda r: f"[{r[inferior]:.3f}; {r[superior]:.3f}]", axis=1)
    return distribuciones_medias, bootstrap_medias, tabla_ic

