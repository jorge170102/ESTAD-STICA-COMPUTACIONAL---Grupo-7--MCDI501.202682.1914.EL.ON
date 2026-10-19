"""Estadística computacional Grupo 7: módulo de Sumativa 2 (no modifica S1)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

class SimuladorMonteCarlo:
    """Ensayos independientes para comparar familias y cobertura de Welch.

    Calibración con n/media/DE de S1; RNG por llamada (semilla explícita).
    Nunca modifica los parámetros originales.
    """
    def __init__(self, parametros, *, iteraciones=10_000, lote=50, alfa=0.05):
        if iteraciones < 2 or lote < 1 or not 0 < alfa < 1:
            raise ValueError("iteraciones, lote o alfa no válidos")
        requeridos = {"A tiempo", "Fuera de plazo"}
        if not requeridos.issubset(set(parametros.index)):
            raise ValueError("Faltan grupos de entrega de S1")
        if not {"n", "media", "de"}.issubset(parametros.columns):
            raise ValueError("Faltan parámetros de S1")
        self.parametros = parametros[["n", "media", "de"]].copy(deep=True)
        if (self.parametros[["media", "de"]] <= 0).any().any() or (self.parametros["n"] < 2).any():
            raise ValueError("Medias y desviaciones deben ser positivas, n >= 2")
        self.iteraciones, self.lote, self.alfa = int(iteraciones), int(lote), float(alfa)

    def simular(self, familia, semilla, media_fuera=None, *, devolver_ic=False):
        """Simula M estudios independientes condicionados a momentos y tamaños de S1.

        Siempre devuelve delta_simulada (fuera de plazo - a tiempo). Si devolver_ic=True,
        devuelve además los IC Welch del 95 % de CADA estudio, construidos con las
        varianzas muestrales SIMULADAS, no las varianzas fijas usadas para generar.
        La cobertura se evalúa respecto al delta verdadero definido por el modelo.
        """
        rng = np.random.default_rng(semilla)
        salida = np.empty(self.iteraciones)
        limites_inf = np.empty(self.iteraciones) if devolver_ic else None
        limites_sup = np.empty(self.iteraciones) if devolver_ic else None
        parametros = self.parametros[["n", "media", "de"]].copy()
        if media_fuera is not None:
            parametros.loc["Fuera de plazo", "media"] = media_fuera
        n0 = int(parametros.loc["A tiempo", "n"])
        n1 = int(parametros.loc["Fuera de plazo", "n"])
        delta_modelo = float(parametros.loc["Fuera de plazo", "media"]
                              - parametros.loc["A tiempo", "media"])
        for inicio in range(0, self.iteraciones, self.lote):
            cantidad = min(self.lote, self.iteraciones - inicio)
            medias = []
            varianzas_medias = []
            for _, r in parametros.iterrows():
                forma = (cantidad, int(r.n))
                if familia == "lognormal":
                    sigma2 = np.log1p((r.de / r.media)**2)
                    muestra = rng.lognormal(np.log(r.media) - sigma2 / 2,
                                           np.sqrt(sigma2), size=forma)
                elif familia == "gamma":
                    muestra = rng.gamma(shape=(r.media/r.de)**2,
                                        scale=r.de**2/r.media, size=forma)
                else:
                    raise ValueError("Familia no reconocida.")
                medias.append(muestra.mean(axis=1))
                if devolver_ic:
                    varianzas_medias.append(muestra.var(axis=1, ddof=1) / int(r.n))
            delta = medias[1] - medias[0]
            salida[inicio:inicio+cantidad] = delta
            if devolver_ic:
                v0, v1 = varianzas_medias
                error = np.sqrt(v0 + v1)
                grados = (v0 + v1)**2 / (v0**2 / (n0 - 1) + v1**2 / (n1 - 1))
                tcrit = stats.t.ppf(1 - self.alfa/2, grados)
                limites_inf[inicio:inicio+cantidad] = delta - tcrit*error
                limites_sup[inicio:inicio+cantidad] = delta + tcrit*error
        if not devolver_ic:
            return salida
        cubre = (limites_inf <= delta_modelo) & (limites_sup >= delta_modelo)
        p_cubre = float(cubre.mean())
        return salida, {
            "familia": familia,
            "iteraciones": self.iteraciones,
            "parametro_delta_verdadero_modelo_g": delta_modelo,
            "cobertura_ic_welch_95_pct": 100*p_cubre,
            "error_montecarlo_cobertura_pp": 100*float(np.sqrt(p_cubre*(1-p_cubre)/self.iteraciones)),
            "ancho_medio_ic_welch_g": float(np.mean(limites_sup-limites_inf)),
            "porcentaje_ic_excluye_cero": 100*float(np.mean((limites_inf > 0) | (limites_sup < 0))),
        }

def resumir_distribucion(valores):
    """Resumen numérico sin confundir DE y error Monte Carlo de la media."""
    q = np.quantile(valores, [0.025, 0.975])
    return {"iteraciones": len(valores), "media": valores.mean(),
            "de": valores.std(ddof=1), "p2_5": q[0], "p97_5": q[1],
            "error_mc_media": valores.std(ddof=1) / np.sqrt(len(valores))}


def analizar_escenarios(simulador, grupos_s1, ic_delta_s1, delta_s1, mc_lognormal, semillas):
    escenarios = []
    distribuciones_escenarios = {}
    for nombre, objetivo_delta, semilla in [
        ("Extremo inferior IC S1", ic_delta_s1[0], semillas["escenario_inferior"]),
        ("Central S1", delta_s1, semillas["lognormal"]),
        ("Extremo superior IC S1", ic_delta_s1[1], semillas["escenario_superior"])]:
        nueva_media = grupos_s1.loc["A tiempo", "media"] + objetivo_delta
        valores = mc_lognormal if nombre == "Central S1" else simulador.simular("lognormal", semilla, nueva_media)
        distribuciones_escenarios[nombre] = valores
        escenarios.append({"escenario": nombre, "delta_entrada": objetivo_delta,
            "media_fuera_entrada": nueva_media, **resumir_distribucion(valores)})
    sensibilidad_entradas = pd.DataFrame(escenarios).set_index("escenario")
    return sensibilidad_entradas, distribuciones_escenarios
