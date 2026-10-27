"""Bootstrap POO para coeficientes y OR de las especificaciones S3.

Método equivalente a S3: bootstrap de filas con reemplazo; GLM binomial
maxiter=100, intervalos percentiles y signos, réplicas comunes para las dos
especificaciones. Por defecto NO usa warm starts ni multiproceso para
preservar las semillas y la lógica del notebook original.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class ResultadoBootstrap:
    tabla_estabilidad: pd.DataFrame
    tabla_resumen: pd.DataFrame
    resultados: dict


class BootstrapLogistico:
    def __init__(self, n_boot: int = 2000, seed: int = 142, nivel_ic: float = 0.95,
                 maxiter: int = 100, minimo_validas: float = 0.90):
        if n_boot < 1 or not 0 < nivel_ic < 1 or not 0 < minimo_validas <= 1:
            raise ValueError("Parámetros bootstrap fuera de rango")
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.nivel_ic = float(nivel_ic)
        self.maxiter = int(maxiter)
        self.minimo_validas = float(minimo_validas)

    def evaluar(self, modelos: dict, columnas: dict,
               X: pd.DataFrame, y: pd.Series) -> ResultadoBootstrap:
        if X.index.equals(y.index) is False:
            raise ValueError("Índices X/y desalineados")
        rng = np.random.default_rng(self.seed)
        yy = y.to_numpy()
        # Mismas réplicas para cada especificación, como en notebook original.
        indices = [rng.integers(0, len(yy), size=len(yy)) for _ in range(self.n_boot)]
        resultados, tablas, resumen = {}, [], []
        alfa = 1 - self.nivel_ic
        for nombre, modelo in modelos.items():
            if nombre not in columnas:
                raise ValueError(f"Sin columnas del modelo {nombre}")
            D = sm.add_constant(X[list(columnas[nombre])], has_constant="add")
            A = D.to_numpy(dtype=float)
            replicas, fallidas = [], 0
            print(f"\nProcesando modelo: {nombre}")
            for ids in indices:
                try:
                    fit = sm.GLM(yy[ids], A[ids], family=sm.families.Binomial()).fit(
                        maxiter=self.maxiter, disp=0)
                    if fit.converged and np.all(np.isfinite(fit.params)):
                        replicas.append(np.asarray(fit.params))
                    else:
                        fallidas += 1
                except Exception:
                    fallidas += 1
            replicas = np.asarray(replicas)
            n_validas = len(replicas)
            print(f"Réplicas válidas: {n_validas}")
            print(f"Réplicas fallidas: {fallidas}")
            if n_validas < self.minimo_validas * self.n_boot:
                raise RuntimeError(f"Demasiadas réplicas fallidas en {nombre}")
            li, ls = np.quantile(replicas, [alfa / 2, 1 - alfa / 2], axis=0)
            coef = modelo.params.to_numpy()
            ci = modelo.conf_int().to_numpy()
            cambios = 100 * np.mean(np.sign(replicas) != np.sign(coef), axis=0)
            tablas.append(pd.DataFrame({
                "modelo": nombre, "variable": D.columns, "coef": coef,
                "IC_Wald_inf": ci[:, 0], "IC_Wald_sup": ci[:, 1],
                "IC_boot_inf": li, "IC_boot_sup": ls,
                "OR": np.exp(coef), "OR_boot_inf": np.exp(li),
                "OR_boot_sup": np.exp(ls),
                "incluye_0_boot": (li <= 0) & (ls >= 0),
                "cambio_signo_pct": cambios,
            }))
            resultados[nombre] = {"replicas": replicas, "columnas": list(D.columns),
                                  "validas": n_validas, "fallidas": fallidas}
            resumen.append({"modelo": nombre, "replicas_solicitadas": self.n_boot,
                            "replicas_validas": n_validas, "replicas_fallidas": fallidas,
                            "porcentaje_validas": 100 * n_validas / self.n_boot})
        return ResultadoBootstrap(pd.concat(tablas, ignore_index=True),
                                  pd.DataFrame(resumen), resultados)


def ejecutar_bootstrap(contexto: dict) -> None:
    """Conserva los nombres y los cuadros originales usados en las demás celdas."""
    from IPython.display import display
    SEED = contexto["SEED"]
    modelos_boot = {"S1_S2": contexto["models"]["S1_S2"],
                    "Stepwise_AIC": contexto["models"]["stepwise"]}
    columnas_boot = {"S1_S2": contexto["selected"]["S1_S2"],
                     "Stepwise_AIC": contexto["selected"]["stepwise"]}
    N_BOOT = 2000
    bootstrap_s3 = BootstrapLogistico(n_boot=N_BOOT, seed=SEED + 100)
    res = bootstrap_s3.evaluar(modelos_boot, columnas_boot, contexto["Ztr"], contexto["ytr"])
    print("\nRESUMEN DE EJECUCIÓN")
    display(res.tabla_resumen.round(2))
    print("\nESTABILIDAD DE COEFICIENTES")
    display(res.tabla_estabilidad.round(4))
    contexto.update({"N_BOOT": N_BOOT, "NIVEL_IC": .95, "ALFA": .05,
                     "modelos_boot": modelos_boot, "columnas_boot": columnas_boot,
                     "resultados_boot": res.resultados,
                     "tabla_estabilidad": res.tabla_estabilidad,
                     "tabla_resumen_boot": res.tabla_resumen,
                     "bootstrap_s3": bootstrap_s3})
