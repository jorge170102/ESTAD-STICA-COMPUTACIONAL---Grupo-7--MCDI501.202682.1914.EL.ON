"""Imputación determinista por dos regresiones múltiples de S3.

Replica exactamente el orden y las decisiones del notebook original:
OLS con HC3 sobre observaciones completas, inicialización por mediana,
2 pasadas secuenciales costo/peso y restricción explícita >= 1.
La salida NO incorpora incertidumbre de imputación múltiple.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping
import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class ImputadorRegresionMultiple:
    predictores: Mapping[str, list[str]]
    variables_faltantes: tuple[str, ...] = ("Cost_of_the_Product", "Weight_in_gms")
    pasadas: int = 2
    minimo_positivo: float = 1.0
    modelos_: dict = field(default_factory=dict, init=False)
    resumen_: list[dict] = field(default_factory=list, init=False)
    medianas_: dict = field(default_factory=dict, init=False)

    def ajustar(self, X_entrenamiento: pd.DataFrame) -> "ImputadorRegresionMultiple":
        self.modelos_.clear()
        self.resumen_.clear()
        self.medianas_ = {v: X_entrenamiento[v].median() for v in self.variables_faltantes}
        for variable, predictores in self.predictores.items():
            observados = X_entrenamiento[[variable] + list(predictores)].dropna()
            if not len(observados):
                raise ValueError(f"No hay casos completos para {variable}")
            modelo = sm.OLS(
                observados[variable],
                sm.add_constant(observados[list(predictores)], has_constant="add"),
            ).fit(cov_type="HC3")
            self.modelos_[variable] = modelo
            residuos = modelo.resid
            from statsmodels.stats.diagnostic import het_breuschpagan
            self.resumen_.append({
                "variable": variable, "n_completos": len(observados),
                "R2_ajustado": modelo.rsquared_adj,
                "RMSE_entrenamiento": np.sqrt(np.mean(residuos**2)),
                "p_Breusch_Pagan": het_breuschpagan(modelo.resid, modelo.model.exog)[1],
            })
        return self

    def transformar(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.modelos_:
            raise RuntimeError("Debe llamar a ajustar() antes de transformar()")
        d = frame.copy()
        faltantes = {v: d[v].isna().to_numpy() for v in self.variables_faltantes}
        for v in self.variables_faltantes:
            d[v] = d[v].fillna(self.medianas_[v])
        for _ in range(self.pasadas):
            for variable, predictores in self.predictores.items():
                if faltantes[variable].any():
                    pred = self.modelos_[variable].predict(
                        sm.add_constant(d.loc[faltantes[variable], list(predictores)], has_constant="add")
                    )
                    d.loc[faltantes[variable], variable] = np.maximum(pred, self.minimo_positivo)
        return d

    def ajustar_transformar(self, X_entrenamiento: pd.DataFrame) -> pd.DataFrame:
        return self.ajustar(X_entrenamiento).transformar(X_entrenamiento)


def ejecutar_imputacion(contexto: dict) -> None:
    """Interfaz compacta para el notebook; conserva los nombres de S3 original."""
    Xtr = contexto["Xtr"]
    MISS = contexto["MISS"]
    IMP_PRED = {
        "Cost_of_the_Product": ["Customer_care_calls", "Prior_purchases", "Discount_offered", "Weight_in_gms"],
        "Weight_in_gms": ["Discount_offered", "Cost_of_the_Product", "Customer_care_calls", "Prior_purchases"],
    }
    imputador = ImputadorRegresionMultiple(IMP_PRED, tuple(MISS))
    Xreg = imputador.ajustar_transformar(Xtr)
    for v, modelo in imputador.modelos_.items():
        print("\n", v, "\n", modelo.summary().tables[1])
    print(pd.DataFrame(imputador.resumen_).round(4))
    Xsimple = Xtr.copy()
    for v in MISS:
        Xsimple[v] = Xsimple[v].fillna(Xtr[v].median())
    complete_mask = Xtr[MISS].notna().all(axis=1)
    Xdel = Xtr.loc[complete_mask]
    ydel = contexto["ytr"].loc[complete_mask]
    assert not Xreg.isna().any().any() and not Xsimple.isna().any().any()
    print("Imputados por regresión:", Xtr[MISS].isna().sum().to_dict())
    contexto.update({
        "IMP_PRED": IMP_PRED, "imputador_s3": imputador,
        "models_imp": imputador.modelos_, "imp_stats": imputador.resumen_,
        "regression_impute": imputador.transformar,
        "Xreg": Xreg, "Xsimple": Xsimple, "complete_mask": complete_mask,
        "Xdel": Xdel, "ydel": ydel,
    })
