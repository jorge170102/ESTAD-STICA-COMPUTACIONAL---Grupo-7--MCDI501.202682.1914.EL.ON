# Grupo 7 / S3 — Implementación original de celda 11.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

# ============================================================
# Diagnóstico de los modelos de regresión para imputación
# ============================================================

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd

for variable, modelo in models_imp.items():

    predichos = modelo.fittedvalues
    residuos = modelo.resid

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gráfico 1: Residuos frente a valores predichos
    axes[0].scatter(
        predichos,
        residuos,
        alpha=0.25,
        s=12
    )

    axes[0].axhline(0, color="red", linestyle="--")

    axes[0].set_title(f"Residuos vs. predichos: {variable}")
    axes[0].set_xlabel("Valores predichos")
    axes[0].set_ylabel("Residuos")

    # Gráfico 2: Q-Q plot
    sm.qqplot(
        residuos,
        line="45",
        fit=True,
        ax=axes[1]
    )

    axes[1].set_title(f"Q-Q plot: {variable}")

    plt.tight_layout()
    plt.show()

    # Resumen de diagnóstico
    print(f"\nVariable: {variable}")
    print(f"R² ajustado: {modelo.rsquared_adj:.4f}")
    print(f"RMSE entrenamiento: {np.sqrt(np.mean(residuos**2)):.2f}")
    print(f"Media de residuos: {residuos.mean():.4f}")
