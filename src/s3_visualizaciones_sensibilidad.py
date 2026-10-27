# Grupo 7 / S3 — Implementación original de celda 32.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

# ============================================================
# 6.1. Comparación gráfica de odds ratios
# ============================================================

for nombre_modelo in especificaciones_imp:

    datos = tabla_coef_imp.loc[
        (tabla_coef_imp["modelo"] == nombre_modelo)
        & (tabla_coef_imp["variable"] != "const")
    ]

    comparacion = datos.pivot(
        index="variable",
        columns="estrategia",
        values="OR"
    )

    ax = comparacion.plot(
        kind="barh",
        figsize=(10, 5),
        width=0.75
    )

    ax.axvline(
        1,
        color="black",
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        f"Comparación de OR según estrategia — {nombre_modelo}"
    )
    ax.set_xlabel("Odds ratio")
    ax.set_ylabel("Variable")
    ax.legend(title="Estrategia")

    plt.tight_layout()
    plt.show()
