# Grupo 7 / S3 — Implementación original de celda 36.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

# ============================================================
# 7.2. Síntesis de resultados
# ============================================================

# A. Desempeño predictivo según estrategia y modelo
resumen_predictivo = tabla_imp[
    [
        "modelo", "estrategia", "n_train", "n_test",
        "AUC", "accuracy", "F1"
    ]
].copy()

print("SÍNTESIS DE DESEMPEÑO PREDICTIVO")
display(
    resumen_predictivo.sort_values(
        ["modelo", "estrategia"]
    ).round(4)
)

# B. Rango de variación de métricas entre estrategias
variacion_metricas = (
    tabla_imp.groupby("modelo")
    .agg(
        AUC_min=("AUC", "min"),
        AUC_max=("AUC", "max"),
        accuracy_min=("accuracy", "min"),
        accuracy_max=("accuracy", "max"),
        F1_min=("F1", "min"),
        F1_max=("F1", "max")
    )
    .reset_index()
)

variacion_metricas["rango_AUC"] = (
    variacion_metricas["AUC_max"]
    - variacion_metricas["AUC_min"]
)

variacion_metricas["rango_accuracy"] = (
    variacion_metricas["accuracy_max"]
    - variacion_metricas["accuracy_min"]
)

variacion_metricas["rango_F1"] = (
    variacion_metricas["F1_max"]
    - variacion_metricas["F1_min"]
)

print("\nVARIACIÓN ENTRE ESTRATEGIAS")
display(variacion_metricas.round(4))

# C. Estabilidad bootstrap de los predictores
resumen_estabilidad = (
    tabla_estabilidad.loc[
        tabla_estabilidad["variable"] != "const",
        [
            "modelo", "variable", "OR",
            "OR_boot_inf", "OR_boot_sup",
            "incluye_0_boot", "cambio_signo_pct"
        ]
    ]
    .copy()
)

print("\nSÍNTESIS DE ESTABILIDAD BOOTSTRAP")
display(resumen_estabilidad.round(4))

# D. Consistencia del signo entre estrategias
signos = tabla_coef_imp.loc[
    tabla_coef_imp["variable"] != "const"
].copy()

signos["signo"] = np.sign(signos["coef"])

consistencia_signos = (
    signos.groupby(["modelo", "variable"])
    .agg(
        signos_distintos=("signo", "nunique"),
        coef_min=("coef", "min"),
        coef_max=("coef", "max")
    )
    .reset_index()
)

consistencia_signos["signo_estable"] = (
    consistencia_signos["signos_distintos"] == 1
)

print("\nCONSISTENCIA DE SIGNOS ENTRE ESTRATEGIAS")
display(consistencia_signos.round(4))

# E. Comprobación de coherencia de signos
assert consistencia_signos["signo_estable"].all(), (
    "Existen variables cuyo signo cambia "
    "entre estrategias de tratamiento."
)
