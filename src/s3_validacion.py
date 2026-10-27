# Grupo 7 / S3 — Implementación original de celda 35.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

# ============================================================
# 7.1. Validación de coherencia global
# ============================================================

controles = []

def registrar_control(nombre, condicion, detalle=""):
    controles.append({
        "control": nombre,
        "resultado": "OK" if bool(condicion) else "REVISAR",
        "detalle": detalle
    })

# 1. Integridad de entrenamiento y prueba
registrar_control(
    "Separación entrenamiento/prueba",
    len(Xtr.index.intersection(Xte.index)) == 0,
    "Sin índices compartidos"
)

registrar_control(
    "Alineación entrenamiento",
    Xtr.index.equals(ytr.index),
    f"N = {len(ytr)}"
)

registrar_control(
    "Alineación prueba",
    Xte.index.equals(yte.index),
    f"N = {len(yte)}"
)

# 2. Estrategias de tratamiento de faltantes
registrar_control(
    "Tamaño imputación por regresión",
    len(Xreg) == len(Xtr),
    f"N = {len(Xreg)}"
)

registrar_control(
    "Tamaño imputación por mediana",
    len(Xsimple) == len(Xtr),
    f"N = {len(Xsimple)}"
)

registrar_control(
    "Casos completos",
    Xdel.index.isin(Xtr.index).all()
    and not Xdel.isna().any().any(),
    f"N = {len(Xdel)}"
)

registrar_control(
    "Ausencia de faltantes tras imputación",
    not Xreg.isna().any().any()
    and not Xsimple.isna().any().any()
)

# 3. Integridad de los modelos logísticos
for nombre, modelo in modelos_boot.items():

    registrar_control(
        f"Convergencia {nombre}",
        modelo.converged
    )

    registrar_control(
        f"Coeficientes finitos {nombre}",
        np.isfinite(modelo.params).all()
    )

# 4. Bootstrap
for nombre, info in resultados_boot.items():

    registrar_control(
        f"Bootstrap {nombre}",
        info["validas"] + info["fallidas"] == N_BOOT
        and info["validas"] >= 0.90 * N_BOOT,
        f"{info['validas']}/{N_BOOT} réplicas válidas"
    )

# 5. Comparación de estrategias
estrategias_esperadas = {
    "Regresion", "Mediana", "Casos_completos"
}

modelos_esperados = {
    "S1_S2", "Stepwise_AIC"
}

registrar_control(
    "Cobertura de estrategias",
    set(tabla_imp["estrategia"]) == estrategias_esperadas
)

registrar_control(
    "Cobertura de especificaciones",
    set(tabla_imp["modelo"]) == modelos_esperados
)

registrar_control(
    "Número de comparaciones",
    len(tabla_imp) == 6
)

registrar_control(
    "Conjunto de prueba común",
    tabla_imp["n_test"].eq(len(yte)).all()
)

registrar_control(
    "Métricas predictivas válidas",
    tabla_imp[
        ["AUC", "accuracy", "precision", "recall", "F1"]
    ].apply(lambda s: s.between(0, 1).all()).all()
)

# 6. Resumen de controles
tabla_controles = pd.DataFrame(controles)

display(tabla_controles)

n_ok = (tabla_controles["resultado"] == "OK").sum()
n_total = len(tabla_controles)

print(f"\nControles satisfactorios: {n_ok}/{n_total}")

assert n_ok == n_total, (
    "Se detectaron inconsistencias. "
    "Revisar los controles marcados como REVISAR."
)
