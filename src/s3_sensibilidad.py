# Grupo 7 / S3 — Implementación original de celda 31.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

# ============================================================
# 6. Impacto de las estrategias de tratamiento de faltantes
# ============================================================

# 1. Estrategias de tratamiento de datos faltantes
frames_imp = {
    "Regresion": Xreg,
    "Mediana": Xsimple,
    "Casos_completos": Xdel
}

# 2. Especificaciones logísticas previamente definidas
especificaciones_imp = {
    "S1_S2": selected["S1_S2"],
    "Stepwise_AIC": selected["stepwise"]
}

# 3. Estructuras de resultados
imp_compare = []
coef_compare = []
modelos_imp = {}

# 4. Evaluación de cada estrategia
for estrategia, d in frames_imp.items():

    print(f"\nProcesando estrategia: {estrategia}")

    # Alinear la variable objetivo con el entrenamiento
    yy = ytr.loc[d.index]

    # Crear un preprocesador independiente
    pp = clone(pre)

    # Ajustar transformaciones sobre entrenamiento
    z = pd.DataFrame(
        pp.fit_transform(d),
        columns=pp.get_feature_names_out(),
        index=d.index
    ).astype(float)

    # Aplicar las mismas transformaciones a prueba
    ztest = pd.DataFrame(
        pp.transform(Xte),
        columns=pp.get_feature_names_out(),
        index=Xte.index
    ).astype(float)

    # Comprobaciones básicas
    assert z.index.equals(yy.index)
    assert z.columns.equals(ztest.columns)
    assert not z.isna().any().any()
    assert not ztest.isna().any().any()

    # 5. Ajustar cada especificación logística
    for nombre_modelo, columnas in especificaciones_imp.items():

        print(f"  Modelo: {nombre_modelo}")

        # Verificar predictores seleccionados
        faltantes = [
            c for c in columnas
            if c not in z.columns
        ]

        assert not faltantes, (
            f"Columnas ausentes en {estrategia}: {faltantes}"
        )

        # Matrices de diseño
        Dtr = sm.add_constant(
            z[columnas],
            has_constant="add"
        )

        Dte = sm.add_constant(
            ztest[columnas],
            has_constant="add"
        )

        # Verificar rango completo
        assert np.linalg.matrix_rank(
            Dtr.to_numpy()
        ) == Dtr.shape[1], (
            f"Matriz sin rango completo: "
            f"{estrategia} / {nombre_modelo}"
        )

        # Ajustar regresión logística
        m = sm.GLM(
            yy,
            Dtr,
            family=sm.families.Binomial()
        ).fit(
            maxiter=150,
            disp=0
        )

        assert m.converged, (
            f"Modelo no convergió: "
            f"{estrategia} / {nombre_modelo}"
        )

        # Guardar modelo
        modelos_imp[(estrategia, nombre_modelo)] = m

        # Predicciones sobre el mismo conjunto de prueba
        pred = m.predict(Dte)

        pred_clase = (pred >= 0.5).astype(int)

        # 6. Métricas predictivas
        imp_compare.append({
            "estrategia": estrategia,
            "modelo": nombre_modelo,
            "n_train": len(d),
            "n_test": len(yte),
            "AUC": roc_auc_score(yte, pred),
            "accuracy": accuracy_score(yte, pred_clase),
            "precision": precision_score(
                yte, pred_clase, zero_division=0
            ),
            "recall": recall_score(
                yte, pred_clase, zero_division=0
            ),
            "F1": f1_score(
                yte, pred_clase, zero_division=0
            ),
            "AIC_train": m.aic
        })

        # 7. Coeficientes e intervalos Wald
        conf = m.conf_int()

        for variable in m.params.index:

            coef_compare.append({
                "estrategia": estrategia,
                "modelo": nombre_modelo,
                "variable": variable,
                "coef": m.params[variable],
                "IC_inf": conf.loc[variable, 0],
                "IC_sup": conf.loc[variable, 1],
                "OR": np.exp(m.params[variable])
            })

# 8. Consolidar resultados
tabla_imp = pd.DataFrame(imp_compare)
tabla_coef_imp = pd.DataFrame(coef_compare)

print("\nCOMPARACIÓN DE DESEMPEÑO PREDICTIVO")
display(
    tabla_imp.sort_values(
        ["modelo", "estrategia"]
    ).round(4)
)

print("\nCOMPARACIÓN DE COEFICIENTES")
display(
    tabla_coef_imp.sort_values(
        ["modelo", "variable", "estrategia"]
    ).round(4)
)
