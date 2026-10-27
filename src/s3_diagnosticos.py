# Grupo 7 / S3 — Implementación original de celda 22.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

# ============================================================
# 4.3. DIAGNÓSTICO DE LOS MODELOS LOGÍSTICOS
# ============================================================

from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from patsy import dmatrix

# Dos especificaciones distintas:
# Stepwise y AIC seleccionaron las mismas variables.
modelos_diagnostico = {
    "S1_S2": models["S1_S2"],
    "Stepwise_AIC": models["stepwise"]
}

columnas_diagnostico = {
    "S1_S2": selected["S1_S2"],
    "Stepwise_AIC": selected["stepwise"]
}

resultados_vif = []
resultados_influencia = []
resultados_linealidad = []

n = len(Ztr)

for nombre, modelo in modelos_diagnostico.items():

    columnas = columnas_diagnostico[nombre]

    # Matriz de diseño utilizada por el modelo
    D = sm.add_constant(
        Ztr[columnas],
        has_constant="add"
    )

    print("\n" + "=" * 65)
    print(f"DIAGNÓSTICO DEL MODELO: {nombre}")
    print("=" * 65)
    print(f"Observaciones: {n}")
    print(f"Predictores: {len(columnas)}")

    # --------------------------------------------------------
    # 1. MULTICOLINEALIDAD (VIF)
    # --------------------------------------------------------

    vif = pd.DataFrame({
        "modelo": nombre,
        "variable": columnas,
        "VIF": [
            variance_inflation_factor(
                D.to_numpy(),
                j + 1
            )
            for j in range(len(columnas))
        ]
    })

    resultados_vif.append(vif)

    print("\n1. FACTOR DE INFLACIÓN DE LA VARIANZA (VIF)")
    print(
        vif[["variable", "VIF"]]
        .round(3)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 2. RESIDUOS Y OBSERVACIONES INFLUYENTES
    # --------------------------------------------------------

    influencia = modelo.get_influence(observed=True)

    leverage = np.asarray(
        influencia.hat_matrix_diag
    )

    cook = np.asarray(
        influencia.cooks_distance[0]
    )

    residuos = np.asarray(
        modelo.resid_pearson
    )

    probabilidades = np.asarray(
        modelo.fittedvalues
    )

    umbral_cook = 4 / n

    # Regla exploratoria habitual:
    # leverage > 2p/n, con p = parámetros + intercepto.
    p = D.shape[1]
    umbral_leverage = 2 * p / n

    resumen = {
        "modelo": nombre,
        "n": n,
        "Cook_mayor_4_n": int(
            np.sum(cook > umbral_cook)
        ),
        "Cook_max": float(np.max(cook)),
        "Leverage_mayor_2p_n": int(
            np.sum(leverage > umbral_leverage)
        ),
        "Leverage_max": float(np.max(leverage)),
        "Residuos_abs_mayor_3": int(
            np.sum(np.abs(residuos) > 3)
        )
    }

    resultados_influencia.append(resumen)

    print("\n2. RESIDUOS E INFLUENCIA")
    print(f"Cook > 4/n: {resumen['Cook_mayor_4_n']}")
    print(f"Cook máximo: {resumen['Cook_max']:.5f}")
    print(
        "Leverage > 2p/n:",
        resumen["Leverage_mayor_2p_n"]
    )
    print(
        f"Leverage máximo: {resumen['Leverage_max']:.5f}"
    )
    print(
        "|Residuo Pearson| > 3:",
        resumen["Residuos_abs_mayor_3"]
    )

    # --------------------------------------------------------
    # 3. GRÁFICOS DE DIAGNÓSTICO
    # --------------------------------------------------------

    fig, axs = plt.subplots(
        1, 2,
        figsize=(12, 4)
    )

    # Residuos de Pearson
    axs[0].scatter(
        probabilidades,
        residuos,
        s=7,
        alpha=0.30
    )

    axs[0].axhline(
        0,
        linestyle="--",
        color="black",
        linewidth=1
    )

    axs[0].set(
        xlabel="Probabilidad ajustada",
        ylabel="Residuo de Pearson",
        title="Residuos vs. probabilidades"
    )

    # Leverage y distancia de Cook
    axs[1].scatter(
        leverage,
        cook,
        s=7,
        alpha=0.30
    )

    axs[1].axhline(
        umbral_cook,
        linestyle="--",
        color="red",
        linewidth=1,
        label="Umbral 4/n"
    )

    axs[1].axvline(
        umbral_leverage,
        linestyle=":",
        color="black",
        linewidth=1,
        label="Umbral 2p/n"
    )

    axs[1].set(
        xlabel="Leverage",
        ylabel="Distancia de Cook",
        title="Observaciones influyentes"
    )

    axs[1].legend()

    fig.suptitle(
        f"Diagnóstico del modelo: {nombre}"
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 4. LINEALIDAD DEL LOGIT MEDIANTE SPLINES
    # --------------------------------------------------------

    print("\n3. LINEALIDAD DEL LOGIT")

    variables_numericas = [
        v for v in columnas
        if v in num_cols
    ]

    for variable in variables_numericas:

        try:
            x = Ztr[variable].astype(float)

            # Evitar splines en variables con muy pocos
            # valores distintos.
            if x.nunique() < 5:

                resultados_linealidad.append({
                    "modelo": nombre,
                    "variable": variable,
                    "LR": np.nan,
                    "gl": np.nan,
                    "p_valor": np.nan,
                    "estado": "Pocos valores distintos"
                })

                print(
                    f"{variable}: no evaluada "
                    "(menos de 5 valores distintos)"
                )
                continue

            # Spline cúbico natural centrado:
            # evita redundancia con el intercepto.
            spline = dmatrix(
                "cr(x, df=4, constraints='center') - 1",
                {"x": x.to_numpy()},
                return_type="dataframe"
            )

            spline.index = Ztr.index

            spline.columns = [
                f"{variable}_spline_{j}"
                for j in range(spline.shape[1])
            ]

            # Mantener los demás predictores del modelo.
            otras_columnas = [
                c for c in columnas
                if c != variable
            ]

            D_spline = pd.concat(
                [
                    Ztr[otras_columnas],
                    spline
                ],
                axis=1
            )

            D_spline = sm.add_constant(
                D_spline,
                has_constant="add"
            )

            # Verificar independencia lineal.
            if (
                np.linalg.matrix_rank(D_spline.to_numpy())
                < D_spline.shape[1]
            ):
                raise ValueError(
                    "Matriz spline sin rango completo"
                )

            modelo_spline = sm.GLM(
                ytr,
                D_spline,
                family=sm.families.Binomial()
            ).fit(maxiter=150, disp=0)

            if not modelo_spline.converged:
                raise RuntimeError(
                    "Modelo spline sin convergencia"
                )

            # Comparación con el modelo logístico original.
            LR = max(
                0.0,
                2 * (modelo_spline.llf - modelo.llf)
            )

            gl = int(
                modelo_spline.df_model - modelo.df_model
            )

            if gl <= 0:
                raise ValueError(
                    "Sin grados de libertad adicionales"
                )

            p_valor = stats.chi2.sf(LR, gl)

            resultados_linealidad.append({
                "modelo": nombre,
                "variable": variable,
                "LR": LR,
                "gl": gl,
                "p_valor": p_valor,
                "estado": "Evaluada"
            })

            print(
                f"{variable}: "
                f"LR={LR:.3f}, "
                f"gl={gl}, "
                f"p={p_valor:.4f}"
            )

        except (
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError
        ) as e:

            resultados_linealidad.append({
                "modelo": nombre,
                "variable": variable,
                "LR": np.nan,
                "gl": np.nan,
                "p_valor": np.nan,
                "estado": str(e)[:80]
            })

            print(
                f"{variable}: diagnóstico no estimable "
                f"({str(e)[:80]})"
            )

# ============================================================
# 5. TABLAS CONSOLIDADAS
# ============================================================

tabla_vif = pd.concat(
    resultados_vif,
    ignore_index=True
)

tabla_influencia = pd.DataFrame(
    resultados_influencia
)

tabla_linealidad = pd.DataFrame(
    resultados_linealidad
)

print("\n" + "=" * 65)
print("RESUMEN COMPARATIVO DE INFLUENCIA")
print("=" * 65)

print(
    tabla_influencia
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 65)
print("RESUMEN DE LINEALIDAD DEL LOGIT")
print("=" * 65)

print(
    tabla_linealidad
    .round(4)
    .to_string(index=False)
)
