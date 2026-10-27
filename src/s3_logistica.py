# Grupo 7 / S3 — Implementación original de celda 16.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

# ============================================================
# 4. CONSTRUCCIÓN Y SELECCIÓN DE MODELOS LOGÍSTICOS
# ============================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ------------------------------------------------------------
# 4.1. Definición de variables
# ------------------------------------------------------------

num_cols = [
    'Customer_care_calls',
    'Customer_rating',
    'Cost_of_the_Product',
    'Prior_purchases',
    'Discount_offered',
    'Weight_in_gms'
]

cat_cols = [
    'Warehouse_block',
    'Mode_of_Shipment',
    'Product_importance',
    'Gender'
]


# ------------------------------------------------------------
# 4.2. Estandarización y codificación
# ------------------------------------------------------------

pre = ColumnTransformer(
    transformers=[
        (
            'num',
            StandardScaler(),
            num_cols
        ),
        (
            'cat',
            OneHotEncoder(
                drop='first',
                handle_unknown='ignore',
                sparse_output=False
            ),
            cat_cols
        )
    ],
    verbose_feature_names_out=False
)

# Ajustar transformaciones solo con entrenamiento
Ztr = pd.DataFrame(
    pre.fit_transform(Xreg),
    columns=pre.get_feature_names_out(),
    index=Xreg.index
).astype(float)

# Aplicar las mismas transformaciones al test
Zte = pd.DataFrame(
    pre.transform(Xte),
    columns=Ztr.columns,
    index=Xte.index
).astype(float)

allcols = list(Ztr.columns)

print("Variables originales:", len(num_cols) + len(cat_cols))
print("Columnas transformadas:", len(allcols))
print("Entrenamiento:", Ztr.shape)
print("Prueba:", Zte.shape)


# ------------------------------------------------------------
# 4.3. Definición de bloques de variables
# ------------------------------------------------------------

# Cada variable numérica forma un bloque individual.
# Cada variable categórica forma un bloque con todas sus columnas dummy.

feature_groups = {
    variable: [variable]
    for variable in num_cols
}

encoder = pre.named_transformers_['cat']

cat_features = list(
    encoder.get_feature_names_out(cat_cols)
)

for variable in cat_cols:

    # Identificar las columnas dummy de cada variable
    columnas = [
        c for c in cat_features
        if c.startswith(variable + '_')
    ]

    feature_groups[variable] = columnas


# Comprobar que todas las columnas están representadas
grouped_cols = [
    c
    for columnas in feature_groups.values()
    for c in columnas
]

assert len(grouped_cols) == len(set(grouped_cols)), (
    "Existen columnas duplicadas entre los bloques."
)

assert set(grouped_cols) == set(allcols), (
    "Existen columnas transformadas sin asignar a un bloque."
)

print("\nBLOQUES DE VARIABLES")

for variable, columnas in feature_groups.items():
    print(f"{variable}: {columnas}")


# ------------------------------------------------------------
# 4.4. Función de ajuste logístico
# ------------------------------------------------------------

def fit_logit(cols, Z=None, yy=None):

    if Z is None:
        Z = Ztr

    if yy is None:
        yy = ytr

    cols = list(cols)

    # Construcción de la matriz de diseño
    mat = sm.add_constant(
        Z[cols],
        has_constant='add'
    )

    # Verificar colinealidad perfecta
    rango = np.linalg.matrix_rank(mat.to_numpy())

    if rango < mat.shape[1]:
        raise ValueError(
            "La matriz de diseño no tiene rango completo. "
            f"Rango={rango}, columnas={mat.shape[1]}"
        )

    # Ajustar modelo logístico
    modelo = sm.GLM(
        yy,
        mat,
        family=sm.families.Binomial()
    ).fit(
        maxiter=150,
        disp=0
    )

    # Verificar convergencia
    if not modelo.converged:
        raise RuntimeError(
            "El modelo logístico no alcanzó convergencia."
        )

    return modelo


# ------------------------------------------------------------
# 4.5. Modelo informado por S1/S2
# ------------------------------------------------------------

base_cols = [
    'Customer_care_calls',
    'Cost_of_the_Product',
    'Discount_offered',
    'Weight_in_gms'
]

assert set(base_cols).issubset(set(allcols)), (
    "Alguna variable del modelo S1/S2 no está disponible."
)

models = {
    'S1_S2': fit_logit(base_cols)
}

selected = {
    'S1_S2': base_cols.copy()
}


# ------------------------------------------------------------
# 4.6. Selección progresiva por bloques
# ------------------------------------------------------------

def forward_select(criterion='lr', alpha=0.05):

    if criterion not in ['lr', 'aic']:
        raise ValueError("El criterio debe ser 'lr' o 'aic'.")

    chosen_groups = []

    remaining = list(feature_groups.keys())

    # Modelo inicial: solo intercepto
    current = fit_logit([])

    while remaining:

        candidates = []

        for variable in remaining:

            # Columnas ya seleccionadas
            cols_actuales = [
                c
                for grupo in chosen_groups
                for c in feature_groups[grupo]
            ]

            # Agregar el bloque candidato
            cols_trial = (
                cols_actuales
                + feature_groups[variable]
            )

            try:

                trial = fit_logit(cols_trial)

                # Estadístico de razón de verosimilitudes
                lr_stat = max(
                    0,
                    2 * (trial.llf - current.llf)
                )

                # Grados de libertad del bloque incorporado
                df_diff = (
                    trial.df_model - current.df_model
                )

                if df_diff <= 0:
                    continue

                pval = stats.chi2.sf(
                    lr_stat,
                    df_diff
                )

                score = (
                    trial.aic
                    if criterion == 'aic'
                    else pval
                )

                candidates.append(
                    (score, variable, trial)
                )

            except (
                ValueError,
                RuntimeError,
                np.linalg.LinAlgError
            ):
                continue

        if not candidates:
            break

        # Seleccionar el mejor bloque candidato
        score, variable, trial = min(
            candidates,
            key=lambda x: x[0]
        )

        # Regla de incorporación
        if criterion == 'aic':

            incorporar = (
                trial.aic < current.aic - 1e-8
            )

        else:

            incorporar = score < alpha

        if incorporar:

            chosen_groups.append(variable)

            remaining.remove(variable)

            current = trial

        else:
            break

    # Convertir grupos seleccionados en columnas
    chosen_cols = [
        c
        for grupo in chosen_groups
        for c in feature_groups[grupo]
    ]

    return chosen_cols, current


# ------------------------------------------------------------
# 4.7. Ajustar los tres modelos
# ------------------------------------------------------------

for key, criterion in [
    ('stepwise', 'lr'),
    ('AIC', 'aic')
]:

    selected[key], models[key] = forward_select(
        criterion=criterion
    )


# ------------------------------------------------------------
# 4.8. Presentación de resultados
# ------------------------------------------------------------

for key, modelo in models.items():

    print("\n" + "=" * 65)
    print(f"MODELO: {key}")
    print("=" * 65)

    print("AIC:", round(modelo.aic, 2))
    print("BIC (log-verosimilitud):", round(modelo.bic_llf, 2))
    print("Convergencia:", modelo.converged)
    print("Variables:", selected[key])

    # Coeficientes y odds ratios
    intervalos = modelo.conf_int()

    tabla = pd.DataFrame({
        'coef': modelo.params,
        'OR': np.exp(modelo.params),
        'IC_OR_inf': np.exp(intervalos.iloc[:, 0]),
        'IC_OR_sup': np.exp(intervalos.iloc[:, 1])
    })

    print("\nCOEFICIENTES Y ODDS RATIOS")
    print(tabla.round(3).to_string())


# ------------------------------------------------------------
# 4.9. Tabla comparativa de modelos
# ------------------------------------------------------------

comparacion_modelos = pd.DataFrame([
    {
        'modelo': nombre,
        'n_predictores': len(selected[nombre]),
        'AIC': modelo.aic,
        'BIC': modelo.bic_llf,
        'logLik': modelo.llf,
        'convergencia': modelo.converged
    }
    for nombre, modelo in models.items()
])

print("\n" + "=" * 65)
print("COMPARACIÓN GENERAL DE MODELOS")
print("=" * 65)

print(
    comparacion_modelos.round(3).to_string(index=False)
)
