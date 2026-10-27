# Grupo 7 / S3 — Implementación original de celda 13.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

# ============================================================
# VALIDACIÓN DE IMPUTACIONES CON VALORES ORIGINALES DE S1
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Recuperar el dataset original de S1
ruta_original = ROOT / "data" / "raw" / "customer_analytics.csv"

df_original = pd.read_csv(ruta_original)

# 2. Verificar identificadores únicos
assert df_original["ID"].is_unique, "Existen ID duplicados en el original."
assert df["ID"].is_unique, "Existen ID duplicados en el dataset MCAR."

# 3. Vincular los registros por ID
# Se conserva el orden y los índices de Xtr.
ids_train = df.loc[Xtr.index, "ID"]

original_train = (
    df_original
    .set_index("ID")
    .reindex(ids_train.to_numpy())
)

original_train.index = Xtr.index

# 4. Validar correspondencia entre datasets
assert original_train[MISS].notna().all().all(), (
    "El dataset original contiene faltantes en las variables evaluadas."
)

assert original_train.index.equals(Xtr.index)

# Verificar que los valores observados no hayan cambiado
for variable in MISS:

    observados = Xtr[variable].notna()

    assert np.allclose(
        Xtr.loc[observados, variable].to_numpy(),
        original_train.loc[observados, variable].to_numpy()
    ), f"Los valores observados no coinciden en {variable}"

# 5. Identificar los faltantes originales de S1
mascaras = {
    variable: Xtr[variable].isna()
    for variable in MISS
}

costo = "Cost_of_the_Product"
peso = "Weight_in_gms"

# Escenarios mutuamente excluyentes
escenarios = {
    "Solo costo": mascaras[costo] & ~mascaras[peso],
    "Solo peso": mascaras[peso] & ~mascaras[costo],
    "Costo y peso": mascaras[costo] & mascaras[peso]
}

# 6. Comparar imputaciones con los valores originales
resultados_validacion = []

for escenario, mascara_escenario in escenarios.items():

    for variable in MISS:

        # Evaluar únicamente valores originalmente faltantes
        mascara = mascara_escenario & mascaras[variable]

        if not mascara.any():
            continue

        verdaderos = original_train.loc[mascara, variable]

        for metodo, datos_imputados in [
            ("Regresión", Xreg),
            ("Mediana", Xsimple)
        ]:

            estimados = datos_imputados.loc[mascara, variable]

            rmse = np.sqrt(
                mean_squared_error(verdaderos, estimados)
            )

            mae = mean_absolute_error(
                verdaderos, estimados
            )

            resultados_validacion.append({
                "escenario": escenario,
                "variable": variable,
                "metodo": metodo,
                "n_evaluados": len(verdaderos),
                "RMSE": rmse,
                "MAE": mae
            })

# 7. Presentar resultados por escenario
tabla_validacion = pd.DataFrame(resultados_validacion)

print("VALIDACIÓN POR ESCENARIO")
print(
    tabla_validacion.round(2).to_string(index=False)
)

# 8. Evaluación global por variable
resultados_globales = []

for variable in MISS:

    mascara = mascaras[variable]

    verdaderos = original_train.loc[mascara, variable]

    for metodo, datos_imputados in [
        ("Regresión", Xreg),
        ("Mediana", Xsimple)
    ]:

        estimados = datos_imputados.loc[mascara, variable]

        resultados_globales.append({
            "variable": variable,
            "metodo": metodo,
            "n_imputados": int(mascara.sum()),
            "RMSE": np.sqrt(
                mean_squared_error(verdaderos, estimados)
            ),
            "MAE": mean_absolute_error(
                verdaderos, estimados
            )
        })

tabla_global = pd.DataFrame(resultados_globales)

print("\nVALIDACIÓN GLOBAL DE IMPUTACIONES")
print(
    tabla_global.round(2).to_string(index=False)
)
