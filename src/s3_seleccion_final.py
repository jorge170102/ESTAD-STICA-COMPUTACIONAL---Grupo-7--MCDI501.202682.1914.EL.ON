# Grupo 7 / S3 — Implementación original de celda 25.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

# ============================================================
# 4.4. Identificación del modelo logístico final
# ============================================================

FINAL = "stepwise"

final_model = models[FINAL]
final_cols = selected[FINAL].copy()

assert final_model.converged
assert selected["stepwise"] == selected["AIC"]

print("Modelo final: Stepwise/AIC")
print("Predictores seleccionados:", final_cols)
print(f"AIC entrenamiento: {final_model.aic:.2f}")
print(f"Número de columnas predictoras: {len(final_cols)}")
