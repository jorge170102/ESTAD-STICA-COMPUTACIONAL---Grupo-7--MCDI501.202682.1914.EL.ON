# Sumativa 3 — Modelamiento e integración S1 → S2 → S3

**Asignatura:** Estadística Computacional (MCDI501)  
**Programa:** Magíster en Ciencia de Datos e Inteligencia Artificial, Universidad Andrés Bello  
**Equipo:** Grupo 7  
**Etapa:** Evaluación Sumativa 3 — Fase 4: cierre y comunicación del proyecto

## 1. Propósito

Integrar los hallazgos de S1 (análisis descriptivo e inferencial) y S2 (simulación y remuestreo) para estudiar el estado de entrega de los envíos mediante imputación por regresión lineal múltiple y modelos de regresión logística. La variable `Reached.on.Time_Y.N` se interpreta como **1 = no llegó a tiempo** y **0 = llegó a tiempo**. Los resultados describen asociaciones, no efectos causales.

## 2. Insumos y trazabilidad

- `data/raw/customer_analytics.csv`: datos originales, 10.999 registros.
- `data/processed/ecommerce_mcar.csv`: datos preparados en S1 con 1.100 valores ocultados artificialmente en cada una de `Cost_of_the_Product` y `Weight_in_gms` (MCAR, aproximadamente 10 % por variable).
- `data/processed/valores_originales_mcar.csv`: auditoría de los valores ocultados, usada para evaluar reconstrucción, no para entrenar los imputadores.
- `data/processed/resultados_validos_s2.json`: antecedentes de correlación y estabilidad de S2.
- `notebooks/sumativa_3.ipynb`: notebook maestro de S3; se mantiene en su ubicación canónica, sin duplicarlo dentro de `S3/`.

## 3. Flujo metodológico

1. **Evidencia S1/S2.** Recuperación de patrones de faltantes MCAR y relaciones entre predictores validadas mediante remuestreo.
2. **Imputación.** Comparación de regresión lineal múltiple, mediana y eliminación de casos completos. La regresión se seleccionó como estrategia principal por su menor error de reconstrucción sobre valores ocultados artificialmente.
3. **Modelos logísticos.** Tres enfoques: predictores informados por S1/S2, selección forward stepwise por razón de verosimilitudes y selección forward por AIC. Stepwise y AIC seleccionaron la misma especificación.
4. **Evaluación y diagnósticos.** Métricas en entrenamiento y prueba, matrices de confusión, curvas ROC/AUC, VIF, residuos, observaciones influyentes y pruebas exploratorias de linealidad del logit mediante splines cúbicos naturales.
5. **Bootstrap.** 2.000 réplicas por especificación única, intervalos percentiles al 95 % para coeficientes y odds ratios, comparación con intervalos Wald y frecuencia de cambios de signo.
6. **Sensibilidad.** Comparación de las tres estrategias de tratamiento de faltantes sobre las dos especificaciones logísticas y un mismo conjunto de prueba.
7. **Cierre.** Comprobaciones programáticas de coherencia, conclusiones y limitaciones.

## 4. Resultados principales

| Indicador | S1/S2 | Stepwise/AIC |
|---|---:|---:|
| AIC en entrenamiento (imputación por regresión) | 8.446,98 | 8.422,20 |
| AUC en prueba | 0,7171 | 0,7185 |
| F1 en prueba | 0,6878 | 0,6886 |
| Réplicas bootstrap válidas | 2.000 / 2.000 | 2.000 / 2.000 |

La comparación de imputaciones mostró AUC entre **0,7167 y 0,7191** en las seis combinaciones de modelo y tratamiento de faltantes. Las asociaciones principales conservaron sus signos; `Cost_of_the_Product` mostró mayor sensibilidad relativa, y bajo imputación por mediana su intervalo Wald en Stepwise/AIC incluyó ligeramente cero.

**Selección propuesta para el cierre:** Stepwise/AIC como especificación principal, por convergencia de dos procedimientos de selección y menor AIC en entrenamiento; S1/S2 se conserva como referencia parsimoniosa. Las diferencias de AUC en prueba son marginales y no justifican por sí solas la elección.

## 5. Limitaciones

- El mecanismo MCAR fue introducido artificialmente; los resultados no se extrapolan automáticamente a MAR o MNAR.
- El test evaluado contiene **2.691 casos completos** en las dos variables con faltantes artificiales. Se excluyen los registros incompletos de prueba, limitando la generalización a dichos casos.
- La imputación por regresión es determinista; el bootstrap se realizó con valores ya imputados y predictores fijos, sin incorporar íntegramente la incertidumbre de imputación ni selección de variables.
- Los diagnósticos detectaron posibles relaciones no lineales en el logit, especialmente en descuento y peso, y en compras previas para el modelo ampliado.
- El desempeño discriminativo es moderado; no se infieren relaciones causales.

## 6. Reproducción

Desde la raíz del repositorio, en Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m jupyter lab
```

Abrir `notebooks/sumativa_3.ipynb`, seleccionar el kernel `.venv` y ejecutar **Restart Kernel → Run All**. El notebook busca la raíz del repositorio desde la carpeta raíz o `notebooks/`. Requiere los archivos procesados y el JSON de S2 ya incluidos en el repositorio. La ejecución completa con versiones recién instaladas debe verificarse antes de la entrega.

Para las pruebas automatizadas disponibles en el repositorio:

```powershell
python -m pytest -q
```

## 7. Productos de entrega

- Notebook maestro: `notebooks/sumativa_3.ipynb`.
- Informe técnico integrado: **pendiente de incorporación a `reports/`**.
- Presentación o video de diez minutos: **pendiente de entrega**.
- Repositorio: contiene los insumos y notebooks de S1, S2 y S3.

## 8. Relación entre las sumativas

**S1:** auditoría de datos, faltantes MCAR, descriptiva, inferencia y correlaciones.  
**S2:** validación de asociaciones mediante bootstrap, permutaciones y simulación.  
**S3:** imputación guiada por S1/S2, modelamiento logístico, diagnósticos, bootstrap de coeficientes, sensibilidad y comunicación integrada.
