# Estadística Computacional — Grupo 7

**Universidad Andrés Bello (UNAB)**  
**Magíster en Ciencia de Datos e Inteligencia Artificial**  
**Asignatura:** Estadística Computacional (`MCDI501`)  
**Evaluaciones:** Sumativa 1 (Fase 2), Sumativa 2 (Fase 3) y Sumativa 3 (Fase 4) — Estadística inferencial, simulación, remuestreo y modelamiento predictivo.
**Periodo:** Octubre de 2026

**Integrantes:** Felipe Palma Barrientos · Vicente Aguila Rojas · Cristian Grandon Grandon · Jorge Gutierrez Jaramillo

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Estado](https://img.shields.io/badge/Sumativas%201%20y%202-Implementadas-238636)](#10-estado-del-proyecto)

## 1. Descripción del proyecto

Este repositorio documenta el desarrollo integrado de las evaluaciones Sumativa 1 (Fase 2), Sumativa 2 (Fase 3) y Sumativa 3 (Fase 4) del proyecto de **Estadística Computacional** del Grupo 7. A partir del conjunto de datos **E-Commerce Shipping / Customer Analytics**, se desarrolla un proceso analítico que comprende estadística descriptiva e inferencial, validación mediante simulación y remuestreo, tratamiento de datos faltantes y modelamiento predictivo mediante regresión logística.

El trabajo utiliza Python y Jupyter Notebook, con funciones reutilizables en `src/`, controles automatizados en `tests/`, conservación de los datos originales y un informe técnico en `reports/`. El propósito es obtener resultados **trazables, interpretables y reproducibles**; las asociaciones encontradas no se interpretan como relaciones causales.

**Pregunta general:** ¿qué características de los productos y de su logística se relacionan con el estado de entrega de los envíos incluidos en el dataset?

## 2. Objetivos

**Objetivo general:** analizar estadísticamente las características de los envíos y su asociación con la puntualidad de entrega mediante herramientas computacionales reproducibles.

**Objetivos específicos:**

1. Auditar la estructura, integridad y calidad del dataset original.
2. Generar de forma reproducible faltantes MCAR en dos variables cuantitativas, conservando una auditoría de los valores retirados.
3. Describir las distribuciones de variables cuantitativas, categóricas y ordinales.
4. Explorar asociaciones mediante correlaciones y comparaciones bivariadas.
5. Estimar tres medias poblacionales con intervalos de confianza del 95 %.
6. Contrastar dos hipótesis estadísticas pertinentes al estado de entrega.
7. Comunicar los resultados, supuestos, limitaciones y conclusiones en un notebook y un informe técnico.
8. Validar los resultados de S1 mediante bootstrap, permutaciones, Monte Carlo y análisis de robustez, y documentar los insumos para S3.

## 3. Dataset y diccionario de variables

- **Nombre:** E-Commerce Shipping / Customer Analytics.
- **Fuente:** [Kaggle — Customer Analytics](https://www.kaggle.com/datasets/prachi13/customer-analytics).
- **Unidad de observación:** un envío identificado por `ID`.
- **Tamaño original:** 10.999 registros y 12 columnas.
- **Calidad inicial:** sin valores faltantes y sin filas duplicadas exactas.
- **Variable objetivo:** `Reached.on.Time_Y.N`; en el análisis del proyecto, **0 = entrega a tiempo** y **1 = entrega fuera de plazo**.

| Variable | Tipo | Descripción |
|---|---|---|
| `ID` | Identificador | Identificador único del envío; no se incluye entre los predictores cuantitativos. |
| `Warehouse_block` | Categórica nominal | Bloque del almacén de origen. |
| `Mode_of_Shipment` | Categórica nominal | Medio de envío: `Ship`, `Flight` o `Road`. |
| `Customer_care_calls` | Cuantitativa discreta | Número de llamadas a atención al cliente. |
| `Customer_rating` | Ordinal | Calificación de 1 a 5. |
| `Cost_of_the_Product` | Cuantitativa | Costo del producto, expresado en USD según la documentación utilizada. |
| `Prior_purchases` | Cuantitativa discreta | Compras previas del cliente. |
| `Product_importance` | Ordinal | Importancia `low`, `medium` o `high`. |
| `Gender` | Categórica nominal | Categoría de género registrada. |
| `Discount_offered` | Cuantitativa | Descuento ofrecido; la fuente consultada no precisa su unidad, por lo que **no se interpreta automáticamente como porcentaje**. |
| `Weight_in_gms` | Cuantitativa | Peso del producto en gramos. |
| `Reached.on.Time_Y.N` | Binaria | Estado de entrega según la codificación usada por el proyecto. |

**Observación de calidad:** `Warehouse_block` contiene la categoría `F`, aunque ciertas descripciones de la fuente mencionan bloques A–E. Se conserva tal como está registrada; no se elimina ni reemplaza automáticamente.

## 4. Estructura del repositorio

**Los archivos de trabajo y las rutas canónicas se mantienen en `notebooks/`, `src/` y `data/`.**
Las carpetas `S1/` y `S2/` son índices de entrega que enlazan estos archivos sin
copiarlos ni producir versiones divergentes.

```text
.
├── .gitignore
├── .mailmap
├── Minutas/
│   └── 20261007_Minuta.md
├── S1/
│   └── README.md                       # Índice de evidencias S1
├── S2/
│   └── README.md                       # Índice de evidencias S2
├── S3/
│   └── README.md                       # Índice de evidencias S2
├── data/
│   ├── raw/customer_analytics.csv
│   └── processed/
│       ├── ecommerce_mcar.csv
│       ├── valores_originales_mcar.csv
│       └── resultados_validos_s2.json
├── notebooks/
│   ├── sumativa_1.ipynb
│   ├── sumativa_2.ipynb
│   └── sumativa_3.ipynb
├── reports/
│   └── Informe_Sumativa_1_Grupo_7_Formato_Institucional_UNAB.pdf
│       # Informe S2: pendiente de publicación por el equipo
├── src/
│   ├── __init__.py
│   ├── carga.py
│   ├── validacion.py
│   ├── transformacion.py
│   ├── estadistica_descriptiva.py
│   ├── analisis_bivariado.py
│   ├── inferencia.py
│   ├── validacion_s2.py
│   ├── s2_alineacion.py
│   ├── s2_bootstrap.py
│   ├── s2_permutaciones.py
│   ├── s2_correlaciones.py
│   ├── s2_montecarlo.py
│   ├── s2_robustez.py
│   ├── s2_visualizaciones.py
│   └── s2_resultados.py
├── tests/
│   ├── test_preparacion.py
│   ├── test_inferencia.py
│   ├── test_validacion_s2.py
│   ├── test_modulos_s2.py
│   ├── test_montecarlo_cobertura.py
│   └── test_correlaciones_optimizadas.py
├── tools/
│   ├── actualizar_readme_s2.py
│   └── verificar_cierre_s2.py
├── docs/
│   └── ARQUITECTURA_S2.md
├── requirements.txt
└── README.md
```

### Responsabilidad de los módulos

| Archivo | Responsabilidad |
|---|---|
| `src/carga.py`, `src/validacion.py`, `src/transformacion.py` | Carga, controles de datos y MCAR de S1. |
| `src/estadistica_descriptiva.py`, `src/analisis_bivariado.py`, `src/inferencia.py` | Descripción, asociaciones e inferencia de S1. |
| `src/s2_alineacion.py` | Reconstrucción y contraste de parámetros procedentes de S1. |
| `src/s2_bootstrap.py` | Bootstrap de tres medias: percentil y BCa, con 10.000 remuestras por media. |
| `src/s2_permutaciones.py` y `src/validacion_s2.py` | Permutaciones de hipótesis, diagnóstico de cuantiles y diferencia recortada. |
| `src/s2_correlaciones.py` | Bootstrap de tres correlaciones de Spearman por pares completos, con tratamiento de empates. |
| `src/s2_montecarlo.py` | Clase `SimuladorMonteCarlo`: lognormal/Gamma, convergencia, cobertura Welch y escenarios. |
| `src/s2_robustez.py` | Jackknife, valores extremos, comparación de familias y medias recortadas. |
| `src/s2_visualizaciones.py` | Gráficos utilizados en el notebook S2. |
| `src/s2_resultados.py` | Consolidación de resultados, recomendaciones S3 y exportación JSON. |
| `notebooks/sumativa_1.ipynb` | Informe computacional reproducible de S1. |
| `notebooks/sumativa_2.ipynb` | Presentación de métodos, ejecuciones desde módulos, figuras e interpretación de S2. |
| `notebooks/sumativa_3.ipynb` | Modelamiento e integración de los resultados de S1 y S2. |
| `tests/test_*.py` | 18 pruebas identificables: 6 de S1 y 12 adicionales de S2. |
| `tools/verificar_cierre_s2.py` | Auditoría local de integridad, ejecución registrada, carpetas y pruebas. |

La modularización separa cálculo, visualización y presentación. **No se cambió la
metodología, el tamaño de remuestreo, las semillas ni los datos MCAR de S1.**

## 5. Preparación y trazabilidad de datos

El archivo de `data/raw/` **no se modifica**. La preparación parte del original, valida su esquema, audita los datos y genera valores faltantes **MCAR (Missing Completely At Random)** con las siguientes características:

| Parámetro | Valor aplicado |
|---|---|
| Columnas afectadas | `Weight_in_gms` y `Cost_of_the_Product` |
| Proporción por columna | 10 % |
| Semilla reproducible | `42` |
| Selección | Índices aleatorios sin reemplazo, de manera independiente por variable |
| Faltantes introducidos en peso | 1.100 de 10.999 |
| Faltantes introducidos en costo | 1.100 de 10.999 |
| Datos válidos para cada variable afectada | 9.899 |
| Dataset resultante | `data/processed/ecommerce_mcar.csv` |
| Registro de los valores retirados | `data/processed/valores_originales_mcar.csv` |

La selección aleatoria se implementa en `generar_faltantes_mcar()` de `src/transformacion.py`. Los valores originales retirados se guardan para su auditoría y una posible evaluación de imputaciones en etapas posteriores; **no se utilizan para completar los faltantes en esta sumativa**.

El análisis descriptivo e inferencial usa el DataFrame preparado (`df_eda`), no la copia original. En cada cálculo se excluyen únicamente los valores ausentes necesarios para esa variable o contraste, en vez de eliminar globalmente todos los registros con cualquier faltante.

## 6. Entorno, instalación y ejecución

### Requisitos

- Python 3.14, versión empleada en el notebook publicado: **3.14.8**.
- Git y Visual Studio Code con extensiones de Python/Jupyter, o Jupyter Notebook.
- Dependencias disponibles en `requirements.txt`.

> `requirements.txt` **fija versiones** de las dependencias directas. Las salidas de S2 registran NumPy 2.5.3, pandas 3.0.6, SciPy 1.17.0 y Matplotlib 3.11.2. Para reproducibilidad estricta, registrar también el intérprete y las versiones realmente instaladas. `pytest` no está incluido en ese archivo; su instalación para ejecutar toda la suite se indica abajo.

### Instalación (Windows PowerShell)

```powershell
git clone https://github.com/jorge170102/ESTAD-STICA-COMPUTACIONAL---Grupo-7--MCDI501.202682.1914.EL.ON.git
cd ESTAD-STICA-COMPUTACIONAL---Grupo-7--MCDI501.202682.1914.EL.ON

py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Si `py -3.14` no está disponible, utiliza `python -m venv .venv` con un intérprete compatible.

### Ejecutar el notebook

1. Abrir la **carpeta raíz del repositorio** en Visual Studio Code o Jupyter.
2. Abrir `notebooks/sumativa_1.ipynb` o `notebooks/sumativa_2.ipynb`, según la evaluación que se desea reproducir, y seleccionar el kernel del entorno virtual `.venv`.
3. Ejecutar **Restart Kernel → Run All** en orden, desde la preparación hasta las conclusiones.
4. Comprobar que las celdas se ejecuten sin errores y guardar el notebook con sus salidas actualizadas.

Para reproducir el flujo completo desde el CSV original, ejecutar primero S1 y después S2. S2 también puede ejecutarse directamente con el CSV MCAR publicado: comprueba su huella antes de calcular resultados y requiere `src/validacion_s2.py`.

También se puede ejecutar sin sobrescribir el notebook original:

```powershell
jupyter nbconvert --to notebook --execute notebooks/sumativa_1.ipynb --output sumativa_1_verificado.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute notebooks/sumativa_2.ipynb --output sumativa_2_verificado.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=900
```

> **Importante:** la celda de preparación de S1 vuelve a generar `data/processed/ecommerce_mcar.csv` y la auditoría a partir del original, con semilla 42. S2 conserva esos datos y actualiza `data/processed/resultados_validos_s2.json`. Los comandos anteriores conservan los notebooks originales, pero sí generan esos archivos derivados; comprueba el estado de Git antes de registrarlos.

### Pruebas automatizadas

Desde la raíz del repositorio:

```powershell
python -m pip install pytest==9.1.1
python -m pytest -q
```

El proyecto contiene **18 pruebas automatizadas**: 3 de preparación S1,
3 de inferencia S1, 4 de validación S2, 3 de integración de módulos,
2 del simulador POO/cobertura de Welch y 3 de equivalencia/empates de Spearman.
Las pruebas utilizan casos sintéticos y **no sustituyen** la ejecución integral
sobre el CSV procesado de S1.

Desde la raíz del repositorio:

```powershell
python -m pip install pytest
python -m pytest -q
python tools/verificar_cierre_s2.py --tests
```

La última celda del notebook S2 comprueba **15 controles computacionales**
independientes de las pruebas unitarias. La cifra de 18 pruebas es la
cantidad de funciones/métodos `test_*` actualmente versionados; se recomienda
comprobar el resultado efectivo de `pytest` en cada entorno de entrega.

## 7. Metodología estadística

### 7.1. Análisis descriptivo y exploratorio

Se calculan medidas de tendencia central (media, mediana y moda), dispersión (desviación estándar, varianza, rango e IQR), cuartiles, frecuencia absoluta y porcentajes. Se muestran histogramas y diagramas de caja por variable cuantitativa, y gráficos de barras para variables categóricas u ordinales.

Las observaciones fuera de las cercas de **1,5 × IQR** se identifican como posibles valores extremos, **sin eliminarlas automáticamente**.

### 7.2. Análisis bivariado

- Matriz de **correlaciones de Spearman**, usando los pares con datos disponibles.
- Comparación de peso, costo y descuento por estado de entrega.
- Tasas descriptivas de entregas fuera de plazo según transporte, bloque e importancia.
- Visualización conjunta de peso y descuento mediante un **gráfico hexbin**.

### 7.3. Estimación de parámetros

Se estiman las medias de `Weight_in_gms`, `Cost_of_the_Product` y `Discount_offered`, con intervalos de confianza bilaterales del **95 %** basados en la distribución *t* de Student:

\[
IC_{95\%}=\bar{x}\pm t_{0.975,n-1}\frac{s}{\sqrt{n}}
\]

Se utiliza la desviación estándar muestral, con exclusión de los valores faltantes **por variable**. La interpretación de los intervalos depende de supuestos como independencia y representatividad del mecanismo de muestreo.

### 7.4. Contrastes de hipótesis

**Contraste A — Peso promedio y puntualidad:** prueba *t* bilateral de **Welch**, con las hipótesis `H0: μ_fuera − μ_a_tiempo = 0` y `H1: μ_fuera − μ_a_tiempo ≠ 0`. Se reportan estadístico *t*, grados de libertad aproximados, valor *p*, diferencia de medias, IC del 95 % y **d de Cohen**. Welch no presupone varianzas iguales.

**Contraste B — Medio de transporte y estado de entrega:** prueba **chi-cuadrado de independencia** en tabla de contingencia 3 × 2. `H0`: independencia entre variables; `H1`: asociación estadística. Se examinan frecuencias esperadas y se informa **V de Cramér**.

En ambos casos se emplea **α = 0,05** como nivel de significación exploratorio. No se aplicó un ajuste conjunto por comparaciones múltiples. Una diferencia o asociación estadísticamente significativa **no demuestra causalidad**; no rechazar `H0` tampoco demuestra igualdad o independencia.

## 8. Principales resultados de la Sumativa 1

Las siguientes cifras corresponden a las **salidas guardadas en el notebook de la rama `main`**, actualizadas el 11 de octubre de 2026, sobre los datos preparados con MCAR.

### 8.1. Calidad y estado de entrega

| Indicador | Resultado |
|---|---:|
| Registros originales y procesados | 10.999 |
| Variables originales | 12 |
| Duplicados exactos originales | 0 |
| Faltantes MCAR en peso | 1.100 (10 %) |
| Faltantes MCAR en costo | 1.100 (10 %) |
| Entregas a tiempo (`0`) | 4.436 (40,33 %) |
| Entregas fuera de plazo (`1`) | 6.563 (59,67 %) |

### 8.2. Estimaciones e intervalos de confianza

| Variable | N válido | Media muestral | IC del 95 % |
|---|---:|---:|---|
| Peso del producto (g) | 9.899 | 3.633,880 | [3.601,646; 3.666,115] |
| Costo del producto (USD) | 9.899 | 210,239 | [209,293; 211,186] |
| Descuento ofrecido (unidad no especificada) | 10.999 | 13,373 | [13,070; 13,676] |

### 8.3. Prueba t de Welch: peso y puntualidad

| Estadístico | Resultado |
|---|---:|
| N válido, a tiempo | 3.992 |
| N válido, fuera de plazo | 5.907 |
| Peso medio, a tiempo | 4.166,349 g |
| Peso medio, fuera de plazo | 3.274,033 g |
| Diferencia (fuera de plazo − a tiempo) | −892,316 g |
| Estadístico t | −27,635 |
| Grados de libertad aproximados | 8.576,338 |
| Valor p | < 0,001 |
| IC del 95 % de la diferencia | [−955,611; −829,021] g |
| d de Cohen | −0,566 |

**Interpretación:** al 5 % se rechaza la hipótesis nula de igualdad de medias. En esta muestra, los productos de entregas fuera de plazo presentan menor peso medio. La magnitud estandarizada es moderada, pero este resultado **no prueba** que el peso cause el retraso.

### 8.4. Chi-cuadrado: medio de transporte y puntualidad

| Estadístico | Resultado |
|---|---:|
| Chi-cuadrado | 0,743 |
| Grados de libertad | 2 |
| Valor p | 0,690 |
| V de Cramér | 0,008 |
| Frecuencia esperada mínima | 709,825 |
| Registros analizados | 10.999 |

| Medio de envío | N envíos | % fuera de plazo |
|---|---:|---:|
| `Ship` | 7.462 | 59,76 % |
| `Flight` | 1.777 | 60,16 % |
| `Road` | 1.760 | 58,81 % |

**Interpretación:** no se rechaza la hipótesis nula de independencia al 5 %. Las tasas observadas son similares y V de Cramér es cercano a cero; esto **no demuestra** que las variables sean independientes en todas las poblaciones.

## 9. Limitaciones y proyecciones

El dataset no documenta suficientemente el procedimiento de selección de la muestra ni permite comprobar de forma definitiva la independencia de los registros. No se dispone de destinos, fechas, distancias ni magnitud de los atrasos, variables que podrían intervenir en las asociaciones observadas. Los análisis se consideran **exploratorios** y sus resultados no deben generalizarse sin una población de referencia definida.

La introducción de faltantes MCAR es una transformación **artificial y controlada** con fines académicos. No reproduce necesariamente el mecanismo de ausencia que aparecería en una operación real. En futuras fases puede utilizarse la auditoría de valores retirados para evaluar métodos de imputación, evitando acceder a esos valores durante el ajuste de los métodos.

La Sumativa 1 **no implementa todavía un modelo predictivo ni una evaluación de imputaciones**; son líneas de continuidad, no resultados actuales.

### Continuidad del proyecto: Sumativa 2 — Fase 3

La Sumativa 2 profundiza y valida los resultados inferenciales obtenidos en la Sumativa 1 mediante métodos computacionales de simulación y remuestreo. Su objetivo es evaluar la incertidumbre, estabilidad y sensibilidad de las estimaciones estadísticas, manteniendo la trazabilidad con los resultados originales.

El análisis se desarrolla en `notebooks/sumativa_2.ipynb` e incluye:

| Procedimiento | Implementación |
|---|---|
| Bootstrap no paramétrico | 10.000 remuestras para tres medias; IC percentil y BCa al 95 %, comparados con los IC clásicos de S1 |
| Pruebas de permutación | 10.000 permutaciones por contraste para validar chi-cuadrado y comparar la diferencia de medias con Welch de S1 |
| Estabilidad de correlaciones | Tres correlaciones de Spearman evaluadas mediante 10.000 remuestras bootstrap e IC al 95 % |
| Simulación Monte Carlo | 10.000 iteraciones por escenario a partir de S1, convergencia, sensibilidad lognormal/Gamma y cobertura del IC Welch al 95 % |
| Análisis de robustez | Jackknife, sensibilidad frente a descuentos altos, distribuciones alternativas, variación de parámetros y medias recortadas al 10 % por cola |
| Preparación para S3 | Consolidación de resultados validados, limitaciones y recomendaciones metodológicas |

**Trazabilidad:** Los procedimientos utilizan los parámetros, intervalos de confianza, pruebas de hipótesis y correlaciones obtenidos en la Sumativa 1. Las comparaciones permiten evaluar la concordancia entre los resultados paramétricos originales y los métodos computacionales.

**Resultados para la Sumativa 3:** El notebook genera `data/processed/resultados_validos_s2.json`, que contiene los resultados estadísticos validados y recomendaciones metodológicas para la siguiente fase.

**Referencia de datos:** S2 identifica la línea base de S1 mediante el commit `0c7ffaca4e45525ab434f6c9305128221e4ef1c4` y la huella SHA-256 del CSV MCAR, normalizando saltos de línea a LF: `1a7f8a5c4da839e481ff2dc93e30c7f64205407a2bd62ae94be4265027ca0714`. El JSON registra fecha, versiones, semillas, tamaños de remuestreo y el commit disponible al ejecutar. Ese commit identifica `HEAD`, pero no acredita por sí solo que no hubiera cambios sin confirmar.

### Principales resultados validados de S2

Las cifras proceden de [resultados_validos_s2.json](data/processed/resultados_validos_s2.json) y se desarrollan en [sumativa_2.ipynb](notebooks/sumativa_2.ipynb).

| Resultado | Evidencia e interpretación |
|---|---|
| Tres medias e intervalos | Los anchos BCa cambian −1,23 % en peso, −1,19 % en costo y +0,94 % en descuento frente a los IC clásicos de S1. La concordancia se interpreta bajo los supuestos de muestreo e independencia. |
| Transporte y puntualidad | p clásico = 0,689549 y p de permutación = 0,689231; ambos mantienen la decisión de no rechazar H₀. Esto no demuestra independencia ni equivalencia. |
| Diferencia de peso | Fuera de plazo menos a tiempo = −892,32 g; Welch y permutación rechazan H₀. El p de permutación ≈ 0,0001 está limitado por la resolución de 10.000 permutaciones; no equivale al p paramétrico de 4,14 × 10⁻¹⁶¹. |
| Correlaciones | Peso–descuento: ρ = −0,3177, IC95 % [−0,3357; −0,2994]. Llamadas–costo: ρ = 0,2932, IC95 % [0,2731; 0,3126]. Valoración–peso: ρ = −0,0104, IC95 % [−0,0299; 0,0092]; su signo no queda determinado. |
| Monte Carlo | Diferencia media lognormal = −892,84 g (error MC de 0,319 g); cobertura Welch al 95 % de 95,35 % en lognormal y 94,94 % en Gamma. Converger y obtener cobertura nominal **no valida** los cuantiles completos de las distribuciones ni establece causalidad. |
| Robustez y continuidad | La influencia individual sobre las medias es pequeña. Excluir 2.209 descuentos > 19 reduce su media de 13,373 a 5,975 y atenúa ρ peso–descuento a −0,0776. La diferencia recortada es −1.094,85 g: conserva el signo, pero cambia la magnitud. Mantener todos los registros en el análisis principal y revisar composición y variables operativas en S3. |

### Ejecución de la Sumativa 2

Desde la raíz del repositorio, una vez instalado el entorno descrito en la sección 6:

```powershell
jupyter lab notebooks/sumativa_2.ipynb
```

En Jupyter, seleccionar el kernel del entorno virtual y ejecutar todas las celdas en orden mediante **Restart Kernel and Run All Cells**.

El notebook utiliza semillas aleatorias definidas para favorecer la reproducibilidad de los procedimientos de remuestreo y simulación.

### Interpretación y limitaciones

Los resultados bootstrap y de permutación permiten contrastar la estabilidad de las conclusiones inferenciales de S1. Sin embargo, no eliminan las limitaciones del dataset ni garantizan representatividad poblacional.

Las simulaciones Monte Carlo corresponden a escenarios hipotéticos condicionados a los parámetros y distribuciones utilizadas. Por tanto, sus resultados no constituyen predicciones operacionales ni evidencia causal.

Los resultados validados servirán como referencia metodológica para la Sumativa 3, especialmente en el tratamiento de datos faltantes, evaluación de sensibilidad y selección de procedimientos estadísticos.

## 10. Estado del proyecto

| Componente | Estado en `main` |
|---|---|
| Selección y documentación del dataset | Completado |
| Carga, esquema y auditoría | Implementados |
| Generación de faltantes MCAR y archivo de auditoría | Implementados |
| Estadística descriptiva y gráficos | Implementados |
| Análisis bivariado y visualizaciones | Implementados |
| Tres intervalos de confianza del 95 % | Implementados |
| Welch y chi-cuadrado, con tamaños del efecto | Implementados |
| Discusión, supuestos y conclusiones | Incorporados al notebook |
| Pruebas automatizadas (18 casos) | Disponibles en `tests/`; ejecutar con `python -m pytest -q` |
| Notebook S1 con resultados actualizados | Disponible; 10 celdas de código en Notebook S1 con salidas consecutivas guardadas |
| Informe técnico institucional S1 | Disponible en `reports/Informe_Sumativa_1_Grupo_7_Formato_Institucional_UNAB.pdf` |
| Modelo predictivo e imputación | Fuera del alcance de Sumativa 1 |
| Sumativa 2 — Validación bootstrap de IC (percentil y BCa) | Completado |
| Sumativa 2 — Validación de pruebas mediante permutación | Completado |
| Sumativa 2 — Estabilidad de correlaciones de Spearman | Completado |
| Sumativa 2 — Simulación Monte Carlo y convergencia | Completado |
| Sumativa 2 — Análisis de robustez (jackknife, outliers y sensibilidad) | Completado |
| Sumativa 2 — Consolidación de resultados validados para S3 | Completado |
| Sumativa 2 — Generación de resultados_validos_s2.json | Completado |
| Notebook S2 y controles internos | 19 celdas de código ejecutadas (1–19) y 15 controles finales satisfactorios |
| Informe técnico S2 en el repositorio | Pendiente de publicación en la versión revisada |
| Carpetas de entrega S1 y S2 | Índices de evidencias `S1/README.md` y `S2/README.md`, sin duplicar archivos canónicos |

**Estado general:** Los análisis de las evaluaciones Sumativa 1 y Sumativa 2 se encuentran implementados. La segunda evaluación incorpora procedimientos de simulación y remuestreo para validar y profundizar los resultados inferenciales obtenidos en la primera, incluyendo análisis de sensibilidad y consolidación de resultados para la Sumativa 3.

Los procedimientos y resultados se encuentran documentados en notebooks/sumativa_2.ipynb, mientras que los insumos validados para la siguiente fase se almacenan en data/processed/resultados_validos_s2.json.

**Nota de reproducción:** Los notebooks S1 y S2 presentan diez y diecinueve celdas de código ejecutadas, respectivamente, con salidas guardadas sin errores. Para verificar la reproducibilidad de ambas evaluaciones en un entorno nuevo, ejecutar **Restart Kernel → Run All** en cada notebook, las 18 pruebas con pytest y los 15 controles internos de S2; comprobar que se reproduzcan las tablas, figuras y archivos de resultados.

**Cierre formal de S2:** La pauta solicita PDF de máximo siete páginas, notebook ejecutable, enlace al repositorio, carpetas S1/S2 y archivo de resultados validados. A octubre de 2026, el notebook S2 modular registra 19 ejecuciones sin errores y 15 controles satisfactorios; el JSON está publicado. **El PDF de S2 permanece pendiente de publicación y entrega por el equipo**. El cumplimiento del límite de siete páginas debe comprobarse sobre la versión PDF definitiva.


### Verificación de cierre y arquitectura modular (octubre de 2026)

La Sumativa 2 separa cálculos estadísticos (`src/s2_*.py`), visualizaciones,
clase `SimuladorMonteCarlo` y presentación (`notebooks/sumativa_2.ipynb`).
El JSON consolidado se encuentra en [resultados_validos_s2.json](data/processed/resultados_validos_s2.json).
La última ejecución publicada registra 19 celdas, 15 controles satisfactorios
y escenarios con cobertura de Welch cercana al 95 % bajo los modelos simulados.

- [Índice de evidencias S1](S1/README.md)
- [Índice de evidencias S2](S2/README.md)
- [Detalles de arquitectura](docs/ARQUITECTURA_S2.md)

**Comprobación local:** `python tools/verificar_cierre_s2.py --tests`.
Este comando muestra advertencia mientras falte el PDF final de S2; por acuerdo
del equipo, no se incluye ni modifica en esta actualización documental.

## 11. Equipo y control de versiones

El proyecto se desarrolla colaborativamente mediante Git y GitHub. Los cambios en preparación, estadística descriptiva, análisis bivariado, inferencia, notebook, pruebas y documentación se han registrado en commits separados. El directorio `Minutas/` documenta acuerdos del equipo, mientras que `.mailmap` facilita la consistencia de nombres de autor.

Para registrar una futura actualización del README de manera independiente:

```powershell
git add README.md
git commit -m "docs: actualiza README con resultados de sumativas 1 y 2"
git pull --rebase origin main
git push origin main
```

## 12. Referencias

1. Universidad Andrés Bello. (2026). *Sumativa 1: informe técnico de avance del proyecto* [Guía y rúbrica de Estadística Computacional, MCDI501]. Material de la asignatura.
2. Universidad Andrés Bello. (2026). *Bases de datos disponibles para el proyecto del curso*. Material de la asignatura.
3. Prachi13. (s. f.). [*Customer Analytics — E-Commerce Shipping*](https://www.kaggle.com/datasets/prachi13/customer-analytics) [Conjunto de datos]. Kaggle.
4. [pandas — Documentation](https://pandas.pydata.org/docs/).
5. [NumPy — Documentation](https://numpy.org/doc/stable/).
6. [SciPy — Statistical functions](https://docs.scipy.org/doc/scipy/reference/stats.html).
7. [Matplotlib — Documentation](https://matplotlib.org/stable/).
8. [Seaborn — Documentation](https://seaborn.pydata.org/).
9. Universidad Andrés Bello. (2026). *Sumativa 2. Informe 3: instrucciones y rúbrica, Fase 3* [Material docente de MCDI501; seis criterios, 87 puntos].

## Sumativa 3 — Modelamiento e integración final

La Sumativa 3 integra los resultados de las fases anteriores mediante imputación por regresión lineal múltiple, clasificación logística, evaluación predictiva, diagnósticos estadísticos, bootstrap de coeficientes y análisis de sensibilidad frente a distintas estrategias de tratamiento de valores faltantes.

Se implementaron tres procedimientos de construcción de modelos logísticos: selección informada por S1/S2, selección progresiva mediante razón de verosimilitudes y selección mediante AIC. Los dos últimos identificaron la misma especificación, propuesta como modelo final por su menor AIC en entrenamiento y la concordancia entre procedimientos.

El análisis bootstrap se desarrolló con 2.000 réplicas por especificación, mientras que la comparación de estrategias de tratamiento de faltantes mostró diferencias predictivas reducidas y asociaciones relativamente estables.

La documentación metodológica, los resultados y las limitaciones se presentan en [S3/README.md](S3/README.md). El notebook maestro se encuentra en [notebooks/sumativa_3.ipynb](notebooks/sumativa_3.ipynb).

---

**Proyecto académico — Grupo 7**  
Magíster en Ciencia de Datos e Inteligencia Artificial · Universidad Andrés Bello · 2026
