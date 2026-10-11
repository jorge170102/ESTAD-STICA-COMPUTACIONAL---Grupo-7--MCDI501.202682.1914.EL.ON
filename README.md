# Estadística Computacional — Grupo 7

**Universidad Andrés Bello (UNAB)**  
**Magíster en Ciencia de Datos e Inteligencia Artificial**  
**Asignatura:** Estadística Computacional (`MCDI501`)  
**Evaluación:** Sumativa 1 — Informe técnico de avance del proyecto  
**Periodo:** Octubre de 2026

**Integrantes:** Felipe Palma Barrientos · Vicente Aguila Rojas · Cristian Grandon Grandon · Jorge Gutierrez Jaramillo

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Estado](https://img.shields.io/badge/Sumativa%201-An%C3%A1lisis%20implementado-238636)](#10-estado-del-proyecto)

## 1. Descripción del proyecto

Este repositorio documenta la primera fase del proyecto de **Estadística Computacional** del Grupo 7. Se utiliza el dataset público **E-Commerce Shipping / Customer Analytics** para describir las características de los envíos, estudiar la puntualidad de las entregas y explorar relaciones mediante estadística descriptiva e inferencial.

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

```text
.
├── .gitignore
├── .mailmap
├── Minutas/
│   └── 20261007_Minuta.md
├── data/
│   ├── raw/
│   │   └── customer_analytics.csv
│   └── processed/
│       ├── ecommerce_mcar.csv
│       └── valores_originales_mcar.csv
├── notebooks/
│   └── sumativa_1.ipynb
├── reports/
│   └── Sumativa_1.pdf
├── src/
│   ├── __init__.py
│   ├── carga.py
│   ├── validacion.py
│   ├── transformacion.py
│   ├── estadistica_descriptiva.py
│   ├── analisis_bivariado.py
│   └── inferencia.py
├── tests/
│   ├── test_preparacion.py
│   └── test_inferencia.py
├── requirements.txt
├── README.md
└── hola.py                  # Archivo auxiliar histórico; no participa en el flujo analítico
```

### Responsabilidad de los módulos

| Archivo | Responsabilidad principal |
|---|---|
| `src/carga.py` | Lectura y escritura de archivos CSV. |
| `src/validacion.py` | Validación de esquema y auditoría de calidad. |
| `src/transformacion.py` | Generación controlada de faltantes MCAR y registro de valores retirados. |
| `src/estadistica_descriptiva.py` | Resúmenes numéricos, histogramas, diagramas de caja y frecuencias categóricas. |
| `src/analisis_bivariado.py` | Correlación de Spearman, comparación por estado, tasas por categorías y gráfico hexbin. |
| `src/inferencia.py` | Intervalos t de Student, prueba t de Welch, chi-cuadrado y tamaños del efecto. |
| `tests/test_preparacion.py` | Pruebas del esquema, conservación del original y reproducibilidad MCAR. |
| `tests/test_inferencia.py` | Pruebas de intervalos con faltantes, Welch y chi-cuadrado sobre datos sintéticos. |
| `notebooks/sumativa_1.ipynb` | Orquestación del análisis, figuras, salidas e interpretaciones. |
| `reports/Sumativa_1.pdf` | Informe técnico institucional de la Sumativa 1, máximo ocho páginas. |

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

> Las dependencias están declaradas **sin versiones fijadas**. Las salidas guardadas registran pandas 3.0.6 y NumPy 2.5.3; otro entorno puede tener versiones diferentes. Para reproducibilidad estricta, conviene documentar también las versiones realmente instaladas.

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
2. Abrir `notebooks/sumativa_1.ipynb` y seleccionar el kernel de `.venv`.
3. Ejecutar **Restart Kernel → Run All** en orden, desde la preparación hasta las conclusiones.
4. Comprobar que las celdas se ejecuten sin errores y guardar el notebook con sus salidas actualizadas.

También se puede ejecutar sin sobrescribir el notebook original:

```powershell
jupyter nbconvert --to notebook --execute notebooks/sumativa_1.ipynb --output sumativa_1_verificado.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=600
```

> **Importante:** la celda de preparación vuelve a generar `data/processed/ecommerce_mcar.csv` y la auditoría a partir del original. Se usa la semilla 42 para reproducir el mismo mecanismo; comprueba el estado de Git antes de registrar archivos generados.

### Pruebas automatizadas

Desde la raíz del repositorio:

```powershell
python -m unittest discover -s tests -v
```

El proyecto contiene **seis pruebas unitarias**: tres de preparación y tres de inferencia. Revisan integridad del original, esquema, semilla MCAR, intervalos con valores faltantes, signo y tamaño del efecto de Welch, y condiciones básicas de chi-cuadrado. Las pruebas inferenciales utilizan un conjunto sintético diseñado para verificar funciones; complementan, pero **no reemplazan**, la ejecución integral del notebook sobre los datos reales.

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
| Pruebas automatizadas (6 casos) | Disponibles en `tests/` |
| Notebook con resultados actualizados | Disponible; 10 celdas de código con salidas consecutivas guardadas |
| Informe técnico institucional | Disponible en `reports/Sumativa_1.pdf` |
| Modelo predictivo e imputación | Fuera del alcance de Sumativa 1 |

**Nota de reproducción:** el notebook disponible en GitHub presenta diez celdas de código ejecutadas y salidas guardadas sin errores. La comprobación definitiva en un equipo nuevo corresponde a ejecutar **Restart Kernel → Run All** e inspeccionar que se reproduzcan las mismas tablas y figuras.

## 11. Equipo y control de versiones

El proyecto se desarrolla colaborativamente mediante Git y GitHub. Los cambios en preparación, estadística descriptiva, análisis bivariado, inferencia, notebook, pruebas y documentación se han registrado en commits separados. El directorio `Minutas/` documenta acuerdos del equipo, mientras que `.mailmap` facilita la consistencia de nombres de autor.

Para registrar una futura actualización del README de manera independiente:

```powershell
git add README.md
git commit -m "docs(readme): documenta analisis y resultados finales de sumativa 1"
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

---

**Proyecto académico — Grupo 7**  
Magíster en Ciencia de Datos e Inteligencia Artificial · Universidad Andrés Bello · 2026
