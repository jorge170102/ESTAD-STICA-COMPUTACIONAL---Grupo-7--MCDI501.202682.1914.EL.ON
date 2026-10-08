# Estadística Computacional — Grupo 7

**Universidad Andrés Bello (UNAB)**  
**Magíster en Ciencia de Datos e Inteligencia Artificial**  
**Asignatura:** Estadística Computacional (MCDI501)  
**Equipo:** Grupo 7  
**Periodo académico:** 2026

---

## 1. Descripción del proyecto

Este repositorio contiene el desarrollo del proyecto académico de la asignatura **Estadística Computacional**, utilizando el conjunto de datos *E-Commerce Shipping / Customer Analytics*.

El proyecto tiene como propósito aplicar técnicas de estadística descriptiva, inferencia estadística y modelamiento para estudiar los factores asociados con la puntualidad de las entregas de productos en una empresa de comercio electrónico.

El desarrollo se realiza mediante Python, priorizando la reproducibilidad, modularidad, trazabilidad de las transformaciones e interpretación estadística de los resultados.

## 2. Conjunto de datos

**Dataset:** E-Commerce Shipping / Customer Analytics  
**Fuente:** [Kaggle — Customer Analytics](https://www.kaggle.com/datasets/prachi13/customer-analytics)

El conjunto contiene:

| Característica | Descripción |
|---|---|
| Observaciones | 10.999 |
| Variables | 12 |
| Tipo de problema | Clasificación binaria |
| Variable objetivo | `Reached.on.Time_Y.N` |
| Datos faltantes originales | 0 |
| Registros duplicados exactos | 0 |

Entre las variables disponibles se encuentran:

- `Warehouse_block`: bloque de almacenamiento.
- `Mode_of_Shipment`: método de transporte.
- `Customer_care_calls`: llamadas a atención al cliente.
- `Customer_rating`: valoración del cliente.
- `Cost_of_the_Product`: costo del producto.
- `Prior_purchases`: compras anteriores.
- `Product_importance`: importancia del producto.
- `Discount_offered`: descuento ofrecido.
- `Weight_in_gms`: peso del producto.

La variable objetivo permite analizar la puntualidad del envío mediante dos clases.

## 3. Objetivos

### Objetivo general

Analizar estadísticamente los factores asociados con la puntualidad de las entregas en el comercio electrónico, aplicando técnicas descriptivas e inferenciales mediante herramientas computacionales reproducibles.

### Objetivos específicos

1. Evaluar la integridad, estructura y calidad del conjunto de datos.
2. Aplicar procedimientos documentados de preparación y transformación.
3. Caracterizar las variables mediante estadística descriptiva.
4. Explorar asociaciones entre las características de los envíos y su puntualidad.
5. Calcular estimaciones puntuales e intervalos de confianza del 95 %.
6. Formular y contrastar hipótesis estadísticas relevantes.
7. Interpretar los resultados considerando sus limitaciones y aplicaciones prácticas.

## 4. Estructura del repositorio

```text
.
├── data/
│   ├── raw/
│   │   └── customer_analytics.csv
│   └── processed/
│       ├── ecommerce_mcar.csv
│       └── valores_originales_mcar.csv
│
├── notebooks/
│   └── sumativa_1.ipynb
│
├── src/
│   ├── __init__.py
│   ├── carga.py
│   ├── validacion.py
│   └── transformacion.py
│
├── tests/
│   └── test_preparacion.py
│
│
├── requirements.txt
└── README.md
```

### Organización del código

- **`data/raw/`:** conjunto de datos original, sin modificaciones.
- **`data/processed/`:** versiones procesadas y registros de auditoría.
- **`notebooks/`:** análisis interactivos, resultados e interpretaciones.
- **`src/carga.py`:** lectura y almacenamiento de datos.
- **`src/validacion.py`:** validación estructural y auditoría de calidad.
- **`src/transformacion.py`:** generación reproducible de datos faltantes.
- **`tests/`:** pruebas automatizadas para verificar el funcionamiento de las operaciones.
- **`reports/`:** documentos de entrega e informes técnicos.

Esta arquitectura separa la lógica de procesamiento de su presentación, favoreciendo la mantenibilidad y reutilización del código.

## 5. Preparación de los datos

La preparación implementada para la primera evaluación contempla las siguientes operaciones:

### 5.1. Carga y validación

- Lectura del dataset mediante Pandas.
- Validación de columnas requeridas.
- Verificación de identificadores únicos.
- Revisión de tipos de datos.
- Identificación de registros duplicados.
- Cuantificación de valores faltantes.

### 5.2. Generación de valores faltantes MCAR

Debido a que el conjunto de datos original no contiene valores faltantes, se aplica un mecanismo **MCAR (*Missing Completely At Random*)** siguiendo las instrucciones metodológicas del curso.

| Parámetro | Configuración |
|---|---|
| Variables | `Weight_in_gms` y `Cost_of_the_Product` |
| Proporción | 10 % por variable |
| Semilla | 42 |
| Procedimiento | Selección aleatoria sin reemplazo |
| Auditoría | Conservación de valores originales |

Este procedimiento permite disponer de datos incompletos para el estudio posterior de técnicas de imputación.

La versión original permanece inalterada y las transformaciones se registran para facilitar su reproducción.

**Nota:** la generación de faltantes no constituye imputación. El tratamiento de los valores ausentes durante cada análisis debe justificarse metodológicamente.

## 6. Entorno de desarrollo

### Requisitos

- Python 3
- Visual Studio Code o Jupyter Notebook
- Git

### Instalación

**1. Clonar el repositorio**

```bash
git clone https://github.com/jorge170102/ESTAD-STICA-COMPUTACIONAL---Grupo-7--MCDI501.202682.1914.EL.ON.git
```

**2. Ingresar al directorio del proyecto**

```bash
cd ESTAD-STICA-COMPUTACIONAL---Grupo-7--MCDI501.202682.1914.EL.ON
```

**3. Crear un entorno virtual**

```bash
python -m venv .venv
```

**4. Activar el entorno virtual en Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**5. Instalar las dependencias**

```bash
python -m pip install -r requirements.txt
```

### Bibliotecas principales

- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Statsmodels
- Scikit-learn
- Jupyter

## 7. Ejecución del proyecto

Abrir `notebooks/sumativa_1.ipynb` desde Visual Studio Code o Jupyter.

Seleccionar el intérprete del entorno virtual y ejecutar las celdas secuencialmente.

El notebook utiliza funciones definidas en `src/` y rutas relativas a la raíz del repositorio.

La preparación produce los siguientes archivos:

- `data/processed/ecommerce_mcar.csv`: versión del dataset con valores faltantes generados.
- `data/processed/valores_originales_mcar.csv`: registro de los valores originales retirados artificialmente.

### Pruebas automatizadas

Desde la raíz del repositorio:

```bash
python -m unittest discover -s tests -v
```

Las pruebas permiten comprobar propiedades del proceso de preparación, incluyendo reproducibilidad y conservación de los datos originales.

## 8. Análisis estadístico propuesto

### Estadística descriptiva

Se estudiarán distribuciones, frecuencias, medidas de tendencia central y dispersión, además de relaciones entre variables.

### Estimación de parámetros

Se contempla estimar parámetros poblacionales e intervalos de confianza del 95 % para variables numéricas como:

- Peso del producto.
- Costo del producto.
- Descuento ofrecido.

### Pruebas de hipótesis

Se proponen inicialmente las siguientes investigaciones:

**Hipótesis 1: Método de transporte y puntualidad**

Evaluar si existe asociación estadísticamente significativa entre el método de envío y la puntualidad de entrega, mediante una prueba chi-cuadrado de independencia.

**Hipótesis 2: Peso del producto y puntualidad**

Evaluar si existen diferencias estadísticamente significativas entre el peso promedio de los productos de envíos puntuales y no puntuales, considerando una prueba t de Welch cuando corresponda.

Los procedimientos se seleccionarán y justificarán después de examinar los datos y verificar sus supuestos.

## 9. Estado del proyecto

| Componente | Estado |
|---|---|
| Selección del dataset | Completado |
| Configuración del repositorio | Completado |
| Carga y validación | Implementado |
| Auditoría de calidad | Implementado |
| Generación MCAR | Implementado |
| Pruebas automatizadas de preparación | Implementadas |
| Estadística descriptiva | Pendiente |
| Análisis bivariado | Pendiente |
| Intervalos de confianza | Pendiente |
| Pruebas de hipótesis | Pendiente |
| Informe técnico final | En desarrollo |

## 10. Consideraciones metodológicas

El proyecto prioriza:

- **Reproducibilidad:** utilización de semillas aleatorias y dependencias documentadas.
- **Trazabilidad:** conservación de fuentes y registros de transformación.
- **Modularidad:** funciones independientes y reutilizables.
- **Validación:** controles estructurales y pruebas automatizadas.
- **Rigor estadístico:** verificación de supuestos e interpretación de resultados.

Los análisis de asociación no deben interpretarse automáticamente como relaciones causales.

## 11. Referencias

- Universidad Andrés Bello (2026). *Bases de datos disponibles para el proyecto del curso*. Material académico de Estadística Computacional.
- Kaggle. [E-Commerce Shipping Data — Customer Analytics](https://www.kaggle.com/datasets/prachi13/customer-analytics).
- Documentación oficial de [Python](https://docs.python.org/3/), [Pandas](https://pandas.pydata.org/docs/) y [SciPy](https://docs.scipy.org/doc/scipy/).

---

**Proyecto académico — Grupo 7**  
Magíster en Ciencia de Datos e Inteligencia Artificial  
Universidad Andrés Bello · 2026
