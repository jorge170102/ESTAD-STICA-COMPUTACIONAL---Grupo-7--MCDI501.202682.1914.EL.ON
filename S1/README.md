# Sumativa 1 — Fase 2

**Grupo 7 · MCDI501 · Magíster en Ciencia de Datos e Inteligencia Artificial · UNAB**

Esta carpeta funciona como **índice de evidencia** de la Sumativa 1. Se mantienen
las rutas canónicas del repositorio para no duplicar notebooks, generar copias
desactualizadas ni romper las importaciones de Python.

## Evidencias

- [Notebook S1 ejecutable](../notebooks/sumativa_1.ipynb).
- [Informe institucional S1 (PDF)](../reports/Informe_Sumativa_1_Grupo_7_Formato_Institucional_UNAB.pdf).
- [Datos originales](../data/raw/customer_analytics.csv).
- [Datos procesados S1 con MCAR](../data/processed/ecommerce_mcar.csv).
- [Auditoría de valores retirados](../data/processed/valores_originales_mcar.csv).
- [Módulos de carga y transformación](../src/transformacion.py).
- [Pruebas de preparación](../tests/test_preparacion.py) y
  [pruebas de inferencia](../tests/test_inferencia.py).

## Continuidad S1 → S2

S1 introduce faltantes MCAR controlados en peso y costo (1.100 por variable),
sin imputación. S2 reutiliza exactamente el archivo `data/processed/ecommerce_mcar.csv`
y contrasta los resultados de S1. La codificación observada en el proyecto es:
**0 = a tiempo; 1 = fuera de plazo**.

El notebook permanece en `notebooks/sumativa_1.ipynb`: no se debe mover para
mantener las rutas relativas y la trazabilidad.
