# Sumativa 2 — Fase 3 · Informe 3

**Grupo 7 · MCDI501 · Magíster en Ciencia de Datos e Inteligencia Artificial · UNAB**

Índice de evidencia para la Sumativa 2. Los archivos canónicos siguen en
`notebooks/`, `src/`, `tests/`, `data/processed/` y `reports/`.

## Evidencias para la rúbrica

| Criterio | Evidencia |
|---|---|
| Bootstrap de 3 medias | [Notebook S2](../notebooks/sumativa_2.ipynb), [`s2_bootstrap.py`](../src/s2_bootstrap.py) |
| Permutaciones de hipótesis de S1 | [Notebook S2](../notebooks/sumativa_2.ipynb), [`s2_permutaciones.py`](../src/s2_permutaciones.py), [`validacion_s2.py`](../src/validacion_s2.py) |
| 3 correlaciones bootstrap | [Notebook S2](../notebooks/sumativa_2.ipynb), [`s2_correlaciones.py`](../src/s2_correlaciones.py) |
| Monte Carlo 10.000 iteraciones, convergencia y cobertura Welch | [Notebook S2](../notebooks/sumativa_2.ipynb), [`s2_montecarlo.py`](../src/s2_montecarlo.py) |
| Sensibilidad y robustez | [`s2_robustez.py`](../src/s2_robustez.py), [Notebook S2](../notebooks/sumativa_2.ipynb) |
| Resultados validados para S3 | [JSON oficial S2](../data/processed/resultados_validos_s2.json), [`s2_resultados.py`](../src/s2_resultados.py) |
| Pruebas unitarias | [Suite `tests/`](../tests/), 18 casos identificados: S1 + S2 |
| Informe técnico institucional | [Ver informe](../reports/Informe_Sumativa_2_Grupo_7.pdf) |

## Reproducibilidad

Ejecutar desde la raíz, con las dependencias instaladas:

```powershell
python -m pytest -q
python tools/verificar_cierre_s2.py --tests
```

Después abrir `notebooks/sumativa_2.ipynb`, seleccionar el entorno correcto y
realizar **Restart Kernel → Run All**. Verificar que las 19 celdas terminen
sin error y los 15 controles internos sean `True`. Guardar el notebook y el
JSON regenerado si cambian sus resultados.

La referencia de S1 se comprueba mediante la huella del CSV procesado. Los
intervalos por Monte Carlo corresponden a modelos condicionados a parámetros
de S1 y **no prueban causalidad**.

## Informe técnico de la Sumativa 2

El informe técnico definitivo de la Sumativa 2 se encuentra disponible en:

reports/Informe_Sumativa_2_Grupo_7_UNAB.pdf

El documento cumple con la extensión máxima de siete páginas establecida en las instrucciones de evaluación y presenta los principales resultados de validación mediante bootstrap, pruebas de permutación, estabilidad de correlaciones, simulación Monte Carlo y análisis de robustez.

Las evidencias computacionales se encuentran en:

notebooks/sumativa_2.ipynb: notebook con 19 celdas de código y 15 controles internos satisfactorios.

data/processed/resultados_validos_s2.json: resultados consolidados y recomendaciones para la Sumativa 3.

src/: módulos de implementación de los procedimientos estadísticos.

Los resultados pueden reproducirse utilizando las semillas establecidas y el conjunto de datos procesado en la Sumativa 1.

*No duplicar notebook ni JSON dentro de S2*: los enlaces apuntan a los archivos
que se ejecutan y mantienen efectivamente.
