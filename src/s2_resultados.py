"""Estadística computacional Grupo 7: módulo de Sumativa 2 (no modifica S1)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import json
import subprocess
from datetime import datetime, timezone

def crear_sintesis():
    sintesis = pd.DataFrame([
        {"resultado": "Media de peso", "evaluacion": "Consistente entre IC; baja influencia individual",
         "alcance": "Sin extremos según 1,5 RIC; independencia y representatividad pendientes",
         "recomendacion_s3": "Conservar media e IC BCa como referencia; revisar sesgos y variables operativas"},
        {"resultado": "Media de costo", "evaluacion": "Consistente entre IC; baja influencia individual",
         "alcance": "Sin extremos según 1,5 RIC; análisis sobre casos disponibles",
         "recomendacion_s3": "Conservar media e IC; evaluar imputación sin acceder a verdad reservada durante el ajuste"},
        {"resultado": "Media de descuento", "evaluacion": "Precisa con la muestra completa; sensible a descuentos altos",
         "alcance": "Excluir 20,08% cambia la población descrita; unidad sin confirmar",
         "recomendacion_s3": "Conservar casos; verificar unidad y complementar con mediana, cuantiles y segmentos"},
        {"resultado": "Transporte y puntualidad", "evaluacion": "Pruebas coinciden: no se rechaza independencia",
         "alcance": "No prueba equivalencia ni ausencia de asociación en otra población",
         "recomendacion_s3": "No priorizar cambios de transporte por este resultado aislado"},
        {"resultado": "Peso y descuento", "evaluacion": "Signo negativo estable; magnitud sensible al filtro",
         "alcance": "La asociación se debilita al retirar descuentos altos",
         "recomendacion_s3": "Analizar composición y posibles variables de confusión; evitar interpretación causal"},
        {"resultado": "Llamadas y costo", "evaluacion": "Signo positivo estable frente al remuestreo",
         "alcance": "Asociación limitada; no indica causalidad ni utilidad predictiva asegurada",
         "recomendacion_s3": "Mantener como relación exploratoria; comprobar con validación fuera de muestra"},
        {"resultado": "Valoración y peso", "evaluacion": "Asociación pequeña; IC incluye cero",
         "alcance": "No se confirma el signo; no se observa alta variabilidad",
         "recomendacion_s3": "No priorizar como asociación monotónica fuerte; no descartar otras relaciones"},
        {"resultado": "Diferencia de peso por puntualidad", "evaluacion": "Dirección examinada con Welch, permutación, familias simuladas y media recortada",
         "alcance": "Modelos calibrados con S1; no identifican causas del retraso",
         "recomendacion_s3": "Investigar perfiles, destinos y distancia antes de recomendar intervenciones"},
    ])
    return sintesis


def registros(tabla, incluir_indice=False):
    tabla = tabla.reset_index() if incluir_indice else tabla
    return json.loads(tabla.to_json(orient="records", double_precision=15))



def guardar_resultados(raiz, ruta_salida, datos):
    REPO_URL = datos["REPO_URL"]
    COMMIT_S1 = datos["COMMIT_S1"]
    hash_datos = datos["hash_datos"]
    VERSIONES = datos["VERSIONES"]
    B = datos["B"]
    M = datos["M"]
    LOTE = datos["LOTE"]
    SEMILLAS = datos["SEMILLAS"]
    bootstrap_medias = datos["bootstrap_medias"]
    permutacion = datos["permutacion"]
    resumen_permutacion_peso = datos["resumen_permutacion_peso"]
    correlaciones = datos["correlaciones"]
    grupos_s1 = datos["grupos_s1"]
    delta_s1 = datos["delta_s1"]
    ic_delta_s1 = datos["ic_delta_s1"]
    welch = datos["welch"]
    ee_delta = datos["ee_delta"]
    resumen_mc = datos["resumen_mc"]
    convergencia = datos["convergencia"]
    controles_convergencia = datos["controles_convergencia"]
    cobertura_welch_mc = datos["cobertura_welch_mc"]
    control_cobertura = datos["control_cobertura"]
    comparacion_modelos = datos["comparacion_modelos"]
    diagnostico_distribuciones = datos["diagnostico_distribuciones"]
    sensibilidad_entradas = datos["sensibilidad_entradas"]
    influencia = datos["influencia"]
    resumen_recortadas = datos["resumen_recortadas"]
    resumen_influencia = datos["resumen_influencia"]
    auditoria_extremos = datos["auditoria_extremos"]
    sensibilidad_extremos = datos["sensibilidad_extremos"]
    n_excluidos = datos["n_excluidos"]
    df = datos["df"]
    pct_desc_filtrado = datos["pct_desc_filtrado"]
    sintesis = datos["sintesis"]
    try:
        commit_ejecucion = subprocess.run(["git", "-C", str(raiz), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit_ejecucion = "No disponible; referencia S1 identificada por commit y hash de datos"

    resultados_s2 = {
        "schema_version": "1.2", "evaluacion": "Sumativa 2", "equipo": "Grupo 7",
        "procedencia": {
            "repositorio": REPO_URL, "commit_referencia_s1": COMMIT_S1,
            "commit_en_ejecucion": commit_ejecucion,
            "notebook_s1": "notebooks/sumativa_1.ipynb",
            "datos": "data/processed/ecommerce_mcar.csv", "sha256_datos_lf": hash_datos,
            "fecha_utc": datetime.now(timezone.utc).isoformat(), "versiones": VERSIONES,
            "bootstrap_por_analisis": B, "permutaciones": B, "monte_carlo_por_escenario": M,
            "lote": LOTE, "semillas": SEMILLAS, "confianza": 0.95,
            "codificacion": {"0": "a tiempo", "1": "fuera de plazo"},
            "tratamiento_faltantes": "Casos disponibles por variable; pares completos en correlaciones; sin imputación",
            "fuentes_s1": {"medias": "notebook sección 6.1; informe VII",
                "pruebas": "notebook 6.2 y 6.3; informe VIII",
                "correlaciones_y_grupos": "notebook 5.3; informe VI"},
        },
        "medias_validadas": registros(bootstrap_medias, True),
        "permutacion": permutacion,
        "permutacion_peso": resumen_permutacion_peso,
        "correlaciones": registros(correlaciones, True),
        "monte_carlo": {
            "pregunta": "Variación de la diferencia de peso medio fuera de plazo menos a tiempo en estudios repetidos",
            "parametros_s1": registros(grupos_s1[["n", "media", "de"]], True),
            "diferencia_s1": delta_s1, "ic_welch_s1": ic_delta_s1.tolist(),
            "p_welch_s1": float(welch.pvalue), "error_estandar_s1": ee_delta,
            "modelo_principal": "Lognormal por grupo ajustada por momentos de S1",
            "resultado_principal": resumen_mc, "convergencia": registros(convergencia),
            "controles_convergencia": {k: bool(v) for k, v in controles_convergencia.items()},
            "validacion_cobertura_welch": registros(cobertura_welch_mc),
            "control_cobertura_condicional": bool(control_cobertura),
            "interpretacion_cobertura": (
                "Fracción de intervalos Welch que incluyen el parámetro de entrada en 10.000 "
                "estudios por modelo. Evalúa calibración condicional, no validez externa ni causalidad."
            ),
            "comparacion_modelos": registros(comparacion_modelos, True),
            "diagnostico_ajuste_cuantiles": registros(diagnostico_distribuciones),
            "sensibilidad_entradas": registros(sensibilidad_entradas, True),
            "regla_escenarios": "Media a tiempo, DE y n fijos en S1; media fuera = media a tiempo + extremo IC Welch S1",
            "interpretacion_intervalos": "Monte Carlo: variación condicional al modelo; bootstrap: IC percentil de la diferencia",
        },
        "robustez": {
            "influencia_jackknife": registros(influencia),
            "comparacion_medias_recortadas": resumen_recortadas,
            "resumen_influencia": registros(resumen_influencia, True),
            "auditoria_extremos": registros(auditoria_extremos, True),
            "sensibilidad_extremos": registros(sensibilidad_extremos),
            "n_excluidos_escenario": n_excluidos,
            "porcentaje_excluido_escenario": 100*n_excluidos/len(df),
            "ic_percentil_descuento_filtrado": pct_desc_filtrado.tolist(),
            "politica": "Retener todos los registros en el análisis principal; la exclusión es solo sensibilidad",
        },
        "sintesis_y_recomendaciones_s3": registros(sintesis),
        "limitaciones": ["Muestreo e independencia no documentados", "Asociaciones no causales",
            "Faltantes MCAR artificiales; incertidumbre condicionada a datos disponibles",
            "Unidad del descuento sin confirmar", "IC y pruebas individuales sin ajuste conjunto",
            "Simulación calibrada con S1: convergencia y cobertura condicional no implican validez externa del modelo"],
        "entrega_formal_pendiente": ["Ejecutar desde cero y actualizar este JSON en el repositorio oficial",
            "Publicar los entregables y el JSON en GitHub y presentar el enlace en Canvas"],
    }

    # allow_nan=False impide exportar números no válidos como si fueran resultados.
    contenido_json = json.dumps(resultados_s2, ensure_ascii=False, indent=2, allow_nan=False)
    ruta_salida.write_text(contenido_json, encoding="utf-8")
    print(f"Resultados para S3 guardados: {ruta_salida.relative_to(raiz)}")
    return resultados_s2
