"""Puente de compatibilidad para las etapas S3 extraídas sin alterar su lógica.

Cada etapa vive como Python legible en src/s3_*.py. Se evalúa con el espacio
compartido del notebook; las clases nuevas de imputación y bootstrap no usan
este puente. No descargar archivos ni modificar el repositorio desde aquí.
"""
from __future__ import annotations
from pathlib import Path
import runpy
from typing import MutableMapping, Any


def ejecutar_etapa(nombre_modulo: str, contexto: MutableMapping[str, Any]) -> None:
    """Ejecuta una etapa de src y sincroniza sus resultados con el notebook.

    Usa runpy con copias de las variables previamente definidas para preservar
    el comportamiento del notebook original, incluido el de funciones
    existentes que esperan variables definidas en etapas anteriores.
    """
    if not nombre_modulo.startswith("s3_") or not nombre_modulo.replace("_", "").isalnum():
        raise ValueError("Nombre de módulo S3 no válido")
    ruta = Path(__file__).resolve().parent / (nombre_modulo + ".py")
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encuentra {ruta.name}. Ejecuta tools/refactorizar_s3.py")
    entrada = {k: v for k, v in contexto.items() if not k.startswith("__")}
    salida = runpy.run_path(str(ruta), init_globals=entrada)
    contexto.update({k: v for k, v in salida.items() if not k.startswith("__")})
