"""Equivalencia exacta con SciPy 1.17 y pruebas de empates en bootstrap emparejado."""
import numpy as np
import pandas as pd
from scipy import stats
from src.s2_correlaciones import bootstrap_correlacion, spearman_por_lote


def prueba_referencia(datos, seed=301, B=150, LOTE=25):
    x, y = datos.to_numpy(dtype=float).T
    referencia = stats.bootstrap(
        (x, y), spearman_por_lote, vectorized=True, paired=True,
        n_resamples=B, batch=LOTE, method="percentile",
        confidence_level=.95, rng=np.random.default_rng(seed))
    return referencia.bootstrap_distribution, np.array(referencia.confidence_interval)


def test_igualdad_scipy_empates_y_semillas():
    rng = np.random.default_rng(1)
    datos = pd.DataFrame({
        "Peso": rng.integers(1000, 5001, size=800),
        "Descuento": rng.integers(1, 66, size=800),
    })
    actual, intervalo = bootstrap_correlacion(datos, 301, B=150, LOTE=25)
    esperado, ci_esperado = prueba_referencia(datos)
    np.testing.assert_array_equal(actual, esperado)
    np.testing.assert_allclose(intervalo, ci_esperado, rtol=0, atol=1e-14)


def test_igualdad_scipy_variables_discretas():
    rng=np.random.default_rng(11)
    datos=pd.DataFrame({"llamadas":rng.integers(1,6,800),
                        "costo":rng.integers(96,311,800)})
    esperado, ci_esperado = prueba_referencia(datos,seed=302,B=251,LOTE=50)
    actual, intervalo = bootstrap_correlacion(datos, 302, B=251, LOTE=50)
    np.testing.assert_array_equal(actual, esperado)
    np.testing.assert_allclose(intervalo, ci_esperado, rtol=0, atol=1e-14)


def test_parcial_ultimo_lote():
    datos=pd.DataFrame({"a":[1,2,3,4,5,6,7,8]*40,
                        "b":[8,7,6,5,4,3,2,1]*40})
    esperado, _=prueba_referencia(datos,seed=9,B=137,LOTE=50)
    actual, _=bootstrap_correlacion(datos,9,B=137,LOTE=50)
    np.testing.assert_array_equal(actual, esperado)
