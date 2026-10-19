"""Tests del motor POO: reproducibilidad y cobertura bajo modelos sintéticos."""
import numpy as np
import pandas as pd
from src.s2_montecarlo import SimuladorMonteCarlo, resumir_distribucion


def parametros_ejemplo():
    return pd.DataFrame({"n": [40, 60], "media": [4000., 3200.],
                         "de": [1200., 1300.]}, index=["A tiempo", "Fuera de plazo"])


def test_montecarlo_poo_reproducibilidad_y_cobertura():
    par = parametros_ejemplo()
    original = par.copy(deep=True)
    modelo = SimuladorMonteCarlo(par, iteraciones=500, lote=25)
    a, cobertura = modelo.simular('lognormal', 401, devolver_ic=True)
    b, c2 = modelo.simular('lognormal', 401, devolver_ic=True)
    np.testing.assert_array_equal(a, b)
    assert cobertura == c2
    assert cobertura['iteraciones'] == 500
    assert 75 < cobertura['cobertura_ic_welch_95_pct'] <= 100
    assert 0 < cobertura['ancho_medio_ic_welch_g']
    assert np.isfinite(a).all()
    pd.testing.assert_frame_equal(par, original)
    gamma = modelo.simular('gamma', 402)
    assert len(gamma) == 500
    assert abs(gamma.mean() - (-800)) < 200
    assert resumir_distribucion(a)['iteraciones'] == 500


def test_motor_rechaza_entradas_invalidas():
    import pytest
    with pytest.raises(ValueError):
        SimuladorMonteCarlo(parametros_ejemplo(), iteraciones=1)
    with pytest.raises(ValueError):
        SimuladorMonteCarlo(parametros_ejemplo(), iteraciones=5).simular('normal', 123)
