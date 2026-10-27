"""Reproducibilidad, mismos índices y mismos coeficientes que GLM original."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pytest
from src.s3_bootstrap import BootstrapLogistico


def caso():
    rng=np.random.default_rng(401);n=140
    z=pd.DataFrame({'Discount_offered':rng.normal(size=n),'Weight_in_gms':rng.normal(size=n)})
    p=1/(1+np.exp(-(.25+1.1*z['Discount_offered']-.65*z['Weight_in_gms'])))
    y=pd.Series(rng.binomial(1,p))
    cols=['Discount_offered','Weight_in_gms']
    model=sm.GLM(y,sm.add_constant(z[cols],has_constant='add'),family=sm.families.Binomial()).fit()
    return z,y,cols,model


def test_semillas_y_equivalencia_con_el_bucle_original():
    X,y,cols,model=caso();B=32;seed=142
    engine=BootstrapLogistico(n_boot=B,seed=seed)
    r=engine.evaluar({'prueba':model},{'prueba':cols},X,y)
    again=engine.evaluar({'prueba':model},{'prueba':cols},X,y)
    A=sm.add_constant(X[cols],has_constant='add').to_numpy(dtype=float)
    yy=y.to_numpy();rng=np.random.default_rng(seed)
    indices=[rng.integers(0,len(yy),size=len(yy)) for _ in range(B)]
    reference=[]
    for ids in indices:
        f=sm.GLM(yy[ids],A[ids],family=sm.families.Binomial()).fit(maxiter=100,disp=0)
        assert f.converged
        reference.append(np.asarray(f.params))
    np.testing.assert_allclose(r.resultados['prueba']['replicas'],reference,rtol=0,atol=0)
    pd.testing.assert_frame_equal(r.tabla_estabilidad,again.tabla_estabilidad)
    assert r.tabla_resumen['replicas_validas'].iat[0] == B
    assert {'OR_boot_inf','OR_boot_sup','IC_Wald_inf','IC_boot_inf','cambio_signo_pct'} <= set(r.tabla_estabilidad)


def test_rechaza_incoherencias():
    X,y,cols,m=caso()
    with pytest.raises(ValueError): BootstrapLogistico(n_boot=0)
    with pytest.raises(ValueError): BootstrapLogistico().evaluar({'x':m},{'x':cols},X,y.rename(index={0:99999}))
