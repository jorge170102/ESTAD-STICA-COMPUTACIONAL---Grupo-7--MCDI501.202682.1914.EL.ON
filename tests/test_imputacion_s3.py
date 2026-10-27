"""Comparación frente a la fórmula de imputación original S3, datos sintéticos."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.s3_imputacion import ImputadorRegresionMultiple, ejecutar_imputacion


def datos():
    rng=np.random.default_rng(42)
    n=350
    d=pd.DataFrame({
      'Customer_care_calls':rng.integers(2,7,n),
      'Prior_purchases':rng.integers(2,12,n),
      'Discount_offered':rng.integers(1,65,n),
      'Weight_in_gms':rng.normal(3400,750,n).clip(1),
      'Cost_of_the_Product':rng.normal(215,38,n).clip(1)
    })
    for v in ['Weight_in_gms','Cost_of_the_Product']:
        d.loc[rng.choice(n,size=30,replace=False),v]=np.nan
    return d


def test_imputacion_equivalente_original():
    X=datos()
    m={
      'Cost_of_the_Product':['Customer_care_calls','Prior_purchases','Discount_offered','Weight_in_gms'],
      'Weight_in_gms':['Discount_offered','Cost_of_the_Product','Customer_care_calls','Prior_purchases']
    }
    def original():
        modelos={}
        for target,preds in m.items():
            obs=X[[target]+preds].dropna()
            modelos[target]=sm.OLS(obs[target],sm.add_constant(obs[preds],has_constant='add')).fit(cov_type='HC3')
        d=X.copy()
        missing={v:d[v].isna().to_numpy() for v in m}
        for v in m:d[v]=d[v].fillna(X[v].median())
        for _ in range(2):
            for v,preds in m.items():
                if missing[v].any():
                    pred=modelos[v].predict(sm.add_constant(d.loc[missing[v],preds],has_constant='add'))
                    d.loc[missing[v],v]=np.maximum(pred,1)
        return d
    cl=ImputadorRegresionMultiple(m).ajustar(X)
    out=cl.transformar(X)
    np.testing.assert_allclose(out[list(m)].to_numpy(),original()[list(m)].to_numpy(),rtol=0,atol=1e-10)
    assert (out[list(m)]>0).all().all()
    for v in m:
        assert out.loc[X[v].notna(),v].equals(X.loc[X[v].notna(),v])


def test_imputacion_misma_transformacion_al_repetir():
    X=datos();m={'Cost_of_the_Product':['Customer_care_calls','Prior_purchases','Discount_offered','Weight_in_gms'], 'Weight_in_gms':['Discount_offered','Cost_of_the_Product','Customer_care_calls','Prior_purchases']}
    cl=ImputadorRegresionMultiple(m).ajustar(X)
    pd.testing.assert_frame_equal(cl.transformar(X),cl.transformar(X))
