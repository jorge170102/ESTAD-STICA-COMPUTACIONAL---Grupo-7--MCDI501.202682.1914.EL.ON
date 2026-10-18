import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.validacion_s2 import permutacion_diferencia_medias, diferencia_recortada, diagnostico_cuantiles_peso


def test_permutacion_bilateral_reproducible_y_etiquetas():
    df = pd.DataFrame({'Weight_in_gms':[2,3,4,5,14,15,16,17],
                       'Reached.on.Time_Y.N':[0,0,0,0,1,1,1,1]})
    a=permutacion_diferencia_medias(df,iteraciones=1000,semilla=42)
    b=permutacion_diferencia_medias(df,iteraciones=1000,semilla=42)
    assert a['n_a_tiempo']==4 and a['n_fuera_plazo']==4
    assert a['diferencia']==12
    assert np.array_equal(a['replicas'],b['replicas'])
    assert a['rechaza_permutacion']==(a['p_permutacion']<.05)
    assert a['rechaza_welch_s1']==(a['p_welch']<.05)
    assert (1/1001)<=a['p_permutacion']<=1
    assert a['intervalo_central_nulo_95'][0]<=a['intervalo_central_nulo_95'][1]


def test_diferencia_recortada_controla_outliers():
    a=np.array([1]*10+[10000]);b=np.array([4]*11)
    assert abs(diferencia_recortada(a,b,proporcion=.1)+3)<1e-9
    assert abs(a.mean()-b.mean())>100


def test_diagnostico_cuantiles_calibracion_y_formato():
    a=np.array([1,2,3,4,5],float); b=np.array([5,6,7,8,9],float)
    gr=pd.DataFrame({'n':[5,5], 'media':[a.mean(),b.mean()],
                     'de':[a.std(ddof=1),b.std(ddof=1)]},index=['A tiempo','Fuera de plazo'])
    d=diagnostico_cuantiles_peso(a,b,gr)
    assert len(d)==10
    assert (d[['cuantil_lognormal_g','cuantil_gamma_g']]>0).all().all()
    assert set(d.grupo)=={'A tiempo','Fuera de plazo'}


def test_both_groups_required():
    df=pd.DataFrame({'Weight_in_gms':[1,2,3,4], 'Reached.on.Time_Y.N':[0]*4})
    import pytest
    with pytest.raises(ValueError):permutacion_diferencia_medias(df,iteraciones=20)
