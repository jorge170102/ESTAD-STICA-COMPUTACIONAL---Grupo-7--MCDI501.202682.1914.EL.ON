"""Pruebas aisladas de integración de módulos sobre datos ficticios.

NO reemplazan ejecución contra los datos originales S1.
"""
import numpy as np
import pandas as pd
from scipy import stats
from src.s2_alineacion import reconstruir_s1
from src.s2_bootstrap import analizar_medias
from src.s2_correlaciones import analizar_correlaciones
from src.s2_permutaciones import permutar_transporte
from src.s2_robustez import analizar_jackknife, analizar_extremos, analizar_medias_recortadas, comparar_familias
from src.s2_resultados import crear_sintesis
from src.s2_montecarlo import SimuladorMonteCarlo, analizar_escenarios, resumir_distribucion


def datos_sinteticos():
    rng=np.random.default_rng(111)
    n=240
    a=pd.DataFrame({
        'ID': np.arange(1,n+1),
        'Weight_in_gms': rng.lognormal(8,0.32,n),
        'Cost_of_the_Product': rng.normal(200,40,n),
        'Discount_offered':rng.integers(1,35,n).astype(float),
        'Customer_care_calls':rng.integers(1,7,n),
        'Customer_rating':rng.integers(1,6,n),
        'Reached.on.Time_Y.N':np.tile([0,1],n//2),
        'Mode_of_Shipment':np.tile(['Ship','Road','Flight'],n//3)
    })
    return a


def test_alineacion_bootstrap_correlaciones_jackknife():
    df=datos_sinteticos()
    vars_=['Weight_in_gms','Cost_of_the_Product','Discount_offered']
    s1=reconstruir_s1(df,vars_,'Reached.on.Time_Y.N')
    assert len(s1.pares)==3
    assert len(s1.tabla_contingencia)==3
    np.testing.assert_allclose(s1.delta,s1.peso_fuera.mean()-s1.peso_tiempo.mean())
    dist,medias,table=analizar_medias(df,vars_,s1.medias,[101,102,103],B=250,LOTE=25)
    assert len(dist)==3 and (medias.bca_inf < medias.bca_sup).all()
    rhos,corr,tabla=analizar_correlaciones(df,s1.pares,[301,302,303],B=250,LOTE=25)
    assert len(corr)==3 and (corr.ic_inf < corr.ic_sup).all()
    infl,res=analizar_jackknife(df,vars_,s1.medias)
    assert len(infl)==9 and len(res)==3


def test_permutaciones_y_robustez_modular():
    df=datos_sinteticos()
    s1=reconstruir_s1(df,['Weight_in_gms','Cost_of_the_Product','Discount_offered'],'Reached.on.Time_Y.N')
    a,p=permutar_transporte(df,s1.tabla_contingencia,s1.chi,s1.p_chi,s1.gl_chi,s1.esperados,
                             s1.v_cramer,objetivo='Reached.on.Time_Y.N',B=250,semilla=201)
    assert len(a)==250 and 0<p['p_permutacion']<=1
    sims=SimuladorMonteCarlo(s1.grupos,iteraciones=250,lote=25)
    log=sims.simular('lognormal',401)
    gamma=sims.simular('gamma',402)
    diff,cmp=comparar_familias(s1.peso_fuera,s1.peso_tiempo,log,gamma,s1.delta,s1.ee_delta,B=250,LOTE=25)
    assert len(cmp)==3 and len(diff)==250
    sed,esc=analizar_escenarios(sims,s1.grupos,s1.ic_delta,s1.delta,log,
                              {'escenario_inferior':601,'lognormal':401,'escenario_superior':602})
    assert len(sed)==3 and len(esc)==3
    dd,bb,intervalo,res=analizar_medias_recortadas(s1.peso_fuera,s1.peso_tiempo,s1.delta,B=250,LOTE=25)
    assert np.isfinite(intervalo).all() and res['iteraciones']==250
    assert len(crear_sintesis())==8


def test_semillas_preservan_secundarios():
    # La misma semilla debe producir el mismo estudio, independientemente de la sección.
    pars=pd.DataFrame({'n':[75,100],'media':[4200,3300],'de':[1600,1400]},
                      index=['A tiempo','Fuera de plazo'])
    sim=SimuladorMonteCarlo(pars,iteraciones=500,lote=50)
    x,_=sim.simular('lognormal',401,devolver_ic=True)
    y=sim.simular('lognormal',401,devolver_ic=False)
    np.testing.assert_array_equal(x,y)
