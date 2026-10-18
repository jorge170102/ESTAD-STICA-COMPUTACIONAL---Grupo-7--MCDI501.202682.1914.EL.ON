"""Funciones de Sumativa 2 del Grupo 7: validación, simulación y remuestreo."""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

TARGET='Reached.on.Time_Y.N'
VARIABLES=['Weight_in_gms','Cost_of_the_Product','Discount_offered']

def correlaciones_bootstrap(df,pares=None,B=3000,semilla=42):
    if pares is None:pares=[('Cost_of_the_Product','Weight_in_gms'),('Discount_offered','Weight_in_gms'),('Customer_care_calls','Prior_purchases'),('Customer_rating','Cost_of_the_Product'),('Weight_in_gms',TARGET)]
    rng=np.random.default_rng(semilla)
    filas=[]
    for a,b in pares:
        sub=df[[a,b]].dropna().to_numpy(float)
        n=len(sub)
        r=stats.spearmanr(sub[:,0],sub[:,1]).statistic
        reps=np.empty(B)
        for i in range(B):
            samp=sub[rng.integers(0,n,n)]
            reps[i]=stats.spearmanr(samp[:,0],samp[:,1]).statistic
        lo,hi=np.quantile(reps,[.025,.975]);
        filas.append({'Par':f'{a} / {b}','N':n,'Spearman':r,'IC inferior':lo,'IC superior':hi,'Incluye cero':lo<=0<=hi})
    return pd.DataFrame(filas)
