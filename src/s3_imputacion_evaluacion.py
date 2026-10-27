# Grupo 7 / S3 — Implementación original de celda 9.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

for v,preds in IMP_PRED.items():
    obs=Xtr[[v]+preds].dropna()
    a,b=train_test_split(obs.index,test_size=.20,random_state=SEED)
    f=LinearRegression().fit(obs.loc[a,preds],obs.loc[a,v]); pred=f.predict(obs.loc[b,preds])
    baseline=np.repeat(obs.loc[a,v].median(),len(b))
    print(
    v,
    'RMSE regresión:', round(np.sqrt(np.mean((pred - obs.loc[b, v])**2)), 2),
    'RMSE mediana:', round(np.sqrt(np.mean((baseline - obs.loc[b, v])**2)), 2),
    'MAE regresión:', round(np.mean(np.abs(pred - obs.loc[b, v])), 2),
    'MAE mediana:', round(np.mean(np.abs(baseline - obs.loc[b, v])), 2)
)
frames={'eliminacion':Xdel,'mediana':Xsimple,'regresion':Xreg}
rows=[]
for name,d in frames.items():
    for v in MISS:
        rows.append({'estrategia':name,'variable':v,'n':len(d),'media':d[v].mean(),'desv':d[v].std(),
                     'rho_peso_descuento':d['Weight_in_gms'].corr(d['Discount_offered'],method='spearman'),
                     'rho_llamadas_costo':d['Customer_care_calls'].corr(d['Cost_of_the_Product'],method='spearman')})
comparison=pd.DataFrame(rows)
print(comparison.round(3).to_string(index=False))
