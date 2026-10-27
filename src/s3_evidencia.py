# Grupo 7 / S3 — Implementación original de celda 5.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

print(pd.DataFrame({'faltantes':df.isna().sum(),'porcentaje':100*df.isna().mean()}).query('faltantes>0').round(2))
print(pd.DataFrame(s2['correlaciones'])[['relacion','rho_s1','ic_inf','ic_sup','incluye_cero']].round(3))
features=[c for c in df if c not in [TARGET,'ID']]
X=df[features].copy(); y=df[TARGET].astype(int)
idx_train,idx_test=train_test_split(np.arange(len(df)),test_size=.30,random_state=SEED,stratify=y)
# Test común: solo observaciones completas en los predictores afectados por MCAR.
idx_test=idx_test[X.iloc[idx_test][MISS].notna().all(axis=1).to_numpy()]
Xtr,Xte=X.iloc[idx_train].copy(),X.iloc[idx_test].copy(); ytr,yte=y.iloc[idx_train],y.iloc[idx_test]
print('Train:',len(Xtr),'Test común completo:',len(Xte),'Prevalencia test:',round(yte.mean(),4))
