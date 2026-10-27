# Grupo 7 / S3 — Implementación original de celda 19.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

metrics=[]
for key,m in models.items():
    for split,Z,yy in [('train',Ztr,ytr),('test',Zte,yte)]:
        prob=m.predict(sm.add_constant(Z[selected[key]],has_constant='add'))
        pred=(prob>=.5).astype(int)
        metrics.append({'modelo':key,'split':split,'n':len(yy),'accuracy':accuracy_score(yy,pred),
                        'precision':precision_score(yy,pred,zero_division=0),'recall':recall_score(yy,pred,zero_division=0),
                        'F1':f1_score(yy,pred,zero_division=0),'AUC':roc_auc_score(yy,prob)})
        print(key,split,'confusión [filas reales 0/1; columnas predichas 0/1]:\n',confusion_matrix(yy,pred))
print(pd.DataFrame(metrics).round(4).to_string(index=False))
fig,ax=plt.subplots(figsize=(7,5))
for key,m in models.items():
    p=m.predict(sm.add_constant(Zte[selected[key]],has_constant='add'))
    RocCurveDisplay.from_predictions(yte,p,name=key,ax=ax)
ax.plot([0,1],[0,1],'--',alpha=.4);ax.set_title('ROC · prueba común');plt.show()

score = (pd.DataFrame(metrics).query("split == 'test'").sort_values(['AUC', 'F1'],ascending=False))
         
print("\nCOMPARACIÓN DE DESEMPEÑO EN PRUEBA")
print(score.round(4).to_string(index=False))
