"""Inferencia reproducible para E-Commerce Shipping (sin imputación)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

OBJETIVO = "Reached.on.Time_Y.N"


def _valores(serie: pd.Series) -> pd.Series:
    x = serie.dropna()
    if not pd.api.types.is_numeric_dtype(x):
        raise TypeError(f"{serie.name} debe ser numérica")
    x = x.astype(float)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(f"{serie.name}: valores insuficientes o no finitos")
    return x


def mostrar_estimaciones(df: pd.DataFrame, confianza: float = .95) -> pd.DataFrame:
    """IC bilateral t de Student para tres medias: casos válidos por variable."""
    if not 0 < confianza < 1:
        raise ValueError("confianza debe pertenecer al intervalo (0, 1)")
    filas=[]
    for col in ("Weight_in_gms", "Cost_of_the_Product", "Discount_offered"):
        x=_valores(df[col]);n=len(x);media=x.mean();de=x.std(ddof=1)
        se=de/np.sqrt(n); tcrit=stats.t.ppf((1+confianza)/2,n-1)
        filas.append({"Variable":col,"N válido":n,"Faltantes excluidos":int(df[col].isna().sum()),
                      "Media estimada":media,"DE":de,"Error estándar":se,"Margen":tcrit*se,
                      "IC inferior":media-tcrit*se,"IC superior":media+tcrit*se})
    tabla=pd.DataFrame(filas).set_index("Variable")
    from IPython.display import display
    display(tabla.round(3))
    return tabla


def prueba_peso_por_entrega(df: pd.DataFrame, alpha: float = .05, mostrar_graficos: bool = True) -> pd.DataFrame:
    """Welch bilateral, diferencia fuera-de-plazo menos a-tiempo, IC y d de Cohen."""
    if not 0 < alpha < 1:raise ValueError("alpha debe estar entre 0 y 1")
    if df[OBJETIVO].isna().any() or not df[OBJETIVO].isin([0,1]).all():
        raise ValueError("Codificación objetivo inválida")
    a=_valores(df.loc[df[OBJETIVO]==0,"Weight_in_gms"])
    b=_valores(df.loc[df[OBJETIVO]==1,"Weight_in_gms"])
    n0,n1=len(a),len(b); v0,v1=a.var(ddof=1),b.var(ddof=1)
    se=np.sqrt(v0/n0+v1/n1)
    if se==0:raise ValueError("No hay variabilidad")
    gl=(v0/n0+v1/n1)**2 / ((v0/n0)**2/(n0-1)+(v1/n1)**2/(n1-1))
    t=stats.ttest_ind(b,a,equal_var=False)
    delta=b.mean()-a.mean();crit=stats.t.ppf(1-alpha/2,gl)
    s_pool=np.sqrt(((n0-1)*v0+(n1-1)*v1)/(n0+n1-2));d=delta/s_pool
    res=pd.DataFrame([{"N a tiempo":n0,"N fuera plazo":n1,"Media a tiempo":a.mean(),
      "Media fuera plazo":b.mean(),"Diferencia g (fuera - a tiempo)":delta,
      "t":t.statistic,"gl":gl,"p":t.pvalue,"IC inferior":delta-crit*se,
      "IC superior":delta+crit*se,"d Cohen":d,"alpha":alpha}])
    from IPython.display import display,Markdown
    display(res.T.round(4))
    if mostrar_graficos:
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for i,(label,x) in enumerate((("A tiempo",a),("Fuera de plazo",b))):
            axes[i,0].hist(x,bins="auto",color="#267FA3");axes[i,0].set_title(label)
            stats.probplot(x,dist="norm",plot=axes[i,1]);axes[i,1].set_title(f"Q-Q: {label}")
        fig.tight_layout();plt.show();plt.close(fig)
    decision="Se rechaza H0" if t.pvalue<alpha else "No se rechaza H0"
    display(Markdown(f"**{decision}** (p={t.pvalue:.3g}). Diferencia: {delta:.2f} g; "
        f"IC del {100*(1-alpha):.0f} %: [{delta-crit*se:.2f}, {delta+crit*se:.2f}] g; "
        f"d de Cohen={d:.3f}. La relación observada no establece causalidad. "
        "La independencia de las observaciones debe justificarse con el diseño de muestreo; "
        "Welch no exige igualdad de varianzas y los tamaños muestrales grandes favorecen la aproximación asintótica."))
    return res


def prueba_modo_por_entrega(df: pd.DataFrame, alpha: float = .05, mostrar_graficos: bool = True) -> pd.DataFrame:
    """Chi-cuadrado de independencia 3x2 con chequeo de esperados y V de Cramér."""
    if not 0<alpha<1:raise ValueError("alpha fuera de rango")
    if df[OBJETIVO].isna().any() or not df[OBJETIVO].isin([0,1]).all():
        raise ValueError("Codificación objetivo inválida")
    d=df[["Mode_of_Shipment",OBJETIVO]].dropna()
    tab=pd.crosstab(d["Mode_of_Shipment"],d[OBJETIVO])
    if tab.shape[0]<2 or tab.shape[1]!=2:raise ValueError("Contingencia incompleta")
    chi,p,gl,esperados=stats.chi2_contingency(tab,correction=False)
    if (esperados<5).any():raise ValueError("Esperados < 5: usar contraste exacto o simulación")
    v=np.sqrt(chi/(len(d)*min(tab.shape[0]-1,tab.shape[1]-1)))
    res=pd.DataFrame([{"Chi-cuadrado":chi,"gl":gl,"p":p,"V Cramer":v,
      "N":len(d),"N excluidos":len(df)-len(d),"Esperado mínimo":esperados.min(),"alpha":alpha}])
    from IPython.display import display,Markdown
    display(Markdown("**Frecuencias observadas:**"));display(tab)
    display(Markdown("**Frecuencias esperadas:**"));display(pd.DataFrame(esperados,index=tab.index,columns=tab.columns).round(2))
    display(res.round(5))
    tasas=100*tab[1]/tab.sum(axis=1)
    if mostrar_graficos:
        fig,ax=plt.subplots(figsize=(8,3.5));tasas.plot.bar(ax=ax,color="#267FA3")
        ax.set_ylim(0,100);ax.set_ylabel("Entregas fuera de plazo (%)")
        fig.tight_layout();plt.show();plt.close(fig)
    decision="Se rechaza H0" if p<alpha else "No se rechaza H0"
    display(Markdown(f"**{decision}** (p={p:.4f}, V de Cramér={v:.4f}); "
        "esto no demuestra causalidad ni, cuando no se rechaza H0, independencia demostrada. "
        "Se asume independencia entre registros; las frecuencias esperadas son adecuadas."))
    return res
