# Grupo 7 / S3 — Implementación original de celda 3.
# Extraída sin alterar la lógica: se ejecuta con s3_runtime.ejecutar_etapa.

from pathlib import Path
import json, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.base import clone
SEED=42; rng=np.random.default_rng(SEED)
ROOT=next((p for p in [Path.cwd(),Path.cwd().parent] if (p/'data/processed/ecommerce_mcar.csv').exists()),None)
if ROOT is None: raise FileNotFoundError('Abrir notebook desde raíz o notebooks/ del repositorio')
df=pd.read_csv(ROOT/'data/processed/ecommerce_mcar.csv')
with open(ROOT/'data/processed/resultados_validos_s2.json',encoding='utf-8') as f: s2=json.load(f)
TARGET='Reached.on.Time_Y.N'; MISS=['Cost_of_the_Product','Weight_in_gms']
assert len(df)==10999 and df[TARGET].notna().all()
assert all(df[c].isna().sum()==1100 for c in MISS)
assert set(df[TARGET].unique())=={0,1}
print('Datos:',df.shape,' | Faltantes:',df[MISS].isna().sum().to_dict())
print('S2 relaciones:',[(x['relacion'],round(x['rho_s1'],3),x['incluye_cero']) for x in s2['correlaciones']])
