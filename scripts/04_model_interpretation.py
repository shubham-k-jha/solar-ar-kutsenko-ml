"""Interpretation-only all-feature XGBoost: SHAP + permutation + interactions."""
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

PATH=Path("data/KUTSENKO_2021_243_AR_model_ready.csv")
FEATURES=["Phi_max_1e21_Mx","R_av_1e20_Mx_per_h","alpha_sat","alpha_mean","alpha_sat_std","alpha_mean_std"]

df=pd.read_csv(PATH)
df["flare_index"]=pd.to_numeric(df["flare_index"],errors="coerce")
df=df[df["flare_index"]>0].copy()
df["log_flare_index"]=np.log10(df["flare_index"])
d=df[FEATURES+["log_flare_index"]].replace([np.inf,-np.inf],np.nan).dropna()
X,y=d[FEATURES],d["log_flare_index"]

def make_model():
    return Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("model",XGBRegressor(n_estimators=500,max_depth=2,learning_rate=0.03,
                              min_child_weight=5,subsample=0.8,colsample_bytree=0.8,
                              reg_alpha=0.5,reg_lambda=2.0,
                              objective="reg:squarederror",random_state=42,n_jobs=-1))
    ])

model=make_model()
model.fit(X,y)
xgb=model.named_steps["model"]

imp=pd.DataFrame({"feature":FEATURES,"importance":xgb.feature_importances_}).sort_values("importance",ascending=False)
print("\nXGBoost importance\n",imp.to_string(index=False))
imp.to_csv("results/xgb_feature_importance_recomputed.csv",index=False)

X_imp=model.named_steps["imputer"].transform(X)
X_shap=pd.DataFrame(X_imp,columns=FEATURES)
explainer=shap.TreeExplainer(xgb)
sv=explainer.shap_values(X_shap)

si=pd.DataFrame({"feature":FEATURES,"mean_abs_SHAP":np.abs(sv).mean(axis=0)}).sort_values("mean_abs_SHAP",ascending=False)
print("\nSHAP importance\n",si.to_string(index=False))
si.to_csv("results/shap_importance_recomputed.csv",index=False)

rows=[]
cv=KFold(n_splits=5,shuffle=True,random_state=42)
for tr,te in cv.split(X):
    m=make_model()
    m.fit(X.iloc[tr],y.iloc[tr])
    p=permutation_importance(m,X.iloc[te],y.iloc[te],scoring="r2",n_repeats=20,random_state=42,n_jobs=-1)
    for f,a,b in zip(FEATURES,p.importances_mean,p.importances_std):
        rows.append({"feature":f,"importance":a,"std":b})
perm=(pd.DataFrame(rows).groupby("feature").agg(mean_importance=("importance","mean"),std_importance=("importance","std")).sort_values("mean_importance",ascending=False))
print("\nPermutation importance\n",perm.to_string())
perm.to_csv("results/permutation_importance_recomputed.csv")

iv=np.asarray(explainer.shap_interaction_values(X_shap))
strength=np.abs(iv).mean(axis=0)
rows=[]
for i in range(len(FEATURES)):
    for j in range(i+1,len(FEATURES)):
        rows.append({"feature_1":FEATURES[i],"feature_2":FEATURES[j],"mean_abs_interaction":strength[i,j]})
inter=pd.DataFrame(rows).sort_values("mean_abs_interaction",ascending=False)
print("\nSHAP interactions\n",inter.to_string(index=False))
inter.to_csv("results/shap_interactions_recomputed.csv",index=False)
