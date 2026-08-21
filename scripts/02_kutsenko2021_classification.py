"""Kutsenko 2021 binary flare-productivity benchmark."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, brier_score_loss
from xgboost import XGBClassifier

PATH=Path("data/KUTSENKO_2021_243_AR_model_ready.csv")
df=pd.read_csv(PATH)
df["flare_index"]=pd.to_numeric(df["flare_index"],errors="coerce")
df["flare_productive"]=(df["flare_index"]>=1).astype(int)

SETS={
"R_av only":["R_av_1e20_Mx_per_h"],
"Phi only":["Phi_max_1e21_Mx"],
"Phi + R_av":["Phi_max_1e21_Mx","R_av_1e20_Mx_per_h"],
"R_av + alpha":["R_av_1e20_Mx_per_h","alpha_sat","alpha_mean"],
"R_av + alpha uncertainty":["R_av_1e20_Mx_per_h","alpha_sat","alpha_mean","alpha_sat_std","alpha_mean_std"],
"All features":["Phi_max_1e21_Mx","R_av_1e20_Mx_per_h","alpha_sat","alpha_mean","alpha_sat_std","alpha_mean_std"]
}
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

def brier(estimator,X,y):
    return brier_score_loss(y,estimator.predict_proba(X)[:,1])

rows=[]
for name,features in SETS.items():
    d=df[features+["flare_productive"]].replace([np.inf,-np.inf],np.nan)
    X,y=d[features],d["flare_productive"]
    model=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("model",XGBClassifier(n_estimators=400,max_depth=2,learning_rate=0.03,
                               min_child_weight=5,subsample=0.8,colsample_bytree=0.8,
                               reg_alpha=0.5,reg_lambda=2.0,eval_metric="logloss",
                               random_state=42,n_jobs=-1))
    ])
    s=cross_validate(model,X,y,cv=cv,scoring={
        "roc_auc":"roc_auc","pr_auc":"average_precision",
        "brier":make_scorer(brier,response_method="predict_proba")
    },n_jobs=-1)
    rows.append({"Model":name,"ROC_AUC":s["test_roc_auc"].mean(),
                 "PR_AUC":s["test_pr_auc"].mean(),"Brier":s["test_brier"].mean()})
out=pd.DataFrame(rows).sort_values("ROC_AUC",ascending=False)
print(out.to_string(index=False))
out.to_csv("results/classification_results_recomputed.csv",index=False)
