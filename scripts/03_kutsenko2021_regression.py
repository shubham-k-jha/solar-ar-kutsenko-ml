"""Controlled Kutsenko 2021 continuous flare-productivity XGBoost ablation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

PATH=Path("data/KUTSENKO_2021_243_AR_model_ready.csv")
df=pd.read_csv(PATH)
df["flare_index"]=pd.to_numeric(df["flare_index"],errors="coerce")
df=df[df["flare_index"]>0].copy()
df["log_flare_index"]=np.log10(df["flare_index"])

SETS={
"Phi_max only":["Phi_max_1e21_Mx"],
"R_av only":["R_av_1e20_Mx_per_h"],
"Phi_max + R_av":["Phi_max_1e21_Mx","R_av_1e20_Mx_per_h"],
"Phi + R_av + alpha":["Phi_max_1e21_Mx","R_av_1e20_Mx_per_h","alpha_sat","alpha_mean"],
"All features":["Phi_max_1e21_Mx","R_av_1e20_Mx_per_h","alpha_sat","alpha_mean","alpha_sat_std","alpha_mean_std"]
}
cv=KFold(n_splits=5,shuffle=True,random_state=42)
rows=[]
for name,features in SETS.items():
    d=df[features+["log_flare_index"]].replace([np.inf,-np.inf],np.nan)
    d=d.dropna(subset=["log_flare_index"])
    X,y=d[features],d["log_flare_index"]
    model=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("model",XGBRegressor(n_estimators=500,max_depth=2,learning_rate=0.03,
                              min_child_weight=5,subsample=0.8,colsample_bytree=0.8,
                              reg_alpha=0.5,reg_lambda=2.0,
                              objective="reg:squarederror",random_state=42,n_jobs=-1))
    ])
    s=cross_validate(model,X,y,cv=cv,scoring={"R2":"r2","MAE":"neg_mean_absolute_error","RMSE":"neg_root_mean_squared_error"},n_jobs=-1)
    rows.append({"Model":name,"N_features":len(features),
                 "CV_R2":s["test_R2"].mean(),"CV_R2_std":s["test_R2"].std(),
                 "CV_MAE":-s["test_MAE"].mean(),"CV_RMSE":-s["test_RMSE"].mean()})
out=pd.DataFrame(rows).sort_values("CV_R2",ascending=False)
print(out.to_string(index=False))
out.to_csv("results/regression_results_recomputed.csv",index=False)
