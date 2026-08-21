"""Kutsenko 2019 scaling benchmark."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

PATH = Path("data/KUTSENKO_2019_423_AR_clean.csv")
df = pd.read_csv(PATH)

phi = pd.to_numeric(df["Phi_max_1e21_Mx"], errors="coerce")
rav = pd.to_numeric(df["R_av_1e20_Mx_per_h"], errors="coerce")
mask = (phi > 0) & (rav > 0)

print("Rows:", len(df))
print("Unique HARPs:", df["HARP"].nunique())
print("Spearman:", spearmanr(phi[mask], rav[mask]))
print("Pearson:", pearsonr(np.log10(phi[mask]), np.log10(rav[mask])))

coef = np.polyfit(np.log10(phi[mask]), np.log10(rav[mask]), 1)
print("Log-log power-law exponent:", coef[0])

data = df[["Phi_max_1e21_Mx", "R_av_1e20_Mx_per_h"]].replace([np.inf,-np.inf],np.nan).dropna()
X, y = data[["Phi_max_1e21_Mx"]], data["R_av_1e20_Mx_per_h"]
cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Linear regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(n_estimators=500,max_depth=2,learning_rate=0.03,
                            subsample=0.8,colsample_bytree=0.8,
                            objective="reg:squarederror",random_state=42,n_jobs=-1)
}
rows=[]
for name, model in models.items():
    pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",model)])
    s=cross_validate(pipe,X,y,cv=cv,scoring="r2",n_jobs=-1)
    rows.append({"Model":name,"CV_R2":s["test_score"].mean(),"CV_R2_std":s["test_score"].std()})
out=pd.DataFrame(rows).sort_values("CV_R2",ascending=False)
print(out.to_string(index=False))
out.to_csv("results/kutsenko2019_ml_benchmark.csv",index=False)
