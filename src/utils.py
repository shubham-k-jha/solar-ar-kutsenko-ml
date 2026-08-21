"""Shared utilities for the Kutsenko ML benchmark."""
import numpy as np
import pandas as pd

DEFAULT_XGB_PARAMS = dict(
    n_estimators=500, max_depth=2, learning_rate=0.03,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=2.0,
    objective="reg:squarederror", random_state=42, n_jobs=-1
)

def replace_nonfinite(df):
    return df.replace([np.inf, -np.inf], np.nan)
