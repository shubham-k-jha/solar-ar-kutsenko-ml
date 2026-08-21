# Kutsenko Kaggle Notebook — Exact Analysis Record

This file records results directly from the **uploaded executed Kaggle notebook**:
`flare-ml-kutsenko-2019-2021.ipynb`.

The notebook is the canonical source for the project. The numbers below are not reconstructed from memory.

## Dataset audit

### Kutsenko 2019
- Rows: 423
- Unique HARPs: 406
- Kappa for `R_av ~ Phi_max^kappa`: 0.4811376714
- Polarity-separation exponent: 0.3615895588

### Kutsenko 2021
- Rows: 243
- Unique HARPs: 235
- Flare-productive (`FI >= 1`): 34
- Non-productive: 209
- Flare-index mean: 0.959730
- Flare-index maximum: 41.4636

## 2019 emergence-rate modelling

The notebook compares a published/scaling-law baseline with linear regression, Random Forest and Gradient Boosting.

### Single split benchmark

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Kutsenko power law | 0.574550 | 0.786792 | 0.598724 |
| Linear regression | 0.411222 | 0.516446 | 0.827109 |
| Random Forest | 0.486517 | 0.650323 | 0.725854 |
| Gradient Boosting | 0.182277 | 0.341316 | **0.924484** |

### 5-fold CV

| Model | R2 mean | R2 std | MAE mean | RMSE mean |
|---|---:|---:|---:|---:|
| Gradient Boosting | **0.916882** | 0.021417 | 0.189561 | 0.373385 |
| Linear regression | 0.793143 | 0.039158 | 0.429678 | 0.587991 |
| Random Forest | 0.641663 | 0.056963 | 0.532627 | 0.775471 |

### Repeated 5-fold CV

Gradient Boosting:

- R2 = **0.917381 ± 0.028053**
- MAE = 0.194191 ± 0.034392
- RMSE = 0.368234 ± 0.086682

### 2019 feature importance

Random Forest importance:

| Feature | Importance |
|---|---:|
| Phi_max | 0.442783 |
| Polarity separation | 0.200603 |
| Emergence duration | 0.112375 |
| Emergence longitude | 0.063623 |
| Peak-flux longitude | 0.052100 |
| alpha_av | 0.050421 |
| Latitude | 0.041088 |
| Tilt | 0.037007 |

## Critical 2019 finding: emergence duration

The notebook's feature ablation gives:

| Features | R2 |
|---|---:|
| Phi only | 0.473380 |
| Phi + separation | 0.464837 |
| Phi + separation + duration | **0.931531** |
| All features | 0.916882 |

The notebook then explicitly tests:

`R_av × emergence_duration`

against `Phi_max` and obtains:

**Spearman rho = 0.999982.**

This is a crucial scientific/data-science finding.

It indicates that the extremely strong predictive contribution of emergence duration is closely tied to the relationship:

`Phi_max ~ R_av × emergence_duration`.

Therefore, the 2019 ML result should **not** be presented as discovering an independent causal role for emergence duration. It is largely recovering the underlying flux-emergence bookkeeping/scaling relationship.

This is exactly why the repository distinguishes **predictive performance** from **physical interpretation**.

---

# 2021 flare productivity

The notebook finds:

- Pearson correlation between `R_av` and flare index: **0.4742**, p = 5.0185e-15
- Spearman correlation: **0.3805**, p = 8.6570e-10

Productive vs non-productive regions:

| Statistic | Productive | Non-productive |
|---|---:|---:|
| Median R_av | 3.05615 | 1.77770 |
| Mean R_av | 3.514203 | 2.062833 |

## Classification

The baseline R_av-only model gives:

- ROC-AUC = **0.761751**
- PR-AUC = **0.415394**
- Brier = **0.193874**

The notebook's main feature-ablation model is **balanced Logistic Regression**, not XGBoost.

| Features | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| **Phi only** | **0.824233** | **0.597748** | 0.150029 |
| All features | 0.805094 | 0.587706 | **0.149604** |
| Phi + R_av | 0.817478 | 0.582288 | 0.151245 |
| R_av + alpha | 0.787222 | 0.448225 | 0.184198 |
| R_av + alpha uncertainty | 0.770616 | 0.445228 | 0.185084 |
| R_av only | 0.761751 | 0.415394 | 0.193874 |

The notebook also evaluates separate Random Forest and XGBoost classifiers using all features:

### Random Forest
- ROC-AUC = 0.795947
- PR-AUC = 0.502697
- Brier = 0.105146

### XGBoost
- ROC-AUC = 0.735435
- PR-AUC = 0.459571
- Brier = 0.124038

The **feature-ablation Logistic Regression table** is the appropriate result for the question "which feature set performs best under the controlled linear classification benchmark."

---

# Continuous flare-productivity modelling

The target is:

`log10(flare_index)`

The notebook compares univariate, multivariate linear, Random Forest and XGBoost models.

## Univariate linear CV

| Feature | CV R2 mean | CV R2 std |
|---|---:|---:|
| Phi_max | 0.175753 | 0.227606 |
| R_av | 0.060193 | 0.221928 |
| alpha_mean | -0.024216 | 0.043671 |
| alpha_sat | -0.037605 | 0.139427 |

## Multivariate models

| Model | CV R2 |
|---|---:|
| Phi_max only | 0.175753 |
| Multivariate Linear | 0.317348 |
| Random Forest | 0.362991 |
| XGBoost | **0.405107** |

The XGBoost result:

- R2 = **0.405107 ± 0.238933**
- MAE = 0.116534
- RMSE = 0.191327

---

# Fair XGBoost feature ablation

This is the strongest controlled regression experiment in the notebook because the model, hyperparameters and CV procedure are held fixed while the feature set changes.

| Model | Features | CV R2 | CV R2 std | CV MAE | CV RMSE |
|---|---:|---:|---:|---:|---:|
| **Phi_max only** | 1 | **0.543534** | 0.062342 | **0.102843** | **0.177074** |
| Phi_max + R_av | 2 | 0.441793 | 0.143321 | 0.113394 | 0.191544 |
| All features | 6 | 0.405107 | 0.238933 | 0.116534 | 0.191327 |
| Phi + R_av + alpha | 4 | 0.399921 | 0.232429 | 0.113951 | 0.192623 |
| R_av only | 1 | 0.057356 | 0.183377 | 0.151392 | 0.250825 |

## Repeated CV robustness

The notebook repeats 5-fold CV 20 times:

| Model | Mean R2 | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| **Phi_max only** | **0.499284** | 0.231466 | 0.553795 | -0.969091 | 0.816640 |
| Phi_max + R_av | 0.400925 | 0.313614 | 0.474032 | -1.467333 | 0.783536 |
| All features | 0.384298 | 0.295278 | 0.466921 | -1.205482 | 0.782816 |

This is an important robustness check: the advantage of Phi_max-only is not just a single favorable 5-fold split, although the very wide repeated-CV range shows that the small sample size produces substantial fold-to-fold uncertainty.

---

# Explainable ML

## XGBoost feature importance

| Feature | Importance |
|---|---:|
| Phi_max | 0.471633 |
| R_av | 0.240777 |
| alpha_sat | 0.127491 |
| alpha_mean | 0.068416 |
| alpha_mean_std | 0.047338 |
| alpha_sat_std | 0.044344 |

## SHAP mean absolute contribution

| Feature | Mean abs(SHAP) |
|---|---:|
| Phi_max | 0.105927 |
| alpha_sat | 0.028953 |
| alpha_mean | 0.025565 |
| R_av | 0.022639 |
| alpha_mean_std | 0.014195 |
| alpha_sat_std | 0.006440 |

## Spearman feature correlations

Important correlations include:

- Phi_max vs R_av: **0.672144**
- alpha_sat vs alpha_mean: **0.682734**
- Phi_max vs alpha_sat: -0.237331
- Phi_max vs alpha_mean: -0.245485

This correlation structure helps explain why adding features does not automatically improve generalization.

---

# Out-of-fold permutation importance

| Feature | Mean importance | Std |
|---|---:|---:|
| **Phi_max** | **0.760692** | 0.155417 |
| alpha_mean | 0.029775 | 0.048483 |
| alpha_sat | 0.027016 | 0.031302 |
| alpha_mean_std | 0.005788 | 0.011729 |
| R_av | -0.011643 | 0.100036 |
| alpha_sat_std | -0.016148 | 0.023891 |

Negative permutation importance does not mean a feature has a physically negative effect. It means that, in the held-out permutation experiment, shuffling the feature did not reduce the score and may improve it slightly due to sampling/model interactions.

---

# SHAP interaction strength

Top interactions:

| Feature pair | Mean abs interaction |
|---|---:|
| R_av × alpha_sat | 0.006008 |
| Phi_max × alpha_sat | 0.005945 |
| Phi_max × alpha_mean | 0.004395 |
| Phi_max × R_av | 0.003366 |
| R_av × alpha_mean | 0.001619 |

These are nonlinear model-interaction findings, not causal claims.

---

# Strongest defensible scientific conclusions

## 1. Maximum flux is the dominant aggregate predictor of flare productivity

Across classification, XGBoost feature importance, SHAP and permutation importance, `Phi_max` consistently emerges as the strongest aggregate predictor.

## 2. Emergence rate is scientifically relevant but partly redundant with flux

`Phi_max` and `R_av` have Spearman rho ≈ 0.672.

R_av is useful for binary discrimination, but it adds little or negative incremental value in the controlled continuous regression once Phi_max is included.

## 3. Alpha parameters contain model-visible nonlinear information

The all-feature XGBoost model uses alpha-related quantities and finds interactions involving them.

However, adding these features **does not improve the controlled CV regression benchmark**.

## 4. The 2019 emergence-rate result needs careful interpretation

The enormous improvement when emergence duration is added is explained by the near-exact relationship:

`R_av × emergence_duration ≈ Phi_max`.

This is an excellent scientific-ML lesson: a high-performing model can recover a mathematical/physical identity rather than discover a new independent predictive mechanism.

## 5. Classification and regression are different questions

R_av has ROC-AUC ≈ 0.762 for productive/non-productive classification but only CV R2 ≈ 0.057 as a standalone continuous predictor in the controlled XGBoost regression.

Therefore, discrimination and magnitude prediction should not be conflated.

---

# What this project does NOT claim

It does not demonstrate that:

- Phi_max causes flares;
- SHAP establishes causality;
- aggregate Phi_max is an operational early-warning feature;
- the Kutsenko sample automatically generalizes to every solar cycle;
- the current model is an operational space-weather forecasting system.

The current catalogue analysis is retrospective.

---

# Phase III — SHARP forecasting

The Kaggle notebook ends by discovering the user's SHARP/emerging-AR tabular inputs.

The next stage should transform the existing project into a genuinely predictive time-series problem:

`early SHARP measurements -> future flare/CME event`

Candidate inputs:

- Phi(t)
- Delta E(t)
- Delta H(t)
- Delta H / Phi^2
- temporal derivatives
- rolling-window statistics
- variability
- recent slopes

Forecast cutoffs can be evaluated at:

- 6 h
- 12 h
- 24 h
- 48 h

The forecast dataset must be constructed so that:

`feature timestamp <= forecast cutoff`

and:

`label window > forecast cutoff`.

Training/testing must be grouped by HARP/active region to prevent observations from the same AR leaking into both sets.

---

# CV-ready project statement

**Physics-Informed Machine Learning for Solar Active Regions and Flare Productivity**

- Analysed **423 emerging solar active regions / 406 unique HARPs** using Kutsenko catalogues and reproduced emergence-rate scaling relationships.
- Benchmarked linear, Random Forest and Gradient Boosting models; Gradient Boosting achieved **5-fold CV R2 = 0.917** for emergence-rate prediction.
- Identified a near-exact relationship between `R_av × emergence_duration` and `Phi_max` (**Spearman rho = 0.999982**), preventing over-interpretation of duration as an independent discovery.
- Built leakage-controlled flare-productivity classification models; `Phi_max` achieved **ROC-AUC 0.824 and PR-AUC 0.598**.
- Built XGBoost regression models for log flare productivity; controlled feature ablation showed **Phi_max-only CV R2 = 0.544**, outperforming larger feature sets.
- Applied **SHAP, out-of-fold permutation importance and SHAP interaction analysis** to distinguish predictive signal from redundant/non-generalizing features.
- Designed the next phase as a **HARP-grouped, leakage-free HMI/SHARP time-series forecasting pipeline** for future flare/CME activity.

---

# Repository provenance

The original executed Kaggle notebook is retained in:

`notebooks/flare-ml-kutsenko-2019-2021.ipynb`

This should be treated as the **canonical computational record**.

The standalone scripts are secondary reproducibility/organization artifacts. For exact reproduction of the current project state, run the Kaggle notebook with its original Kaggle dataset inputs.
