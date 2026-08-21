# ☀️ Physics-Informed Machine Learning for Solar Active Regions

<p align="center">

**Solar Physics • Statistical Learning • Machine Learning • Explainable AI • Flare Productivity**

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189FDD)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-8A2BE2)
![Solar Physics](https://img.shields.io/badge/Domain-Solar%20Physics-FFB000)
![Status](https://img.shields.io/badge/Status-Active%20Research-2EA44F)

</p>

> A physics-informed machine-learning study of **emerging solar active regions, magnetic-flux emergence, and flare productivity**, using published Kutsenko catalogues as a benchmark and building toward time-resolved HMI/SHARP forecasting.

---

## 🔭 Project Overview

Solar active regions evolve through magnetic-flux emergence, polarity separation, and magnetic-field restructuring. Some regions subsequently produce powerful solar flares and CMEs, while others remain relatively quiet.

This project investigates:

> **How much can measurable properties of emerging active regions explain magnetic-flux emergence and flare productivity, and which variables provide genuinely useful predictive information?**

The project combines solar physics with a complete data-science workflow:

```text
Published Solar-Physics Catalogues
              │
              ▼
       Data Quality Audit
              │
              ▼
     Statistical Relationships
              │
       ┌──────┴──────┐
       ▼             ▼
Emergence-Rate    Flare-Productivity
    Modelling         Modelling
       │             │
       └──────┬──────┘
              ▼
      Feature Ablation
              │
              ▼
     Explainable Machine
          Learning
       ┌──────┼──────┐
       ▼      ▼      ▼
     SHAP   Perm.   Interactions
            Importance
              │
              ▼
      Physical Interpretation
              │
              ▼
     HMI/SHARP Time-Series
          Forecasting
```

---

# 🎯 Research Objectives

### 1. Emerging active regions

Investigate how maximum magnetic flux, emergence rate, emergence duration, polarity separation, and other properties are related.

### 2. Flare productivity

Determine whether magnetic properties can distinguish flare-productive and non-productive active regions.

### 3. Continuous prediction

Predict the magnitude of flare productivity rather than treating the problem only as binary classification.

### 4. Explainable ML

Determine which features the models actually use and whether apparently important variables contain independent information.

### 5. Future forecasting

Extend the retrospective catalogue analysis into a genuinely time-aware forecasting system using HMI/SHARP measurements.

---

# 📚 Scientific Foundation

The project uses published studies by Kutsenko and collaborators as its scientific foundation.

| Study | Role |
|---|---|
| **Kutsenko et al. 2019** | Emerging active-region magnetic-flux and emergence-rate scaling |
| **Kutsenko et al. 2021** | Early-stage flare-productivity relationships |
| **Kutsenko, Abramenko & Plotnikov 2024** | Statistical context for emerging regions and strong flares |
| **ApJ 806, 245** | Energy/helicity-flux methodology used in the SHARP extension |

### References

- **Kutsenko et al. (2019)** — [DOI: 10.1093/mnras/stz308](https://doi.org/10.1093/mnras/stz308)
- **Kutsenko et al. (2021)** — [DOI: 10.1093/mnras/staa3548](https://doi.org/10.1093/mnras/staa3548)
- **Kutsenko, Abramenko & Plotnikov (2024)** — [DOI: 10.1088/1674-4527/ad2e4d](https://doi.org/10.1088/1674-4527/ad2e4d)
- **Energy/helicity-flux methodology** — [DOI: 10.1088/0004-637X/806/2/245](https://doi.org/10.1088/0004-637X/806/2/245)

---

# 🗂️ Dataset Overview

## Kutsenko 2019

| Quantity | Value |
|---|---:|
| Catalogue rows | **423** |
| Unique HARPs | **406** |
| Kappa | **0.48114** |
| Polarity-separation exponent | **0.36159** |

Important variables include:

- `Phi_max` — maximum unsigned magnetic flux
- `R_av` — average magnetic-flux emergence rate
- emergence duration
- polarity separation
- tilt
- latitude
- longitude
- magnetic-field parameters

---

## Kutsenko 2021

| Quantity | Value |
|---|---:|
| Catalogue rows | **243** |
| Unique HARPs | **235** |
| Flare-productive regions | **34** |
| Non-productive regions | **209** |
| Mean flare index | **0.95973** |
| Maximum flare index | **41.4636** |

---

# 📈 Kutsenko 2019 — Emergence-Rate Machine Learning

The first ML experiment predicts average flux-emergence rate from active-region properties.

## Model benchmark

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Kutsenko power law | 0.599 | 0.575 | 0.787 |
| Linear Regression | 0.827 | 0.411 | 0.516 |
| Random Forest | 0.726 | 0.487 | 0.650 |
| **Gradient Boosting** | **0.924** | **0.182** | **0.341** |

### 5-Fold Cross-Validation

Gradient Boosting achieved:

- **CV R² = 0.9169**
- R² standard deviation = **0.0214**
- MAE = **0.1896**
- RMSE = **0.3734**

This demonstrates that the catalogue's emergence-rate relationship is highly predictable.

---

# 🧪 Critical Finding — Emergence Duration

Feature ablation produced:

| Feature Set | CV R² |
|---|---:|
| Phi only | 0.473 |
| Phi + separation | 0.465 |
| **Phi + separation + duration** | **0.932** |
| All features | 0.917 |

At first glance, this suggests a dramatic independent contribution from emergence duration.

However, the notebook performs a crucial diagnostic:

\[
\Phi_{\max} \sim R_{av}\times t_{\mathrm{emergence}}
\]

with:

\[
\boxed{\rho_{\mathrm{Spearman}}=0.999982}
\]

### Interpretation

The ML model is largely recovering the underlying relationship:

\[
\boxed{\Phi_{\max}\approx R_{av}\times t_{\mathrm{emergence}}}
\]

rather than demonstrating that emergence duration is an independent causal predictor.

This is an important example of why **feature importance must be interpreted together with physical relationships and feature dependence**.

---

# 🔥 Kutsenko 2021 — Flare Productivity

The second part investigates whether emerging-region properties can identify flare-productive active regions.

## Statistical relationships

The notebook finds:

| Relationship | Result |
|---|---:|
| Pearson correlation: `R_av` vs flare index | **0.4742** |
| Spearman correlation: `R_av` vs flare index | **0.3805** |

Productive regions show higher emergence rates:

| Statistic | Productive | Non-productive |
|---|---:|---:|
| Median R_av | **3.056** | 1.778 |
| Mean R_av | **3.514** | 2.063 |

---

# 🎯 Binary Classification

The R_av-only baseline gives:

- ROC-AUC = **0.7618**
- PR-AUC = **0.4154**
- Brier score = **0.1939**

Controlled feature-ablation results:

| Features | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| 🥇 **Phi only** | **0.8242** | **0.5977** | 0.1500 |
| All features | 0.8051 | 0.5877 | **0.1496** |
| Phi + R_av | 0.8175 | 0.5823 | 0.1512 |
| R_av + alpha | 0.7872 | 0.4482 | 0.1842 |
| R_av + alpha uncertainty | 0.7706 | 0.4452 | 0.1851 |
| R_av only | 0.7618 | 0.4154 | 0.1939 |

### Key result

\[
\boxed{\mathrm{ROC\!-\!AUC}=0.8242}
\]

\[
\boxed{\mathrm{PR\!-\!AUC}=0.5977}
\]

for the Phi-only benchmark.

This indicates that maximum unsigned magnetic flux contains substantial information about flare productivity in this retrospective catalogue.

---

# 📊 Continuous Flare-Productivity Prediction

The continuous target is:

\[
y=\log_{10}(\mathrm{flare\ index})
\]

## Model comparison

| Model | CV R² |
|---|---:|
| Phi_max only | 0.176 |
| Multivariate Linear | 0.317 |
| Random Forest | 0.363 |
| **XGBoost** | **0.405** |

The best general nonlinear model reaches:

\[
\boxed{CV-R^2=0.405}
\]

---

# 🧬 Controlled XGBoost Feature Ablation

The same XGBoost model and validation procedure were used while changing only the feature set.

| Features | CV R² | R² Std | MAE | RMSE |
|---|---:|---:|---:|---:|
| 🥇 **Phi_max only** | **0.5435** | 0.0623 | **0.1028** | **0.1771** |
| Phi_max + R_av | 0.4418 | 0.1433 | 0.1134 | 0.1915 |
| All features | 0.4051 | 0.2389 | 0.1165 | 0.1913 |
| Phi + R_av + alpha | 0.3999 | 0.2324 | 0.1140 | 0.1926 |
| R_av only | 0.0574 | 0.1834 | 0.1514 | 0.2508 |

### Repeated 5-Fold Cross-Validation

| Model | Mean R² | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| **Phi_max only** | **0.4993** | 0.2315 | 0.5538 | -0.9691 | 0.8166 |
| Phi_max + R_av | 0.4009 | 0.3136 | 0.4740 | -1.4673 | 0.7835 |
| All features | 0.3843 | 0.2953 | 0.4669 | -1.2055 | 0.7828 |

The wide repeated-CV range highlights the uncertainty associated with the relatively small flare-productive sample.

---

# 🔍 Explainable Machine Learning

## XGBoost Feature Importance

| Feature | Importance |
|---|---:|
| 🥇 **Phi_max** | **0.4716** |
| R_av | 0.2408 |
| alpha_sat | 0.1275 |
| alpha_mean | 0.0684 |
| alpha_mean_std | 0.0473 |
| alpha_sat_std | 0.0443 |

## SHAP Importance

| Feature | Mean \|SHAP\| |
|---|---:|
| 🥇 **Phi_max** | **0.1059** |
| alpha_sat | 0.0290 |
| alpha_mean | 0.0256 |
| R_av | 0.0226 |
| alpha_mean_std | 0.0142 |
| alpha_sat_std | 0.0064 |

## Out-of-Fold Permutation Importance

| Feature | Mean Importance |
|---|---:|
| 🥇 **Phi_max** | **0.7607** |
| alpha_mean | 0.0298 |
| alpha_sat | 0.0270 |
| alpha_mean_std | 0.0058 |
| R_av | -0.0116 |
| alpha_sat_std | -0.0161 |

The agreement between these interpretation methods strengthens the conclusion that **Phi_max is the dominant aggregate signal** in the current catalogue analysis.

---

# 🧩 SHAP Interaction Analysis

Strongest interaction strengths:

| Feature Pair | Mean Absolute Interaction |
|---|---:|
| R_av × alpha_sat | 0.00601 |
| Phi_max × alpha_sat | 0.00595 |
| Phi_max × alpha_mean | 0.00440 |
| Phi_max × R_av | 0.00337 |
| R_av × alpha_mean | 0.00162 |

These are **model interactions**, not evidence of causal physical interactions.

---

# 📸 Visual Results

The principal figures below are extracted directly from the executed Kaggle notebook.

### Emergence and Scaling

![Kutsenko analysis](figures/fig_01_cell_05_plot.png)

![Kutsenko analysis](figures/fig_02_cell_07_plot.png)

### Flare Productivity

![Flare productivity](figures/fig_03_cell_12_plot.png)

![Flare productivity](figures/fig_04_cell_15_plot.png)

### Machine Learning

![Machine learning](figures/fig_05_cell_18_plot.png)

![Machine learning](figures/fig_06_cell_20_plot.png)

### Explainable AI

![SHAP analysis](figures/fig_07_cell_24_plot.png)

![SHAP analysis](figures/fig_08_cell_25_plot.png)

![Feature analysis](figures/fig_09_cell_27_plot.png)

![Model analysis](figures/fig_10_cell_29_plot.png)

> The complete set of notebook-generated figures is available in [`figures/`](figures/).

---

# 🧠 Key Scientific Findings

### 1. Magnetic flux is the dominant aggregate predictor

Across classification, XGBoost importance, SHAP and permutation importance, `Phi_max` consistently provides the strongest predictive signal.

### 2. Emergence rate is useful but partly redundant

`Phi_max` and `R_av` have a Spearman correlation of approximately:

\[
\rho\approx0.672
\]

R_av provides useful information for binary discrimination, but its incremental contribution becomes weak in the controlled continuous regression once Phi_max is included.

### 3. Alpha contains model-visible nonlinear information

Alpha-related parameters appear in XGBoost importance and SHAP interactions.

However, adding them **does not improve the controlled continuous-regression benchmark**.

### 4. Emergence duration requires careful interpretation

The spectacular improvement from adding emergence duration is largely explained by:

\[
\Phi_{\max}\approx R_{av}\times t_{\mathrm{emergence}}
\]

Therefore, duration should not be presented as an independent causal discovery.

### 5. Classification and regression answer different questions

R_av achieves useful binary discrimination:

\[
ROC\text{-}AUC\approx0.762
\]

but performs poorly as a standalone continuous flare-productivity predictor:

\[
CV-R^2\approx0.057
\]

This is why both classification and regression are evaluated.

---

# ⚠️ Scientific Limitations

The current catalogue study is **retrospective**, not an operational forecasting system.

### Aggregate features

`Phi_max` summarizes the evolution of an active region and may not be available at the time an operational forecast is required.

### Sample size

The 2021 catalogue contains only **34 flare-productive regions**, making uncertainty and class imbalance important.

### Correlated variables

Magnetic flux, emergence rate and emergence duration are physically and statistically related.

### Validation

Random cross-validation can overestimate generalization when observations from related active regions appear across training and testing.

### Causality

SHAP, permutation importance and feature importance describe model behavior. They **do not establish causal physical relationships**.

---

# 🛡️ Leakage-Aware Forecasting Design

The next stage changes the problem from:

```text
Completed active-region evolution
             ↓
      Predict productivity
```

to:

```text
Information available at time t
             ↓
       Forecast cutoff
             ↓
       Future time window
             ↓
     Flare / CME occurrence
```

For every prediction snapshot:

\[
t_{\mathrm{feature}}\leq t_{\mathrm{cutoff}}
\]

and the target event must occur strictly after the cutoff.

## Planned forecast horizons

- 6 hours
- 12 hours
- 24 hours
- 48 hours

## Planned SHARP features

The emerging-region extension will use time-dependent quantities such as:

- magnetic flux `Phi(t)`
- energy flux `ΔE(t)`
- helicity flux `ΔH(t)`
- normalized helicity `ΔH / Phi²`
- temporal derivatives
- rolling-window statistics
- variability
- acceleration/change rates
- pre-event evolution

The eventual dataset will use **HARP/active-region grouped validation** so measurements from the same active region cannot leak into both training and testing.

---

# 🧪 Reproducibility

The **executed Kaggle notebook is the canonical computational record**:

[`notebooks/flare-ml-kutsenko-2019-2021.ipynb`](notebooks/flare-ml-kutsenko-2019-2021.ipynb)

Standalone analysis scripts are provided for organization and reproducibility:

```text
scripts/
├── 01_kutsenko2019_scaling.py
├── 02_kutsenko2021_classification.py
├── 03_kutsenko2021_regression.py
└── 04_model_interpretation.py
```

## Kaggle / Local Portability

The notebook automatically detects the Kaggle dataset mount:

```python
from pathlib import Path

KAGGLE_KUTSENKO = Path("/kaggle/input/datasets/sj3267/kutsenko")
LOCAL_DATA = Path("../data")

DATA_DIR = (
    KAGGLE_KUTSENKO
    if KAGGLE_KUTSENKO.exists()
    else LOCAL_DATA
)
```

This allows the analysis to run in Kaggle while remaining portable to a cloned GitHub repository.

---

# 📁 Repository Structure

```text
solar-ar-kutsenko-ml/
│
├── README.md
├── METHODOLOGY.md
├── PROJECT_ROADMAP.md
├── LICENSE
├── requirements.txt
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── flare-ml-kutsenko-2019-2021.ipynb
│
├── scripts/
│   ├── 01_kutsenko2019_scaling.py
│   ├── 02_kutsenko2021_classification.py
│   ├── 03_kutsenko2021_regression.py
│   └── 04_model_interpretation.py
│
├── src/
│   └── utils.py
│
├── results/
│   ├── classification_results.csv
│   ├── regression_results.csv
│   ├── repeated_cv.csv
│   ├── shap_importance.csv
│   ├── shap_interactions.csv
│   ├── permutation_importance.csv
│   └── KAGGLE_NOTEBOOK_EXACT_RESULTS.md
│
└── figures/
    ├── notebook-generated figures
    └── README.md
```

---

# 🚀 Research Roadmap

### ✅ Completed

- [x] Kutsenko 2019 catalogue analysis
- [x] Emergence-rate scaling analysis
- [x] Linear Regression
- [x] Random Forest
- [x] Gradient Boosting
- [x] Kutsenko 2021 flare classification
- [x] Continuous flare-productivity regression
- [x] XGBoost benchmarking
- [x] Controlled feature ablation
- [x] Repeated cross-validation
- [x] SHAP analysis
- [x] Permutation importance
- [x] SHAP interaction analysis
- [x] Scientific interpretation
- [x] Leakage-aware forecasting design

### 🔄 Next Research Stage

- [ ] Build an emerging-active-region catalogue from HMI/SHARP time series
- [ ] Integrate magnetic-energy and helicity-flux measurements
- [ ] Construct time-resolved features
- [ ] Define strict prediction cutoffs
- [ ] Construct future flare/CME labels
- [ ] Use HARP-grouped train/test splitting
- [ ] Evaluate 6/12/24/48-hour forecasts
- [ ] Probability calibration
- [ ] Uncertainty estimation
- [ ] Cross-cycle validation

---

# ⭐ Project Takeaway

The central result of this project is not simply that one machine-learning model achieved a high score.

The important result is the **combination of statistical modelling, controlled ML experiments and physical interpretation**:

\[
\boxed{
\text{Physical hypothesis}
\rightarrow
\text{Statistical analysis}
\rightarrow
\text{ML}
\rightarrow
\text{Feature ablation}
\rightarrow
\text{Explainability}
\rightarrow
\text{Physical interpretation}
}
\]

The current catalogue analysis establishes the benchmark.

The next stage is to move from retrospective quantities to **time-resolved HMI/SHARP measurements**, where the model must predict future solar activity using only information available before the forecast time.
