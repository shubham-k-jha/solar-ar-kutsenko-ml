# Methodology notes

## Benchmark target
The 2021 continuous regression benchmark uses log10(flare_index) on positive flare-index observations.

## Validation
The controlled regression ablation uses the same XGBoost configuration and 5-fold shuffled KFold across feature sets. The classification benchmark uses StratifiedKFold.

## Interpretation
SHAP and full-data feature importance are interpretation tools. Out-of-fold permutation importance and CV scores are used to assess independent/generalizing predictive contribution.

## Leakage warning
These catalogue benchmarks are retrospective. The future SHARP forecasting study must not use Phi_max or any quantity measured after the forecast cutoff.

## Forecast validation
For the time-series extension, groups must be HARP/AR identifiers. Randomly splitting individual timestamps would leak AR-specific information across train and test sets.
