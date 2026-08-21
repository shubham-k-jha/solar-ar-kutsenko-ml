# Data

Raw Kutsenko catalogue CSVs are intentionally not committed by default.

Expected benchmark files:

- `KUTSENKO_2019_423_AR_clean.csv`
- `KUTSENKO_2021_243_AR_model_ready.csv`
- `KUTSENKO_2021_243_AR_supplement_available.csv`
- `KUTSENKO_2019_2021_common_HARP_index.csv`

The future SHARP forecasting phase will additionally use the user's calculated time-series data containing HARP, timestamp, Phi, Delta E, Delta H and Delta H/Phi^2.

Do not commit unpublished/private/large raw datasets without permission.

## Portable notebook paths

The canonical Kaggle notebook now detects the Kaggle dataset mount automatically.
If `/kaggle/input/datasets/sj3267/kutsenko` exists, it is used; otherwise the notebook
falls back to the repository's local `data/` directory.

This keeps the notebook usable both on Kaggle and after cloning the GitHub repository.
