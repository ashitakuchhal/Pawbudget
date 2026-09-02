# Pet Project

A Flask web app that predicts pet insurance risk, medical costs, and claims probability using trained ML models.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Files

- `app.py` — Flask application / API
- `train_model.py`, `train_claims_models.py`, `generate_data.py` — model training scripts
- `*.pkl` — trained models (tracked via Git LFS: `cost_model`, `risk_model`, `claims_probability_model`, `medical_cost_model`, plus `encoders.pkl`)
- `*.csv` — training/reference datasets
- `templates/` — HTML templates for the web UI

## Model size

`claims_probability_model.pkl` and `medical_cost_model.pkl` were originally trained as RandomForests with 150 unlimited-depth trees, producing ~300MB and ~250MB files. Those settings were overfitting (test AUC 0.677, cost R² of -0.02 — worse than predicting the mean). They've been retrained with 60 trees / max depth 10, which improved both metrics (AUC 0.72, R² 0.06) and shrank the combined file size to ~1MB. Retrain with `train_claims_models.py`.
