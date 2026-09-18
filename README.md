# Sydney Housing Price Prediction and Decision Support System

SIT720 Task 8.1D – Machine Learning Mini Project.

The project predicts residential sale prices in **Blacktown, Parramatta and Mosman** from 105 sold properties, and provides a simple Streamlit app that gives a model estimate for a property.

## Project files

| Path | Description |
|---|---|
| `data/sydney_housing_final.csv` | Dataset: 105 sold properties (35 per suburb). Read-only; its SHA-256 hash is checked by the notebook. |
| `data/property_sources.csv` | Source log for each sale (listing URL, collection method, verification flags). |
| `notebooks/housing_price_prediction.ipynb` | Main notebook: data quality, EDA, feature analysis, models, evaluation, error analysis, Part 5 comparison and model export. |
| `src/features.py` | Feature engineering and custom scikit-learn components shared by the notebook and the app. |
| `models/gradient_boosting_model.joblib` | Final Gradient Boosting Regressor pipeline, trained on all 105 properties (created by the notebook). |
| `app/app.py` | Streamlit decision-support app. |
| `report/report.md` | Written report. |
| `report/llm_human_comparison.csv` | Holdout estimates from Gradient Boosting, ChatGPT and Human. |
| `report/final_metrics.json` | Model performance figures shown in the app (created by the notebook). |
| `report/figures/`, `report/screenshots/` | Notebook figures and app screenshots. |

## Models

A suburb × property-type median baseline, **Ridge Regression**, **Random Forest Regressor** and **Gradient Boosting Regressor**, compared with 5-fold cross-validation. Gradient Boosting was selected for the final prototype using the repeated-cross-validation mean-MAE criterion. However, model rankings varied across metrics, and the model showed substantial overfitting risk (training R² 0.992 vs CV R² 0.513).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the notebook

```bash
source .venv/bin/activate
jupyter lab
```

Open `notebooks/housing_price_prediction.ipynb` and run all cells. It takes a few minutes, uses `SEED = 42`, and recreates the model file, metrics and figures.

## Run the app

```bash
source .venv/bin/activate
python -m streamlit run app/app.py
```

Run this from the project root, then open http://localhost:8501. Choose a suburb and property type, enter the property details (sizes can be marked unknown), and click **Estimate Property Value**. The estimate is a model estimate, not a professional property valuation.
