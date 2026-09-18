# Sydney Housing Price Prediction and Decision Support System

SIT720 Task 8.1D – Machine Learning Mini Project

This project develops a machine learning model for estimating residential property prices across **Blacktown, Parramatta and Mosman**. The dataset contains 105 sold properties, with 35 properties collected from each suburb.

The project covers data preparation, exploratory data analysis, feature engineering, regression modelling, model evaluation and deployment through a Streamlit application.

## Repository Structure

- app/app.py – Streamlit application
- data/sydney_housing_final.csv – Main housing dataset
- data/property_sources.csv – Property source records
- notebooks/housing_price_prediction.ipynb – Complete machine learning analysis
- src/features.py – Feature engineering and preprocessing
- models/gradient_boosting_model.joblib – Final trained model
- report/final_metrics.json – Final model metrics
- report/llm_human_comparison.csv – ML, LLM and human comparison
- report/figures/ – Figures generated during the analysis
- report/screenshots/ – Screenshots of the Streamlit application
- tests/test_app_ui.py – Streamlit UI tests
- requirements.txt – Python dependencies

## Analysis

I compared three regression models:

- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

I also used a suburb × property-type median baseline as a reference.

The models were evaluated using 5-fold cross-validation with MAE, RMSE, R² and MdAPE. Since the dataset is relatively small, I also repeated cross-validation across six shuffled splits to check whether the model ranking was stable.

Gradient Boosting achieved the lowest mean MAE across the repeated cross-validation runs and was selected for the final application. However, it also showed evidence of overfitting, which is discussed in more detail in the notebook and report.

The final 10-property holdout results were:

| Metric | Result |
|---|---:|
| MAE | $528,200 |
| RMSE | $1,098,488 |
| R² | 0.034 |
| MdAPE | 12.3% |

The holdout set contains only 10 properties and includes an unusual high-value Mosman property, so these results are interpreted cautiously.

## Setup

Clone the repository:

```bash
git clone https://github.com/sahil296/SIT720-Sydney-Housing-Price-Prediction.git
cd SIT720-Sydney-Housing-Price-Prediction
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```

## Run the Notebook

The complete machine learning analysis is available in notebooks/housing_price_prediction.ipynb.

From the repository root, run:

```bash
python3 -m jupyter lab notebooks/housing_price_prediction.ipynb
```

Once the notebook opens, select **Kernel → Restart Kernel and Run All Cells**.

This runs the complete analysis from the beginning and reproduces the outputs used in the report.

## Run the Streamlit Application

From the repository root, run:

```bash
python3 -m streamlit run app/app.py
```

The application should open automatically in the browser at http://localhost:8501.

The application allows the user to select a suburb and property type and enter information such as bedrooms, bathrooms, parking, land size and building area. It then uses the final Gradient Boosting pipeline to generate an estimated sale price.

The application also displays supporting information including suburb and property-type medians, charts, model performance and known limitations. CSV files can also be uploaded for batch predictions.

The predicted value is provided as a model estimate and should not be treated as a professional property valuation.

## Tests

The Streamlit UI tests can be run from the repository root:

```bash
python3 -m unittest tests/test_app_ui.py -v
```

## Data and Sources

The main dataset is stored in data/sydney_housing_final.csv, while the source information for the collected properties is recorded in data/property_sources.csv.

The dataset contains 105 sold properties:

- Blacktown – 35
- Parramatta – 35
- Mosman – 35

The dataset includes sale price, sale date, property type, bedrooms, bathrooms, car spaces, land size, building area and transport-related location features.

The main limitations are the relatively small sample size, missing building-area information and limited representation of luxury properties and townhouses. Some characteristics that may influence property prices, such as condition, views and renovation quality, are also not consistently available in the dataset.

## GenAI Acknowledgement

Generative AI tools were used as a supporting resource for brainstorming, reviewing explanations and improving the clarity of written material. I wrote and implemented the project code and carried out the machine learning workflow. Any suggestions were reviewed before being incorporated, and I tested and verified the final notebook, application and outputs.