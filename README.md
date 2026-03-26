# Air Quality Prediction and Respiratory Health Impact Modeling in California (2020–2024)

## Overview

This project analyzes air quality trends across California and models their relationship with respiratory health outcomes. By integrating EPA air quality data with California public health records, we develop predictive models to forecast AQI and assess its impact on asthma-related hospitalizations and emergency department visits.

**Team:** Alli McKernan, Paola Rodriguez, Vinh Dao

---

## Objectives

- Forecast future population-weighted statewide AQI using time-series models
- Quantify the lagged impact of air pollution on asthma-related health outcomes
- Compare statistical and machine learning approaches
- Provide insights for public health and policy decision-making

---

## Data Sources

| Source | Description |
|--------|-------------|
| [EPA Air Quality System (AQS)](https://www.epa.gov/aqs) | Daily county-level AQI data (2020–2024) |
| [California Dept. of Public Health (CDPH)](https://www.cdph.ca.gov/) | Asthma hospitalization and ED visit rates by county |

> **Note:** Raw EPA AQS files ( for 2020–2024) must be downloaded separately from the EPA AQS website and placed in . All other data files are included in this repository.

---

## Repository Structure

```
Air-Quality-Prediction-and-Respiratory-Health-Impact-Modeling-in-California-2020-2024-/
|
+-- data/
|   +-- aqs/                          # Raw EPA AQS daily CSVs (download separately)
|   +-- asthma_hosp_by_county.csv     # Raw CDPH hospitalization data
|   +-- asthma_ed_by_county.csv       # Raw CDPH ED visit data
|   +-- asthma_hosp_clean.csv         # Cleaned hospitalization data (generated)
|   +-- asthma_ed_clean.csv           # Cleaned ED visit data (generated)
|   +-- weekly_statewide_aqi.csv      # Population-weighted weekly AQI (generated)
|   +-- weekly_county_aqi.csv         # Weekly AQI by county (generated)
|   +-- annual_county_aqi.csv         # Annual mean AQI by county (generated)
|   +-- joint_aqi_health_county.csv   # Merged AQI + health dataset (generated)
|   +-- aqi_forecast_results.csv      # AQI model forecasts (generated)
|   +-- aqi_model_metrics.csv         # AQI model evaluation metrics (generated)
|   +-- health_model_metrics.csv      # Health model evaluation metrics (generated)
|   +-- health_model_predictions.csv  # Health model predictions (generated)
|   +-- data_dictionary.md            # Variable descriptions
|
+-- notebooks/
|   +-- 01_eda.ipynb                  # Data loading, cleaning, EDA
|   +-- 02_modeling.ipynb             # Time-series forecasting + health impact models
|
+-- src/
|   +-- data_preprocessing.py         # Load, filter, and aggregate AQS/CDPH data
|   +-- feature_engineering.py        # Lag features and county encoding
|   +-- modeling.py                   # ARIMA, SARIMA, RF, and regression models
|   +-- evaluation.py                 # MAE, RMSE, R2 metric helpers
|   +-- visualization.py              # Reusable plotting functions
|
+-- app/
|   +-- app.py                        # Streamlit interactive dashboard
|
+-- images/
|   +-- daily_mean_aqi.png
|   +-- forecast_model_comparison.png
|   +-- feature_importance.png
|
+-- requirements.txt
+-- README.md
```

---

## How to Run

### Prerequisites
- Python 3.12+
- EPA AQS data files in  (see Data Sources above)

### 1. Clone the repository
```bash
git clone https://github.com/alligmckernan/Air-Quality-Prediction-and-Respiratory-Health-Impact-Modeling-in-California-2020-2024-.git
cd Air-Quality-Prediction-and-Respiratory-Health-Impact-Modeling-in-California-2020-2024-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebooks (in order)
```bash
# Step 1 — EDA and data preprocessing (generates all files in data/)
jupyter notebook notebooks/01_eda.ipynb

# Step 2 — Modeling (requires outputs from step 1)
jupyter notebook notebooks/02_modeling.ipynb
```

### 4. Launch the interactive dashboard
```bash
streamlit run app/app.py
```

---

## Modeling Approach

### Model 1 — AQI Forecast Model
Forecasts future population-weighted statewide AQI using weekly time-series data.

| Model | MAE | RMSE |
|-------|-----|------|
| ARIMA(1,1,1) | 17.74 | 21.69 |
| SARIMA(1,1,1)(1,1,1,52) | 12.20 | 16.05 |
| Random Forest (lag features) | 12.75 | 16.05 |

### Model 2 — Lagged Respiratory Health Impact Model
Predicts county-level asthma hospitalization and ED visit rates from lagged AQI exposure.

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | 0.72 | 0.90 | 0.02 |
| Random Forest | 0.78 | 1.01 | -0.22 |
| Gradient Boosting | 0.84 | 1.09 | -0.42 |

---

## Visualizations

**Daily Mean AQI (2020–2024)**
![Daily Mean AQI (2020–2024)](images/daily_mean_aqi.png)

**AQI Forecast Model Comparison**
![AQI Forecast Model Comparison](images/forecast_model_comparison.png)

**Random Forest Feature Importance**
![Random Forest Feature Importance](images/feature_importance.png)

---

## Data Dictionary

See [](data/data_dictionary.md) for descriptions of all variables and data sources.

---

## Future Work

- Incorporate wildfire smoke and weather variables
- Expand health outcome targets (COPD, cardiovascular)
- Improve county-level spatial modeling
- Deploy dashboard publicly via Streamlit Cloud
