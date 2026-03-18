# Air Quality Prediction and Respiratory Health Impact Modeling in California (2020–2024)

## 📌 Overview

This project analyzes air quality trends across California and models their relationship with respiratory health outcomes. By integrating environmental and public health datasets, we develop predictive models to forecast air quality and assess its impact on asthma-related health outcomes.

---

## 🎯 Objectives

* Forecast future air quality trends using time-series models
* Quantify the impact of air pollution on respiratory health
* Compare statistical and machine learning approaches
* Provide insights for public health and policy decision-making

---

## 📊 Data Sources

* **EPA Air Quality System (AQS)** – Air pollution data
* **California Department of Public Health (CDPH)** – Respiratory health outcomes

**Timeframe:** 2020–2024

---

## 🧪 Modeling Approach

### 🔹 Model 1: AQI Forecast Model

**Goal:** Forecast future population-weighted statewide AQI

* Dataset: `weekly_statewide_aqi.csv`
* Models:

  * ARIMA (baseline time-series model)
  * SARIMA (seasonal model)
  * Random Forest (using time-based features)

**Outputs:**

* AQI forecasts
* Evaluation metrics (RMSE, MAE)

---

### 🔹 Model 2: Lagged Respiratory Health Impact Model

**Goal:** Predict asthma-related health outcomes based on air pollution exposure

* Dataset: `joint_aqi_health_county.csv`
* Models:

  * Linear Regression
  * Random Forest Regression
  * Gradient Boosting Regression

**Outputs:**

* Predicted health outcomes
* Model evaluation (R², RMSE, MAE)

---

## 📈 Key Results

* **SARIMA** performed best for capturing seasonal AQI trends
* Random Forest captured non-linear relationships but showed limited R² improvement
* Air pollution exposure showed measurable relationships with asthma-related outcomes

---

## 📁 Repository Structure

```
Air-Quality-Project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── data_dictionary.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── evaluation.py
│
├── images/
│   ├── eda_trends.png
│   ├── correlation_heatmap.png
│   └── model_performance.png
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## 📊 Visualizations

### Air Quality Trends

![Air Quality Trends](images/eda_trends.png)

### Model Performance

![Model Performance](images/model_performance.png)

---

## ▶️ How to Run

### 1. Clone the repository

```
git clone <your-repo-link>
cd Air-Quality-Project
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run notebooks

* Open `notebooks/01_eda.ipynb`
* Open `notebooks/02_modeling.ipynb`

### 4. Run the application (optional)

```
python app/app.py
```

---

## 🌐 Interactive Application

An interactive script (`app.py`) is included to explore model outputs and predictions.

---

## 📚 Data Dictionary

See `/data/data_dictionary.md` for detailed descriptions of all variables and data sources.

---

## 👥 Team Members

* Alli McKernan
* Paola Rodriguez
* Vinh Dao

---

## 📌 Future Work

* Incorporate additional environmental variables (e.g., weather, wildfire data)
* Improve model performance using advanced machine learning techniques
* Expand analysis beyond California to other regions
