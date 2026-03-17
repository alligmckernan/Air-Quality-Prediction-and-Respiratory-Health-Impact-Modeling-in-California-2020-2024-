import pandas as pd

files = [
    "data/weekly_statewide_aqi.csv",
    "data/annual_county_aqi.csv",
    "data/weekly_county_aqi.csv",
    "data/asthma_hosp_clean.csv",
    "data/asthma_ed_clean.csv",
    "data/joint_aqi_health_county.csv",
    "data/aqi_model_metrics.csv",
    "data/aqi_forecast_results.csv",
    "data/health_model_metrics.csv",
    "data/health_model_predictions.csv",
]

for f in files:
    try:
        df = pd.read_csv(f)
        print(f"\n{f}")
        print(df.columns.tolist())
        print(df.head(2))
    except Exception as e:
        print(f"\nERROR reading {f}: {e}")