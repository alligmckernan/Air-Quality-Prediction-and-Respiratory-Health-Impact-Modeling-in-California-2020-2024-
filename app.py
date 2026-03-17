import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# -----------------------------------
# Page setup
# -----------------------------------
st.set_page_config(
    page_title="California Air Quality & Respiratory Health Explorer",
    layout="wide"
)

DATA_DIR = Path("data")

# -----------------------------------
# Load data
# -----------------------------------
@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None

weekly_state = load_csv("weekly_statewide_aqi.csv")
annual_county = load_csv("annual_county_aqi.csv")
weekly_county = load_csv("weekly_county_aqi.csv")
hosp = load_csv("asthma_hosp_clean.csv")
ed = load_csv("asthma_ed_clean.csv")
joint = load_csv("joint_aqi_health_county.csv")
aqi_metrics = load_csv("aqi_model_metrics.csv")
aqi_forecast = load_csv("aqi_forecast_results.csv")
health_metrics = load_csv("health_model_metrics.csv")
health_preds = load_csv("health_model_predictions.csv")

# -----------------------------------
# Basic cleaning
# -----------------------------------
if weekly_state is not None:
    weekly_state["WeekEnd"] = pd.to_datetime(weekly_state["WeekEnd"])
    weekly_state["Year"] = weekly_state["WeekEnd"].dt.year

if weekly_county is not None:
    weekly_county["WeekEnd"] = pd.to_datetime(weekly_county["WeekEnd"])
    weekly_county["Year"] = weekly_county["WeekEnd"].dt.year

if aqi_forecast is not None:
    aqi_forecast["WeekEnd"] = pd.to_datetime(aqi_forecast["WeekEnd"])

# -----------------------------------
# Sidebar navigation
# -----------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Air Quality Explorer",
        "Health Outcomes Explorer",
        "Air Quality vs Health",
        "Forecast & Model Results",
        "Key Takeaways"
    ]
)

# -----------------------------------
# Overview
# -----------------------------------
if page == "Overview":
    st.title("California Air Quality & Respiratory Health Explorer")

    st.markdown("""
    This interactive tool supports exploration of California air quality trends,
    asthma-related hospitalization and emergency department visit outcomes,
    and forecasting/modeling results from our capstone project.
    """)

    c1, c2, c3 = st.columns(3)

    if annual_county is not None:
        c1.metric("County-Year Air Quality Rows", f"{len(annual_county):,}")

    if joint is not None:
        c2.metric("Merged Air Quality + Health Rows", f"{len(joint):,}")

    if joint is not None:
        c3.metric("Matched Counties", f"{joint['County_join'].nunique():,}")

    st.subheader("Data Sources")
    st.markdown("""
    - EPA / AQS air quality data  
    - California Department of Public Health asthma outcome data  
    - Team-cleaned and merged analysis files  
    """)

    st.subheader("Project Goal")
    st.markdown("""
    Examine how county-level air quality patterns relate to asthma-related health outcomes,
    and summarize model results for air quality forecasting and exploratory health prediction.
    """)

    with st.expander("Key Terms / How to Read This Tool"):
        st.markdown("""
        - **Air Quality Index (AQI):** A score that describes how clean or polluted the air is. Higher values usually mean worse air quality.
        - **Population-weighted air quality:** An air quality measure adjusted to reflect where more people live.
        - **Age-adjusted rate:** A rate adjusted so counties can be compared more fairly even if their age distributions differ.
        - **Correlation:** A number showing how strongly two variables move together. Positive values mean they tend to increase together.
        - **Forecast:** A model-based estimate of future values using past data.
        """)

# -----------------------------------
# Air Quality Explorer
# -----------------------------------
elif page == "Air Quality Explorer":
    st.title("Air Quality Explorer")

    tab1, tab2, tab3 = st.tabs([
        "Statewide Weekly Air Quality",
        "County Annual Air Quality",
        "County Weekly Air Quality"
    ])

    with tab1:
        if weekly_state is None:
            st.warning("weekly_statewide_aqi.csv not found.")
        else:
            year_options = ["All Years"] + sorted(weekly_state["Year"].dropna().unique().tolist())
            selected_year = st.selectbox("Select year", year_options, key="statewide_year")

            if selected_year == "All Years":
                df_plot = weekly_state.copy()
            else:
                df_plot = weekly_state[weekly_state["Year"] == selected_year].copy()

            chart = alt.Chart(df_plot).mark_line().encode(
                x=alt.X("WeekEnd:T", title="Week Ending"),
                y=alt.Y("PopWeighted_AQI:Q", title="Population-Weighted Air Quality Score"),
                tooltip=["WeekEnd:T", "PopWeighted_AQI:Q"]
            ).properties(height=420)

            st.altair_chart(chart, use_container_width=True)

            st.caption("Higher values generally indicate worse air quality.")

    with tab2:
        if annual_county is None:
            st.warning("annual_county_aqi.csv not found.")
        else:
            counties = sorted(annual_county["COUNTY"].dropna().unique())
            county = st.selectbox("Select county", counties)

            df_plot = annual_county[annual_county["COUNTY"] == county].copy()

            chart = alt.Chart(df_plot).mark_bar().encode(
                x=alt.X("Year:O", title="Year"),
                y=alt.Y("Mean_AQI:Q", title="Average Air Quality Score"),
                tooltip=["COUNTY", "Year", "Mean_AQI"]
            ).properties(height=420)

            st.altair_chart(chart, use_container_width=True)

    with tab3:
        if weekly_county is None:
            st.warning("weekly_county_aqi.csv not found.")
        else:
            counties = sorted(weekly_county["County"].dropna().unique())
            county = st.selectbox("Select county for weekly trend", counties, key="weekly_county_select")

            year_options = ["All Years"] + sorted(weekly_county["Year"].dropna().unique().tolist())
            selected_year = st.selectbox("Select year for county weekly trend", year_options, key="county_weekly_year")

            df_plot = weekly_county[weekly_county["County"] == county].copy()
            if selected_year != "All Years":
                df_plot = df_plot[df_plot["Year"] == selected_year]

            chart = alt.Chart(df_plot).mark_line().encode(
                x=alt.X("WeekEnd:T", title="Week Ending"),
                y=alt.Y("Mean_AQI:Q", title="Average Air Quality Score"),
                tooltip=["County", "WeekEnd:T", "Mean_AQI"]
            ).properties(height=420)

            st.altair_chart(chart, use_container_width=True)

# -----------------------------------
# Health Outcomes Explorer
# -----------------------------------
elif page == "Health Outcomes Explorer":
    st.title("Health Outcomes Explorer")

    outcome = st.selectbox(
        "Select health outcome",
        ["Hospitalization Rate", "ED Visit Rate"]
    )

    if outcome == "Hospitalization Rate":
        df = hosp
        value_col = "AGE-ADJUSTED HOSPITALIZATION RATE"
        count_col = "NUMBER OF HOSPITALIZATIONS"
    else:
        df = ed
        value_col = "AGE-ADJUSTED ED VISIT RATE"
        count_col = "NUMBER OF ED VISITS"

    if df is None:
        st.warning("Selected health dataset not found.")
    else:
        counties = sorted(df["COUNTY"].dropna().unique())
        county = st.selectbox("Select county", counties, key="health_county")

        df_plot = df[df["COUNTY"] == county].copy()

        chart = alt.Chart(df_plot).mark_line(point=True).encode(
            x=alt.X("YEAR:O", title="Year"),
            y=alt.Y(f"{value_col}:Q", title=outcome),
            tooltip=["COUNTY", "YEAR", value_col, count_col]
        ).properties(height=420)

        st.altair_chart(chart, use_container_width=True)

        latest = df_plot.sort_values("YEAR").iloc[-1]
        c1, c2 = st.columns(2)
        c1.metric("Latest Year", int(latest["YEAR"]))
        c2.metric("Latest Value", f"{latest[value_col]:.2f}")

# -----------------------------------
# Air Quality vs Health
# -----------------------------------
elif page == "Air Quality vs Health":
    st.title("Air Quality vs Health")

    if joint is None:
        st.warning("joint_aqi_health_county.csv not found.")
    else:
        year_options = sorted(joint["Year"].dropna().unique())
        selected_year = st.selectbox("Select year", year_options)

        outcome_label = st.selectbox(
            "Select outcome",
            [
                "Hospitalization Rate",
                "ED Visit Rate"
            ]
        )

        if outcome_label == "Hospitalization Rate":
            outcome_col = "AGE-ADJUSTED HOSPITALIZATION RATE"
        else:
            outcome_col = "AGE-ADJUSTED ED VISIT RATE"

        df_plot = joint[joint["Year"] == selected_year].copy()
        df_plot = df_plot.dropna(subset=["Mean_AQI", outcome_col])

        points = alt.Chart(df_plot).mark_circle(size=90).encode(
            x=alt.X("Mean_AQI:Q", title="Average Air Quality Score"),
            y=alt.Y(f"{outcome_col}:Q", title=outcome_label),
            tooltip=["County_join", "Year", "Mean_AQI", outcome_col]
        )

        trendline = alt.Chart(df_plot).transform_regression(
            "Mean_AQI", outcome_col
        ).mark_line()

        chart = (points + trendline).properties(height=450)

        st.altair_chart(chart, use_container_width=True)

        if len(df_plot) > 1:
            corr = df_plot[["Mean_AQI", outcome_col]].corr().iloc[0, 1]
            st.metric("Correlation", f"{corr:.3f}")

            if corr >= 0.5:
                interp = "a moderately strong positive relationship"
            elif corr >= 0.2:
                interp = "a weak-to-moderate positive relationship"
            elif corr > 0:
                interp = "a weak positive relationship"
            elif corr == 0:
                interp = "no clear relationship"
            else:
                interp = "a negative relationship"

            st.caption(
                f"For {selected_year}, this suggests {interp} between air quality and the selected health outcome across counties. This shows association, not causation."
            )

# -----------------------------------
# Forecast & Model Results
# -----------------------------------
elif page == "Forecast & Model Results":
    st.title("Forecast & Model Results")

    if aqi_metrics is not None:
        st.subheader("Air Quality Model Comparison")
        st.dataframe(aqi_metrics, use_container_width=True)

    if aqi_forecast is not None:
        st.subheader("Actual vs Predicted Air Quality")

        forecast_plot = aqi_forecast.copy()
        plot_df = forecast_plot.melt(
            id_vars=["WeekEnd"],
            value_vars=["Actual", "ARIMA_Forecast", "SARIMA_Forecast", "RandomForest"],
            var_name="Series",
            value_name="AirQuality"
        )

        plot_df = plot_df.dropna(subset=["AirQuality"])

        chart = alt.Chart(plot_df).mark_line().encode(
            x=alt.X("WeekEnd:T", title="Week Ending"),
            y=alt.Y("AirQuality:Q", title="Air Quality Score"),
            color="Series:N",
            tooltip=["WeekEnd:T", "Series:N", "AirQuality:Q"]
        ).properties(height=450)

        st.altair_chart(chart, use_container_width=True)

    if health_metrics is not None:
        st.subheader("Health Model Metrics")
        st.dataframe(health_metrics, use_container_width=True)

    if health_preds is not None:
        st.subheader("Actual vs Predicted Health Outcome")
        model_choice = st.selectbox(
            "Select health prediction series",
            ["Pred_LR", "Pred_RF", "Pred_GB"]
        )

        df_plot = health_preds.copy()

        chart = alt.Chart(df_plot).mark_circle(size=80).encode(
            x=alt.X("Actual:Q", title="Actual"),
            y=alt.Y(f"{model_choice}:Q", title=f"Predicted ({model_choice})"),
            tooltip=["Mean_AQI", "AQI_lag1", "Actual", model_choice]
        ).properties(height=420)

        st.altair_chart(chart, use_container_width=True)

# -----------------------------------
# Key Takeaways
# -----------------------------------
elif page == "Key Takeaways":
    st.title("Key Takeaways")

    st.markdown("""
    - Air quality varies across California counties and over time.  
    - County-level air quality can be compared with asthma hospitalization and emergency department visit outcomes.  
    - SARIMA performed better than ARIMA in the air quality forecasting results shown here.  
    - The health prediction models had limited performance, suggesting air quality alone is not enough to strongly predict hospitalization outcomes.  
    - This tool is intended for exploration and decision support, not clinical or regulatory use.  
    """)