import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================
# Load & Filter AQS Data
# =============================================================

def load_aqs_data(aqs_dir, years):
    """
    Load and concatenate EPA AQS daily AQI CSV files for the given years.
    Skips missing files and warns the user.
    Returns a combined DataFrame of all loaded years.
    """
    frames = []
    for yr in years:
        fp = Path(aqs_dir) / f"daily_aqi_by_county_{yr}.csv"
        if fp.exists():
            df_yr = pd.read_csv(fp)
            frames.append(df_yr)
            print(f"{yr}: {len(df_yr):,} rows loaded")
        else:
            print(f"WARNING: {fp} not found — skipping")
    return pd.concat(frames, ignore_index=True)


def filter_california(df):
    """
    Filter a raw AQS DataFrame to California rows only.
    Adds Date (datetime), Year, and Month columns.
    Returns the filtered California-only DataFrame.
    """
    df = df[df["State Name"] == "California"].copy()
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df


# =============================================================
# Population-Weighted AQI
# =============================================================

def add_population_weights(df, county_col):
    """
    Merge California county population estimates onto the AQS DataFrame.
    Population values are 2020 Census estimates for all 58 CA counties.
    Returns the DataFrame with a Population column added (NaN for unmapped counties).
    """
    ca_county_pop = {
        "Los Angeles": 10014009, "San Diego": 3298634, "Orange": 3186989,
        "Riverside": 2418185, "San Bernardino": 2181654, "Santa Clara": 1936259,
        "Alameda": 1682353, "Sacramento": 1585055, "Contra Costa": 1153526,
        "Fresno": 1008654, "Kern": 909235, "San Francisco": 873965,
        "Ventura": 843843, "San Mateo": 764442, "San Joaquin": 779233,
        "Stanislaus": 550660, "Sonoma": 488863, "Tulare": 473117,
        "Solano": 447643, "Monterey": 434061, "Santa Barbara": 446475,
        "Placer": 398329, "San Luis Obispo": 282424, "Marin": 258826,
        "Shasta": 182155, "Merced": 281202, "Butte": 211632,
        "Yolo": 220500, "El Dorado": 192843, "Imperial": 179702,
        "Kings": 152940, "Madera": 157327, "Napa": 137744,
        "Humboldt": 136310, "Nevada": 103487, "Sutter": 99633,
        "Mendocino": 91305, "Yuba": 83421, "Lake": 68766,
        "Tehama": 65084, "San Benito": 62808, "Tuolumne": 55810,
        "Calaveras": 46221, "Siskiyou": 43724, "Amador": 41472,
        "Lassen": 33159, "Del Norte": 28650, "Glenn": 28917,
        "Colusa": 21547, "Plumas": 20007, "Inyo": 18970,
        "Mariposa": 17203, "Trinity": 12285, "Mono": 14444,
        "Modoc": 8661, "Sierra": 3236, "Alpine": 1204,
    }
    pop_df = pd.DataFrame(list(ca_county_pop.items()), columns=[county_col, "Population"])
    return df.merge(pop_df, on=county_col, how="left")


def compute_weekly_state_aqi(df_aqs_pop):
    """
    Compute population-weighted weekly statewide AQI from daily county data.
    Each week is anchored to its Sunday end date (WeekEnd).
    Returns a sorted DataFrame with columns: WeekEnd, PopWeighted_AQI.
    """
    df = df_aqs_pop.copy()
    df["WeekEnd"] = df["Date"] + pd.to_timedelta(6 - df["Date"].dt.dayofweek, unit="d")
    weekly = (
        df.groupby("WeekEnd")
        .apply(lambda g: np.average(g["AQI"], weights=g["Population"]))
        .reset_index()
    )
    weekly.columns = ["WeekEnd", "PopWeighted_AQI"]
    return weekly.sort_values("WeekEnd")


def compute_annual_county_aqi(df_aqs_pop, county_col):
    """
    Compute annual mean AQI by county from daily population-weighted data.
    COUNTY column is uppercased to match CDPH health data conventions.
    Returns a DataFrame with columns: COUNTY, Year, Mean_AQI.
    """
    annual = (
        df_aqs_pop.groupby([county_col, "Year"])["AQI"]
        .mean()
        .reset_index()
        .rename(columns={"AQI": "Mean_AQI", county_col: "COUNTY"})
    )
    annual["COUNTY"] = annual["COUNTY"].str.upper()
    return annual


# =============================================================
# Health Data Filtering
# =============================================================

def filter_main(df, count_col):
    """
    Filter CDPH health data to the main analysis stratum:
      - Total population / All ages only (removes age/race sub-strata)
      - Years 2020-2023
      - County-level rows only (excludes the statewide California row)
    Cleans the count column by removing commas and converting to numeric.
    Returns the filtered and cleaned DataFrame.
    """
    df = df.copy()
    df = df[
        (df["STRATA"] == "Total population")
        & (df["AGE GROUP"] == "All ages")
        & (df["YEAR"] >= 2020)
        & (df["YEAR"] <= 2023)
        & (df["COUNTY"] != "California")
    ].copy()
    df[count_col] = (
        df[count_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace({"nan": "", "": "0"})
    )
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
    return df
