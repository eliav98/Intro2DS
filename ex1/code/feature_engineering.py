import numpy as np, pandas as pd

COUNTRY_NORMALIZATION_MAP = {
    "Cape Verde": "Cabo Verde", "Czechia": "Czech Republic (Czechia)", "Cote D'Ivoire": "Côte D'Ivoire",
    "Democratic Republic Of Congo": "Dr Congo", "Reunion": "Réunion", "Curacao": "Curaçao",
    "Micronesia (Country)": "Micronesia", "Palestine": "State Of Palestine",
    "Saint Vincent And The Grenadines": "St. Vincent & Grenadines", "Sao Tome And Principe": "Sao Tome & Principe",
    "United States Virgin Islands": "U.S. Virgin Islands", "East Timor": "Timor-Leste",
}



def apply_manual_country_mapping(df: pd.DataFrame, map_dict: dict) -> pd.DataFrame:
    """
    Apply manual country name remapping to the index of df.
    Saves the mapping dictionary as a CSV for audit.
    """
    df = df.rename(index=map_dict)
    # pd.DataFrame(sorted(map_dict.items()), columns=["Original", "MappedTo"]).to_csv(
    #     OUTPUT_DIR / f"{label}_country_mapping.csv", index=False
    # )
    return df


def merge_dfs(d: pd.DataFrame, g: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    df = d.join(g, how="inner").join(p, how="inner")
    return df


def z(s: pd.Series) -> pd.Series:
    """
    Compute z-score normalization for a given numeric series.
    Resulting values have mean 0 and standard deviation 1.
    """
    return (s - s.mean()) / s.std(ddof=0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on merged country-level data.

    Adds the following columns:
    - TotalGDP: GDP_per_capita_PPP * Population
    - LogGDPperCapita: log10 of GDP per capita
    - LogPopulation: log10 of Population
    - LifeExpectancy_z: z-score normalized LifeExpectancy_Both
    - LogGDPpc_z: z-score normalized LogGDPperCapita
    - LogPop_z: z-score normalized LogPopulation

    Parameters:
        df (pd.DataFrame): The merged dataset with demographics, GDP, and population.

    Returns:
        pd.DataFrame: DataFrame with engineered features added.
    """
    df = df.copy()

    df["TotalGDP"] = df["GDP_per_capita_PPP"] * df["Population"]
    df["LogGDPperCapita"] = np.log10(df["GDP_per_capita_PPP"])
    df["LogPopulation"] = np.log10(df["Population"])

    df["LifeExpectancy_z"] = z(df["LifeExpectancy_Both"])
    df["LogGDPpc_z"] = z(df["LogGDPperCapita"])
    df["LogPop_z"] = z(df["LogPopulation"])

    return df
