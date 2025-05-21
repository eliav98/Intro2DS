import numpy as np, pandas as pd

from ex1.code.data_cleaning import clean_demographics, clean_gdp, clean_population
from ex1.code.utils import DEMOGRAPHICS_DATA_FILENAME, GDP_DATA_FILENAME, POP_DATA_FILENAME, OUTPUT_DIR, DATA_DIR


def load(name: str):
    return pd.read_csv(OUTPUT_DIR/ name)


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


def main():
    d = clean_demographics(load(DATA_DIR / DEMOGRAPHICS_DATA_FILENAME))
    g = clean_gdp(load(DATA_DIR / GDP_DATA_FILENAME))
    p = clean_population(load(DATA_DIR / POP_DATA_FILENAME))

    df = d.join(g, how="inner").join(p, how="inner")
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df["TotalGDP"] = df["GDP_per_capita_PPP"] * df["Population"]
    df["LogGDPperCapita"] = np.log10(df["GDP_per_capita_PPP"])
    df["LogPopulation"] = np.log10(df["Population"])
    df["LifeExpectancy_z"] = z(df["LifeExpectancy_Both"])
    df["LogGDPpc_z"] = z(df["LogGDPperCapita"])
    df["LogPop_z"] = z(df["LogPopulation"])
    X = df.sort_index()[["LifeExpectancy_z", "LogGDPpc_z", "LogPop_z"]].to_numpy()
    np.save(OUTPUT_DIR / "X.npy", X)

if __name__ == "__main__":
    main()
