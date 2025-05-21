import numpy as np
import pandas as pd

from utils import OUTPUT_DIR

def standardize_country(name: str) -> str:
    name = name.strip()
    if name.lower().startswith("the "):
        name = name[4:]
    return name.title()

def clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    original = df.copy()

    # numeric conversion
    numeric_cols = [
        "LifeExpectancy_Both",
        "LifeExpectancy_Female",
        "LifeExpectancy_Male",
        "UrbanPopulation_Percentage",
        "UrbanPopulation_Absolute",
        "PopulationDensity"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # remove invalid LifeExpectancyBoth
    mask_valid = df["LifeExpectancy_Both"].between(40, 100)
    cleaned = df[mask_valid].copy()

    # country names
    cleaned = cleaned.reset_index()
    cleaned["Country"] = cleaned["Country"].apply(standardize_country)
    mismatches = pd.DataFrame({
        "Original": original.reset_index()["Country"],
        "Standardized": cleaned["Country"],
    }).query("Original != Standardized")
    mismatches.to_csv(OUTPUT_DIR / "name_mismatches.csv", index=False)

    cleaned = cleaned.set_index("Country")
    return cleaned


def tukey_outliers(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ~series.between(lo, hi)


def clean_gdp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # df["GDP_per_capita_PPP"] = pd.to_numeric(df["GDP_per_capita_PPP"].str.replace(",", ""), errors="coerce")
    removed = df[df["GDP_per_capita_PPP"].isna()]
    removed.to_csv(OUTPUT_DIR / "dropped_gdp.csv", index=False)
    df = df.dropna(subset=["GDP_per_capita_PPP"])
    # outliers
    n_out = tukey_outliers(df["GDP_per_capita_PPP"]).sum()
    print(f"[GDP] Tukey outliers count: {n_out}")
    # duplicates
    df = df.drop_duplicates(subset=["Country"], keep="first")
    df["Country"] = df["Country"].apply(standardize_country)
    return df.set_index("Country")


def clean_population(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # df["Population"] = pd.to_numeric(df["Population"].str.replace(",", ""), errors="coerce")
    removed = df[df["Population"].isna()]
    removed.to_csv(OUTPUT_DIR / "dropped_population.csv", index=False)
    df = df.dropna(subset=["Population"])

    # log‑Tukey outliers
    logpop = np.log10(df["Population"])
    n_out = tukey_outliers(logpop).sum()
    print(f"[Population] Log‑Tukey outliers count: {n_out}")

    df = df.drop_duplicates(subset=["Country"], keep="first")
    df["Country"] = df["Country"].apply(standardize_country)
    return df.set_index("Country")