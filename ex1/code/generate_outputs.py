# generate_outputs.py
import pandas as pd
import numpy as np
from pathlib import Path
from demographics_crawler import crawl_demographics
from data_cleaning import clean_demographics, clean_gdp, clean_population
from feature_engineering import engineer_features, apply_manual_country_mapping, COUNTRY_NORMALIZATION_MAP
from utils import (
    DEMOGRAPHICS_DATA_FILENAME, GDP_DATA_FILENAME, POP_DATA_FILENAME,
    OUTPUT_DIR, DATA_DIR
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Crawl and save demographics
    df_demographics = crawl_demographics(save=True)

    # 2. Load raw GDP and Population files
    df_gdp = pd.read_csv(DATA_DIR / GDP_DATA_FILENAME, na_values=["None"])
    df_pop = pd.read_csv(DATA_DIR / POP_DATA_FILENAME, na_values=["None"])

    df_gdp.head().to_csv(OUTPUT_DIR / "gdp_before_sort.csv", index=False)
    df_pop.head().to_csv(OUTPUT_DIR / "pop_before_sort.csv", index=False)

    df_gdp.sort_values("Country").head().to_csv(OUTPUT_DIR / "gdp_after_sort.csv", index=False)
    df_pop.sort_values("Country").head().to_csv(OUTPUT_DIR / "pop_after_sort.csv", index=False)

    df_gdp.describe().to_csv(OUTPUT_DIR / "gdp_describe.csv")
    df_pop.describe().to_csv(OUTPUT_DIR / "pop_describe.csv")

    # 3. Clean datasets
    df_demographics_clean = clean_demographics(df_demographics)
    df_gdp_clean = clean_gdp(df_gdp)
    df_pop_clean = clean_population(df_pop)

    # 4. Join all datasets (inner join)
    df_gdp_clean_normalized = apply_manual_country_mapping(df_gdp_clean, COUNTRY_NORMALIZATION_MAP)
    df_pop_clean_normalized = apply_manual_country_mapping(df_pop_clean, COUNTRY_NORMALIZATION_MAP)
    df_final = df_demographics_clean.join(df_gdp_clean_normalized, how="inner").join(df_pop_clean_normalized, how="inner")

    # 5. Track lost countries
    union = set(df_demographics_clean.index) | set(df_gdp_clean.index) | set(df_pop_clean.index)
    lost = sorted(union - set(df_final.index))
    pd.Series(lost, name="Country").to_csv(OUTPUT_DIR / "lost_countries.csv", index=False)

    # 6. Feature Engineering
    df_fe = engineer_features(df_final)

    # 7. Save scaled stats
    # df_fe[["LifeExpectancy_z", "LogGDPpc_z", "LogPop_z"]].describe().to_csv(OUTPUT_DIR / "X_scaled_describe.csv")

    # 8. Save final matrix
    X = df_fe.sort_index()[["LifeExpectancy_z", "LogGDPpc_z", "LogPop_z"]].to_numpy()
    np.save(OUTPUT_DIR / "X.npy", X)

    print(f"✅ Done! Countries in final dataset: {df_fe.shape[0]}")


if __name__ == "__main__":
    main()
