import numpy as np, pandas as pd
from pathlib import Path

def load(name: str):
    return pd.read_csv(Path("output") / name, index_col="Country")

def z(s: pd.Series):
    return (s - s.mean()) / s.std(ddof=0)

def main():
    o = Path("output")
    d = load("demographics_clean.csv")
    g = load("gdp_clean.csv")
    p = load("pop_clean.csv")
    df = d.join(g, how="inner").join(p, how="inner")
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df["TotalGDP"] = df["GDP per capita PPP"] * df["Population"]
    df["LogGDPperCapita"] = np.log10(df["GDP per capita PPP"])
    df["LogPopulation"] = np.log10(df["Population"])
    df["LifeExpectancy_z"] = z(df["LifeExpectancy Both"])
    df["LogGDPpc_z"] = z(df["LogGDPperCapita"])
    df["LogPop_z"] = z(df["LogPopulation"])
    X = df.sort_index()[["LifeExpectancy_z", "LogGDPpc_z", "LogPop_z"]].to_numpy()
    np.save(o / "X.npy", X)

if __name__ == "__main__":
    main()
