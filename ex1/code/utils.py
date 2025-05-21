from pathlib import Path
import re
import time
from typing import Dict, List, Optional
import requests
import pandas as pd


BASE_URL = "https://www.worldometers.info"
DEMOGRAPHICS_INDEX = f"{BASE_URL}/demographics/"
DEMOGRAPHICS_HEADER = "Demographics of Countries"

GDP_DATA_FILENAME = "gdp_per_capita_2021.csv"
POP_DATA_FILENAME = "population_2021.csv"
DEMOGRAPHICS_DATA_FILENAME = "demographics_data.csv"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

RE_NUMBER = re.compile(r"[-+]??\d*[\d,]*\.?\d+")  # matches 1,234.56 too
_Number = Optional[float | int]

def _to_float(x: str | None) -> Optional[float]:
    return float(x.replace(",", "")) if x else None

def fetch(url: str, delay: float = 0) -> str:
    """Download *url* (respect robots.txt ‑ this site allows GET). Cache to disk."""
    fname = CACHE_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", url) + ".html")
    if fname.exists():
        return fname.read_text(encoding="utf‑8", errors="ignore")

    time.sleep(delay)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    fname.write_text(resp.text, encoding="utf‑8")
    return resp.text

def load_csv(filename: str, expected_cols: List[str], sort_prefix: str, save: bool = False) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / filename, na_values=["None"], keep_default_na=True)
    assert all(col in df.columns for col in expected_cols), f"{filename}: missing cols"
    print(f"{expected_cols} cols exist in dataframe")
    for col in ["GDP_per_capita_PPP", "Population"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print("Ensured numeric types")
    if save:
        df.head(5).to_csv(OUTPUT_DIR / f"{sort_prefix}_before_sort.csv", index=False)
    print(df.head())
    df = df.sort_values("Country")
    print(df.head())
    if save:
        df.head(5).to_csv(OUTPUT_DIR / f"{sort_prefix}_after_sort.csv", index=False)
        df.describe().to_csv(OUTPUT_DIR / f"{sort_prefix}_describe.csv")
    return df

def confirm_cols(df, expected_cols: List[str]) -> pd.DataFrame:
        assert all(col in df.columns for col in expected_cols)

def read_csv(filename: str):
    return pd.read_csv(DATA_DIR / filename, na_values=["None"], keep_default_na=True)

def preview():
    import pandas as pd
    from pathlib import Path

    DATA = Path("data")  # adjust if raw CSVs live elsewhere
    OUT = Path("output");
    OUT.mkdir(exist_ok=True)



    gdp = read_csv("gdp_per_capita_2021.csv")
    pop = read_csv("population_2021.csv")

    # numeric fix
    for col in ["GDP_per_capita_PPP", "Population"]:
        if col in gdp.columns:
            gdp[col] = pd.to_numeric(gdp[col], errors="coerce")
        if col in pop.columns:
            pop[col] = pd.to_numeric(pop[col], errors="coerce")

    # before sort snapshots
    gdp.head().to_csv(OUT / "gdp_before_sort.csv", index=False)
    pop.head().to_csv(OUT / "pop_before_sort.csv", index=False)

    # after sort snapshots
    (gdp.sort_values("Country").head()
     .to_csv(OUT / "gdp_after_sort.csv", index=False))
    (pop.sort_values("Country").head()
     .to_csv(OUT / "pop_after_sort.csv", index=False))

    # describe tables
    gdp.describe().to_csv(OUT / "gdp_describe.csv")
    pop.describe().to_csv(OUT / "pop_describe.csv")

if __name__ == "__main__":
    print("!")