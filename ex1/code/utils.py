from pathlib import Path
import re
import time
from typing import Dict, List, Optional
import requests
import pandas as pd


BASE_URL = "https://www.worldometers.info"
DEMOGRAPHICS_INDEX = f"{BASE_URL}/demographics/"
DEMOGRAPHICS_HEADER = "Demographics of Countries"

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

def load_csv(filename: str, expected_cols: List[str], sort_prefix: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / filename, na_values=["None"], keep_default_na=True)
    assert all(col in df.columns for col in expected_cols), f"{filename}: missing cols"
    df.head(5).to_csv(OUTPUT_DIR / f"{sort_prefix}_before_sort.csv", index=False)
    df = df.sort_values("Country")
    df.head(5).to_csv(OUTPUT_DIR / f"{sort_prefix}_after_sort.csv", index=False)
    df.describe().to_csv(OUTPUT_DIR / f"{sort_prefix}_describe.csv")
    return df