import re
from typing import Dict, List, Optional
from urllib.parse import urljoin

import pandas as pd

from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from utils import (CACHE_DIR, BASE_URL, DEMOGRAPHICS_INDEX, OUTPUT_DIR,
                   _to_float, _Number,
                   fetch)



def extract_country_links(index_html: str) -> Dict[str, str]:
    soup = BeautifulSoup(index_html, 'html.parser')  # 'lxml' is faster than html.parser but requires `pip install lxml`
    section = soup.find(lambda tag: tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]
                                    and "Demographics of Countries".lower() in tag.get_text().lower())
    sibling = section.find_next()
    links = {
        a.text.strip(): urljoin(BASE_URL, a["href"])
        for a in sibling.find_all("a", href=True)
    }
    return links


def parse_country_page(html: str) -> Dict[str, _Number]:
    soup = BeautifulSoup(html, "html.parser")

    out = {
        "LifeExpectancy_Both": None,
        "LifeExpectancy_Female": None,
        "LifeExpectancy_Male": None,
        "UrbanPopulation_Percentage": None,
        "UrbanPopulation_Absolute": None,
        "PopulationDensity": None,
    }

    # ---- 1) Life‑expectancy cards -------------------------------------------------
    label_map = {
        "Both Sexes": "LifeExpectancy_Both",
        "Females": "LifeExpectancy_Female",
        "Males": "LifeExpectancy_Male",
    }
    for card in soup.select("div.bg-zinc-50"):
        label = card.select_one("span")
        value = card.find_next("div", class_=re.compile(r"text-2xl"))
        if label and value and (key := label_map.get(label.get_text(strip=True))):
            out[key] = _to_float(value.get_text(strip=True))

    # ---- 2) Urbanisation paragraph ------------------------------------------------
    urb_h2 = soup.find(id="urb")
    p = urb_h2.find_next("p")
    if urb_h2 and p:
        txt = p.get_text(" ", strip=True)
        pct = re.search(r"([0-9][0-9.,]*)%", txt)
        abs_ = re.search(r"\(([0-9][0-9,]*)\s*people", txt)
        out["UrbanPopulation_Percentage"] = _to_float(pct.group(1)) if pct else None
        out["UrbanPopulation_Absolute"] = int(abs_.group(1).replace(",", "")) if abs_ else None

    # ---- 3) Population‑density paragraph -----------------------------------------
    dens_h2 = soup.find(id="population-density")
    p = dens_h2.find_next("p")
    if dens_h2 and p:
        m = re.search(r"\b([0-9][0-9.,]*)\s*people per Km", p.get_text())
        out["PopulationDensity"] = _to_float(m.group(1)) if m else None

    return out


def crawl_demographics(delay: float = 0, save: bool = False) -> pd.DataFrame:
    """Return df_demographics and write sorted/unsorted CSVs."""
    index_html = fetch(DEMOGRAPHICS_INDEX, delay=delay)
    links = extract_country_links(index_html)

    records = {}
    for country, url in tqdm(links.items(), desc="Crawling", unit="country", disable=True):
        html = fetch(url, delay=delay)
        records[country] = parse_country_page(html)

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "Country"

    # persist
    if save:
        unsorted = df.copy()
        unsorted.head(10).to_csv(OUTPUT_DIR / "demographics_before_sort.csv")
        df = df.sort_index()
        df.head(10).to_csv(OUTPUT_DIR / "demographics_after_sort.csv")
        df.to_csv(OUTPUT_DIR / "demographics_data.csv")
    return df


if __name__ == "__main__":
    print("DONE!")
