"""
demographics_crawler.py

Crawls country-level demographic data from https://www.worldometers.info/demographics/.

This script assumes the structure of the site as of May 2025, specifically:
- The index page includes a header containing the phrase "Demographics of Countries"
- All country links are found immediately after this header
- Country-specific data lives in structured HTML blocks (divs, paragraphs) labeled with specific IDs or headings

If the website structure changes, the specific CSS selectors and regex used below may fail.
"""

import re
from typing import Dict, Optional
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from utils import (
    BASE_URL, DEMOGRAPHICS_INDEX, OUTPUT_DIR,
    fetch, _to_float, _Number
)


def extract_country_links(index_html: str) -> Dict[str, str]:
    """
    Extract country page links from the demographics index page.

    The site has a header ("Demographics of Countries") followed by a <div>
    containing <a> tags to each country's page. This structure is very specific:
    if they ever redesign the page, this will silently break.
    """
    soup = BeautifulSoup(index_html, 'html.parser')

    # Look for the header containing "Demographics of Countries"
    section = soup.find(lambda tag: tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]
                        and "Demographics of Countries".lower() in tag.get_text().lower())

    # Immediately following this header is the <div> or <ul> of links
    sibling = section.find_next()

    # Collect all <a> tags within that section and build absolute URLs
    return {
        a.text.strip(): urljoin(BASE_URL, a["href"])
        for a in sibling.find_all("a", href=True)
    }


def parse_country_page(html: str) -> Dict[str, _Number]:
    """
    Parse structured demographic fields from a single country page.

    The layout follows 3 main sections:
    1. Life expectancy cards
    2. Urban population paragraph
    3. Population density paragraph

    The scraping logic here is fragile — it depends on ID-based headings
    and string patterns that match only under very specific HTML formatting.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Output template: all fields default to None if missing
    out = {
        "LifeExpectancy_Both": None,
        "LifeExpectancy_Female": None,
        "LifeExpectancy_Male": None,
        "UrbanPopulation_Percentage": None,
        "UrbanPopulation_Absolute": None,
        "PopulationDensity": None,
    }

    # === 1. LIFE EXPECTANCY DATA ======================================
    # These show up in divs with class="bg-zinc-50"
    # Each card has a span with label ("Both Sexes", "Females", "Males")
    # followed by a div with class "text-2xl" containing the value

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

    # === 2. URBAN POPULATION SECTION ==================================
    # This section starts at a heading with id="urb", followed by a paragraph.
    # We extract:
    #   - percentage using regex on "[number]%"
    #   - absolute value inside "([number] people)"

    urb_h2 = soup.find(id="urb")
    if urb_h2:
        p = urb_h2.find_next("p")
        if p:
            txt = p.get_text(" ", strip=True)

            # Regex for capturing "58.3%" (percentages)
            pct = re.search(r"([0-9][0-9.,]*)%", txt)

            # Regex for capturing "(1,234,567 people)"
            abs_ = re.search(r"\(([0-9][0-9,]*)\s*people", txt)

            out["UrbanPopulation_Percentage"] = _to_float(pct.group(1)) if pct else None
            out["UrbanPopulation_Absolute"] = int(abs_.group(1).replace(",", "")) if abs_ else None

    # === 3. POPULATION DENSITY SECTION ================================
    # Heading has id="population-density" and is followed by a <p> tag
    # We're looking for phrases like "421 people per Km"

    dens_h2 = soup.find(id="population-density")
    if dens_h2:
        p = dens_h2.find_next("p")
        if p:
            # We only care about numeric part before "people per Km"
            m = re.search(r"\b([0-9][0-9.,]*)\s*people per Km", p.get_text())
            out["PopulationDensity"] = _to_float(m.group(1)) if m else None

    return out


def crawl_demographics(delay: float = 0, save: bool = False) -> pd.DataFrame:
    """
    Crawl all country pages and compile into a structured DataFrame.

    If save=True, outputs include:
    - demographics_before_sort.csv
    - demographics_after_sort.csv
    - demographics_data.csv
    """
    index_html = fetch(DEMOGRAPHICS_INDEX, delay=delay)
    links = extract_country_links(index_html)

    records = {}
    for country, url in tqdm(links.items(), desc="Crawling", unit="country", disable=False):
        html = fetch(url, delay=delay)
        records[country] = parse_country_page(html)

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "Country"

    if save:
        df.head(10).to_csv(OUTPUT_DIR / "demographics_before_sort.csv")
        df = df.sort_index()
        df.head(10).to_csv(OUTPUT_DIR / "demographics_after_sort.csv")
        df.to_csv(OUTPUT_DIR / "demographics_data.csv")

    return df


if __name__ == "__main__":
    print("Starting demographics crawl...")
    crawl_demographics(delay=0.1, save=True)
    print("✅ Done.")
