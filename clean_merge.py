

import re
from pathlib import Path
import pandas as pd


def clean_city_name(name: str, is_census: bool = False) -> str:
    """
    Standardizes Missouri city/place names across Census and Zillow datasets
    to resolve naming mismatches (e.g., 'St. Louis city' -> 'st louis').
    """
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    if is_census:
        s = re.sub(r",\s*missouri$", "", s)
        s = re.sub(r"\s+(city|town|village|cdp|municipality)(\s*\(pt\.\))?$", "", s)
    s = s.replace(".", "").replace("'", "")
    s = re.sub(r"^saint\b", "st", s)
    s = re.sub(r"^sainte\b", "ste", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def merge_datasets(
    census_path: str = "datasets/sub-est2025_29.csv",
    zhvi_path: str = "datasets/ZHVI.csv",
    target_year: int = 2025,
    output_path: str = "datasets/missouri_housing_population_merged.csv",
) -> pd.DataFrame:
    """
    Cleans and merges the Missouri Census city population dataset and Zillow City ZHVI dataset
    into a single CSV file with one row per city for the target year.
    """
    print(f"[INFO] Loading Census file: {census_path}")
    print(f"[INFO] Loading ZHVI file:   {zhvi_path}")

    # 1. Load and clean Census city-level population (SUMLEV == 162: Incorporated Places)
    census = pd.read_csv(census_path, encoding="latin-1")
    cities = census[census["SUMLEV"] == 162].copy()
    cities["city_clean"] = cities["NAME"].apply(lambda x: clean_city_name(x, is_census=True))

    pop_col = f"POPESTIMATE{target_year}"
    cities_year = cities[["PLACE", "NAME", "city_clean", pop_col]].rename(
        columns={pop_col: "Population"}
    )
    cities_year["Population"] = pd.to_numeric(cities_year["Population"], errors="coerce")

    # 2. Load and filter Zillow City ZHVI to Missouri and target year
    zhvi = pd.read_csv(zhvi_path)
    if "State" not in zhvi.columns or "RegionName" not in zhvi.columns:
        raise ValueError(
            f"The file '{zhvi_path}' is the State-level ZHVI file, not the City-level ZHVI file. "
            "Please ensure you are using the City-level ZHVI CSV."
        )

    zhvi_mo = zhvi[zhvi["State"] == "MO"].copy()

    date_cols = [
        c for c in zhvi_mo.columns if re.match(f"^{target_year}-\\d{{2}}-\\d{{2}}$", str(c))
    ]
    zhvi_mo["ZHVI"] = zhvi_mo[date_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    zhvi_mo = zhvi_mo.dropna(subset=["ZHVI"]).copy()
    zhvi_mo["city_clean"] = zhvi_mo["RegionName"].apply(
        lambda x: clean_city_name(x, is_census=False)
    )

    # 3. Merge into a single dataset (1 row per city)
    merged = pd.merge(
        zhvi_mo[["RegionID", "RegionName", "CountyName", "Metro", "city_clean", "ZHVI"]],
        cities_year,
        on="city_clean",
        how="inner",
    )
    merged["Year"] = target_year
    merged["Metro"] = merged["Metro"].fillna("Non-Metro / Rural")

    # Reorder columns cleanly and sort alphabetically by city
    final_cols = [
        "RegionID",
        "RegionName",
        "CountyName",
        "Metro",
        "Year",
        "Population",
        "ZHVI",
    ]
    merged = merged[final_cols].sort_values("RegionName").reset_index(drop=True)

    merged.to_csv(output_path, index=False)
    print(f"[INFO] Successfully merged {len(merged)} Missouri cities for {target_year}.")
    print(f"[INFO] Saved single combined dataset to '{output_path}'.")
    return merged
