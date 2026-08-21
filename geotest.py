import re
import unicodedata
import pandas as pd

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

QUOTE_CHARS = re.compile(r"[’‘‛ʻ`']")
NON_ALNUM = re.compile(r"[^a-z0-9]+")
MULTI_SPACE = re.compile(r"\s+")
class Geocoder:

    def __init__(self, worldcities_path: str):
        (
            self.exact_lookup,
            self.city_lookup,
            self.capital_lookup,
        ) = build_reference(worldcities_path)

    def get_coordinates(
        self,
        city: str,
        country: str,
    ) -> tuple[float | None, float | None, str]:

        city_norm = normalize_text(city)
        country_norm = normalize_country(country)

        key = (city_norm, country_norm)

        if key in self.exact_lookup.index:
            row = self.exact_lookup.loc[key]
            return float(row["lat"]), float(row["lng"])

        if city_norm in self.city_lookup.index:
            row = self.city_lookup.loc[city_norm]
            return float(row["lat"]), float(row["lng"])

        if country_norm in self.capital_lookup.index:
            row = self.capital_lookup.loc[country_norm]
            return float(row["lat"]), float(row["lng"])

        return None, None
def normalize_text(s):
    
    if pd.isna(s):
        return ""

    s = str(s)

    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")

    s = QUOTE_CHARS.sub("", s)

    s = s.lower()

    s = NON_ALNUM.sub(" ", s)

    s = MULTI_SPACE.sub(" ", s).strip()

    return s

def normalize_country(s):
    """
    Normalize country names and handle worldcities-specific country names.
    """
    s = normalize_text(s)

    aliases = {
        "south korea": "korea south",
        "north korea": "korea north",
        "myanmar": "burma",
        "ivory coast": "Côte d'Ivoire",
        "democratic republic of the congo": "congo kinshasa",
    }

    return aliases.get(s, s)
# ---------------------------------------------------------------------------
# Build reference lookup
# ---------------------------------------------------------------------------

def build_reference(worldcities_path):

    wc = pd.read_csv(worldcities_path)

    wc["city_norm"] = wc["city_ascii"].map(normalize_text)
    wc["country_norm"] = wc["country"].map(normalize_country)

    wc["population"] = (
        pd.to_numeric(wc["population"], errors="coerce")
        .fillna(-1)
    )

    wc_sorted = wc.sort_values("population", ascending=False)
    # ----------------------------------------------------------
    # Exact city + country
    # ----------------------------------------------------------

    exact_lookup = (
        wc_sorted
        .drop_duplicates(
            subset=["city_norm", "country_norm"],
            keep="first"
        )
        .set_index(["city_norm", "country_norm"])[
            ["lat", "lng", "city", "country"]
        ]
    )

    # ----------------------------------------------------------
    # City only
    # ----------------------------------------------------------

    city_only_lookup = (
        wc_sorted
        .drop_duplicates(
            subset=["city_norm"],
            keep="first"
        )
        .set_index("city_norm")[
            ["lat", "lng", "city", "country"]
        ]
    )

    # ----------------------------------------------------------
    # Country capital
    # ----------------------------------------------------------

    capital_lookup = (
        wc_sorted[
            wc_sorted["capital"] == "primary"
        ]
        .drop_duplicates(
            subset=["country_norm"],
            keep="first"
        )
        .set_index("country_norm")[
            ["lat", "lng", "city", "country"]
        ]
    )

    return exact_lookup, city_only_lookup, capital_lookup


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

def join_user_data(
    user_df,
    exact_lookup,
    city_only_lookup,
    capital_lookup,
):

    df = user_df.copy()

    df["_city_norm"] = df["City"].map(normalize_text)
    df["_country_norm"] = df["Country"].map(normalize_country)

    lat = []
    lng = []
    match_type = []
    matched_city = []
    matched_country = []

    for city_n, country_n in zip(
        df["_city_norm"],
        df["_country_norm"],
    ):

        key = (city_n, country_n)

        # ------------------------------------------------------
        # Exact match
        # ------------------------------------------------------

        if key in exact_lookup.index:

            row = exact_lookup.loc[key]

            lat.append(row["lat"])
            lng.append(row["lng"])

            match_type.append("exact")

            matched_city.append(row["city"])
            matched_country.append(row["country"])

        # ------------------------------------------------------
        # City only
        # ------------------------------------------------------

        elif city_n in city_only_lookup.index:

            row = city_only_lookup.loc[city_n]

            lat.append(row["lat"])
            lng.append(row["lng"])

            match_type.append("city_only_fallback")

            matched_city.append(row["city"])
            matched_country.append(row["country"])

        # ------------------------------------------------------
        # Country capital
        # ------------------------------------------------------

        elif country_n in capital_lookup.index:

            row = capital_lookup.loc[country_n]

            lat.append(row["lat"])
            lng.append(row["lng"])

            match_type.append("country_capital_fallback")

            matched_city.append(row["city"])
            matched_country.append(row["country"])

        # ------------------------------------------------------
        # No match
        # ------------------------------------------------------

        else:

            lat.append(pd.NA)
            lng.append(pd.NA)

            match_type.append("no_match")

            matched_city.append(pd.NA)
            matched_country.append(pd.NA)

    df["lat"] = lat
    df["lng"] = lng

    df["match_type"] = match_type
    df["matched_city"] = matched_city
    df["matched_country"] = matched_country

    df.drop(
        columns=["_city_norm", "_country_norm"],
        inplace=True,
    )

    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(df):

    total = len(df)

    counts = df["match_type"].value_counts()

    print(f"Total rows: {total}")

    for label in [
        "exact",
        "city_only_fallback",
        "country_capital_fallback",
        "no_match",
    ]:

        n = counts.get(label, 0)

        print(
            f"  {label:26s}: {n:6d} ({n/total:.1%})"
        )

    unmatched = df[
        df["match_type"] == "no_match"
    ][["City", "Country"]]

    if len(unmatched):

        print("\nSample unmatched rows:")

        print(
            unmatched.head(20).to_string(index=False)
        )
if __name__ == "__main__":

    worldcities_path = r"D:\X_user_recommendation\data\raw\worldcities.csv"
    user_path = r"D:\X_user_recommendation\data\raw\Assessment_TwitterDataset.csv"
    out_path = r"D:\X_user_recommendation\data\processed\geouser.csv"

    print("Loading world cities...")

    exact_lookup, city_only_lookup, capital_lookup = build_reference(
        worldcities_path
    )

    print("Loading user dataset...")

    user_df = pd.read_csv(user_path)

    print("Matching cities...")

    result = join_user_data(
        user_df,
        exact_lookup,
        city_only_lookup,
        capital_lookup,
    )

    print_report(result)
    result = result[result["match_type"] != "no_match"].reset_index(drop=True)

    result.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")