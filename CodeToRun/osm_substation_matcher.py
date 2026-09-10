
"""
Find candidate OpenStreetMap substations for unmapped PyPSA buses.

This script:
1. Reads buses_not_mapped.csv.
2. Downloads OSM objects tagged power=substation from an Ireland/Northern
   Ireland bounding box using the Overpass API.
3. Extracts each substation's centre coordinate and useful OSM tags.
4. Fuzzy-matches each PyPSA station name against ONLY power=substation objects.
5. Writes buses_osm_substation_candidates.csv for manual review.

Install:
    pip install pandas requests rapidfuzz

Run:
    python osm_substation_matcher.py
"""

from pathlib import Path
import re
import time
import unicodedata

import pandas as pd
import requests
from rapidfuzz import fuzz, process


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

DATA_DIR = Path(
    "data/"
)

# Change this if buses_not_mapped.csv is elsewhere.
INPUT_FILE = Path(
    DATA_DIR / "buses_not_mapped.csv"
)

OUTPUT_FILE = DATA_DIR / "buses_osm_substation_candidates.csv"
OSM_CACHE_FILE = DATA_DIR / "osm_power_substations_ireland.csv"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Covers the island of Ireland with a small margin.
SOUTH = 51.2
WEST = -11.0
NORTH = 55.6
EAST = -5.0

# How many candidate substations to keep for each PyPSA station.
TOP_N = 5

# Text score threshold for a candidate to be considered plausible.
MIN_TEXT_SCORE = 55

# You can increase this if your Overpass connection is slow.
REQUEST_TIMEOUT = 180

HEADERS = {
    "User-Agent": "EIEG-Hackathon-PyPSA-OSM-Substation-Matcher/1.0"
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def normalise_name(value):
    """
    Normalise names for fuzzy matching.

    Examples:
        'ARIGNA_T'    -> 'arigna'
        'W_ARKLOW_OFF' -> 'w arklow'
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Remove accents.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Replace separators with spaces.
    text = re.sub(r"[_\-/]+", " ", text)

    # Remove common network suffixes that often occur in PSS/E names.
    tokens = text.split()

    suffixes = {
        "t", "off", "on", "stn", "station",
        "sub", "ss", "kv"
    }

    while tokens and tokens[-1].lower() in suffixes:
        tokens.pop()

    text = " ".join(tokens)

    # Keep only letters/numbers/spaces.
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


def voltage_tokens(value):
    """
    Convert an OSM voltage string such as:
        110000;220000
    to:
        {110, 220}
    """
    if pd.isna(value):
        return set()

    values = set()

    for token in re.split(r"[;,/ ]+", str(value)):
        token = token.strip()

        if not token:
            continue

        try:
            number = float(token)

            # OSM normally stores volts.
            if number >= 1000:
                number = number / 1000.0

            values.add(int(round(number)))

        except ValueError:
            continue

    return values


def preferred_osm_name(tags):
    """
    Choose the most useful OSM text field for station matching.
    """
    for key in [
        "name",
        "name:en",
        "operator:ref",
        "ref",
        "reference",
    ]:
        value = tags.get(key)

        if value:
            return str(value)

    return ""


def build_osm_url(osm_type, osm_id):
    if not osm_type or not osm_id:
        return None

    if osm_type == "node":
        return f"https://www.openstreetmap.org/node/{osm_id}"

    if osm_type == "way":
        return f"https://www.openstreetmap.org/way/{osm_id}"

    if osm_type == "relation":
        return f"https://www.openstreetmap.org/relation/{osm_id}"

    return None


# ---------------------------------------------------------------------
# DOWNLOAD SUBSTATIONS
# ---------------------------------------------------------------------

def download_substations():
    """
    Download only OSM elements tagged power=substation.

    'out center' makes Overpass return a centre coordinate for ways
    and relations, so nodes, areas and relations can all be handled
    consistently.
    """

    query = f"""
    [out:json][timeout:120];
    (
      node["power"="substation"]({SOUTH},{WEST},{NORTH},{EAST});
      way["power"="substation"]({SOUTH},{WEST},{NORTH},{EAST});
      relation["power"="substation"]({SOUTH},{WEST},{NORTH},{EAST});
    );
    out tags center;
    """

    print("Downloading OSM power=substation objects...")
    print(f"Overpass endpoint: {OVERPASS_URL}")

    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    records = []

    for element in payload.get("elements", []):
        tags = element.get("tags", {})

        osm_type = element.get("type")
        osm_id = element.get("id")

        if osm_type == "node":
            lat = element.get("lat")
            lon = element.get("lon")
        else:
            centre = element.get("center", {})
            lat = centre.get("lat")
            lon = centre.get("lon")

        if lat is None or lon is None:
            continue

        name = preferred_osm_name(tags)

        record = {
            "osm_type": osm_type,
            "osm_id": osm_id,
            "osm_name": name,
            "osm_name_normalised": normalise_name(name),
            "osm_lat": float(lat),
            "osm_lon": float(lon),
            "osm_voltage": tags.get("voltage"),
            "osm_substation": tags.get("substation"),
            "osm_operator": tags.get("operator"),
            "osm_ref": tags.get("ref"),
            "osm_location": tags.get("location"),
            "osm_url": build_osm_url(osm_type, osm_id),
        }

        records.append(record)

    osm = pd.DataFrame(records)

    if osm.empty:
        raise RuntimeError(
            "Overpass returned no power=substation objects. "
            "Check your internet connection or Overpass service status."
        )

    osm.to_csv(OSM_CACHE_FILE, index=False)

    print(f"Downloaded {len(osm):,} substations.")
    print(f"Saved OSM substation cache to:\n{OSM_CACHE_FILE}")

    return osm


# ---------------------------------------------------------------------
# LOAD OSM CACHE OR DOWNLOAD
# ---------------------------------------------------------------------

if OSM_CACHE_FILE.exists():
    print(f"Loading cached OSM substations from:\n{OSM_CACHE_FILE}")
    osm = pd.read_csv(OSM_CACHE_FILE)



# ---------------------------------------------------------------------
# LOAD PYPSA UNMAPPED BUSES
# ---------------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find buses_not_mapped.csv at:\n{INPUT_FILE}"
    )

buses = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(buses):,} unmapped PyPSA buses.")

if "name" not in buses.columns:
    raise ValueError(
        "Expected a 'name' column containing the PyPSA bus ID."
    )


# ---------------------------------------------------------------------
# PREPARE OSM NAMES FOR MATCHING
# ---------------------------------------------------------------------

osm["osm_name"] = osm["osm_name"].fillna("").astype(str)

if "osm_name_normalised" not in osm.columns:
    osm["osm_name_normalised"] = osm["osm_name"].map(normalise_name)

# Keep named OSM substations for fuzzy text matching.
named_osm = osm[
    osm["osm_name_normalised"].fillna("").str.len() > 0
].copy()

print(f"OSM substations with usable names: {len(named_osm):,}")


# ---------------------------------------------------------------------
# MATCH PYPSA STATIONS TO OSM SUBSTATIONS
# ---------------------------------------------------------------------

output_rows = []

osm_choices = named_osm["osm_name_normalised"].tolist()


for index, row in buses.iterrows():

    station = row.get("station")
    psse_name = row.get("psse_name")

    search_name = station

    if pd.isna(search_name) or not str(search_name).strip():
        search_name = psse_name

    search_norm = normalise_name(search_name)

    print(
        f"[{index + 1}/{len(buses)}] "
        f"Matching: {search_name}"
    )

    base = row.to_dict()
    base["search_name"] = search_name
    base["search_name_normalised"] = search_norm

    if not search_norm:

        candidate = base.copy()
        candidate["candidate_rank"] = None
        candidate["text_score"] = None
        candidate["voltage_match"] = False
        candidate["review_status"] = "no station name"
        output_rows.append(candidate)

        continue

    matches = process.extract(
        search_norm,
        osm_choices,
        scorer=fuzz.WRatio,
        limit=TOP_N,
    )

    pypsa_voltage = pd.to_numeric(
        pd.Series([row.get("v_nom")]),
        errors="coerce"
    ).iloc[0]

    for rank, (_, text_score, osm_choice_index) in enumerate(
        matches,
        start=1,
    ):

        osm_row = named_osm.iloc[osm_choice_index]

        candidate = base.copy()

        osm_voltages = voltage_tokens(
            osm_row.get("osm_voltage")
        )

        voltage_match = False

        if pd.notna(pypsa_voltage) and osm_voltages:
            voltage_match = (
                int(round(float(pypsa_voltage)))
                in osm_voltages
            )

        # Give a modest boost when the OSM voltage agrees with PyPSA.
        combined_score = float(text_score)

        if voltage_match:
            combined_score += 10

        combined_score = min(combined_score, 100)

        candidate.update(
            {
                "candidate_rank": rank,
                "text_score": round(float(text_score), 1),
                "combined_score": round(combined_score, 1),
                "voltage_match": voltage_match,
                "osm_type": osm_row.get("osm_type"),
                "osm_id": osm_row.get("osm_id"),
                "osm_name": osm_row.get("osm_name"),
                "osm_lon": osm_row.get("osm_lon"),
                "osm_lat": osm_row.get("osm_lat"),
                "osm_voltage": osm_row.get("osm_voltage"),
                "osm_substation": osm_row.get("osm_substation"),
                "osm_operator": osm_row.get("osm_operator"),
                "osm_ref": osm_row.get("osm_ref"),
                "osm_location": osm_row.get("osm_location"),
                "osm_url": osm_row.get("osm_url"),
            }
        )

        if text_score >= 85 and voltage_match:
            status = "strong candidate"

        elif text_score >= 85:
            status = "good name match - check voltage"

        elif text_score >= MIN_TEXT_SCORE and voltage_match:
            status = "possible candidate - voltage agrees"

        elif text_score >= MIN_TEXT_SCORE:
            status = "possible candidate - manual review"

        else:
            status = "weak match"

        candidate["review_status"] = status

        output_rows.append(candidate)


# ---------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------

results = pd.DataFrame(output_rows)

# Put the most useful columns first.
preferred_columns = [
    "name",
    "psse_name",
    "station",
    "v_nom",
    "search_name",
    "candidate_rank",
    "text_score",
    "combined_score",
    "voltage_match",
    "review_status",
    "osm_name",
    "osm_lon",
    "osm_lat",
    "osm_voltage",
    "osm_substation",
    "osm_operator",
    "osm_ref",
    "osm_type",
    "osm_id",
    "osm_url",
]

ordered = [
    col for col in preferred_columns
    if col in results.columns
]

remaining = [
    col for col in results.columns
    if col not in ordered
]

results = results[ordered + remaining]

results.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("Finished.")
print(f"Saved candidate matches to:\n{OUTPUT_FILE}")

print()
print("Candidate status counts:")
print(results["review_status"].value_counts(dropna=False))

print()
print(
    "IMPORTANT: Review the matches before copying osm_lon/osm_lat "
    "into your PyPSA buses.csv."
)
