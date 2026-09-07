from pathlib import Path
import time
import pandas as pd
import requests



INPUT_FILE = "outputbuses_not_mapped.csv"
OUTPUT_FILE =  "output/buses_osm_candidates.csv"

df = pd.read_csv(INPUT_FILE)

# Important: identify your application to Nominatim.
HEADERS = {
    "User-Agent": "EIEG-Hackathon-PyPSA-Geocoder/1.0"
}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def clean_station_name(row):
    """
    Prefer the station column if present.
    Fall back to psse_name.

    Modify this function later if your PSS/E names contain
    abbreviations such as _T, _OFF, etc.
    """
    station = row.get("station")

    if pd.notna(station) and str(station).strip():
        return str(station).strip()

    psse_name = row.get("psse_name")

    if pd.notna(psse_name):
        return str(psse_name).strip()

    return None


def search_osm(name):
    """
    Search OpenStreetMap/Nominatim.

    We restrict results to Ireland + UK because the network may
    include both Republic of Ireland and Northern Ireland.
    """

    if not name:
        return None

    queries = [
        f"{name} substation Ireland",
        f"{name} electricity substation Ireland",
        f"{name} Ireland",
    ]

    for query in queries:

        params = {
            "q": query,
            "format": "jsonv2",
            "limit": 5,
            "addressdetails": 1,

            # Ireland and United Kingdom
            "countrycodes": "ie,gb",
        }

        try:
            response = requests.get(
                NOMINATIM_URL,
                params=params,
                headers=HEADERS,
                timeout=20,
            )

            response.raise_for_status()

            results = response.json()

        except Exception as exc:
            print(f"Error searching {name}: {exc}")
            return None

        if results:

            # Prefer something that looks electrically relevant
            for result in results:

                text = (
                    str(result.get("display_name", ""))
                    + " "
                    + str(result.get("type", ""))
                    + " "
                    + str(result.get("category", ""))
                ).lower()

                if (
                    "substation" in text
                    or "power" in text
                    or "electric" in text
                ):
                    return result

            # Otherwise retain the first result,
            # but mark it for manual review.
            return results[0]

        # Public Nominatim policy requires <= 1 request/sec.
        time.sleep(1.1)

    return None


results = []

for i, row in df.iterrows():

    station_name = clean_station_name(row)

    print(
        f"[{i + 1}/{len(df)}] "
        f"Searching OSM for: {station_name}"
    )

    result = search_osm(station_name)

    output = row.to_dict()

    if result is None:

        output["osm_found"] = False
        output["osm_lon"] = None
        output["osm_lat"] = None
        output["osm_display_name"] = None
        output["osm_type"] = None
        output["osm_category"] = None
        output["osm_id"] = None
        output["osm_url"] = None
        output["review_status"] = "not found"

    else:

        lon = float(result["lon"])
        lat = float(result["lat"])

        output["osm_found"] = True
        output["osm_lon"] = lon
        output["osm_lat"] = lat
        output["osm_display_name"] = result.get("display_name")
        output["osm_type"] = result.get("type")
        output["osm_category"] = result.get("category")
        output["osm_id"] = result.get("osm_id")

        osm_type = result.get("osm_type")
        osm_id = result.get("osm_id")

        if osm_type and osm_id:
            output["osm_url"] = (
                f"https://www.openstreetmap.org/"
                f"{osm_type}/{osm_id}"
            )
        else:
            output["osm_url"] = None

        # Sanity check: Ireland / Northern Ireland bounding box
        if -11 <= lon <= -5 and 51 <= lat <= 56:
            output["review_status"] = "candidate"
        else:
            output["review_status"] = "outside Ireland bounds"

    results.append(output)

    # Be polite to the public Nominatim server.
    time.sleep(1.1)


result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(f"Saved OSM candidates to:")
print(OUTPUT_FILE)

print()
print("Summary:")
print(result_df["review_status"].value_counts(dropna=False))