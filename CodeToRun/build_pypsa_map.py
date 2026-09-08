"""
Build static and interactive geographic maps from a PyPSA CSV network folder.

Expected files include:
    network.csv
    buses.csv
    lines.csv
    transformers.csv
    generators.csv
    loads.csv
    links.csv
    carriers.csv
    snapshots.csv
    sub_networks.csv

Install:
    pip install pypsa pandas matplotlib folium

Run:
    python build_pypsa_map.py

Edit DATA_DIR below if the CSV files are in a different folder.
"""

from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt
import pypsa


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data" / "TYTFS2024_WP2033_V35_transmission"
OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_OUTPUT = OUTPUT_DIR / "pypsa_network_map.png"
INTERACTIVE_OUTPUT = OUTPUT_DIR / "pypsa_network_map.html"

print(f"Loading PyPSA network from: {DATA_DIR}")
print(f"Exists: {DATA_DIR.exists()}")
print(f"Is directory: {DATA_DIR.is_dir()}")

n = pypsa.Network(DATA_DIR)

SHOW_GENERATORS = True
SHOW_TRANSFORMERS = True

# Bus marker sizes by nominal voltage
BUS_SIZE = {
    110: 9,
    220: 16,
    275: 20,
    400: 28,
}

# Matplotlib will assign its normal default colours automatically.
# We deliberately use different line styles/widths for voltage levels.
LINE_WIDTH = {
    110: 0.7,
    220: 1.1,
    275: 1.4,
    400: 1.8,
}


# ---------------------------------------------------------------------
# LOAD NETWORK
# ---------------------------------------------------------------------

print(f"Loading PyPSA network from: {DATA_DIR.resolve()}")
n = pypsa.Network(DATA_DIR)

print(f"Network:      {n.name}")
print(f"Buses:        {len(n.buses):,}")
print(f"Lines:        {len(n.lines):,}")
print(f"Transformers: {len(n.transformers):,}")
print(f"Generators:   {len(n.generators):,}")
print(f"Loads:        {len(n.loads):,}")
print(f"Links:        {len(n.links):,}")


# ---------------------------------------------------------------------
# CLEAN / PREPARE COORDINATES
# ---------------------------------------------------------------------

# PyPSA uses x = longitude and y = latitude.
buses = n.buses.copy()

buses["x"] = pd.to_numeric(buses["x"], errors="coerce")
buses["y"] = pd.to_numeric(buses["y"], errors="coerce")
buses["v_nom"] = pd.to_numeric(buses["v_nom"], errors="coerce")


# ---------------------------------------------------------------------
# FILL MISSING BUS COORDINATES FROM OSM CANDIDATES
# ---------------------------------------------------------------------

OSM_CANDIDATES_FILE = DATA_DIR / "buses_osm_candidates.csv"

if OSM_CANDIDATES_FILE.exists():

    osm = pd.read_csv(OSM_CANDIDATES_FILE)

    # PyPSA bus IDs are usually stored in the index.
    buses.index = buses.index.astype(str)

    # Work out which column in the OSM CSV contains the original bus ID.
    # Common possibilities are "name", "bus", or an unnamed CSV index column.
    candidate_id_column = None

    for col in ["name", "bus", "Bus", "Unnamed: 0"]:
        if col in osm.columns:
            candidate_id_column = col
            break

    if candidate_id_column is None:
        print(
            "WARNING: Could not identify a bus-ID column in "
            "buses_osm_candidates.csv"
        )

    else:
        osm[candidate_id_column] = osm[candidate_id_column].astype(str)

        # Keep only candidates that passed your initial sanity check.
        if "review_status" in osm.columns:
            osm = osm[
                osm["review_status"].astype(str).str.lower() == "candidate"
            ].copy()

        # Make sure OSM coordinates are numeric.
        osm["osm_lon"] = pd.to_numeric(
            osm["osm_lon"],
            errors="coerce"
        )

        osm["osm_lat"] = pd.to_numeric(
            osm["osm_lat"],
            errors="coerce"
        )

        osm = osm.dropna(
            subset=["osm_lon", "osm_lat"]
        )

        # Index by bus ID for easy lookup.
        osm = osm.set_index(candidate_id_column)

        filled_count = 0

        for bus_id in buses.index:

            x = buses.at[bus_id, "x"]
            y = buses.at[bus_id, "y"]

            # Treat missing values and (0,0) as needing a fallback.
            needs_coordinates = (
                pd.isna(x)
                or pd.isna(y)
                or (x == 0 and y == 0)
            )

            if not needs_coordinates:
                continue

            if bus_id not in osm.index:
                continue

            candidate = osm.loc[bus_id]

            # If duplicate candidates exist, just take the first for now.
            if isinstance(candidate, pd.DataFrame):
                candidate = candidate.iloc[0]

            lon = candidate["osm_lon"]
            lat = candidate["osm_lat"]

            # Final Ireland sanity check.
            if not (
                -11 <= lon <= -5
                and 51 <= lat <= 56
            ):
                continue

            buses.at[bus_id, "x"] = lon
            buses.at[bus_id, "y"] = lat

            # Optional: record where the coordinate came from
            buses.at[bus_id, "coordinate_source"] = "OSM candidate"

            filled_count += 1

        print(
            f"Filled {filled_count} bus coordinates "
            "from OSM candidates."
        )

else:
    print(
        f"No OSM candidate file found at:\n"
        f"{OSM_CANDIDATES_FILE}"
    )



valid_buses = buses[
    buses["x"].notna()
    & buses["y"].notna()
    & buses["x"].between(-11, -5)
    & buses["y"].between(51, 56)
].copy()

missing_buses = buses.index.difference(valid_buses.index)

print(f"Buses with coordinates: {len(valid_buses):,}")
print(f"Buses missing coordinates: {len(missing_buses):,}")

# Convert indices to strings to avoid CSV type differences such as 1021 vs "1021".
valid_buses.index = valid_buses.index.astype(str)

lines = n.lines.copy()
lines.index = lines.index.astype(str)
lines["bus0"] = lines["bus0"].astype(str)
lines["bus1"] = lines["bus1"].astype(str)

transformers = n.transformers.copy()
transformers.index = transformers.index.astype(str)
transformers["bus0"] = transformers["bus0"].astype(str)
transformers["bus1"] = transformers["bus1"].astype(str)

generators = n.generators.copy()
generators.index = generators.index.astype(str)
generators["bus"] = generators["bus"].astype(str)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def bus_voltage(bus_name):
    """Return nominal bus voltage, or NaN if unavailable."""
    if bus_name not in valid_buses.index:
        return math.nan
    return valid_buses.at[bus_name, "v_nom"]


def connection_voltage(bus0, bus1):
    """
    Pick the higher nominal voltage of the two connected buses.
    Useful for styling a transmission connection.
    """
    voltages = [bus_voltage(bus0), bus_voltage(bus1)]
    voltages = [v for v in voltages if pd.notna(v)]
    return max(voltages) if voltages else math.nan


def nearest_voltage_style(v):
    """Map an arbitrary voltage to one of our display categories."""
    levels = [110, 220, 275, 400]
    if pd.isna(v):
        return 110
    return min(levels, key=lambda level: abs(level - float(v)))


# ---------------------------------------------------------------------
# STATIC MATPLOTLIB MAP
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 13))

# Transmission lines
for _, line in lines.iterrows():
    bus0 = line["bus0"]
    bus1 = line["bus1"]

    if bus0 not in valid_buses.index or bus1 not in valid_buses.index:
        continue

    x0, y0 = valid_buses.loc[bus0, ["x", "y"]]
    x1, y1 = valid_buses.loc[bus1, ["x", "y"]]

    level = nearest_voltage_style(connection_voltage(bus0, bus1))

    ax.plot(
        [x0, x1],
        [y0, y1],
        linewidth=LINE_WIDTH[level],
        alpha=0.65,
        zorder=1,
    )

# Transformers
if SHOW_TRANSFORMERS:
    for _, trafo in transformers.iterrows():
        bus0 = trafo["bus0"]
        bus1 = trafo["bus1"]

        if bus0 not in valid_buses.index or bus1 not in valid_buses.index:
            continue

        x0, y0 = valid_buses.loc[bus0, ["x", "y"]]
        x1, y1 = valid_buses.loc[bus1, ["x", "y"]]

        ax.plot(
            [x0, x1],
            [y0, y1],
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
            zorder=2,
        )

# Buses, separated by voltage so the legend is useful
for voltage in sorted(valid_buses["v_nom"].dropna().unique()):
    subset = valid_buses[valid_buses["v_nom"] == voltage]

    size = BUS_SIZE.get(
        int(round(voltage)),
        8,
    )

    ax.scatter(
        subset["x"],
        subset["y"],
        s=size,
        label=f"{voltage:g} kV buses",
        alpha=0.8,
        zorder=3,
    )

# Generators: marker size scales approximately with installed capacity
if SHOW_GENERATORS and not generators.empty:
    gen_rows = []

    for name, gen in generators.iterrows():
        bus = gen["bus"]

        if bus not in valid_buses.index:
            continue

        p_nom = pd.to_numeric(pd.Series([gen.get("p_nom")]), errors="coerce").iloc[0]
        if pd.isna(p_nom):
            p_nom = 0.0

        gen_rows.append(
            {
                "name": name,
                "bus": bus,
                "x": valid_buses.at[bus, "x"],
                "y": valid_buses.at[bus, "y"],
                "p_nom": p_nom,
                "carrier": gen.get("carrier", ""),
            }
        )

    gen_map = pd.DataFrame(gen_rows)

    if not gen_map.empty:
        # sqrt scaling avoids enormous markers for very large plants
        marker_sizes = 8 + 2.5 * gen_map["p_nom"].clip(lower=0).pow(0.5)

        ax.scatter(
            gen_map["x"],
            gen_map["y"],
            s=marker_sizes,
            marker="^",
            alpha=0.55,
            label="Generators",
            zorder=4,
        )

ax.set_title(f"{n.name}\nPyPSA transmission network")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal", adjustable="datalim")
ax.grid(True, linewidth=0.3, alpha=0.4)
ax.legend(loc="best", fontsize=8)

plt.tight_layout()
plt.savefig(STATIC_OUTPUT, dpi=250, bbox_inches="tight")
print(f"Saved static map: {STATIC_OUTPUT}")


# ---------------------------------------------------------------------
# OPTIONAL INTERACTIVE FOLIUM MAP
# ---------------------------------------------------------------------

try:
    import folium

    centre_lat = valid_buses["y"].median()
    centre_lon = valid_buses["x"].median()

    fmap = folium.Map(
    location=[centre_lat, centre_lon],
    zoom_start=7,
    tiles=None,
    )

    folium.TileLayer(
    tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr="© OpenStreetMap contributors",
    name="OpenStreetMap",
    max_zoom=19,
    referrer_policy="strict-origin-when-cross-origin",
    ).add_to(fmap)

    # Lines
    for line_name, line in lines.iterrows():
        bus0 = line["bus0"]
        bus1 = line["bus1"]

        if bus0 not in valid_buses.index or bus1 not in valid_buses.index:
            continue

        b0 = valid_buses.loc[bus0]
        b1 = valid_buses.loc[bus1]

        voltage = connection_voltage(bus0, bus1)
        s_nom = line.get("s_nom", "")
        length = line.get("length", "")

        popup = (
            f"<b>Line:</b> {line_name}<br>"
            f"<b>Bus 0:</b> {bus0}<br>"
            f"<b>Bus 1:</b> {bus1}<br>"
            f"<b>Voltage:</b> {voltage:g} kV<br>"
            f"<b>Rating:</b> {s_nom} MVA<br>"
            f"<b>Length:</b> {length}"
        )

        folium.PolyLine(
            locations=[
                [b0["y"], b0["x"]],
                [b1["y"], b1["x"]],
            ],
            weight=max(1.0, LINE_WIDTH[nearest_voltage_style(voltage)]),
            opacity=0.65,
            popup=folium.Popup(popup, max_width=350),
        ).add_to(fmap)

    # Transformers
    if SHOW_TRANSFORMERS:
        for trafo_name, trafo in transformers.iterrows():
            bus0 = trafo["bus0"]
            bus1 = trafo["bus1"]

            if bus0 not in valid_buses.index or bus1 not in valid_buses.index:
                continue

            b0 = valid_buses.loc[bus0]
            b1 = valid_buses.loc[bus1]

            popup = (
                f"<b>Transformer:</b> {trafo_name}<br>"
                f"<b>Bus 0:</b> {bus0}<br>"
                f"<b>Bus 1:</b> {bus1}<br>"
                f"<b>Rating:</b> {trafo.get('s_nom', '')} MVA<br>"
                f"<b>Tap ratio:</b> {trafo.get('tap_ratio', '')}"
            )

            folium.PolyLine(
                locations=[
                    [b0["y"], b0["x"]],
                    [b1["y"], b1["x"]],
                ],
                weight=2,
                opacity=0.55,
                dash_array="5,5",
                popup=folium.Popup(popup, max_width=350),
            ).add_to(fmap)

    # Buses
    for bus_name, bus in valid_buses.iterrows():
        station = bus.get("station", "")
        psse_name = bus.get("psse_name", "")
        voltage = bus.get("v_nom", "")
        geocode_method = bus.get("geocode_method", "")

        popup = (
            f"<b>Bus:</b> {bus_name}<br>"
            f"<b>Station:</b> {station}<br>"
            f"<b>PSS/E name:</b> {psse_name}<br>"
            f"<b>Voltage:</b> {voltage} kV<br>"
            f"<b>Geocode:</b> {geocode_method}<br>"
            f"<b>Longitude:</b> {bus['x']:.6f}<br>"
            f"<b>Latitude:</b> {bus['y']:.6f}"
        )

        radius = max(2.5, BUS_SIZE.get(int(round(voltage)), 8) / 4)

        folium.CircleMarker(
            location=[bus["y"], bus["x"]],
            radius=radius,
            fill=True,
            fill_opacity=0.7,
            weight=1,
            popup=folium.Popup(popup, max_width=350),
            tooltip=f"{station or psse_name or bus_name} — {voltage:g} kV",
        ).add_to(fmap)

    # Generators
    if SHOW_GENERATORS:
        for gen_name, gen in generators.iterrows():
            bus = gen["bus"]

            if bus not in valid_buses.index:
                continue

            b = valid_buses.loc[bus]
            p_nom = gen.get("p_nom", "")
            carrier = gen.get("carrier", "")

            popup = (
                f"<b>Generator:</b> {gen_name}<br>"
                f"<b>Bus:</b> {bus}<br>"
                f"<b>Station:</b> {b.get('station', '')}<br>"
                f"<b>Carrier:</b> {carrier}<br>"
                f"<b>Capacity:</b> {p_nom} MW"
            )

            folium.Marker(
                location=[b["y"], b["x"]],
                popup=folium.Popup(popup, max_width=350),
                tooltip=f"{carrier}: {p_nom} MW",
            ).add_to(fmap)

    fmap.save(INTERACTIVE_OUTPUT)
    print(f"Saved interactive map: {INTERACTIVE_OUTPUT}")
    # Patch Folium-generated HTML so Leaflet sends the required Referer.
    html = Path(INTERACTIVE_OUTPUT).read_text(encoding="utf-8")

    old = '"opacity": 1,'
    new = '"opacity": 1,\n  "referrerPolicy": "strict-origin-when-cross-origin",'

    html = html.replace(old, new, 1)

    Path(INTERACTIVE_OUTPUT).write_text(html, encoding="utf-8")

    print("Added Leaflet referrerPolicy to tile layer.")


except ImportError:
    print(
        "Folium is not installed, so only the PNG was created.\n"
        "Install it with: pip install folium"
    )


# ---------------------------------------------------------------------
# REPORT BUSES NOT MAPPED
# ---------------------------------------------------------------------

not_mapped = buses.loc[missing_buses].copy()

if not not_mapped.empty:

    def coordinate_reason(row):
        # Missing both
        if pd.isna(row["x"]) and pd.isna(row["y"]):
            return "missing longitude and latitude"

        # Missing longitude only
        if pd.isna(row["x"]):
            return "missing longitude"

        # Missing latitude only
        if pd.isna(row["y"]):
            return "missing latitude"

        # Explicit (0, 0)
        if row["x"] == 0 and row["y"] == 0:
            return "coordinates are (0, 0)"

        # Outside our Ireland map bounds
        if not (-11 <= row["x"] <= -5 and 51 <= row["y"] <= 56):
            return "outside Ireland bounds"

        return "unknown"

    not_mapped["reason"] = not_mapped.apply(
        coordinate_reason,
        axis=1
    )

    columns = [
        col for col in [
            "psse_name",
            "station",
            "v_nom",
            "x",
            "y",
            "geocode_method",
            "reason",
        ]
        if col in not_mapped.columns
    ]

    output_file = OUTPUT_DIR / "buses_not_mapped_new.csv"

    not_mapped[columns].to_csv(output_file)

    print(f"Saved unmapped bus report: {output_file}")

    # Useful summary in terminal
    print("\nReasons buses were not mapped:")
    print(not_mapped["reason"].value_counts())