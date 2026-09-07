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

DATA_DIR = Path("/home/seba/Documents/Colleg/Hackathon_2/Hackathons-main/grid_TF_Wind/data/pypsa/TYTFS2024_WP2033_V35_transmission")       # folder containing buses.csv, lines.csv, etc.
STATIC_OUTPUT = "pypsa_network_map.png"
INTERACTIVE_OUTPUT = "pypsa_network_map.html"

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
        tiles="CartoDB positron",
    )

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

except ImportError:
    print(
        "Folium is not installed, so only the PNG was created.\n"
        "Install it with: pip install folium"
    )


# ---------------------------------------------------------------------
# REPORT MISSING GEOCODES
# ---------------------------------------------------------------------

if len(missing_buses):
    missing = buses.loc[missing_buses].copy()

    columns = [
        col
        for col in [
            "v_nom",
            "psse_name",
            "station",
            "jurisdiction",
            "geocode_method",
        ]
        if col in missing.columns
    ]

    missing[columns].to_csv("buses_missing_coordinates.csv")
    print("Saved missing-coordinate report: buses_missing_coordinates.csv")
