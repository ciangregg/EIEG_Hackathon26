import h5py
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


NC_FILE = "/home/seba/Documents/Colleg/Hackathon_2/EIEG_Hackathon26/cian/EIEG_Hackathon26/data/SV2024_all-island.nc"

# Local basemap file, e.g. Ireland + Northern Ireland boundaries
BASEMAP_FILE = "/home/seba/Documents/Colleg/Hackathon_2/EIEG_Hackathon26/ireland_all_island.geojson"

OUTPUT_PNG = "WP2024_network_map.png"


def decode_strings(values):
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_))
        else str(value)
        for value in values
    ]


# ------------------------------------------------------------
# Read network
# ------------------------------------------------------------

with h5py.File(NC_FILE, "r") as f:

    bus_ids = decode_strings(f["buses_i"][:])

    bus_lon = np.asarray(
        f["buses_x"][:],
        dtype=float
    )

    bus_lat = np.asarray(
        f["buses_y"][:],
        dtype=float
    )

    bus_voltage = np.asarray(
        f["buses_v_nom"][:],
        dtype=float
    )

    line_bus0 = decode_strings(
        f["lines_bus0"][:]
    )

    line_bus1 = decode_strings(
        f["lines_bus1"][:]
    )


# ------------------------------------------------------------
# Bus lookup
# ------------------------------------------------------------

buses = {}

for i, bus_id in enumerate(bus_ids):

    if np.isfinite(bus_lon[i]) and np.isfinite(bus_lat[i]):

        buses[bus_id] = {
            "lon": bus_lon[i],
            "lat": bus_lat[i],
            "voltage": bus_voltage[i]
        }


# ------------------------------------------------------------
# Load offline basemap
# ------------------------------------------------------------

basemap = gpd.read_file(BASEMAP_FILE)

# Network coordinates are longitude/latitude
# so convert basemap to WGS84 if necessary

basemap = basemap.to_crs("EPSG:4326")


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 12)
)


# Country outline

basemap.plot(
    ax=ax,
    facecolor="#eeeeee",
    edgecolor="#555555",
    linewidth=0.8
)


# ------------------------------------------------------------
# Voltage colour
# ------------------------------------------------------------

def voltage_colour(voltage):

    if voltage >= 400:
        return "purple"

    if voltage >= 275:
        return "red"

    if voltage >= 200:
        return "orange"

    if voltage >= 100:
        return "blue"

    return "green"


# ------------------------------------------------------------
# Transmission lines
# ------------------------------------------------------------

for bus0, bus1 in zip(
    line_bus0,
    line_bus1
):

    if bus0 not in buses or bus1 not in buses:
        continue

    b0 = buses[bus0]
    b1 = buses[bus1]

    voltage = max(
        b0["voltage"],
        b1["voltage"]
    )

    ax.plot(
        [
            b0["lon"],
            b1["lon"]
        ],
        [
            b0["lat"],
            b1["lat"]
        ],
        color=voltage_colour(voltage),
        linewidth=1.2,
        alpha=0.75
    )


# ------------------------------------------------------------
# Buses
# ------------------------------------------------------------

for bus in buses.values():

    ax.scatter(
        bus["lon"],
        bus["lat"],
        s=6,
        color=voltage_colour(
            bus["voltage"]
        ),
        zorder=5
    )


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.set_title(
    "WP2033 All-Island Transmission Network",
    fontsize=16
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_aspect("equal")

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=300
)

plt.show()