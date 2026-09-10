import h5py
import numpy as np
import geopandas as gpd
import folium
from folium import FeatureGroup
from folium.plugins import Fullscreen, MousePosition
import pypsa
from shapely.geometry import Point



# ============================================================
# Files
# ============================================================

NC_FILE = (
    "/home/seba/Documents/Colleg/Hackathon_2/"
    "EIEG_Hackathon26/cian/EIEG_Hackathon26/"
    "data/SV2024_all-island.nc"
)

BASEMAP_FILE = (
    "/home/seba/Documents/Colleg/Hackathon_2/"
    "EIEG_Hackathon26/ireland_all_island.geojson"
)

OUTPUT_HTML = "SV2024_interactive_network_mk2_NI.html"


# ============================================================
# Helpers
# ============================================================

def decode_value(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.generic):
        return value.item()

    return value


def decode_array(values):
    return [decode_value(v) for v in values]


def voltage_colour(voltage):
    if voltage >= 400:
        return "#7b1fa2"
    elif voltage >= 275:
        return "#d32f2f"
    elif voltage >= 200:
        return "#f57c00"
    elif voltage >= 100:
        return "#1976d2"
    else:
        return "#388e3c"


def clean_value(value):
    value = decode_value(value)

    if value is None:
        return None

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

        # Neater numbers in popup
        if value.is_integer():
            return int(value)

        return round(value, 4)

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return None

    return value


def filter_network_to_region(
    network,
    regions_gdf,
    region_name,
    candidate_columns=("GEOUNIT", "NAME", "NAME_LONG", "ADMIN", "SOVEREIGNT")
):
    """
    Return a PyPSA subnetwork containing only buses inside a named region.

    Parameters
    ----------
    network : pypsa.Network
        Loaded PyPSA network.

    regions_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the regional polygons.

    region_name : str
        Region to select, e.g. "Northern Ireland".

    candidate_columns : tuple
        Columns to search for the region name.

    Returns
    -------
    pypsa.Network
        Filtered PyPSA network.

    geopandas.GeoDataFrame
        The selected region polygon.
    """

    regions = regions_gdf.to_crs("EPSG:4326").copy()

    region = None

    # Find region in likely name columns
    for column in candidate_columns:

        if column not in regions.columns:
            continue

        mask = (
            regions[column]
            .astype(str)
            .str.fullmatch(
                region_name,
                case=False,
                na=False
            )
        )

        if mask.any():
            region = regions.loc[mask].copy()

            print(
                f"Found {region_name} "
                f"using column '{column}'"
            )

            break

    if region is None:
        raise ValueError(
            f"Could not find '{region_name}' "
            f"in the supplied GeoJSON."
        )

    # Combine multiple polygons if necessary
    region_geometry = region.geometry.union_all()

    # Turn PyPSA buses into geographic points
    buses_geo = gpd.GeoDataFrame(
        network.buses.copy(),
        geometry=[
            Point(x, y)
            for x, y in zip(
                network.buses["x"],
                network.buses["y"]
            )
        ],
        crs="EPSG:4326"
    )

    # Select buses geographically inside NI
    inside = buses_geo.geometry.apply(
        region_geometry.covers
    )

    selected_buses = buses_geo.index[
        inside
    ].tolist()

    print(
        f"{len(selected_buses)} buses "
        f"found inside {region_name}"
    )

    # PyPSA creates a subnetwork using these buses
    filtered_network = network.slice_network(
        buses=selected_buses
    )

    return filtered_network, region



# ============================================================
# Read the .nc file
# ============================================================

print("Reading network...")


with h5py.File(NC_FILE, "r") as f:

    bus_ids = decode_array(f["buses_i"][:])

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


    # --------------------------------------------------------
    # Lines
    # --------------------------------------------------------

    line_ids = decode_array(
        f["lines_i"][:]
    )

    line_bus0 = decode_array(
        f["lines_bus0"][:]
    )

    line_bus1 = decode_array(
        f["lines_bus1"][:]
    )


    # --------------------------------------------------------
    # Automatically find ALL bus attributes
    # --------------------------------------------------------

    bus_attributes = {}

    for name in f.keys():

        if not name.startswith("buses_"):
            continue

        if name in [
            "buses_i",
            "buses_x",
            "buses_y"
        ]:
            continue

        try:
            data = f[name][:]

            if len(data) == len(bus_ids):

                key = name.removeprefix("buses_")

                bus_attributes[key] = decode_array(
                    data
                )

        except Exception:
            pass


    # --------------------------------------------------------
    # Automatically find ALL line attributes
    # --------------------------------------------------------

    line_attributes = {}

    for name in f.keys():

        if not name.startswith("lines_"):
            continue

        if name in [
            "lines_i",
            "lines_bus0",
            "lines_bus1"
        ]:
            continue

        try:
            data = f[name][:]

            if len(data) == len(line_ids):

                key = name.removeprefix("lines_")

                line_attributes[key] = decode_array(
                    data
                )

        except Exception:
            pass


print(
    f"Found {len(bus_attributes)} bus attributes"
)

print(
    f"Found {len(line_attributes)} line attributes"
)


# ============================================================
# Build bus lookup
# ============================================================

bus_lookup = {}


for i, bus_id in enumerate(bus_ids):

    if not (
        np.isfinite(bus_lon[i])
        and np.isfinite(bus_lat[i])
    ):
        continue

    bus_lookup[bus_id] = {
        "index": i,
        "lat": float(bus_lat[i]),
        "lon": float(bus_lon[i]),
        "voltage": float(bus_voltage[i])
    }


print(
    f"Buses with coordinates: {len(bus_lookup)}"
)


# ============================================================
# Load network
# ============================================================

n = pypsa.Network(NC_FILE)

basemap = gpd.read_file(
    BASEMAP_FILE
)

n, ni_boundary = filter_network_to_region(
    network=n,
    regions_gdf=basemap,
    region_name="Northern Ireland"
)

print(n)

# ============================================================
# Apply PyPSA NI selection to existing map data
# ============================================================

ni_bus_ids = set(n.buses.index)

bus_lookup = {
    bus_id: bus
    for bus_id, bus in bus_lookup.items()
    if bus_id in ni_bus_ids
}

NI_OUTPUT_NC = "SV2024_northern_ireland.nc"

n.export_to_netcdf(NI_OUTPUT_NC)

print(f"Saved NI-only network to: {NI_OUTPUT_NC}")

print(
    f"Buses remaining for map: {len(bus_lookup)}"
)

# ============================================================
# Load local Ireland outline
# ============================================================

ireland = ni_boundary

# ============================================================
# Create map
# ============================================================

m = folium.Map(
    location=[
    54.65,
    -6.8
],

zoom_start=8,


    tiles=None,

    control_scale=True,

    prefer_canvas=True
)


# ============================================================
# Add proper web-map basemap
#
# NOT OpenStreetMap tiles.
# Uses Esri World Street Map instead.
# ============================================================

folium.TileLayer(

    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Street_Map/MapServer/"
        "tile/{z}/{y}/{x}"
    ),

    attr=(
        "Tiles &copy; Esri — "
        "Source: Esri, HERE, Garmin, "
        "USGS, Intermap, INCREMENT P, "
        "NRCan, Esri Japan, METI, "
        "Esri China (Hong Kong), "
        "NGA, OpenStreetMap contributors, "
        "and the GIS User Community"
    ),

    name="Street Map",

    overlay=False,

    control=True,

    show=True

).add_to(m)


# ============================================================
# Optional clean/light basemap
# ============================================================

folium.TileLayer(

    tiles=(
        "https://{s}.basemaps.cartocdn.com/"
        "light_all/{z}/{x}/{y}{r}.png"
    ),

    attr=(
        "&copy; OpenStreetMap contributors "
        "&copy; CARTO"
    ),

    name="Light Map",

    overlay=False,

    control=True,

    show=False

).add_to(m)


# ============================================================
# Local Ireland boundary
# ============================================================

boundary_layer = FeatureGroup(
    name="Ireland boundary",
    show=True
)


folium.GeoJson(

    ireland,

    style_function=lambda feature: {

        "fillColor": "#eeeeee",

        "color": "#444444",

        "weight": 1.5,

        "fillOpacity": 0.08
    },

    name="Ireland boundary"

).add_to(boundary_layer)


boundary_layer.add_to(m)


# ============================================================
# Voltage groups
# ============================================================

line_layers = {

    "400+ kV":
        FeatureGroup(
            name="400+ kV lines",
            show=True
        ),

    "275–399 kV":
        FeatureGroup(
            name="275–399 kV lines",
            show=True
        ),

    "200–274 kV":
        FeatureGroup(
            name="200–274 kV lines",
            show=True
        ),

    "100–199 kV":
        FeatureGroup(
            name="100–199 kV lines",
            show=True
        ),

    "<100 kV":
        FeatureGroup(
            name="<100 kV lines",
            show=True
        )
}


def voltage_group(v):

    if v >= 400:
        return "400+ kV"

    if v >= 275:
        return "275–399 kV"

    if v >= 200:
        return "200–274 kV"

    if v >= 100:
        return "100–199 kV"

    return "<100 kV"


# ============================================================
# Add transmission lines
# ============================================================

for i, (
    line_id,
    bus0,
    bus1
) in enumerate(
    zip(
        line_ids,
        line_bus0,
        line_bus1
    )
):

    if (
        bus0 not in bus_lookup
        or bus1 not in bus_lookup
    ):
        continue


    b0 = bus_lookup[bus0]
    b1 = bus_lookup[bus1]


    voltage = max(
        b0["voltage"],
        b1["voltage"]
    )


    # --------------------------------------------------------
    # Popup
    # --------------------------------------------------------

    popup = f"""
    <div style="
        font-family: Arial;
        font-size: 13px;
        min-width: 250px;
    ">

    <h4 style="margin-bottom:8px;">
        Transmission Line
    </h4>

    <b>ID:</b> {line_id}<br>
    <b>From:</b> {bus0}<br>
    <b>To:</b> {bus1}<br>
    <b>Voltage:</b> {voltage:g} kV<br>
    """


    # Add every NC line attribute

    for attribute, values in line_attributes.items():

        value = clean_value(
            values[i]
        )

        if value is None:
            continue

        popup += (
            f"<b>{attribute}:</b> "
            f"{value}<br>"
        )


    popup += "</div>"


    group = voltage_group(
        voltage
    )


    folium.PolyLine(

        locations=[
            [
                b0["lat"],
                b0["lon"]
            ],
            [
                b1["lat"],
                b1["lon"]
            ]
        ],

        color=voltage_colour(
            voltage
        ),

        weight=3,

        opacity=0.8,

        tooltip=(
            f"{line_id} "
            f"({voltage:g} kV)"
        ),

        popup=folium.Popup(
            popup,
            max_width=400
        )

    ).add_to(
        line_layers[group]
    )


# Add line groups to map

for layer in line_layers.values():
    layer.add_to(m)


# ============================================================
# Bus layer
# ============================================================

bus_layer = FeatureGroup(
    name="Buses / substations",
    show=True
)


for bus_id, bus in bus_lookup.items():

    i = bus["index"]


    # --------------------------------------------------------
    # Generate popup automatically
    # --------------------------------------------------------

    popup = f"""
    <div style="
        font-family: Arial;
        font-size: 13px;
        min-width: 280px;
    ">

    <h3 style="
        margin-top:0;
        margin-bottom:8px;
    ">
        {bus_id}
    </h3>

    <table style="
        border-collapse:collapse;
        width:100%;
    ">

    <tr>
        <td><b>Latitude</b></td>
        <td>{bus["lat"]:.5f}</td>
    </tr>

    <tr>
        <td><b>Longitude</b></td>
        <td>{bus["lon"]:.5f}</td>
    </tr>
    """


    # --------------------------------------------------------
    # Every bus attribute from the .nc file
    # --------------------------------------------------------

    for attribute, values in bus_attributes.items():

        value = clean_value(
            values[i]
        )

        if value is None:
            continue


        nice_name = (
            attribute
            .replace("_", " ")
            .title()
        )


        popup += f"""
        <tr>
            <td style="
                padding-right:12px;
            ">
                <b>{nice_name}</b>
            </td>

            <td>
                {value}
            </td>
        </tr>
        """


    popup += """
    </table>
    </div>
    """


    # --------------------------------------------------------
    # Bus marker
    # --------------------------------------------------------

    folium.CircleMarker(

        location=[
            bus["lat"],
            bus["lon"]
        ],

        radius=5,

        color="#222222",

        weight=1,

        fill=True,

        fill_color=voltage_colour(
            bus["voltage"]
        ),

        fill_opacity=0.95,

        tooltip=(
            f"{bus_id} — "
            f"{bus['voltage']:g} kV"
        ),

        popup=folium.Popup(
            popup,
            max_width=450
        )

    ).add_to(
        bus_layer
    )


bus_layer.add_to(m)


# ============================================================
# Fit map to network
# ============================================================

all_lats = [
    bus["lat"]
    for bus in bus_lookup.values()
]

all_lons = [
    bus["lon"]
    for bus in bus_lookup.values()
]


m.fit_bounds([
    [
        min(all_lats),
        min(all_lons)
    ],
    [
        max(all_lats),
        max(all_lons)
    ]
])


# ============================================================
# Map controls
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(m)


Fullscreen(
    position="topright"
).add_to(m)


MousePosition(

    position="bottomleft",

    separator=" | ",

    prefix="Coordinates:"

).add_to(m)


# ============================================================
# Voltage legend
# ============================================================

legend = """
<div style="
    position: fixed;
    bottom: 40px;
    right: 20px;
    width: 170px;

    background-color: white;

    border: 2px solid #777;
    border-radius: 6px;

    z-index: 9999;

    font-size: 13px;
    font-family: Arial;

    padding: 10px;

    box-shadow: 0 1px 6px rgba(0,0,0,0.3);
">

<b>Nominal Voltage</b>

<br><br>

<span style="
    display:inline-block;
    width:25px;
    height:4px;
    background:#7b1fa2;
"></span>
&nbsp; 400+ kV

<br>

<span style="
    display:inline-block;
    width:25px;
    height:4px;
    background:#d32f2f;
"></span>
&nbsp; 275–399 kV

<br>

<span style="
    display:inline-block;
    width:25px;
    height:4px;
    background:#f57c00;
"></span>
&nbsp; 200–274 kV

<br>

<span style="
    display:inline-block;
    width:25px;
    height:4px;
    background:#1976d2;
"></span>
&nbsp; 100–199 kV

<br>

<span style="
    display:inline-block;
    width:25px;
    height:4px;
    background:#388e3c;
"></span>
&nbsp; &lt;100 kV

</div>
"""


m.get_root().html.add_child(
    folium.Element(legend)
)


# ============================================================
# Referrer policy
#
# Allows normal browser Referer behaviour for cross-origin
# tile requests without sending the full page path.
# ============================================================

referrer_policy = """
<meta
    name="referrer"
    content="strict-origin-when-cross-origin"
>
"""


m.get_root().header.add_child(
    folium.Element(
        referrer_policy
    )
)


# ============================================================
# Title
# ============================================================

title_html = """
<div style="
    position: fixed;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);

    z-index: 9999;

    background: rgba(255,255,255,0.92);

    padding: 8px 18px;

    border-radius: 7px;

    box-shadow: 0 1px 5px rgba(0,0,0,0.25);

    font-family: Arial;
">

<b style="font-size:18px;">
SV2024 All-Island Transmission Network
</b>

</div>
"""


m.get_root().html.add_child(
    folium.Element(title_html)
)


# ============================================================
# Save
# ============================================================

m.save(
    OUTPUT_HTML
)


print()
print(
    "Interactive map created:"
)

print(
    OUTPUT_HTML
)