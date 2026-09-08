import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import FeatureGroup
from folium.plugins import Fullscreen, MousePosition
import pypsa

BEFORE_NC = (
    "/home/seba/Documents/Colleg/Hackathon_2/"
    "EIEG_Hackathon26/cian/EIEG_Hackathon26/"
    "data/SV2024_northern_ireland.nc"
)

AFTER_NC = (
    "/home/seba/Documents/Colleg/Hackathon_2/"
    "EIEG_Hackathon26/cian/EIEG_Hackathon26/"
    "data/SV2024_NI_after_optimization.nc"
)

BASEMAP_FILE = (
    "/home/seba/Documents/Colleg/Hackathon_2/"
    "EIEG_Hackathon26/ireland_all_island.geojson"
)

OUTPUT_HTML = "SV2024_NI_before_after_difference_mk2.html"
SNAPSHOT_INDEX = 0
CHANGE_TOLERANCE_MW = 1.0


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        if value.is_integer():
            return int(value)
        return round(value, 4)
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    return value


def voltage_colour(voltage):
    if voltage >= 400:
        return "#7b1fa2"
    elif voltage >= 275:
        return "#d32f2f"
    elif voltage >= 200:
        return "#f57c00"
    elif voltage >= 100:
        return "#1976d2"
    return "#388e3c"


def flow_width(flow_mw):
    return 1.5 + min(abs(float(flow_mw)) / 50.0, 8.0)


def change_width(change_mw):
    return 2.0 + min(abs(float(change_mw)) / 20.0, 8.0)


def difference_colour(change_abs_mw, tolerance=CHANGE_TOLERANCE_MW):
    if change_abs_mw > tolerance:
        return "#d73027"   # increased loading
    elif change_abs_mw < -tolerance:
        return "#4575b4"   # decreased loading
    return "#777777"


def safe_timeseries_value(df, snapshot, component_id, default=0.0):
    if df is None or df.empty:
        return float(default)
    if snapshot not in df.index or component_id not in df.columns:
        return float(default)
    value = df.at[snapshot, component_id]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def select_northern_ireland_boundary(
    regions_gdf,
    region_name="Northern Ireland",
    candidate_columns=("GEOUNIT", "NAME", "NAME_LONG", "ADMIN", "SOVEREIGNT")
):
    regions = regions_gdf.to_crs("EPSG:4326").copy()
    for column in candidate_columns:
        if column not in regions.columns:
            continue
        mask = (
            regions[column]
            .astype(str)
            .str.fullmatch(region_name, case=False, na=False)
        )
        if mask.any():
            print(f"Found {region_name} using column '{column}'")
            return regions.loc[mask].copy()
    raise ValueError(f"Could not find '{region_name}' in the supplied GeoJSON.")


def aggregate_generation_by_bus(network, snapshot):
    result = pd.Series(0.0, index=network.buses.index, dtype=float)
    if network.generators.empty:
        return result
    p = network.generators_t.p
    if p.empty or snapshot not in p.index:
        return result
    values = p.loc[snapshot].reindex(network.generators.index).fillna(0.0)
    by_bus = values.groupby(network.generators["bus"]).sum()
    idx = result.index.intersection(by_bus.index)
    result.loc[idx] = by_bus.reindex(idx)
    return result


def aggregate_load_by_bus(network, snapshot):
    result = pd.Series(0.0, index=network.buses.index, dtype=float)
    if network.loads.empty:
        return result

    values = None
    if (
        hasattr(network.loads_t, "p")
        and not network.loads_t.p.empty
        and snapshot in network.loads_t.p.index
    ):
        values = network.loads_t.p.loc[snapshot]
    elif (
        hasattr(network.loads_t, "p_set")
        and not network.loads_t.p_set.empty
        and snapshot in network.loads_t.p_set.index
    ):
        values = network.loads_t.p_set.loc[snapshot]

    if values is None:
        values = network.loads["p_set"]

    values = values.reindex(network.loads.index).fillna(0.0)
    by_bus = values.groupby(network.loads["bus"]).sum()
    idx = result.index.intersection(by_bus.index)
    result.loc[idx] = by_bus.reindex(idx)
    return result


def generator_popup(generator_id, before, after, snapshot):
    before_p = safe_timeseries_value(before.generators_t.p, snapshot, generator_id)
    after_p = safe_timeseries_value(after.generators_t.p, snapshot, generator_id)
    change = after_p - before_p

    if generator_id in after.generators.index:
        row = after.generators.loc[generator_id]
    else:
        row = before.generators.loc[generator_id]

    carrier = clean_value(row.get("carrier", ""))
    bus = clean_value(row.get("bus", ""))
    p_nom = clean_value(row.get("p_nom", np.nan))

    return f"""
    <div style="font-family:Arial;font-size:13px;min-width:260px;">
        <h4 style="margin:0 0 8px 0;">Generator</h4>
        <b>ID:</b> {generator_id}<br>
        <b>Bus:</b> {bus}<br>
        <b>Carrier:</b> {carrier}<br>
        <b>Nominal capacity:</b> {p_nom} MW<br>
        <hr>
        <b>Snapshot:</b> {snapshot}<br>
        <b>Before:</b> {before_p:.2f} MW<br>
        <b>After:</b> {after_p:.2f} MW<br>
        <b>Change:</b> {change:+.2f} MW
    </div>
    """


print("Loading BEFORE network...")
n_before = pypsa.Network(BEFORE_NC)

print("Loading AFTER network...")
n_after = pypsa.Network(AFTER_NC)

common_snapshots = n_before.snapshots.intersection(n_after.snapshots)
if len(common_snapshots) == 0:
    raise ValueError("The BEFORE and AFTER networks do not share any snapshots.")

if SNAPSHOT_INDEX < 0 or SNAPSHOT_INDEX >= len(common_snapshots):
    raise IndexError(
        f"SNAPSHOT_INDEX={SNAPSHOT_INDEX} is invalid. "
        f"Valid range is 0 to {len(common_snapshots) - 1}."
    )

snapshot = common_snapshots[SNAPSHOT_INDEX]
print(f"Comparing snapshot: {snapshot}")

common_buses = n_before.buses.index.intersection(n_after.buses.index)
common_lines = n_before.lines.index.intersection(n_after.lines.index)
print(f"Common buses: {len(common_buses)}")
print(f"Common lines: {len(common_lines)}")

bus_lookup = {}
for bus_id in common_buses:
    after_bus = n_after.buses.loc[bus_id]
    before_bus = n_before.buses.loc[bus_id]

    lon = after_bus.get("x", np.nan)
    lat = after_bus.get("y", np.nan)
    if not np.isfinite(lon) or not np.isfinite(lat):
        lon = before_bus.get("x", np.nan)
        lat = before_bus.get("y", np.nan)
    if not np.isfinite(lon) or not np.isfinite(lat):
        continue

    voltage = after_bus.get("v_nom", before_bus.get("v_nom", 0.0))
    bus_lookup[bus_id] = {
        "lat": float(lat),
        "lon": float(lon),
        "voltage": float(voltage),
    }

print(f"Buses with coordinates: {len(bus_lookup)}")

gen_before_by_bus = aggregate_generation_by_bus(n_before, snapshot)
gen_after_by_bus = aggregate_generation_by_bus(n_after, snapshot)
load_before_by_bus = aggregate_load_by_bus(n_before, snapshot)
load_after_by_bus = aggregate_load_by_bus(n_after, snapshot)

basemap = gpd.read_file(BASEMAP_FILE)
ni_boundary = select_northern_ireland_boundary(basemap)

m = folium.Map(
    location=[54.65, -6.8],
    zoom_start=8,
    tiles=None,
    control_scale=True,
    prefer_canvas=False,
)

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Street_Map/MapServer/"
        "tile/{z}/{y}/{x}"
    ),
    attr=(
        "Tiles &copy; Esri — Source: Esri, HERE, Garmin, USGS, Intermap, "
        "INCREMENT P, NRCan, Esri Japan, METI, Esri China (Hong Kong), "
        "NGA, OpenStreetMap contributors, and the GIS User Community"
    ),
    name="Street Map",
    overlay=False,
    control=True,
    show=True,
).add_to(m)

folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr="&copy; OpenStreetMap contributors &copy; CARTO",
    name="Light Map",
    overlay=False,
    control=True,
    show=False,
).add_to(m)

boundary_layer = FeatureGroup(name="Northern Ireland boundary", show=True)
folium.GeoJson(
    ni_boundary,
    style_function=lambda feature: {
        "fillColor": "#eeeeee",
        "color": "#444444",
        "weight": 1.5,
        "fillOpacity": 0.08,
    },
).add_to(boundary_layer)
boundary_layer.add_to(m)

before_layer = FeatureGroup(name="1 — Before optimisation", show=False)
after_layer = FeatureGroup(name="2 — After optimisation", show=True)
difference_layer = FeatureGroup(name="3 — Difference", show=False)

for line_id in common_lines:
    before_line = n_before.lines.loc[line_id]
    after_line = n_after.lines.loc[line_id]

    bus0 = after_line["bus0"]
    bus1 = after_line["bus1"]
    if bus0 not in bus_lookup or bus1 not in bus_lookup:
        continue

    b0 = bus_lookup[bus0]
    b1 = bus_lookup[bus1]
    voltage = max(b0["voltage"], b1["voltage"])

    before_flow = safe_timeseries_value(n_before.lines_t.p0, snapshot, line_id)
    after_flow = safe_timeseries_value(n_after.lines_t.p0, snapshot, line_id)
    signed_change = after_flow - before_flow
    loading_change = abs(after_flow) - abs(before_flow)

    coords = [[b0["lat"], b0["lon"]], [b1["lat"], b1["lon"]]]

    before_popup = f"""
    <div style="font-family:Arial;font-size:13px;min-width:270px;">
        <h4 style="margin:0 0 8px 0;">Before optimisation</h4>
        <b>Line:</b> {line_id}<br>
        <b>From:</b> {bus0}<br>
        <b>To:</b> {bus1}<br>
        <b>Voltage:</b> {voltage:g} kV<br>
        <b>Snapshot:</b> {snapshot}<br><hr>
        <b>Flow:</b> {before_flow:+.2f} MW<br>
        <b>|Flow|:</b> {abs(before_flow):.2f} MW
    </div>
    """

    folium.PolyLine(
        locations=coords,
        color=voltage_colour(voltage),
        weight=flow_width(before_flow),
        opacity=0.80,
        no_clip=True,
        smooth_factor=0.5,
        tooltip=f"{line_id} — BEFORE: {before_flow:+.1f} MW",
        popup=folium.Popup(before_popup, max_width=400),
    ).add_to(before_layer)

    after_popup = f"""
    <div style="font-family:Arial;font-size:13px;min-width:270px;">
        <h4 style="margin:0 0 8px 0;">After optimisation</h4>
        <b>Line:</b> {line_id}<br>
        <b>From:</b> {bus0}<br>
        <b>To:</b> {bus1}<br>
        <b>Voltage:</b> {voltage:g} kV<br>
        <b>Snapshot:</b> {snapshot}<br><hr>
        <b>Flow:</b> {after_flow:+.2f} MW<br>
        <b>|Flow|:</b> {abs(after_flow):.2f} MW
    </div>
    """

    folium.PolyLine(
        locations=coords,
        color=voltage_colour(voltage),
        weight=flow_width(after_flow),
        opacity=0.80,
        no_clip=True,
        smooth_factor=0.5,
        tooltip=f"{line_id} — AFTER: {after_flow:+.1f} MW",
        popup=folium.Popup(after_popup, max_width=400),
    ).add_to(after_layer)

    diff_popup = f"""
    <div style="font-family:Arial;font-size:13px;min-width:300px;">
        <h4 style="margin:0 0 8px 0;">Optimisation difference</h4>
        <b>Line:</b> {line_id}<br>
        <b>From:</b> {bus0}<br>
        <b>To:</b> {bus1}<br>
        <b>Voltage:</b> {voltage:g} kV<br>
        <b>Snapshot:</b> {snapshot}<br><hr>
        <b>Before:</b> {before_flow:+.2f} MW<br>
        <b>After:</b> {after_flow:+.2f} MW<br>
        <b>Signed change:</b> {signed_change:+.2f} MW<br>
        <b>Loading change:</b> {loading_change:+.2f} MW<br><br>
        <small>Red = increased |flow|<br>Blue = decreased |flow|<br>Grey = little/no change</small>
    </div>
    """

    folium.PolyLine(
        locations=coords,
        color=difference_colour(loading_change),
        weight=change_width(loading_change),
        opacity=0.88,
        no_clip=True,
        smooth_factor=0.5,
        tooltip=f"{line_id} — Δ loading: {loading_change:+.1f} MW",
        popup=folium.Popup(diff_popup, max_width=420),
    ).add_to(difference_layer)

before_layer.add_to(m)
after_layer.add_to(m)
difference_layer.add_to(m)

# ============================================================
# COMPONENT LAYERS
#
# Blue circles     = substations / buses
# Orange triangles = generators
# Green squares    = loads
#
# Generators and loads are offset slightly from their bus so
# all three component types remain visible when they share the
# same geographic coordinate.
# ============================================================

substation_layer = FeatureGroup(
    name="🔵 Substations / buses",
    show=True
)

generator_layer = FeatureGroup(
    name="🟠 Generators",
    show=True
)

load_layer = FeatureGroup(
    name="🟢 Loads",
    show=True
)


# ------------------------------------------------------------
# Substations / buses
# ------------------------------------------------------------

for bus_id, bus in bus_lookup.items():

    gen_before = float(gen_before_by_bus.get(bus_id, 0.0))
    gen_after = float(gen_after_by_bus.get(bus_id, 0.0))
    load_before = float(load_before_by_bus.get(bus_id, 0.0))
    load_after = float(load_after_by_bus.get(bus_id, 0.0))

    popup = f"""
    <div style="font-family:Arial;font-size:13px;min-width:320px;">
        <h3 style="margin:0 0 8px 0;">Substation / Bus</h3>
        <b>ID:</b> {bus_id}<br>
        <b>Latitude:</b> {bus['lat']:.5f}<br>
        <b>Longitude:</b> {bus['lon']:.5f}<br>
        <b>Voltage:</b> {bus['voltage']:g} kV<br>
        <b>Snapshot:</b> {snapshot}<br><hr>

        <table style="border-collapse:collapse;width:100%;">
            <tr>
                <th style="text-align:left;"></th>
                <th style="text-align:right;">Before</th>
                <th style="text-align:right;">After</th>
                <th style="text-align:right;">Change</th>
            </tr>

            <tr>
                <td><b>Generation</b></td>
                <td style="text-align:right;">{gen_before:.2f}</td>
                <td style="text-align:right;">{gen_after:.2f}</td>
                <td style="text-align:right;">{gen_after - gen_before:+.2f}</td>
            </tr>

            <tr>
                <td><b>Load</b></td>
                <td style="text-align:right;">{load_before:.2f}</td>
                <td style="text-align:right;">{load_after:.2f}</td>
                <td style="text-align:right;">{load_after - load_before:+.2f}</td>
            </tr>
        </table>

        <small>Power values in MW</small>
    </div>
    """

    folium.CircleMarker(
        location=[bus["lat"], bus["lon"]],
        radius=5,
        color="#0d47a1",
        weight=2,
        fill=True,
        fill_color="#2196f3",
        fill_opacity=0.90,
        tooltip=(
            f"Substation: {bus_id} — "
            f"{bus['voltage']:g} kV"
        ),
        popup=folium.Popup(
            popup,
            max_width=450
        ),
    ).add_to(substation_layer)

substation_layer.add_to(m)


# ------------------------------------------------------------
# Generators
# ------------------------------------------------------------

all_generators = n_before.generators.index.union(
    n_after.generators.index
)

generator_number_at_bus = {}

for generator_id in all_generators:

    if generator_id in n_after.generators.index:
        gen_row = n_after.generators.loc[generator_id]
    else:
        gen_row = n_before.generators.loc[generator_id]

    bus_id = gen_row.get("bus", None)

    if bus_id not in bus_lookup:
        continue

    before_p = safe_timeseries_value(
        n_before.generators_t.p,
        snapshot,
        generator_id
    )

    after_p = safe_timeseries_value(
        n_after.generators_t.p,
        snapshot,
        generator_id
    )

    change = after_p - before_p

    p_nom = clean_value(
        gen_row.get("p_nom", 0.0)
    )

    try:
        p_nom_number = float(p_nom or 0.0)
    except (TypeError, ValueError):
        p_nom_number = 0.0

    carrier = clean_value(
        gen_row.get("carrier", "")
    )

    bus = bus_lookup[bus_id]

    # Multiple generators can share one bus. Spread them slightly
    # east/north of the actual substation coordinate.
    generator_number_at_bus[bus_id] = (
        generator_number_at_bus.get(bus_id, 0) + 1
    )

    k = generator_number_at_bus[bus_id]

    lat_offset = 0.0025 + ((k - 1) % 4) * 0.0012
    lon_offset = 0.0025 + ((k - 1) // 4) * 0.0012

    marker_lat = bus["lat"] + lat_offset
    marker_lon = bus["lon"] + lon_offset

    # Size represents installed capacity, not dispatch.
    radius = 6.0 + min(
        np.sqrt(max(p_nom_number, 0.0)) / 2.8,
        10.0
    )

    popup = f"""
    <div style="font-family:Arial;font-size:13px;min-width:300px;">
        <h4 style="margin:0 0 8px 0;">Generator</h4>

        <b>ID:</b> {generator_id}<br>
        <b>Bus:</b> {bus_id}<br>
        <b>Carrier:</b> {carrier}<br>
        <b>Nominal capacity:</b> {p_nom_number:.2f} MW<br>
        <b>Snapshot:</b> {snapshot}<br>

        <hr>

        <b>Before generation:</b> {before_p:.2f} MW<br>
        <b>After generation:</b> {after_p:.2f} MW<br>
        <b>Dispatch change:</b> {change:+.2f} MW<br>

        <br>
        <small>
            Orange triangle size represents nominal generator capacity.
            Marker is offset slightly from its actual bus for visibility.
        </small>
    </div>
    """

    folium.RegularPolygonMarker(
        location=[
            marker_lat,
            marker_lon
        ],
        number_of_sides=3,
        rotation=0,
        radius=radius,
        color="#e65100",
        weight=2,
        fill=True,
        fill_color="#ff9800",
        fill_opacity=0.88,
        tooltip=(
            f"Generator: {generator_id} — "
            f"{carrier} — "
            f"{after_p:.1f} MW after"
        ),
        popup=folium.Popup(
            popup,
            max_width=430
        ),
    ).add_to(generator_layer)

generator_layer.add_to(m)


# ------------------------------------------------------------
# Loads
# ------------------------------------------------------------

all_loads = n_before.loads.index.union(
    n_after.loads.index
)

load_number_at_bus = {}

for load_id in all_loads:

    if load_id in n_after.loads.index:
        load_row = n_after.loads.loc[load_id]
    else:
        load_row = n_before.loads.loc[load_id]

    bus_id = load_row.get("bus", None)

    if bus_id not in bus_lookup:
        continue

    before_load = 0.0
    after_load = 0.0

    if load_id in n_before.loads.index:
        if (
            hasattr(n_before.loads_t, "p")
            and not n_before.loads_t.p.empty
            and snapshot in n_before.loads_t.p.index
        ):
            before_load = safe_timeseries_value(
                n_before.loads_t.p,
                snapshot,
                load_id
            )
        elif (
            hasattr(n_before.loads_t, "p_set")
            and not n_before.loads_t.p_set.empty
            and snapshot in n_before.loads_t.p_set.index
        ):
            before_load = safe_timeseries_value(
                n_before.loads_t.p_set,
                snapshot,
                load_id
            )
        else:
            before_load = float(
                n_before.loads.at[load_id, "p_set"]
            )

    if load_id in n_after.loads.index:
        if (
            hasattr(n_after.loads_t, "p")
            and not n_after.loads_t.p.empty
            and snapshot in n_after.loads_t.p.index
        ):
            after_load = safe_timeseries_value(
                n_after.loads_t.p,
                snapshot,
                load_id
            )
        elif (
            hasattr(n_after.loads_t, "p_set")
            and not n_after.loads_t.p_set.empty
            and snapshot in n_after.loads_t.p_set.index
        ):
            after_load = safe_timeseries_value(
                n_after.loads_t.p_set,
                snapshot,
                load_id
            )
        else:
            after_load = float(
                n_after.loads.at[load_id, "p_set"]
            )

    load_change = after_load - before_load

    carrier = clean_value(
        load_row.get("carrier", "")
    )

    bus = bus_lookup[bus_id]

    # Multiple loads can share one bus. Spread them slightly
    # west/south of the actual substation coordinate.
    load_number_at_bus[bus_id] = (
        load_number_at_bus.get(bus_id, 0) + 1
    )

    k = load_number_at_bus[bus_id]

    lat_offset = 0.0025 + ((k - 1) % 4) * 0.0012
    lon_offset = 0.0025 + ((k - 1) // 4) * 0.0012

    marker_lat = bus["lat"] - lat_offset
    marker_lon = bus["lon"] - lon_offset

    # Size represents demand at this snapshot.
    radius = 5.5 + min(
        np.sqrt(max(abs(after_load), 0.0)) / 2.5,
        10.0
    )

    popup = f"""
    <div style="font-family:Arial;font-size:13px;min-width:300px;">
        <h4 style="margin:0 0 8px 0;">Load</h4>

        <b>ID:</b> {load_id}<br>
        <b>Bus:</b> {bus_id}<br>
        <b>Carrier:</b> {carrier}<br>
        <b>Snapshot:</b> {snapshot}<br>

        <hr>

        <b>Before demand:</b> {before_load:.2f} MW<br>
        <b>After demand:</b> {after_load:.2f} MW<br>
        <b>Demand change:</b> {load_change:+.2f} MW<br>

        <br>
        <small>
            Green square size represents demand at this snapshot.
            Marker is offset slightly from its actual bus for visibility.
        </small>
    </div>
    """

    folium.RegularPolygonMarker(
        location=[
            marker_lat,
            marker_lon
        ],
        number_of_sides=4,
        rotation=45,
        radius=radius,
        color="#1b5e20",
        weight=2,
        fill=True,
        fill_color="#4caf50",
        fill_opacity=0.88,
        tooltip=(
            f"Load: {load_id} — "
            f"{after_load:.1f} MW"
        ),
        popup=folium.Popup(
            popup,
            max_width=430
        ),
    ).add_to(load_layer)

load_layer.add_to(m)

all_lats = [bus["lat"] for bus in bus_lookup.values()]
all_lons = [bus["lon"] for bus in bus_lookup.values()]
if all_lats and all_lons:
    m.fit_bounds([
        [min(all_lats), min(all_lons)],
        [max(all_lats), max(all_lons)],
    ])

folium.LayerControl(
    position="topleft",
    collapsed=False,
    autoZIndex=True
).add_to(m)

Fullscreen(
    position="topright"
).add_to(m)
MousePosition(
    position="bottomleft",
    separator=" | ",
    prefix="Coordinates:",
).add_to(m)

legend = f"""
<div style="position:fixed;bottom:40px;right:20px;width:265px;background-color:white;
            border:2px solid #777;border-radius:6px;z-index:9999;font-size:13px;
            font-family:Arial;padding:10px;box-shadow:0 1px 6px rgba(0,0,0,0.3);">
<b>Optimisation comparison</b><br>
Snapshot: {snapshot}<hr style="margin:8px 0;">

<b>Components</b><br>
<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#2196f3;border:2px solid #0d47a1;"></span>
&nbsp; Substation / bus<br>
<span style="display:inline-block;color:#ff9800;font-size:18px;line-height:14px;">▲</span>
&nbsp; Generator<br>
<span style="display:inline-block;width:12px;height:12px;background:#4caf50;border:2px solid #1b5e20;"></span>
&nbsp; Load

<hr style="margin:8px 0;">
<b>Difference layer</b><br>
<span style="display:inline-block;width:25px;height:4px;background:#d73027;"></span>
&nbsp; Increased line loading<br>
<span style="display:inline-block;width:25px;height:4px;background:#4575b4;"></span>
&nbsp; Decreased line loading<br>
<span style="display:inline-block;width:25px;height:4px;background:#777777;"></span>
&nbsp; Little / no change
<hr style="margin:8px 0;">
Line thickness = magnitude<br>
Generator circle size = dispatch change
</div>
"""
m.get_root().html.add_child(folium.Element(legend))

referrer_policy = """
<meta name="referrer" content="strict-origin-when-cross-origin">
"""
m.get_root().header.add_child(folium.Element(referrer_policy))

title_html = f"""
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9999;
            background:rgba(255,255,255,0.94);padding:8px 18px;border-radius:7px;
            box-shadow:0 1px 5px rgba(0,0,0,0.25);font-family:Arial;text-align:center;">
<b style="font-size:18px;">Northern Ireland Network — Before vs After Optimisation</b><br>
<span style="font-size:12px;">Snapshot: {snapshot}</span>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

m.save(OUTPUT_HTML)
print()
print("Interactive comparison map created:")
print(OUTPUT_HTML)
