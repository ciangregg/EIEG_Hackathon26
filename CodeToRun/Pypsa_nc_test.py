import json
import numpy as np
import h5py


# ============================================================
# Configuration
# ============================================================

INPUT_NC = "/home/seba/Documents/Colleg/Hackathon_2/EIEG_Hackathon26/cian/EIEG_Hackathon26/data/SV2024_all-island.nc"
OUTPUT_HTML = "output/WP2024_openstreetmap_overlay.html"


# ============================================================
# Helpers
# ============================================================

def decode_strings(values):
    """Convert NetCDF/HDF5 byte strings to normal Python strings."""
    output = []

    for value in values:
        if isinstance(value, (bytes, np.bytes_)):
            output.append(value.decode("utf-8", errors="replace"))
        else:
            output.append(str(value))

    return output


# ============================================================
# Read the NetCDF file
# ============================================================

with h5py.File(INPUT_NC, "r") as f:

    # ----------------------------
    # Buses / substations
    # ----------------------------

    bus_ids = decode_strings(f["buses_i"][:])

    # Longitude and latitude
    bus_lon = np.asarray(f["buses_x"][:], dtype=float)
    bus_lat = np.asarray(f["buses_y"][:], dtype=float)

    bus_voltage = np.asarray(
        f["buses_v_nom"][:],
        dtype=float
    )

    if "buses_station" in f:
        bus_station = decode_strings(f["buses_station"][:])
    else:
        bus_station = [""] * len(bus_ids)

    if "buses_jurisdiction" in f:
        bus_jurisdiction = decode_strings(
            f["buses_jurisdiction"][:]
        )
    else:
        bus_jurisdiction = [""] * len(bus_ids)

    # ----------------------------
    # Transmission lines
    # ----------------------------

    line_ids = decode_strings(f["lines_i"][:])
    line_bus0 = decode_strings(f["lines_bus0"][:])
    line_bus1 = decode_strings(f["lines_bus1"][:])

    if "lines_s_nom" in f:
        line_rating = np.asarray(
            f["lines_s_nom"][:],
            dtype=float
        )
    else:
        line_rating = np.full(
            len(line_ids),
            np.nan
        )

    if "lines_carrier" in f:
        line_carrier = decode_strings(
            f["lines_carrier"][:]
        )
    else:
        line_carrier = [""] * len(line_ids)

    # ----------------------------
    # Transformers
    # ----------------------------

    transformer_ids = decode_strings(
        f["transformers_i"][:]
    )

    transformer_bus0 = decode_strings(
        f["transformers_bus0"][:]
    )

    transformer_bus1 = decode_strings(
        f["transformers_bus1"][:]
    )


# ============================================================
# Build bus lookup
# ============================================================

bus_lookup = {}

for i, bus_id in enumerate(bus_ids):

    lon = bus_lon[i]
    lat = bus_lat[i]

    if np.isfinite(lon) and np.isfinite(lat):

        bus_lookup[bus_id] = {
            "lat": float(lat),
            "lon": float(lon),
            "index": i
        }


# ============================================================
# Prepare bus data for JavaScript
# ============================================================

buses = []

for bus_id, location in bus_lookup.items():

    i = location["index"]

    voltage = bus_voltage[i]

    buses.append({
        "id": bus_id,
        "lat": location["lat"],
        "lon": location["lon"],
        "voltage": (
            None
            if not np.isfinite(voltage)
            else float(voltage)
        ),
        "station": bus_station[i],
        "jurisdiction": bus_jurisdiction[i]
    })


# ============================================================
# Prepare transmission lines
# ============================================================

lines = []

for i, (bus0, bus1) in enumerate(
    zip(line_bus0, line_bus1)
):

    # Only draw line if both endpoint buses have coordinates
    if bus0 not in bus_lookup or bus1 not in bus_lookup:
        continue

    b0 = bus_lookup[bus0]
    b1 = bus_lookup[bus1]

    voltage0 = bus_voltage[b0["index"]]
    voltage1 = bus_voltage[b1["index"]]

    voltage = max(voltage0, voltage1)

    rating = line_rating[i]

    lines.append({
        "id": line_ids[i],
        "bus0": bus0,
        "bus1": bus1,

        "coords": [
            [b0["lat"], b0["lon"]],
            [b1["lat"], b1["lon"]]
        ],

        "voltage": (
            None
            if not np.isfinite(voltage)
            else float(voltage)
        ),

        "rating": (
            None
            if not np.isfinite(rating)
            else float(rating)
        ),

        "carrier": line_carrier[i]
    })


# ============================================================
# Prepare transformer connections
# ============================================================

transformers = []

for i, (bus0, bus1) in enumerate(
    zip(transformer_bus0, transformer_bus1)
):

    if bus0 not in bus_lookup or bus1 not in bus_lookup:
        continue

    b0 = bus_lookup[bus0]
    b1 = bus_lookup[bus1]

    transformers.append({
        "id": transformer_ids[i],
        "bus0": bus0,
        "bus1": bus1,

        "coords": [
            [b0["lat"], b0["lon"]],
            [b1["lat"], b1["lon"]]
        ]
    })


# ============================================================
# Determine map bounds
# ============================================================

valid_latitudes = [
    bus["lat"] for bus in buses
]

valid_longitudes = [
    bus["lon"] for bus in buses
]

bounds = [
    [
        min(valid_latitudes),
        min(valid_longitudes)
    ],
    [
        max(valid_latitudes),
        max(valid_longitudes)
    ]
]


# ============================================================
# HTML + Leaflet map
# ============================================================

html = """
<!DOCTYPE html>
<html>

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
WP2033 All-Island Network
</title>


<!-- Leaflet CSS -->

<link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
/>


<style>

html,
body,
#map {
    height: 100%;
    width: 100%;
    margin: 0;
}


.legend {
    background: white;
    padding: 10px;
    font-family: Arial, sans-serif;
    font-size: 13px;
    line-height: 20px;

    border-radius: 5px;

    box-shadow:
        0 1px 5px rgba(0,0,0,0.5);
}


.legend i {

    display: inline-block;

    width: 22px;
    height: 4px;

    margin-right: 7px;

    vertical-align: middle;
}

</style>

</head>


<body>

<div id="map"></div>


<!-- Leaflet JavaScript -->

<script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
</script>


<script>


// ==========================================================
// Data from Python
// ==========================================================

const buses = BUS_DATA;

const lines = LINE_DATA;

const transformers = TRANSFORMER_DATA;

const networkBounds = MAP_BOUNDS;


// ==========================================================
// Create map
// ==========================================================

const map = L.map("map");


L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        maxZoom: 19,

        attribution:
            "&copy; OpenStreetMap contributors"
    }
).addTo(map);


// Zoom to network

map.fitBounds(
    networkBounds,
    {
        padding: [20, 20]
    }
);


// ==========================================================
// Voltage colours
// ==========================================================

function voltageColour(voltage) {

    if (voltage >= 400)
        return "#7b1fa2";

    if (voltage >= 275)
        return "#d32f2f";

    if (voltage >= 200)
        return "#f57c00";

    if (voltage >= 100)
        return "#1976d2";

    return "#388e3c";
}


// ==========================================================
// HTML escaping
// ==========================================================

function escapeHTML(value) {

    return String(value ?? "")
        .replace(
            /[&<>"']/g,
            character => ({
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#39;"
            })[character]
        );
}


// ==========================================================
// Layers
// ==========================================================

const lineLayer =
    L.layerGroup().addTo(map);

const transformerLayer =
    L.layerGroup().addTo(map);

const busLayer =
    L.layerGroup().addTo(map);


// ==========================================================
// Transmission lines
// ==========================================================

lines.forEach(line => {

    const voltage =
        line.voltage || 0;


    const polyline = L.polyline(
        line.coords,
        {
            color: voltageColour(voltage),

            weight: 2.5,

            opacity: 0.75
        }
    );


    polyline.bindPopup(

        "<b>Transmission Line</b>" +

        "<br>ID: " +
        escapeHTML(line.id) +

        "<br>From: " +
        escapeHTML(line.bus0) +

        "<br>To: " +
        escapeHTML(line.bus1) +

        "<br>Nominal voltage: " +
        voltage +
        " kV" +

        "<br>Rating: " +
        (
            line.rating === null
            ? "—"
            : line.rating + " MVA"
        ) +

        "<br>Carrier: " +
        escapeHTML(line.carrier)
    );


    polyline.addTo(lineLayer);

});


// ==========================================================
// Transformers
// ==========================================================

transformers.forEach(transformer => {

    const polyline = L.polyline(

        transformer.coords,

        {
            color: "#444444",

            weight: 2,

            opacity: 0.6,

            dashArray: "5, 5"
        }
    );


    polyline.bindPopup(

        "<b>Transformer</b>" +

        "<br>ID: " +
        escapeHTML(transformer.id) +

        "<br>From: " +
        escapeHTML(transformer.bus0) +

        "<br>To: " +
        escapeHTML(transformer.bus1)
    );


    polyline.addTo(transformerLayer);

});


// ==========================================================
// Buses
// ==========================================================

buses.forEach(bus => {

    const voltage =
        bus.voltage || 0;


    const marker = L.circleMarker(

        [bus.lat, bus.lon],

        {

            radius: 3.5,

            color:
                voltageColour(voltage),

            weight: 1,

            fillOpacity: 0.9

        }
    );


    marker.bindPopup(

        "<b>Bus / Substation</b>" +

        "<br>ID: " +
        escapeHTML(bus.id) +

        "<br>Voltage: " +
        (
            bus.voltage === null
            ? "—"
            : bus.voltage + " kV"
        ) +

        "<br>Station: " +
        (
            escapeHTML(bus.station)
            || "—"
        ) +

        "<br>Jurisdiction: " +
        (
            escapeHTML(bus.jurisdiction)
            || "—"
        )

    );


    marker.addTo(busLayer);

});


// ==========================================================
// Layer control
// ==========================================================

L.control.layers(

    null,

    {
        "Transmission lines":
            lineLayer,

        "Transformers":
            transformerLayer,

        "Buses / substations":
            busLayer
    },

    {
        collapsed: false
    }

).addTo(map);


// ==========================================================
// Legend
// ==========================================================

const legend =
    L.control({
        position: "bottomright"
    });


legend.onAdd = function () {

    const div =
        L.DomUtil.create(
            "div",
            "legend"
        );


    div.innerHTML =

        "<b>Nominal Voltage</b><br>" +

        '<i style="background:#7b1fa2"></i>' +
        "400+ kV<br>" +

        '<i style="background:#d32f2f"></i>' +
        "275–399 kV<br>" +

        '<i style="background:#f57c00"></i>' +
        "200–274 kV<br>" +

        '<i style="background:#1976d2"></i>' +
        "100–199 kV<br>" +

        '<i style="background:#388e3c"></i>' +
        "&lt;100 kV";


    return div;
};


legend.addTo(map);


</script>

</body>

</html>
"""


# ============================================================
# Insert network data into HTML
# ============================================================

html = html.replace(
    "BUS_DATA",
    json.dumps(buses)
)

html = html.replace(
    "LINE_DATA",
    json.dumps(lines)
)

html = html.replace(
    "TRANSFORMER_DATA",
    json.dumps(transformers)
)

html = html.replace(
    "MAP_BOUNDS",
    json.dumps(bounds)
)


# ============================================================
# Save output
# ============================================================

with open(
    OUTPUT_HTML,
    "w",
    encoding="utf-8"
) as file:

    file.write(html)


print("Map created successfully.")
print(f"Output: {OUTPUT_HTML}")

print(
    f"{len(buses)} buses, "
    f"{len(lines)} lines, "
    f"{len(transformers)} transformers."
)