import ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import folium

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================================
# Loading
# ============================================================

def load_nodes(path):
    """Load a nodes CSV (baseline or optimised) and parse the tuple columns.

    Note per the CSVs: `groups` (single-element tuple) and `group` (int) are
    the "exclusive" one-group-per-node fields used by the annealer. The full
    official multi-membership WDT groups live in `wdt_groups` when present —
    not every output file includes it, so it's parsed only if the column
    exists rather than assumed.
    """
    df = pd.read_csv(path)
    if "groups" in df.columns:
        df["groups"] = df["groups"].apply(ast.literal_eval)
    if "wdt_groups" in df.columns:
        df["wdt_groups"] = df["wdt_groups"].apply(ast.literal_eval)
    df["group"] = df["group"].astype(int)
    return df


# ============================================================
# Geographic plotting
# ============================================================

def _group_colors(group_ids, cmap_name="tab20"):
    """Stable color mapping from group id -> color, shared across plots
    so the same group id is the same color in before/after comparisons."""
    unique_groups = sorted(set(group_ids))
    cmap = plt.get_cmap(cmap_name, max(len(unique_groups), 1))
    return {g: cmap(i) for i, g in enumerate(unique_groups)}


def plot_geographic_groups(nodes, group_col="group", title="", ax=None,
                            size_col="mec_mw", color_map=None):
    """Scatter nodes at their real lat/lon, colored by constraint group.
    Point size scales with mec_mw (installed capacity) so bigger generators
    stand out."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 8))

    colors = color_map or _group_colors(nodes[group_col])
    sizes = 20 + 4 * nodes[size_col].fillna(nodes[size_col].median())

    for g, sub in nodes.groupby(group_col):
        ax.scatter(sub["longitude"], sub["latitude"],
                   s=sizes.loc[sub.index], color=colors[g],
                   label=f"group {g}", edgecolors="black", linewidths=0.4, alpha=0.85)

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if len(colors) <= 20:
        ax.legend(fontsize=6, ncol=2, loc="best", framealpha=0.8)

    if own_fig:
        fig.tight_layout()
    return ax


def plot_before_after(baseline, optimised, group_col="group",
                       save_path="before_after_groups.png"):
    """Side-by-side geographic comparison: current SONI-style geographic
    zones vs the annealer's granular stochastic groups."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharex=True, sharey=True)
    plot_geographic_groups(baseline, group_col, "Baseline (geographic zones)", ax=axes[0])
    plot_geographic_groups(optimised, group_col, "Optimised (granular stochastic)", ax=axes[1])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ============================================================
# Interactive OSM map (folium)
# ============================================================

def make_osm_map(nodes, group_col="group", title="", save_path="map.html",
                  color_map=None, zoom_start=8):
    """Interactive OpenStreetMap view of the nodes, colored by constraint
    group. Click a marker for name/technology/capacity/group. Saves a
    standalone .html you can open in any browser."""
    colors = color_map or _group_colors(nodes[group_col])
    center = [nodes["latitude"].mean(), nodes["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")

    if title:
        title_html = f'<h3 style="text-align:center;font-family:sans-serif">{title}</h3>'
        m.get_root().html.add_child(folium.Element(title_html))

    for _, row in nodes.iterrows():
        g = row[group_col]
        color_hex = mcolors.to_hex(colors[g])
        radius = 4 + 0.4 * (row["mec_mw"] if pd.notna(row.get("mec_mw")) else 5)
        popup = (
            f"<b>{row.get('name', row.get('node_id'))}</b><br>"
            f"group: {g}<br>"
            f"technology: {row.get('technology', 'n/a')}<br>"
            f"mec_mw: {row.get('mec_mw', 'n/a')}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color="black",
            weight=0.5,
            fill=True,
            fill_color=color_hex,
            fill_opacity=0.85,
            popup=folium.Popup(popup, max_width=250),
        ).add_to(m)

    m.save(save_path)
    return save_path


def make_before_after_osm_maps(baseline, optimised, group_col="group",
                                 baseline_path="baseline_map.html",
                                 optimised_path="optimised_map.html"):
    """Two separate interactive maps (folium can't easily do side-by-side
    panels like matplotlib) -- open both in browser tabs to compare."""
    colors = _group_colors(pd.concat([baseline[group_col], optimised[group_col]]))
    make_osm_map(baseline, group_col, "Baseline (geographic zones)",
                 save_path=baseline_path, color_map=colors)
    make_osm_map(optimised, group_col, "Optimised (granular stochastic)",
                 save_path=optimised_path, color_map=colors)
    return baseline_path, optimised_path


# ============================================================
# Group spread metric
# Quantifies "geographic" vs "granular/scattered" grouping directly:
# average pairwise distance between nodes within the same group.
# Low spread = geographically clustered (like current zones).
# High spread = scattered across the region (the granular goal).
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def mean_within_group_spread(nodes, group_col="group"):
    """Mean pairwise haversine distance (km) within each group, and the
    overall average across groups (skips groups with <2 nodes)."""
    spreads = {}
    for g, sub in nodes.groupby(group_col):
        if len(sub) < 2:
            continue
        lat = sub["latitude"].to_numpy()
        lon = sub["longitude"].to_numpy()
        dists = []
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                dists.append(haversine_km(lat[i], lon[i], lat[j], lon[j]))
        spreads[g] = float(np.mean(dists))
    overall = float(np.mean(list(spreads.values()))) if spreads else float("nan")
    return spreads, overall


def plot_spread_comparison(baseline, optimised, group_col="group",
                            save_path="group_spread_comparison.png"):
    """Bar chart: average within-group geographic spread, baseline vs
    optimised. This is the direct visual evidence for whether the
    annealer actually achieved 'granular stochastic' (scattered) groups."""
    _, base_overall = mean_within_group_spread(baseline, group_col)
    _, opt_overall = mean_within_group_spread(optimised, group_col)

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(["Baseline\n(geographic)", "Optimised\n(granular stochastic)"],
                   [base_overall, opt_overall], color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("mean within-group distance (km)")
    ax.set_title("Group geographic spread")
    for b, v in zip(bars, [base_overall, opt_overall]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f} km",
                ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def merge_groups_onto_baseline(baseline, slim_df):
    """Some annealer outputs are slim — just node_id + group assignments,
    without the lat/lon/mec_mw/technology metadata. This takes baseline's
    full metadata and overlays the new group/groups columns from slim_df,
    aligned by node_id, so plotting always has what it needs."""
    merged = baseline.copy(deep=True).set_index("node_id")
    slim_indexed = slim_df.set_index("node_id")
    merged.loc[slim_indexed.index, "group"] = slim_indexed["group"]
    if "groups" in slim_indexed.columns:
        merged.loc[slim_indexed.index, "groups"] = slim_indexed["groups"]
    return merged.reset_index()


def find_optimised_file(baseline, candidate_paths):
    """Try each candidate path and return the first one whose `group`
    column actually differs from the baseline — i.e. the real
    post-annealing result, not just a copied-through baseline file.
    Aligns on node_id (if present) rather than row position, in case
    the files aren't in the same row order."""
    base_indexed = baseline.set_index("node_id") if "node_id" in baseline.columns else None

    for path in candidate_paths:
        path = Path(path)
        if not path.exists():
            print(f"  [skip] not found: {path}")
            continue
        try:
            df = load_nodes(path)
        except Exception as e:
            print(f"  [skip] couldn't parse: {path} ({e})")
            continue

        if base_indexed is not None and "node_id" in df.columns:
            df_indexed = df.set_index("node_id")
            common = base_indexed.index.intersection(df_indexed.index)
            if len(common) == 0:
                print(f"  [skip] no matching node_id values: {path}")
                continue
            n_diff = (base_indexed.loc[common, "group"].to_numpy()
                      != df_indexed.loc[common, "group"].to_numpy()).sum()
            print(f"  {path}: {n_diff} of {len(common)} matched nodes differ from baseline")
        else:
            if len(df) != len(baseline):
                print(f"  [skip] row count differs and no node_id to align on: {path}")
                continue
            n_diff = (baseline["group"].to_numpy() != df["group"].to_numpy()).sum()
            print(f"  {path}: {n_diff} of {len(df)} groups differ from baseline (by row position)")

        if n_diff > 0:
            return df, path
    raise FileNotFoundError(
        "None of the candidate files differ from the baseline — "
        "the real annealed output isn't among them."
    )


if __name__ == "__main__":
    baseline = load_nodes(OUTPUT_DIR / "annealer_nodes_32_counties_sv2024_wdt_exclusive.csv")

    print("Looking for the real optimised file among other outputs/ candidates:")
    candidates = [
        OUTPUT_DIR / "annealer_nodes_32_counties_exclusive.csv",
        OUTPUT_DIR / "annealer_nodes_current.csv",
        OUTPUT_DIR / "annealer_nodes_exclusive.csv",
    ]
    optimised, optimised_path = find_optimised_file(baseline, candidates)
    print(f"Using: {optimised_path}")

    # optimised file may be slim (just node_id + group) -- merge onto
    # baseline's full metadata (lat/lon/mec_mw/etc.) before plotting
    missing_cols = {"latitude", "longitude", "mec_mw"} - set(optimised.columns)
    if missing_cols:
        print(f"  optimised file is missing {missing_cols} -- "
              f"merging its group assignments onto baseline metadata")
        optimised = merge_groups_onto_baseline(baseline, optimised)

    plot_before_after(baseline, optimised, save_path=OUTPUT_DIR / "before_after_groups.png")
    plot_spread_comparison(baseline, optimised, save_path=OUTPUT_DIR / "group_spread_comparison.png")
    make_before_after_osm_maps(baseline, optimised,
                                baseline_path=OUTPUT_DIR / "baseline_map.html",
                                optimised_path=OUTPUT_DIR / "optimised_map.html")
    print("done")