import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ============================================================
# Loading
# ============================================================

def load_nodes(path):
    """Load a nodes CSV (baseline or optimised) and parse the tuple columns.

    Note per the CSVs: `groups` (single-element tuple) and `group` (int) are
    the "exclusive" one-group-per-node fields used by the annealer. The full
    official multi-membership WDT groups live in `wdt_groups` — this loader
    parses both, but plotting here uses the exclusive `group` for coloring
    since that's what the annealer actually optimises.
    """
    df = pd.read_csv(path)
    df["groups"] = df["groups"].apply(ast.literal_eval)
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


if __name__ == "__main__":
    baseline = load_nodes("sample_data/annealer_nodes_32_counties_sv2024_wdt_exclusive.csv")
    optimised = load_nodes("sample_data/optimised_constraint_groups_32_counties_99_20per.csv")

    plot_before_after(baseline, optimised, save_path="/mnt/user-data/outputs/before_after_groups.png")
    plot_spread_comparison(baseline, optimised, save_path="/mnt/user-data/outputs/group_spread_comparison.png")
    print("done")