"""Simple plotter for the 26/32-county annealer node CSVs.

Usage:
    python3 plot_constraint_nodes.py 26
    python3 plot_constraint_nodes.py 32
    python3 plot_constraint_nodes.py 32 outputs/optimised_constraint_groups.csv

Requires: pandas, matplotlib
"""
from pathlib import Path
import ast
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DEFAULTS = {
    "26": OUTPUTS / "annealer_nodes_26_counties_sv2024_wdt_exclusive.csv",
    "32": OUTPUTS / "annealer_nodes_32_counties_sv2024_wdt_exclusive.csv",
}


def group_number(value):
    """Read 7, '(7,)', '[7]' etc. as integer group 7."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value.strip())
        except Exception:
            pass
    if isinstance(value, (tuple, list, set)):
        value = list(value)[0]
    return int(value)


def main():
    scope = sys.argv[1] if len(sys.argv) > 1 else "32"
    if scope not in DEFAULTS:
        raise SystemExit("Usage: python3 plot_constraint_nodes.py [26|32] [optional_csv]")

    csv_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULTS[scope]
    if not csv_path.is_absolute():
        # First accept a path relative to the current working directory, then ROOT.
        if not csv_path.exists():
            csv_path = ROOT / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find node CSV: {csv_path}")

    nodes = pd.read_csv(csv_path)
    required = {"name", "latitude", "longitude"}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    if "groups" in nodes.columns:
        nodes["plot_group"] = nodes["groups"].apply(group_number)
    elif "group" in nodes.columns:
        nodes["plot_group"] = nodes["group"].apply(group_number)
    else:
        raise ValueError("CSV needs a 'groups' or 'group' column")

    nodes = nodes.dropna(subset=["latitude", "longitude"]).copy()

    fig, ax = plt.subplots(figsize=(9, 11))
    scatter = ax.scatter(
        nodes["longitude"],
        nodes["latitude"],
        c=nodes["plot_group"],
        cmap="turbo",
        s=28,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.25,
    )

    # Label group centroids rather than all 174/240 nodes, keeping the map readable.
    centres = nodes.groupby("plot_group")[["longitude", "latitude"]].mean()
    for group, row in centres.iterrows():
        ax.text(
            row["longitude"], row["latitude"], str(group),
            fontsize=7, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="black", alpha=0.75),
        )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label("Constraint group ID")

    ax.set_title(f"{scope}-County Renewable Constraint Nodes ({len(nodes)} nodes)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    out = OUTPUTS / f"constraint_nodes_{scope}_counties.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")

    print(f"Loaded: {csv_path}")
    print(f"Nodes: {len(nodes)}")
    print(f"Groups: {nodes['plot_group'].nunique()}")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
