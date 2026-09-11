from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union
except Exception as e:  # pragma: no cover
    gpd = None
    Polygon = None
    MultiPolygon = None
    unary_union = None

# ------------------------------------------------------------
# Annealer API import with fallback
# ------------------------------------------------------------
import sys
from pathlib import Path

# This script lives in NI_Grid_Simulator/outputs/
# The API lives in NI_Grid_Simulator/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("RUNNING SCRIPT:", Path(__file__).resolve())
print("PROJECT ROOT:", PROJECT_ROOT)

from all_island_annealer_api import make_emulator

print("API FOUND:", make_emulator)
# ------------------------------------------------------------
# Utility data structures
# ------------------------------------------------------------
@dataclass
class Snapshot:
    eval_index: int
    temperature: float
    current_dd: float
    best_dd: float
    grouping: np.ndarray
    changed_nodes: Tuple[int, int]
    accepted: bool
    is_best: bool


# ------------------------------------------------------------
# Annealer helpers (compatible with your current exclusive-group model)
# ------------------------------------------------------------
def parse_scalar_group(cell) -> int:
    if isinstance(cell, (list, tuple, np.ndarray)):
        return int(cell[0])
    s = str(cell).strip()
    if s.startswith("(") or s.startswith("["):
        vals = eval(s, {"__builtins__": {}})
        if isinstance(vals, int):
            return int(vals)
        return int(vals[0])
    return int(float(s))


def grouping_to_nodes(grouping: np.ndarray, nodes: pd.DataFrame) -> pd.DataFrame:
    out = nodes.copy(deep=False)
    out["groups"] = [(int(g),) for g in grouping]
    out["group"] = [int(g) for g in grouping]
    return out


def propose_swap(grouping: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
    n = len(grouping)
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n))
    while i == j or grouping[i] == grouping[j]:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
    cand = grouping.copy()
    cand[i], cand[j] = cand[j], cand[i]
    return cand, i, j


def prepare_energy_evaluator(nodes: pd.DataFrame, blackbox):
    prepare = getattr(blackbox, "prepare_grouping_evaluator", None)
    if callable(prepare):
        return prepare(nodes)
    return lambda grouping: blackbox(grouping_to_nodes(grouping, nodes))


def estimate_temperature_range(
    grouping: np.ndarray,
    energy_eval,
    rng: np.random.Generator,
    n_samples: int = 24,
    hot_accept_prob: float = 0.5,
    cold_accept_prob: float = 1e-3,
) -> Tuple[float, float]:
    E = energy_eval(grouping)
    dEs: List[float] = []
    g = grouping.copy()
    for _ in range(n_samples):
        cand, _, _ = propose_swap(g, rng)
        E_new = energy_eval(cand)
        if np.isfinite(E_new):
            d = abs(E_new - E)
            if d > 0 and np.isfinite(d):
                dEs.append(float(d))
            g = cand
            E = E_new
    if not dEs:
        return 1e-3, 1e-6
    dEs = np.asarray(dEs, dtype=float)
    d_typ = np.percentile(dEs, 50)
    d_small = np.percentile(dEs, 5)
    T0 = -d_typ / np.log(hot_accept_prob)
    T1 = -d_small / np.log(cold_accept_prob)
    T0 = max(float(T0), 1e-6)
    T1 = max(min(float(T1), T0 * 0.5), 1e-9)
    return T0, T1


def run_annealer_with_snapshots(
    nodes: pd.DataFrame,
    blackbox,
    max_evals: int,
    n_frames: int,
    seed: int,
    hot_accept_prob: float = 0.5,
    cold_accept_prob: float = 1e-3,
    extra_best_frames: bool = True,
) -> Tuple[pd.DataFrame, float, float, List[Snapshot], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    grouping = np.array([parse_scalar_group(x) for x in nodes["groups"]], dtype=np.int64)
    energy_eval = prepare_energy_evaluator(nodes, blackbox)

    current_dd = float(energy_eval(grouping))
    baseline_dd = current_dd
    best_dd = current_dd
    best_grouping = grouping.copy()

    T_start, T_end = estimate_temperature_range(
        grouping=grouping.copy(),
        energy_eval=energy_eval,
        rng=rng,
        n_samples=min(24, max(6, max_evals // 20)),
        hot_accept_prob=hot_accept_prob,
        cold_accept_prob=cold_accept_prob,
    )

    frame_targets = set(np.unique(np.linspace(0, max_evals - 1, max(2, n_frames), dtype=int)).tolist())
    snapshots: List[Snapshot] = []
    history_rows = []

    # Save initial state.
    snapshots.append(
        Snapshot(
            eval_index=0,
            temperature=T_start,
            current_dd=current_dd,
            best_dd=best_dd,
            grouping=grouping.copy(),
            changed_nodes=(-1, -1),
            accepted=True,
            is_best=True,
        )
    )

    for eval_index in range(1, max_evals + 1):

        if eval_index % 100 == 0 or eval_index == max_evals:
            print(f"Annealing: {eval_index}/{max_evals} ({100*eval_index/max_evals:.0f}%) | Best DD: {best_dd:.4f}%")
        frac = 0.0 if max_evals == 1 else (eval_index - 1) / (max_evals - 1)
        T = T_start * ((T_end / T_start) ** frac)

        cand, i, j = propose_swap(grouping, rng)
        E_new = float(energy_eval(cand))
        accepted = False
        is_best = False

        if np.isfinite(E_new):
            dE = E_new - current_dd
            if dE < 0 or rng.random() < math.exp(-dE / max(T, 1e-12)):
                grouping = cand
                current_dd = E_new
                accepted = True
                if current_dd < best_dd:
                    best_dd = current_dd
                    best_grouping = grouping.copy()
                    is_best = True
        else:
            dE = math.inf

        history_rows.append({
            "eval_index": eval_index,
            "temperature": T,
            "current_dd": current_dd,
            "best_dd": best_dd,
            "candidate_dd": E_new,
            "delta_dd": dE,
            "accepted": bool(accepted),
            "best_update": bool(is_best),
            "swap_i": i,
            "swap_j": j,
        })

        if (eval_index - 1) in frame_targets or eval_index in frame_targets or (extra_best_frames and is_best):
            snapshots.append(
                Snapshot(
                    eval_index=eval_index,
                    temperature=T,
                    current_dd=current_dd,
                    best_dd=best_dd,
                    grouping=grouping.copy(),
                    changed_nodes=(i, j),
                    accepted=accepted,
                    is_best=is_best,
                )
            )

    # Deduplicate snapshots with same eval index, keeping last occurrence.
    snap_by_eval = {s.eval_index: s for s in snapshots}
    snapshots = [snap_by_eval[k] for k in sorted(snap_by_eval)]

    best_nodes = grouping_to_nodes(best_grouping, nodes)
    history = pd.DataFrame(history_rows)
    return best_nodes, baseline_dd, best_dd, snapshots, history


# ------------------------------------------------------------
# Map / territory helpers
# ------------------------------------------------------------
def read_buses_from_nc(network_path: Path) -> pd.DataFrame:
    rows = []
    with h5py.File(network_path, "r") as f:
        def dec(x):
            return x.decode("utf-8", errors="replace") if isinstance(x, (bytes, bytearray)) else str(x)
        buses_i = f["buses_i"][:]
        buses_x = f["buses_x"][:]
        buses_y = f["buses_y"][:]
        jurisdictions = f["buses_jurisdiction"][:] if "buses_jurisdiction" in f else [b"?"] * len(buses_i)
        for i in range(len(buses_i)):
            rows.append({
                "bus": dec(buses_i[i]),
                "longitude": float(buses_x[i]),
                "latitude": float(buses_y[i]),
                "jurisdiction": dec(jurisdictions[i]),
            })
    return pd.DataFrame(rows)


def load_boundary(basemap: Optional[Path], nodes_df: pd.DataFrame):
    if basemap is not None and basemap.exists():
        if gpd is None:
            raise RuntimeError("geopandas is required to read the basemap file")

        gdf = gpd.read_file(basemap)

        # Keep only polygonal geometries
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

        if len(gdf) == 0:
            raise RuntimeError("No polygon geometry found in basemap")

        geom = unary_union(gdf.geometry)

        # If multiple polygons exist, keep the largest one
        # so the island outline is used rather than stray shapes / sea extents.
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)

        return geom

    # Fallback: padded convex hull from nodes
    if Polygon is None:
        raise RuntimeError("shapely/geopandas not available for fallback boundary creation")

    pts = [
        (float(lon), float(lat))
        for lon, lat in zip(nodes_df["longitude"], nodes_df["latitude"])
        if np.isfinite(lon) and np.isfinite(lat)
    ]
    hull = Polygon(pts).convex_hull.buffer(0.25)
    return hull


def iter_polygons(geom):
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        for g in geom.geoms:
            yield g
    else:
        try:
            for g in geom.geoms:
                yield from iter_polygons(g)
        except Exception:
            return


def boundary_mask_and_paths(boundary_geom, width: int, height: int, xlim, ylim):
    xs = np.linspace(xlim[0], xlim[1], width)
    ys = np.linspace(ylim[0], ylim[1], height)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    mask = np.zeros(len(pts), dtype=bool)
    paths = []
    for poly in iter_polygons(boundary_geom):
        ext = np.asarray(poly.exterior.coords)
        path = MplPath(ext)
        inside = path.contains_points(pts)
        # subtract holes
        for ring in poly.interiors:
            inside &= ~MplPath(np.asarray(ring.coords)).contains_points(pts)
        mask |= inside
        paths.append(ext)
    return mask.reshape(height, width), xx, yy, paths


def compute_fixed_territories(nodes_df: pd.DataFrame, boundary_geom, width=720, height=900, gamma=0.5):
    lons = pd.to_numeric(nodes_df["longitude"], errors="coerce").to_numpy(float)
    lats = pd.to_numeric(nodes_df["latitude"], errors="coerce").to_numpy(float)
    weights = pd.to_numeric(nodes_df.get("mec_mw", 1.0), errors="coerce").fillna(1.0).to_numpy(float)
    weights = np.maximum(weights, 1e-6)

    minx, miny, maxx, maxy = boundary_geom.bounds
    dx = maxx - minx
    dy = maxy - miny
    pad_x = dx * 0.03
    pad_y = dy * 0.03
    xlim = (minx - pad_x, maxx + pad_x)
    ylim = (miny - pad_y, maxy + pad_y)

    mask, xx, yy, paths = boundary_mask_and_paths(boundary_geom, width, height, xlim, ylim)

    speed = (weights / np.median(weights)) ** gamma
    speed = np.clip(speed, 0.5, 2.0)

    best_score = np.full(xx.shape, np.inf, dtype=np.float32)
    owner = np.full(xx.shape, -1, dtype=np.int32)

    for idx, (x0, y0, s) in enumerate(zip(lons, lats, speed)):
        score = ((xx - x0) ** 2 + (yy - y0) ** 2) / (s ** 2)
        better = score < best_score
        owner[better] = idx
        best_score[better] = score[better]

    owner[~mask] = -1
    arrival = np.sqrt(best_score, where=np.isfinite(best_score), out=np.full_like(best_score, np.inf))
    arrival[~mask] = np.nan
    return {
        "owner": owner,
        "arrival": arrival,
        "mask": mask,
        "xx": xx,
        "yy": yy,
        "paths": paths,
        "xlim": xlim,
        "ylim": ylim,
        "speed": speed,
    }


def make_group_palette(group_ids: Sequence[int]):
    uniq = sorted({int(g) for g in group_ids})
    cmap = plt.get_cmap("tab20", max(20, len(uniq)))
    mapping = {}
    for i, g in enumerate(uniq):
        if i < 20:
            mapping[g] = cmap(i % cmap.N)
        else:
            mapping[g] = plt.cm.hsv(i / max(1, len(uniq)))
    return mapping


def grouping_array_to_colors(grouping: np.ndarray, palette: dict) -> np.ndarray:
    return np.array([palette[int(g)] for g in grouping], dtype=float)


def territory_rgba(owner: np.ndarray, grouping: np.ndarray, palette: dict, alpha: float = 0.82) -> np.ndarray:
    h, w = owner.shape
    rgba = np.zeros((h, w, 4), dtype=float)
    valid = owner >= 0
    if not np.any(valid):
        return rgba
    node_colors = grouping_array_to_colors(grouping, palette)
    idx = owner[valid]
    rgba[valid, :3] = node_colors[idx, :3]
    rgba[valid, 3] = alpha
    return rgba


def render_frame(
    territory,
    nodes_df: pd.DataFrame,
    snapshot: Snapshot,
    baseline_dd: float,
    palette: dict,
    out_path: Path,
    intro_reveal: Optional[float] = None,
    title: str = "All-island annealing",
):
    owner = territory["owner"]
    arrival = territory["arrival"]
    paths = territory["paths"]
    xlim = territory["xlim"]
    ylim = territory["ylim"]

    fig = plt.figure(figsize=(7.5, 9.5), dpi=120)
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.9])
    ax.set_facecolor("#f6f5ef")

    if intro_reveal is None:
        rgba = territory_rgba(owner, snapshot.grouping, palette, alpha=0.86)
    else:
        rgba = territory_rgba(owner, snapshot.grouping, palette, alpha=0.86)
        finite = np.isfinite(arrival)
        if np.any(finite):
            thresh = np.nanquantile(arrival[finite], intro_reveal)
            show = arrival <= thresh
            rgba[~show] = [0, 0, 0, 0]

    ax.imshow(
        rgba,
        origin="lower",
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        interpolation="nearest",
        zorder=1,
    )

    # boundary
    for ext in paths:
        ax.add_patch(MplPolygon(ext, closed=True, fill=False, edgecolor="#2f3b2f", linewidth=1.2, zorder=3))

    # node markers
    lons = pd.to_numeric(nodes_df["longitude"], errors="coerce").to_numpy(float)
    lats = pd.to_numeric(nodes_df["latitude"], errors="coerce").to_numpy(float)
    colors = grouping_array_to_colors(snapshot.grouping, palette)
    ax.scatter(lons, lats, s=10, c=colors, edgecolors="black", linewidths=0.15, zorder=4)

    # highlight changed nodes if any
    changed = [x for x in snapshot.changed_nodes if isinstance(x, (int, np.integer)) and x >= 0 and x < len(nodes_df)]
    if changed and intro_reveal is None:
        ax.scatter(lons[changed], lats[changed], s=80, facecolors="none", edgecolors="white", linewidths=1.6, zorder=5)
        ax.scatter(lons[changed], lats[changed], s=60, facecolors="none", edgecolors="black", linewidths=0.8, zorder=5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, weight="bold", pad=8)

    improvement_pp = baseline_dd - snapshot.best_dd
    rel = (improvement_pp / baseline_dd * 100.0) if baseline_dd else 0.0

    if intro_reveal is None:
        total_groups = len(set(int(g) for g in snapshot.grouping))

        info = (
        f"Iteration: {snapshot.eval_index}\n"
        f"Temperature: {snapshot.temperature:.5f}\n"
        f"Groups: {total_groups}\n"
        f"Current DD: {snapshot.current_dd:.3f}%\n"
        f"Best DD: {snapshot.best_dd:.3f}%\n"
        f"Baseline DD: {baseline_dd:.3f}%\n"
        f"Best improvement: {improvement_pp:.3f} pp ({rel:.1f}%)"
    )
    else:
        info = (
            "Static weighted territories\n"
            "(computed once from location + MEC)\n"
            f"Reveal: {intro_reveal*100:.0f}%"
        )

    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#666666", alpha=0.92),
        zorder=10,
    )


    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_gif(frames: List[Path], gif_path: Path, fps: int = 6):
    images = [imageio.imread(p) for p in frames]
    duration = 1.0 / fps
    imageio.mimsave(gif_path, images, duration=duration, loop=0)


# ------------------------------------------------------------
# Main CLI
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run the all-island annealer and create an OpenFront-style GIF.")
    ap.add_argument("--scope", default="32", help="26 or 32. '36' is treated as 32 for convenience.")
    ap.add_argument("--network", default="data/SV2024_all-island.nc")
    ap.add_argument("--nodes-csv", default=None, help="Optional path to the exclusive-group node CSV to feed make_emulator.")
    ap.add_argument(
    "--basemap",
    default=str(PROJECT_ROOT / "ireland_land_boundary.geojson"),
    help="Ireland land boundary GeoJSON."
)
    ap.add_argument("--seed", type=int, default=76, help="Annealing / weather seed passed to make_emulator.")
    ap.add_argument("--runs", type=int, default=10000, help="Number of weighted emulator runs for make_emulator.")
    ap.add_argument("--max-evals", type=int, default=5000, help="Annealer evaluations.")
    ap.add_argument("--frames", type=int, default=72, help="Annealing frames to save.")
    ap.add_argument("--intro-frames", type=int, default=10, help="Initial static territory reveal frames.")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--canvas-width", type=int, default=720)
    ap.add_argument("--canvas-height", type=int, default=900)
    ap.add_argument("--gamma", type=float, default=0.5, help="Territory weight exponent.")
    ap.add_argument("--output-dir", default="outputs/openfront_gif")
    args = ap.parse_args()

    scope = str(args.scope)
    if scope == "36":
        print("[note] treating requested 36-county scope as 32-county all-island scope.")
        scope = "32"
    if scope not in {"26", "32"}:
        raise SystemExit("--scope must be 26, 32, or 36 (alias to 32)")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"BUILDING EMULATOR | scope={scope} | seed={args.seed}")
    print("=" * 70)
    node_csv = args.nodes_csv
    if node_csv is None:
        node_csv = f"outputs/annealer_nodes_{scope}_counties_sv2024_wdt_exclusive.csv"
    dispatch_down, nodes = make_emulator(
        scope=scope,
        runs=args.runs,
        seed=args.seed,
        network_file=args.network,
        asset_file=node_csv,
    )

    # Ensure coordinates exist by merging with bus data if needed.
    if not {"longitude", "latitude"}.issubset(nodes.columns):
        buses = read_buses_from_nc(Path(args.network))
        nodes = nodes.merge(buses[["bus", "longitude", "latitude"]], on="bus", how="left")

    # Cast capacity if present.
    if "mec_mw" not in nodes.columns:
        nodes["mec_mw"] = 1.0

    print("=" * 70)
    print(f"RUNNING ANNEALER | max_evals={args.max_evals}")
    print("=" * 70)
    best_nodes, baseline_dd, best_dd, snapshots, history = run_annealer_with_snapshots(
        nodes=nodes,
        blackbox=dispatch_down,
        max_evals=args.max_evals,
        n_frames=args.frames,
        seed=args.seed,
    )

    print(f"Baseline DD: {baseline_dd:.4f}%")
    print(f"Best DD:     {best_dd:.4f}%")
    print(f"Improvement: {baseline_dd - best_dd:.4f} pp")

    # Save annealing outputs.
    best_csv = outdir / f"optimised_constraint_groups_{scope}_counties.csv"
    hist_csv = outdir / f"annealing_history_{scope}_counties.csv"
    snap_npz = outdir / f"annealing_snapshots_{scope}_counties.npz"
    meta_json = outdir / f"annealing_snapshot_meta_{scope}_counties.json"
    best_nodes.to_csv(best_csv, index=False)
    history.to_csv(hist_csv, index=False)
    np.savez_compressed(
        snap_npz,
        grouping=np.stack([s.grouping for s in snapshots], axis=0),
        eval_index=np.array([s.eval_index for s in snapshots], dtype=int),
        temperature=np.array([s.temperature for s in snapshots], dtype=float),
        current_dd=np.array([s.current_dd for s in snapshots], dtype=float),
        best_dd=np.array([s.best_dd for s in snapshots], dtype=float),
        changed_nodes=np.array([s.changed_nodes for s in snapshots], dtype=int),
        accepted=np.array([s.accepted for s in snapshots], dtype=bool),
        is_best=np.array([s.is_best for s in snapshots], dtype=bool),
    )
    meta_json.write_text(json.dumps({
        "baseline_dd": baseline_dd,
        "best_dd": best_dd,
        "scope": scope,
        "seed": args.seed,
        "max_evals": args.max_evals,
    }, indent=2))

    # Build fixed territories once.
    print("=" * 70)
    print("COMPUTING FIXED OPENFRONT-STYLE TERRITORIES")
    print("=" * 70)
    boundary = load_boundary(Path(args.basemap) if args.basemap else None, nodes)
    territory = compute_fixed_territories(
        nodes_df=nodes,
        boundary_geom=boundary,
        width=args.canvas_width,
        height=args.canvas_height,
        gamma=args.gamma,
    )
    palette = make_group_palette([parse_scalar_group(g) for g in nodes["groups"]])
    for snap in snapshots:
        for g in snap.grouping:
            palette.setdefault(int(g), plt.cm.hsv((int(g) % 30) / 30))

    # Render frames.
    frames_dir = outdir / "frames"
    frames_dir.mkdir(exist_ok=True)
    frame_paths: List[Path] = []

    # Intro reveal frames using initial snapshot.
    initial_snapshot = snapshots[0]
    if args.intro_frames > 0:
        for k in range(args.intro_frames):
            reveal = (k + 1) / args.intro_frames
            p = frames_dir / f"frame_{len(frame_paths):04d}.png"
            render_frame(
                territory=territory,
                nodes_df=nodes,
                snapshot=initial_snapshot,
                baseline_dd=baseline_dd,
                palette=palette,
                out_path=p,
                intro_reveal=reveal,
                title=f"{scope}-county annealing territories",
            )
            frame_paths.append(p)

    # Main annealing frames.
    for snap in snapshots:
        p = frames_dir / f"frame_{len(frame_paths):04d}.png"
        render_frame(
            territory=territory,
            nodes_df=nodes,
            snapshot=snap,
            baseline_dd=baseline_dd,
            palette=palette,
            out_path=p,
            intro_reveal=None,
            title=f"{scope}-county annealing territories",
        )
        frame_paths.append(p)

    # Pause on final best frame.
    final_snap = snapshots[-1]
    if frame_paths:
        for _ in range(6):
            p = frames_dir / f"frame_{len(frame_paths):04d}.png"
            render_frame(
                territory=territory,
                nodes_df=nodes,
                snapshot=final_snap,
                baseline_dd=baseline_dd,
                palette=palette,
                out_path=p,
                intro_reveal=None,
                title=f"{scope}-county annealing territories",
            )
            frame_paths.append(p)

    gif_path = outdir / f"annealing_openfront_{scope}_counties.gif"
    build_gif(frame_paths, gif_path, fps=args.fps)

    # Save a final still too.
    still_path = outdir / f"annealing_openfront_final_{scope}_counties.png"
    render_frame(
        territory=territory,
        nodes_df=nodes,
        snapshot=final_snap,
        baseline_dd=baseline_dd,
        palette=palette,
        out_path=still_path,
        intro_reveal=None,
        title=f"{scope}-county annealing territories (final)",
    )

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Best grouping CSV: {best_csv}")
    print(f"Annealing history: {hist_csv}")
    print(f"Snapshot NPZ:      {snap_npz}")
    print(f"GIF:               {gif_path}")
    print(f"Final still:       {still_path}")


if __name__ == "__main__":
    main()
