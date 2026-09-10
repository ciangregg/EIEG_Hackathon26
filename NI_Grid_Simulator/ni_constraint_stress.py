"""Fast Monte-Carlo stress testing of SONI Northern Ireland WDT constraint groups.

This module is deliberately a thermal-flow experiment.  Each trial creates a
balanced random spatial pattern of renewable injection and demand, so energy
surplus, SNSP/inertia and economic dispatch do not trigger renewable reductions.
Group dispatch-down is only applied when one of the represented target circuits
exceeds its thermal rating.

The SONI group memberships and purposes are taken from the Wind Dispatch Tool
Constraint Group Overview (01 February 2024).  The supplied NI-only network does
not include the Tandragee-Louth cross-border tie-lines, so Group 4 (All NI) is
retained in the catalogue but cannot be physically triggered by this network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import math
import time

import numpy as np
import pandas as pd

from ni_grid_core import DCGridModel, NetworkData, load_shares


# Transmission-node membership from SONI's published NI WDT groups, mapped to
# bus IDs/aliases present in the supplied network model.  Some stations have
# multiple bus sections, so all relevant local sections are included.
SONI_GROUPS = {
    1: {
        "name": "NW Northern",
        "stations": ["Brockaghboy", "Lisaghmore", "Coleraine", "Loguestown", "Coolkeeragh", "Rasharkin", "Limavady", "Garvagh"],
        "buses": {"73500", "84411", "84412", "75010", "84511", "84512", "75510", "75511", "75514", "75520", "75521", "75522", "87900", "83510", "73510"},
        "target_lines": ["75510-83510-1", "81510-87900-1"],
        "purpose": "NW Northern / Coolkeeragh-Limavady or Kells-Rasharkin thermal relief",
        "model_status": "represented",
    },
    2: {
        "name": "NW Southern",
        "stations": ["Aghyoule", "Gort", "Drumquin", "Magherakeel", "Strabane", "Omagh", "Enniskillen", "Tremoge", "Killymallaght", "Slieve Kirk"],
        "buses": {"70010", "80710", "77210", "85510", "89510", "87510", "79010", "90210", "81810", "89210"},
        "target_lines": ["75510-89510-1"],
        "purpose": "NW Southern / Coolkeeragh-Strabane thermal relief proxy in intact topology",
        "model_status": "represented as intact-network proxy; SONI definition is contingency-driven",
    },
    3: {
        "name": "Oma-Dromore",
        "stations": ["Aghyoule", "Drumquin", "Enniskillen"],
        "buses": {"70010", "77210", "79010"},
        "target_lines": ["76810-87510-1", "76810-87510-2"],
        "purpose": "Dromore-Omagh 110 kV thermal relief",
        "model_status": "represented",
    },
    4: {
        "name": "All NI",
        "stations": ["All controllable renewable NI generation"],
        "buses": set(),  # populated dynamically with all renewable buses
        "target_lines": [],
        "purpose": "Tandragee-Louth 275/220 kV tie-line flow management",
        "model_status": "membership represented; physical trigger unavailable in NI-only network",
    },
    5: {
        "name": "Kells - Rasharkin",
        "stations": ["Brockaghboy", "Garvagh", "Coleraine", "Loguestown", "Limavady", "Rasharkin"],
        "buses": {"73500", "73510", "75010", "84511", "84512", "83510", "87900"},
        "target_lines": ["75510-83510-1", "75010-83510-1"],
        "purpose": "Coolkeeragh-Limavady / Limavady-Coleraine thermal relief",
        "model_status": "represented; SONI definition is especially relevant during local outages",
    },
}


@dataclass
class StressTestConfig:
    runs: int = 10_000
    seed: int = 42
    min_generation_pct_of_mec: float = 100.0
    max_generation_pct_of_mec: float = 300.0
    thermal_scale: float = 1.0
    generation_concentration: float = 35.0
    demand_concentration: float = 120.0
    batch_size: int = 500
    max_group_actions: int = 5


@dataclass
class StressTestResult:
    run_table: pd.DataFrame
    group_summary: pd.DataFrame
    line_summary: pd.DataFrame
    sequence_summary: pd.DataFrame
    group_catalogue: pd.DataFrame
    metadata: dict
    top_cases: list[dict] = field(default_factory=list)


def _renewable_assets(mapped_assets: pd.DataFrame) -> pd.DataFrame:
    a = mapped_assets[
        mapped_assets["market_bucket"].isin(["wind", "solar"])
        & mapped_assets["mapped_bus"].notna()
    ].copy().reset_index(drop=True)
    a["max_export_capacity_mw"] = pd.to_numeric(a["max_export_capacity_mw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return a


def group_catalogue(mapped_assets: pd.DataFrame, model: DCGridModel) -> pd.DataFrame:
    renew = _renewable_assets(mapped_assets)
    all_renew_buses = set(renew["mapped_bus"].astype(str))
    rows = []
    line_ids = set(model.network.lines["branch"].astype(str))
    for gid, info in SONI_GROUPS.items():
        buses = all_renew_buses if gid == 4 else set(info["buses"])
        mask = renew["mapped_bus"].astype(str).isin(buses)
        lines = [x for x in info["target_lines"] if x in line_ids]
        rows.append({
            "group": gid,
            "name": info["name"],
            "published_stations": ", ".join(info["stations"]),
            "mapped_renewable_assets": int(mask.sum()),
            "mapped_mec_mw": float(renew.loc[mask, "max_export_capacity_mw"].sum()),
            "model_target_lines": ", ".join(lines) if lines else "—",
            "model_status": info["model_status"],
        })
    return pd.DataFrame(rows)


def _gamma_weights(rng: np.random.Generator, base: np.ndarray, concentration: float, n: int) -> np.ndarray:
    """Random positive weights centred on base shares, row-normalised."""
    base = np.asarray(base, float)
    base = np.clip(base, 0.0, None)
    if base.sum() <= 0:
        base = np.ones_like(base) / len(base)
    else:
        base = base / base.sum()
    # Keep alpha away from zero so tiny assets do not create numerical pathologies.
    alpha = np.maximum(base * max(float(concentration), 1e-3), 0.08)
    x = rng.gamma(shape=alpha, scale=1.0, size=(n, len(base)))
    sums = x.sum(axis=1, keepdims=True)
    bad = sums[:, 0] <= 1e-15
    if np.any(bad):
        x[bad, :] = base
        sums = x.sum(axis=1, keepdims=True)
    return x / sums


def _required_reduction(flow: float, limit: float, sensitivity: float) -> float:
    """MW reduction required to bring one signed flow to its thermal boundary.

    sensitivity is delta-flow per +1 MW of total group dispatch-down, balanced at
    the model slack.  Returns inf when dispatch-down in this group cannot relieve
    the current signed overload.
    """
    if flow > limit + 1e-9:
        if sensitivity >= -1e-10:
            return math.inf
        return max(0.0, (flow - limit) / (-sensitivity))
    if flow < -limit - 1e-9:
        if sensitivity <= 1e-10:
            return math.inf
        return max(0.0, (-limit - flow) / sensitivity)
    return 0.0


def _single_run_dispatch(
    flow: np.ndarray,
    limits: np.ndarray,
    asset_dispatch: np.ndarray,
    H_asset: np.ndarray,
    group_masks: dict[int, np.ndarray],
    group_targets: dict[int, np.ndarray],
    line_loading_indices: np.ndarray,
    max_actions: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[float], bool]:
    """Sequentially apply represented SONI groups to their overloaded target lines.

    Candidate selection is deterministic: highest target-line loading first, then
    least required MW reduction as a tie-breaker.  Each group is used at most once
    per trial.  This is a transparent stress-test heuristic, not a claim about the
    exact real-time WDT operator sequence.
    """
    f = flow.copy()
    dispatch = asset_dispatch.copy()
    used: set[int] = set()
    seq: list[int] = []
    curtailed: list[float] = []

    for _ in range(max_actions):
        candidates = []
        for gid in (1, 2, 3, 5):
            if gid in used:
                continue
            tidx = group_targets.get(gid, np.array([], dtype=int))
            if len(tidx) == 0:
                continue
            load = np.abs(f[tidx]) / limits[tidx]
            overloaded = load > 1.0 + 1e-9
            if not np.any(overloaded):
                continue
            mask = group_masks[gid]
            available = float(dispatch[mask].sum())
            if available <= 1e-9:
                continue
            weights = dispatch[mask] / available
            sens_all = -(H_asset[mask, :].T @ weights)  # branch delta per 1 MW curtailment
            reqs = []
            helpful = True
            for li in tidx[overloaded]:
                req = _required_reduction(float(f[li]), float(limits[li]), float(sens_all[li]))
                if not np.isfinite(req):
                    helpful = False
                    break
                reqs.append(req)
            if not helpful or not reqs:
                continue
            required = max(reqs)
            severity = float(np.max(load[overloaded]))
            candidates.append((-severity, required, gid, available, sens_all, mask, weights))

        if not candidates:
            break
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, required, gid, available, sens_all, mask, weights = candidates[0]
        delta = min(float(required) * 1.000001 + 1e-7, available)
        if delta <= 1e-8:
            used.add(gid)
            continue
        f += delta * sens_all
        dispatch[mask] -= delta * weights
        dispatch[dispatch < 1e-10] = 0.0
        used.add(gid)
        seq.append(gid)
        curtailed.append(delta)

    unresolved = bool(np.any(np.abs(f[line_loading_indices]) > limits[line_loading_indices] + 1e-7))
    return f, dispatch, seq, curtailed, unresolved


def run_stress_test(
    network: NetworkData,
    model: DCGridModel,
    mapped_assets: pd.DataFrame,
    config: StressTestConfig,
    progress: Optional[Callable[[float, str], None]] = None,
) -> StressTestResult:
    start = time.perf_counter()
    cfg = config
    runs = int(np.clip(cfg.runs, 1, 100_000))
    batch_size = int(np.clip(cfg.batch_size, 25, 5000))
    rng = np.random.default_rng(int(cfg.seed))

    renew = _renewable_assets(mapped_assets)
    if renew.empty:
        raise ValueError("No mapped wind/solar assets are available for the stress test.")

    # Pick the same slack convention as the rest of the simulator.
    ordinary = network.native_generators[network.native_generators["carrier"].ne("load shedding")]
    slack_bus = str(ordinary.sort_values("p_nom_mw", ascending=False).iloc[0].bus) if len(ordinary) else str(network.buses.sort_values("v_nom_kv", ascending=False).iloc[0].bus)
    H = model.ptdf(slack_bus)  # branch x bus
    branch_df = model.branches.reset_index(drop=True)
    line_idx = np.flatnonzero(branch_df["type"].eq("line").to_numpy())
    line_ids = branch_df["branch"].astype(str).to_numpy()
    limits = branch_df["s_nom_mva"].to_numpy(float) * float(cfg.thermal_scale)
    if np.any(limits[line_idx] <= 0):
        raise ValueError("One or more line thermal limits are non-positive.")

    nbus = len(model.bus_ids)
    bus_index = model.bus_index
    asset_buses = renew["mapped_bus"].astype(str).to_numpy()
    asset_bus_idx = np.array([bus_index[b] for b in asset_buses], dtype=int)
    # H response for +1 MW at each asset bus balanced at the slack: asset x branch.
    H_asset = H[:, asset_bus_idx].T.copy()

    # Renewable spatial base weights use MEC only as a location/size prior; in this
    # stress mode it is not an availability cap.
    mec = renew["max_export_capacity_mw"].to_numpy(float)
    total_mec = float(mec.sum())
    gen_base = mec / total_mec if total_mec > 0 else np.ones(len(renew)) / len(renew)

    demand_share_by_bus = load_shares(network)
    demand_buses = np.array([str(b) for b in demand_share_by_bus.index if str(b) in bus_index], dtype=object)
    demand_base = np.array([float(demand_share_by_bus.loc[b]) for b in demand_buses], dtype=float)
    demand_base /= demand_base.sum()
    demand_bus_idx = np.array([bus_index[b] for b in demand_buses], dtype=int)

    # Matrix for fast bus aggregation.
    asset_to_bus = np.zeros((len(renew), nbus), dtype=float)
    asset_to_bus[np.arange(len(renew)), asset_bus_idx] = 1.0
    demand_to_bus = np.zeros((len(demand_buses), nbus), dtype=float)
    demand_to_bus[np.arange(len(demand_buses)), demand_bus_idx] = 1.0

    all_renew_buses = set(asset_buses)
    group_masks = {}
    for gid, info in SONI_GROUPS.items():
        buses = all_renew_buses if gid == 4 else set(info["buses"])
        group_masks[gid] = np.array([b in buses for b in asset_buses], dtype=bool)

    branch_lookup = {str(b): i for i, b in enumerate(line_ids)}
    group_targets = {
        gid: np.array([branch_lookup[x] for x in info["target_lines"] if x in branch_lookup], dtype=int)
        for gid, info in SONI_GROUPS.items()
    }

    # Per-run collectors.
    rows = []
    first_group_counts = {g: 0 for g in SONI_GROUPS}
    activation_counts = {g: 0 for g in SONI_GROUPS}
    group_curtail_sum = {g: 0.0 for g in SONI_GROUPS}
    line_worst_count = np.zeros(len(branch_df), dtype=np.int64)
    line_overload_count = np.zeros(len(branch_df), dtype=np.int64)
    line_loading_sum = np.zeros(len(branch_df), dtype=float)
    line_loading_max = np.zeros(len(branch_df), dtype=float)
    # Compact float32 matrix for percentiles; ~4.9 MB for 10,000 x 122 NI lines.
    line_loading_samples = np.empty((runs, len(line_idx)), dtype=np.float32)
    seq_counts: dict[str, int] = {}
    top_cases: list[dict] = []
    top_case_count = 5

    done = 0
    for b0 in range(0, runs, batch_size):
        nb = min(batch_size, runs - b0)
        gen_w = _gamma_weights(rng, gen_base, cfg.generation_concentration, nb)
        dem_w = _gamma_weights(rng, demand_base, cfg.demand_concentration, nb)
        stress_pct = rng.uniform(float(cfg.min_generation_pct_of_mec), float(cfg.max_generation_pct_of_mec), size=nb)
        totals = total_mec * stress_pct / 100.0
        gen_dispatch = gen_w * totals[:, None]
        demand_dispatch = dem_w * totals[:, None]
        p = gen_dispatch @ asset_to_bus - demand_dispatch @ demand_to_bus
        # Floating error only; balance exactly at slack.
        p[:, bus_index[slack_bus]] -= p.sum(axis=1)
        flows = p @ H.T

        for j in range(nb):
            f0 = flows[j]
            loading0 = 100.0 * np.abs(f0) / limits
            line_load = loading0[line_idx]
            worst_local = int(np.argmax(line_load))
            worst_idx = int(line_idx[worst_local])
            line_worst_count[worst_idx] += 1
            line_overload_count[line_idx] += (line_load > 100.0).astype(np.int64)
            line_loading_sum[line_idx] += line_load
            line_loading_max[line_idx] = np.maximum(line_loading_max[line_idx], line_load)
            line_loading_samples[b0 + j, :] = line_load.astype(np.float32, copy=False)

            f_final, dispatch_final, seq, curts, unresolved = _single_run_dispatch(
                f0, limits, gen_dispatch[j], H_asset, group_masks, group_targets, line_idx, int(cfg.max_group_actions)
            )
            for gid, mw in zip(seq, curts):
                activation_counts[gid] += 1
                group_curtail_sum[gid] += float(mw)
            if seq:
                first_group_counts[seq[0]] += 1
            seq_key = " → ".join(f"G{x}" for x in seq) if seq else "No represented group triggered"
            seq_counts[seq_key] = seq_counts.get(seq_key, 0) + 1

            loading_final = 100.0 * np.abs(f_final[line_idx]) / limits[line_idx]
            initial_over = int(np.sum(line_load > 100.0))
            final_over = int(np.sum(loading_final > 100.0 + 1e-7))
            rows.append({
                "run": b0 + j + 1,
                "stress_pct_of_mapped_mec": float(stress_pct[j]),
                "total_generation_demand_mw": float(totals[j]),
                "initial_worst_line": str(line_ids[worst_idx]),
                "initial_worst_loading_pct": float(line_load[worst_local]),
                "initial_overloaded_lines": initial_over,
                "first_group": (int(seq[0]) if seq else np.nan),
                "activation_sequence": seq_key,
                "groups_activated": len(seq),
                "dispatch_down_mw": float(sum(curts)),
                "dispatch_down_pct": (100.0 * float(sum(curts)) / float(totals[j])) if float(totals[j]) > 1e-9 else 0.0,
                "final_worst_loading_pct": float(np.max(loading_final)),
                "final_overloaded_lines": final_over,
                "unresolved_overload": bool(unresolved),
            })

            case = {
                "run": int(b0 + j + 1),
                "stress_pct_of_mapped_mec": float(stress_pct[j]),
                "total_generation_demand_mw": float(totals[j]),
                "initial_worst_line": str(line_ids[worst_idx]),
                "initial_worst_loading_pct": float(line_load[worst_local]),
                "final_worst_loading_pct": float(np.max(loading_final)),
                "dispatch_down_mw": float(sum(curts)),
                "activation_sequence": seq_key,
                "slack_bus": slack_bus,
                "thermal_scale": float(cfg.thermal_scale),
                "initial_injections_mw": p[j].astype(np.float32, copy=True),
                "initial_flow_mw": f0.astype(np.float32, copy=True),
                "final_flow_mw": f_final.astype(np.float32, copy=True),
            }
            top_cases.append(case)
            top_cases.sort(key=lambda c: (c["initial_worst_loading_pct"], c["dispatch_down_mw"]), reverse=True)
            if len(top_cases) > top_case_count:
                del top_cases[top_case_count:]

        done += nb
        if progress is not None:
            elapsed = max(time.perf_counter() - start, 1e-6)
            rate = done / elapsed
            eta = (runs - done) / rate if rate > 0 else 0.0
            progress(done / runs, f"Stress test {done:,}/{runs:,} runs · {rate:,.0f} runs/s · ETA {eta:.1f}s")

    run_table = pd.DataFrame(rows)

    group_rows = []
    for gid, info in SONI_GROUPS.items():
        act = activation_counts[gid]
        group_rows.append({
            "group": gid,
            "name": info["name"],
            "first_activation_count": first_group_counts[gid],
            "first_activation_pct": 100.0 * first_group_counts[gid] / runs,
            "activation_count": act,
            "activation_pct": 100.0 * act / runs,
            "avg_dispatch_down_when_activated_mw": group_curtail_sum[gid] / act if act else 0.0,
            "total_dispatch_down_mw_across_trials": group_curtail_sum[gid],
            "model_status": info["model_status"],
        })
    group_summary = pd.DataFrame(group_rows).sort_values(["first_activation_count", "activation_count"], ascending=False)

    line_rows = []
    for kk, li in enumerate(line_idx):
        vals = line_loading_samples[:, kk].astype(float, copy=False)
        line_rows.append({
            "line": str(line_ids[li]),
            "worst_line_count": int(line_worst_count[li]),
            "worst_line_pct_of_runs": 100.0 * line_worst_count[li] / runs,
            "overload_count": int(line_overload_count[li]),
            "overload_pct_of_runs": 100.0 * line_overload_count[li] / runs,
            "mean_loading_pct": float(line_loading_sum[li] / runs),
            "p95_loading_pct": float(np.percentile(vals, 95)) if len(vals) else 0.0,
            "max_loading_pct": float(line_loading_max[li]),
            "thermal_limit_mw": float(limits[li]),
        })
    line_summary = pd.DataFrame(line_rows).sort_values(["overload_count", "p95_loading_pct"], ascending=False)

    sequence_summary = pd.DataFrame([
        {"activation_sequence": k, "count": v, "pct_of_runs": 100.0 * v / runs}
        for k, v in seq_counts.items()
    ]).sort_values("count", ascending=False)

    elapsed = time.perf_counter() - start
    metadata = {
        "runs": runs,
        "seed": int(cfg.seed),
        "elapsed_seconds": elapsed,
        "runs_per_second": runs / elapsed if elapsed > 0 else math.inf,
        "total_mapped_renewable_mec_mw": total_mec,
        "slack_bus": slack_bus,
        "thermal_scale": float(cfg.thermal_scale),
        "unresolved_runs": int(run_table["unresolved_overload"].sum()),
        "any_initial_overload_runs": int((run_table["initial_overloaded_lines"] > 0).sum()),
        "total_available_renewable_mw_across_trials": float(run_table["total_generation_demand_mw"].sum()),
        "total_dispatch_down_mw_across_trials": float(run_table["dispatch_down_mw"].sum()),
        "aggregate_dispatch_down_pct": (
            100.0 * float(run_table["dispatch_down_mw"].sum()) / float(run_table["total_generation_demand_mw"].sum())
            if float(run_table["total_generation_demand_mw"].sum()) > 1e-9 else 0.0
        ),
        "mean_run_dispatch_down_pct": float(run_table["dispatch_down_pct"].mean()),
        "p95_run_dispatch_down_pct": float(np.percentile(run_table["dispatch_down_pct"].to_numpy(float), 95)),
    }
    return StressTestResult(
        run_table=run_table,
        group_summary=group_summary.reset_index(drop=True),
        line_summary=line_summary.reset_index(drop=True),
        sequence_summary=sequence_summary.reset_index(drop=True),
        group_catalogue=group_catalogue(mapped_assets, model),
        metadata=metadata,
        top_cases=top_cases,
    )
