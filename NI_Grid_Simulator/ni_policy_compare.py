"""Current-vs-smaller constraint group policy comparison for Northern Ireland.

This is a research approximation, not a reproduction of SONI's control-room model.
It separates the physical operating scenario from the renewable constraint-group
policy so identical cases can be evaluated with the published five NI groups and
with smaller candidate memberships found by simulated annealing.

The operating approximation includes:
- realistic NI demand scaling using the supplied 168-snapshot spatial load shape,
- wind/solar availability capped by mapped MEC,
- an SNSP-style renewable/non-synchronous ceiling (default 75%),
- minimum synchronous thermal units and minimum-stable-output approximation,
- north-south and Moyle export headroom for surplus energy,
- published NI WDT group/corridor contingency states where represented,
- an all-line N-1 diagnostic using DC LODFs,
- diagnostic I^2 R transmission-line losses.

Group 4 is represented through an explicit north-south export limit proxy because
Louth and the physical cross-border circuits are absent from the supplied NI-only
network. This is intentionally labelled as a boundary proxy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import math
import time

import numpy as np
import pandas as pd

from ni_grid_core import DCGridModel, NetworkData, load_shares
from ni_constraint_stress import SONI_GROUPS, _required_reduction


# Published-corridor security states that can be represented in the supplied NI model.
# Each tuple is (monitored line, outage line or None, note).
GROUP_SECURITY_STATES: dict[int, list[tuple[str, Optional[str], str]]] = {
    1: [
        ("81510-87900-1", None, "Kells-Rasharkin intact loading"),
        ("75510-83510-1", "81510-87900-1", "Coolkeeragh-Limavady after Kells-Rasharkin outage"),
    ],
    2: [
        ("75510-89510-1", "75510-81810-1", "Coolkeeragh-Strabane after Coolkeeragh-Killymallaght outage"),
    ],
    3: [
        ("76810-87510-1", None, "Dromore-Omagh circuit 1"),
        ("76810-87510-2", None, "Dromore-Omagh circuit 2"),
    ],
    4: [],  # represented by north-south boundary proxy instead of a physical Louth circuit
    5: [
        ("75010-83510-1", "75510-83510-1", "Limavady-Coleraine after Coolkeeragh-Limavady outage"),
        ("75510-83510-1", "75010-83510-1", "Coolkeeragh-Limavady after Limavady-Coleraine outage"),
    ],
}


@dataclass
class OperationalConfig:
    runs: int = 10_000
    seed: int = 42
    batch_size: int = 250
    demand_scale: float = 2.10
    demand_jitter_pct: float = 6.0
    sns_limit_pct: float = 75.0
    min_sync_units: int = 2
    min_stable_output_pct: float = 20.0
    north_south_export_limit_mw: float = 400.0
    north_south_import_limit_mw: float = 200.0
    moyle_export_limit_mw: float = 410.0
    moyle_import_limit_mw: float = 441.0
    thermal_scale: float = 1.0
    demand_spatial_concentration: float = 220.0
    renewable_local_variability: float = 0.18
    moyle_mean_mw: float = 0.0
    north_south_mean_mw: float = -30.0
    coolkeeragh_must_run_mw: float = 0.0
    planned_outage_exposure_pct: float = 0.0
    include_n1_diagnostic: bool = True
    enforce_all_n1_with_groups: bool = True
    max_group_actions: int = 8


@dataclass
class OperatingScenarioSet:
    renewable_assets: pd.DataFrame
    thermal_assets: pd.DataFrame
    demand_mw: np.ndarray
    renewable_potential_mw: np.ndarray  # runs x renewable assets
    demand_dispatch_mw: np.ndarray      # runs x demand buses
    moyle_mw: np.ndarray                # + import / - export
    north_south_mw: np.ndarray          # + import / - export
    initial_thermal_dispatch_mw: np.ndarray  # runs x thermal assets
    pre_network_renewable_dispatch_mw: np.ndarray
    pre_network_dispatch_down_mw: np.ndarray
    pre_network_dd_sns_mw: np.ndarray
    pre_network_dd_min_sync_mw: np.ndarray
    pre_network_dd_surplus_mw: np.ndarray
    initial_injections_mw: np.ndarray   # runs x buses
    initial_flows_mw: np.ndarray        # runs x branches
    planned_outage_line_index: np.ndarray  # local line index; -1 means intact
    planned_outage_branch: np.ndarray
    metadata: dict


@dataclass
class PolicyResult:
    run_table: pd.DataFrame
    metrics: dict
    group_summary: pd.DataFrame
    line_summary: pd.DataFrame
    top_cases: list[dict] = field(default_factory=list)


@dataclass
class AnnealResult:
    current: PolicyResult
    candidate: PolicyResult
    membership_table: pd.DataFrame
    history: pd.DataFrame
    metadata: dict


def _renewables(mapped_assets: pd.DataFrame) -> pd.DataFrame:
    x = mapped_assets[
        mapped_assets["market_bucket"].isin(["wind", "solar"])
        & mapped_assets["mapped_bus"].notna()
    ].copy().reset_index(drop=True)
    x["max_export_capacity_mw"] = pd.to_numeric(x["max_export_capacity_mw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return x


def _thermals(mapped_assets: pd.DataFrame) -> pd.DataFrame:
    x = mapped_assets[
        mapped_assets["market_bucket"].eq("thermal")
        & mapped_assets["mapped_bus"].notna()
    ].copy().reset_index(drop=True)
    x["max_export_capacity_mw"] = pd.to_numeric(x["max_export_capacity_mw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return x[x["max_export_capacity_mw"] > 0].reset_index(drop=True)


def current_group_masks(renew: pd.DataFrame) -> dict[int, np.ndarray]:
    buses = renew["mapped_bus"].astype(str).to_numpy()
    all_buses = set(buses)
    out: dict[int, np.ndarray] = {}
    for gid, info in SONI_GROUPS.items():
        allowed = all_buses if gid == 4 else set(info["buses"])
        out[gid] = np.array([b in allowed for b in buses], dtype=bool)
    return out


def _gamma_weights(rng: np.random.Generator, base: np.ndarray, concentration: float, n: int) -> np.ndarray:
    base = np.asarray(base, float)
    base = np.clip(base, 0.0, None)
    base = base / base.sum() if base.sum() > 0 else np.ones_like(base) / len(base)
    alpha = np.maximum(base * max(float(concentration), 1e-3), 0.08)
    x = rng.gamma(alpha, 1.0, size=(n, len(base)))
    s = x.sum(axis=1, keepdims=True)
    bad = s[:, 0] <= 1e-15
    if np.any(bad):
        x[bad] = base
        s = x.sum(axis=1, keepdims=True)
    return x / s


def _thermal_dispatch(cap: np.ndarray, residual_mw: float, min_units: int, min_stable_pct: float) -> np.ndarray:
    """Simple deterministic commitment/dispatch proxy.

    Largest units are committed first. Online units carry at least their configured
    minimum stable output; additional requirement is filled by headroom.
    """
    cap = np.asarray(cap, float)
    out = np.zeros_like(cap)
    if len(cap) == 0:
        return out
    order = np.argsort(-cap)
    nmin = int(np.clip(min_units, 0, len(cap)))
    min_stable = cap * float(min_stable_pct) / 100.0
    online: list[int] = list(order[:nmin])
    min_total = float(min_stable[online].sum()) if online else 0.0
    target = max(float(residual_mw), min_total, 0.0)

    # Bring further units online only if the current committed capacity is insufficient.
    pos = nmin
    while online and cap[online].sum() + 1e-9 < target and pos < len(order):
        online.append(int(order[pos])); pos += 1
    if not online and target > 0:
        online = [int(order[0])]
    if online:
        out[online] = min_stable[online]
        remaining = max(0.0, target - float(out.sum()))
        headroom = np.maximum(cap - out, 0.0)
        while remaining > 1e-8:
            active = np.array([i for i in online if headroom[i] > 1e-9], dtype=int)
            if len(active) == 0 and pos < len(order):
                i = int(order[pos]); pos += 1; online.append(i)
                out[i] = min_stable[i]
                remaining = max(0.0, target - float(out.sum()))
                headroom = np.maximum(cap - out, 0.0)
                continue
            if len(active) == 0:
                break
            h = headroom[active]
            alloc = min(remaining, float(h.sum())) * h / h.sum()
            out[active] += alloc
            remaining = max(0.0, target - float(out.sum()))
            headroom = np.maximum(cap - out, 0.0)
    return np.minimum(out, cap)


def _line_lodf(model: DCGridModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return line-only PTDF, LODF, validity mask and line branch indices."""
    branch = model.branches.reset_index(drop=True)
    line_idx = np.flatnonzero(branch["type"].eq("line").to_numpy())
    line = branch.iloc[line_idx].reset_index(drop=True)
    ordinary = model.network.native_generators[model.network.native_generators["carrier"].ne("load shedding")]
    slack = str(ordinary.sort_values("p_nom_mw", ascending=False).iloc[0].bus) if len(ordinary) else str(model.bus_ids[0])
    Hfull = model.ptdf(slack)
    H = Hfull[line_idx, :]
    nline = len(line)
    lodf = np.full((nline, nline), np.nan, dtype=float)
    valid = np.ones(nline, dtype=bool)
    for k, r in line.iterrows():
        b0, b1 = model.bus_index[str(r.bus0)], model.bus_index[str(r.bus1)]
        transfer = H[:, b0] - H[:, b1]
        denom = 1.0 - float(transfer[k])
        if abs(denom) < 1e-5:
            valid[k] = False
            continue
        lodf[:, k] = transfer / denom
        lodf[k, k] = -1.0
    return H, lodf, valid, line_idx


def generate_operating_scenarios(
    network: NetworkData,
    model: DCGridModel,
    mapped_assets: pd.DataFrame,
    cfg: OperationalConfig,
    progress: Optional[Callable[[float, str], None]] = None,
) -> OperatingScenarioSet:
    """Generate a frozen set of plausible NI operating points.

    The supplied load profile is used for shape/spatial distribution, but scaled so
    its maximum aligns with the observed ~1.5 GW 2024/25 sent-out peak rather than
    treating the raw PyPSA values as a calibrated demand magnitude.
    """
    rng = np.random.default_rng(int(cfg.seed))
    runs = int(np.clip(cfg.runs, 1, 100_000))
    renew = _renewables(mapped_assets)
    therm = _thermals(mapped_assets)
    if renew.empty or therm.empty:
        raise ValueError("Mapped renewable and thermal assets are required for operational comparison mode.")

    nbus = len(model.bus_ids)
    rbus_idx = np.array([model.bus_index[str(b)] for b in renew["mapped_bus"]], dtype=int)
    tbus_idx = np.array([model.bus_index[str(b)] for b in therm["mapped_bus"]], dtype=int)
    rmec = renew["max_export_capacity_mw"].to_numpy(float)
    tcap = therm["max_export_capacity_mw"].to_numpy(float)

    shares = load_shares(network)
    dbuses = np.array([str(b) for b in shares.index if str(b) in model.bus_index], dtype=object)
    dbase = np.array([float(shares.loc[b]) for b in dbuses], float)
    dbase /= dbase.sum()
    dbus_idx = np.array([model.bus_index[b] for b in dbuses], dtype=int)

    # Sample supplied temporal demand shape, then scale to realistic NI magnitude.
    raw_total = network.load_profile.sum(axis=1).to_numpy(float)
    snap_idx = rng.integers(0, len(raw_total), size=runs)
    jitter = rng.normal(1.0, float(cfg.demand_jitter_pct) / 100.0, size=runs)
    demand = np.clip(raw_total[snap_idx] * float(cfg.demand_scale) * jitter, 350.0, 1800.0)
    dweights = _gamma_weights(rng, dbase, cfg.demand_spatial_concentration, runs)
    demand_dispatch = dweights * demand[:, None]

    # Shared weather factor + local deviations. Wind mean is deliberately moderate;
    # solar follows a crude daylight envelope so it cannot be high at every hour.
    hour = rng.integers(0, 24, size=runs)
    wind_common = rng.beta(2.2, 3.8, size=runs)
    cloud = rng.beta(4.0, 2.0, size=runs)
    daylight = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
    solar_common = np.clip(daylight * (0.45 + 0.55 * cloud), 0.0, 1.0)
    local_sigma = max(float(cfg.renewable_local_variability), 0.0)
    local = rng.lognormal(mean=-0.5 * local_sigma**2, sigma=local_sigma, size=(runs, len(renew)))
    common = np.where(renew["market_bucket"].to_numpy()[None, :] == "wind", wind_common[:, None], solar_common[:, None])
    ravail = np.clip(common * local, 0.0, 1.0)
    potential = ravail * rmec[None, :]

    # Random scheduled external flow. Positive means import into NI.
    moyle = np.clip(rng.normal(float(cfg.moyle_mean_mw), 110.0, size=runs), -float(cfg.moyle_export_limit_mw), float(cfg.moyle_import_limit_mw))
    ns = np.clip(rng.normal(float(cfg.north_south_mean_mw), 120.0, size=runs), -float(cfg.north_south_export_limit_mw), float(cfg.north_south_import_limit_mw))

    pre_net = potential.copy()
    dd_sns = np.zeros(runs)
    dd_min = np.zeros(runs)
    dd_surplus = np.zeros(runs)
    thermal_dispatch = np.zeros((runs, len(therm)), dtype=float)

    # Boundary bus locations for fixed external injections.
    moyle_candidates = mapped_assets[mapped_assets["market_bucket"].eq("interconnector") & mapped_assets["mapped_bus"].notna()]
    moyle_bus = str(moyle_candidates.iloc[0]["mapped_bus"]) if len(moyle_candidates) else "86220"
    ns_bus = "90020" if "90020" in model.bus_index else str(model.bus_ids[0])

    min_units = int(cfg.min_sync_units)
    min_stable_pct = float(cfg.min_stable_output_pct)
    largest = np.argsort(-tcap)[:max(0, min(min_units, len(tcap)))]
    min_thermal = float((tcap[largest] * min_stable_pct / 100.0).sum()) if len(largest) else 0.0
    thermal_names = therm["point_name"].astype(str).to_numpy()
    cool_candidates = [j for j, name in enumerate(thermal_names) if "Coolkeeragh Power Station GT" in name and "GT8" not in name]
    cool_idx = max(cool_candidates, key=lambda j: tcap[j]) if cool_candidates else None

    injections = np.zeros((runs, nbus), dtype=float)
    for i in range(runs):
        ren = pre_net[i].copy()
        ext_import = max(0.0, moyle[i]) + max(0.0, ns[i])
        ext_export = max(0.0, -moyle[i]) + max(0.0, -ns[i])
        sns_denom = max(demand[i] + ext_export, 1.0)
        allowed_non_sync = float(cfg.sns_limit_pct) / 100.0 * sns_denom
        allowed_ren = max(0.0, allowed_non_sync - ext_import)
        rt = float(ren.sum())
        if rt > allowed_ren + 1e-9:
            cut = rt - allowed_ren
            ren *= allowed_ren / rt if rt > 0 else 0.0
            dd_sns[i] += cut

        # Keep minimum synchronous output. Fixed scheduled imports/exports remain.
        residual = demand[i] - float(ren.sum()) - moyle[i] - ns[i]
        if residual < min_thermal:
            cut = min(min_thermal - residual, float(ren.sum()))
            if cut > 0:
                ren *= (float(ren.sum()) - cut) / float(ren.sum())
                dd_min[i] += cut
                residual += cut

        tdisp = _thermal_dispatch(tcap, residual, min_units, min_stable_pct)
        # If thermal capacity cannot meet residual, use extra import headroom before declaring imbalance.
        deficit = demand[i] - float(ren.sum()) - float(tdisp.sum()) - moyle[i] - ns[i]
        if deficit > 1e-6:
            add_ns = min(deficit, float(cfg.north_south_import_limit_mw) - ns[i])
            ns[i] += max(add_ns, 0.0); deficit -= max(add_ns, 0.0)
            add_m = min(deficit, float(cfg.moyle_import_limit_mw) - moyle[i])
            moyle[i] += max(add_m, 0.0); deficit -= max(add_m, 0.0)
        elif deficit < -1e-6:
            # Extra energy can be exported up to boundary headroom; only the remainder is surplus DD.
            surplus = -deficit
            ns_export_head = float(cfg.north_south_export_limit_mw) + ns[i]
            add_export = min(surplus, max(ns_export_head, 0.0))
            ns[i] -= add_export; surplus -= add_export
            moyle_export_head = float(cfg.moyle_export_limit_mw) + moyle[i]
            add_export_m = min(surplus, max(moyle_export_head, 0.0))
            moyle[i] -= add_export_m; surplus -= add_export_m
            if surplus > 1e-6 and ren.sum() > 0:
                cut = min(surplus, float(ren.sum()))
                ren *= (float(ren.sum()) - cut) / float(ren.sum())
                dd_surplus[i] += cut

        # Recalculate thermal exactly after export adjustments, maintaining the commitment floor.
        residual2 = demand[i] - float(ren.sum()) - moyle[i] - ns[i]
        tdisp = _thermal_dispatch(tcap, residual2, min_units, min_stable_pct)
        # Optional historical must-run proxy: shift synchronous output toward the
        # largest mapped Coolkeeragh GT while preserving total thermal output.
        must = max(0.0, float(cfg.coolkeeragh_must_run_mw))
        if cool_idx is not None and must > 0 and tdisp[cool_idx] + 1e-9 < min(must, tcap[cool_idx]):
            target_c = min(must, tcap[cool_idx])
            need = target_c - tdisp[cool_idx]
            donors = np.array([j for j in range(len(tdisp)) if j != cool_idx and tdisp[j] > 1e-9], dtype=int)
            transferable = float(tdisp[donors].sum()) if len(donors) else 0.0
            shift = min(need, transferable)
            if shift > 0 and transferable > 0:
                tdisp[donors] -= shift * tdisp[donors] / transferable
                tdisp[cool_idx] += shift
        thermal_dispatch[i] = tdisp
        pre_net[i] = ren

        p = np.zeros(nbus, dtype=float)
        np.add.at(p, rbus_idx, ren)
        np.add.at(p, tbus_idx, tdisp)
        p[model.bus_index[moyle_bus]] += moyle[i]
        p[model.bus_index[ns_bus]] += ns[i]
        np.add.at(p, dbus_idx, -demand_dispatch[i])
        # Tiny imbalance or simplified commitment mismatch goes at the largest thermal bus.
        balance_bus = str(therm.iloc[int(np.argmax(tcap))]["mapped_bus"])
        p[model.bus_index[balance_bus]] -= p.sum()
        injections[i] = p

    ordinary = network.native_generators[network.native_generators["carrier"].ne("load shedding")]
    slack = str(ordinary.sort_values("p_nom_mw", ascending=False).iloc[0].bus) if len(ordinary) else str(model.bus_ids[0])
    H = model.ptdf(slack)
    flows = injections @ H.T
    planned_idx = np.full(runs, -1, dtype=int)
    planned_branch = np.full(runs, "Intact", dtype=object)
    exposure = float(np.clip(cfg.planned_outage_exposure_pct, 0.0, 100.0)) / 100.0
    if exposure > 0:
        _Hl, _lodf, _valid, _line_idx = _line_lodf(model)
        _line_names = model.branches.iloc[_line_idx]["branch"].astype(str).to_numpy()
        _lookup = {b: j for j, b in enumerate(_line_names)}
        key_names = [
            "81510-87900-1", "75510-81810-1", "75510-83510-1",
            "75010-83510-1", "76810-87510-1", "76810-87510-2",
        ]
        key = np.array([_lookup[x] for x in key_names if x in _lookup and _valid[_lookup[x]]], dtype=int)
        chosen = np.flatnonzero(rng.random(runs) < exposure)
        if len(key) and len(chosen):
            ks = rng.choice(key, size=len(chosen), replace=True)
            for ii, kk in zip(chosen, ks):
                fl = flows[ii, _line_idx].copy()
                fl = fl + _lodf[:, kk] * fl[kk]
                fl[kk] = 0.0
                flows[ii, _line_idx] = fl
                planned_idx[ii] = int(kk)
                planned_branch[ii] = str(_line_names[kk])
    pre_dd = potential.sum(axis=1) - pre_net.sum(axis=1)

    return OperatingScenarioSet(
        renewable_assets=renew,
        thermal_assets=therm,
        demand_mw=demand,
        renewable_potential_mw=potential,
        demand_dispatch_mw=demand_dispatch,
        moyle_mw=moyle,
        north_south_mw=ns,
        initial_thermal_dispatch_mw=thermal_dispatch,
        pre_network_renewable_dispatch_mw=pre_net,
        pre_network_dispatch_down_mw=pre_dd,
        pre_network_dd_sns_mw=dd_sns,
        pre_network_dd_min_sync_mw=dd_min,
        pre_network_dd_surplus_mw=dd_surplus,
        initial_injections_mw=injections,
        initial_flows_mw=flows,
        planned_outage_line_index=planned_idx,
        planned_outage_branch=planned_branch,
        metadata={
            "runs": runs,
            "seed": int(cfg.seed),
            "moyle_bus": moyle_bus,
            "north_south_bus": ns_bus,
            "min_thermal_mw": min_thermal,
            "demand_mean_mw": float(demand.mean()),
            "demand_max_mw": float(demand.max()),
            "renewable_potential_mean_mw": float(potential.sum(axis=1).mean()),
            "planned_outage_exposure_pct": float(cfg.planned_outage_exposure_pct),
            "coolkeeragh_must_run_mw": float(cfg.coolkeeragh_must_run_mw),
        },
    )


def _security_state_indices(model: DCGridModel, line_idx: np.ndarray) -> tuple[dict[str, int], dict[int, list[tuple[int, Optional[int], str]]]]:
    line_branch = model.branches.iloc[line_idx]["branch"].astype(str).to_numpy()
    lookup = {b: i for i, b in enumerate(line_branch)}
    states: dict[int, list[tuple[int, Optional[int], str]]] = {}
    for gid, defs in GROUP_SECURITY_STATES.items():
        rows = []
        for monitor, outage, note in defs:
            if monitor not in lookup:
                continue
            if outage is not None and outage not in lookup:
                continue
            rows.append((lookup[monitor], lookup[outage] if outage is not None else None, note))
        states[gid] = rows
    return lookup, states


def _state_flow(flow_line: np.ndarray, lodf: np.ndarray, monitor: int, outage: Optional[int]) -> float:
    if outage is None:
        return float(flow_line[monitor])
    v = lodf[monitor, outage]
    if not np.isfinite(v):
        return math.nan
    return float(flow_line[monitor] + v * flow_line[outage])


def _state_sensitivity(sens_line: np.ndarray, lodf: np.ndarray, monitor: int, outage: Optional[int]) -> float:
    if outage is None:
        return float(sens_line[monitor])
    v = lodf[monitor, outage]
    if not np.isfinite(v):
        return math.nan
    return float(sens_line[monitor] + v * sens_line[outage])


def _all_n1_worst(flow_line: np.ndarray, limits: np.ndarray, lodf: np.ndarray, valid_outage: np.ndarray) -> tuple[float, int, int]:
    """Worst post-contingency loading ratio, monitor index, outage index."""
    post = flow_line[:, None] + lodf * flow_line[None, :]
    load = np.abs(post) / limits[:, None]
    load[:, ~valid_outage] = -np.inf
    np.fill_diagonal(load, -np.inf)
    flat = int(np.argmax(load))
    m, k = np.unravel_index(flat, load.shape)
    return float(load[m, k]), int(m), int(k)


def evaluate_policy(
    network: NetworkData,
    model: DCGridModel,
    scenarios: OperatingScenarioSet,
    cfg: OperationalConfig,
    masks: dict[int, np.ndarray],
    progress: Optional[Callable[[float, str], None]] = None,
    progress_start: float = 0.0,
    progress_span: float = 1.0,
) -> PolicyResult:
    start = time.perf_counter()
    renew = scenarios.renewable_assets
    therm = scenarios.thermal_assets
    runs = len(scenarios.demand_mw)
    branch = model.branches.reset_index(drop=True)
    H_line, lodf, valid_outage, line_idx = _line_lodf(model)
    line_df = branch.iloc[line_idx].reset_index(drop=True)
    line_ids = line_df["branch"].astype(str).to_numpy()
    limits = line_df["s_nom_mva"].to_numpy(float) * float(cfg.thermal_scale)
    lookup, group_states = _security_state_indices(model, line_idx)

    ordinary = network.native_generators[network.native_generators["carrier"].ne("load shedding")]
    slack = str(ordinary.sort_values("p_nom_mw", ascending=False).iloc[0].bus) if len(ordinary) else str(model.bus_ids[0])
    Hfull = model.ptdf(slack)
    rbus_idx = np.array([model.bus_index[str(b)] for b in renew["mapped_bus"]], dtype=int)
    tcap = therm["max_export_capacity_mw"].to_numpy(float)
    balance_bus = str(therm.iloc[int(np.argmax(tcap))]["mapped_bus"])
    bidx = model.bus_index[balance_bus]
    # Sensitivity of line flow to +1 MW renewable at asset bus and -1 MW at balancing thermal bus.
    transfer_asset = (Hfull[line_idx[:, None], rbus_idx[None, :]] - Hfull[line_idx, bidx][:, None]).T  # asset x line

    # I^2R diagnostic factors for base-case line flows.
    bus_v = network.buses.set_index("bus")["v_nom_kv"].to_dict()
    loss_factor = np.array([
        float(r.r_ohm) / max(float(bus_v.get(str(r.bus0), 110.0)) ** 2, 1.0)
        for _, r in line_df.iterrows()
    ])

    group_first = {g: 0 for g in masks}
    group_any = {g: 0 for g in masks}
    group_dd = {g: 0.0 for g in masks}
    line_over = np.zeros(len(line_df), dtype=np.int64)
    line_worst = np.zeros(len(line_df), dtype=np.int64)
    line_load_sum = np.zeros(len(line_df), dtype=float)
    line_max = np.zeros(len(line_df), dtype=float)
    rows = []
    top_cases: list[dict] = []
    top_n = 5

    network_dd_total = 0.0
    accepted_renew_total = 0.0
    potential_total = float(scenarios.renewable_potential_mw.sum())
    pre_dd_total = float(scenarios.pre_network_dispatch_down_mw.sum())
    losses_total = 0.0
    generated_total = 0.0
    security_pass = 0
    n1_pass = 0

    for i in range(runs):
        ffull = scenarios.initial_flows_mw[i].copy()
        f = ffull[line_idx].copy()
        planned_k = int(scenarios.planned_outage_line_index[i]) if hasattr(scenarios, "planned_outage_line_index") else -1
        dispatch = scenarios.pre_network_renewable_dispatch_mw[i].copy()
        seq: list[int] = []
        curts: list[float] = []
        used: set[int] = set()

        # Base diagnostics before group actions.
        load0 = 100.0 * np.abs(f) / limits
        wi = int(np.argmax(load0))
        line_worst[wi] += 1
        line_over += (load0 > 100.0).astype(np.int64)
        line_load_sum += load0
        line_max = np.maximum(line_max, load0)

        # Apply published group policy to its represented security states.
        for _ in range(int(cfg.max_group_actions)):
            candidates = []
            for gid in (1, 2, 3, 5):
                if gid in used or gid not in masks:
                    continue
                mask = np.asarray(masks[gid], bool)
                available = float(dispatch[mask].sum())
                if available <= 1e-8 or not group_states.get(gid):
                    continue
                weights = dispatch[mask] / available
                # Curtail renewable and replace it at the balancing thermal bus.
                sens_line = -(transfer_asset[mask].T @ weights)
                if planned_k >= 0 and np.isfinite(lodf[:, planned_k]).any():
                    sens_line = sens_line + lodf[:, planned_k] * sens_line[planned_k]
                    sens_line[planned_k] = 0.0
                required = 0.0
                severity = 0.0
                feasible = True
                violated = False
                for monitor, outage, _note in group_states[gid]:
                    effective_outage = None if planned_k >= 0 else outage
                    sf = _state_flow(f, lodf, monitor, effective_outage)
                    if not np.isfinite(sf):
                        continue
                    lim = limits[monitor]
                    ratio = abs(sf) / lim
                    if ratio > 1.0 + 1e-9:
                        violated = True
                        ss = _state_sensitivity(sens_line, lodf, monitor, effective_outage)
                        req = _required_reduction(sf, lim, ss)
                        if not np.isfinite(req):
                            feasible = False
                            break
                        required = max(required, float(req))
                        severity = max(severity, ratio)
                if violated and feasible:
                    candidates.append((-severity, required, gid, available, weights, sens_line, mask))
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            _, required, gid, available, weights, sens_line, mask = candidates[0]
            delta = min(required * 1.000001 + 1e-7, available)
            if delta <= 1e-8:
                used.add(gid)
                continue
            f += delta * sens_line
            # Keep full branch vector consistent enough for map reconstruction: update line branches only.
            ffull[line_idx] = f
            dispatch[mask] -= delta * weights
            dispatch[dispatch < 1e-10] = 0.0
            used.add(gid)
            seq.append(gid); curts.append(delta)

        # Broader N-1 security action.  The published group memberships are preserved,
        # but when a credible line outage creates another thermal overload the model
        # chooses the existing group with the strongest useful electrical relief.
        # This is a WDT-like approximation rather than an exact reconstruction of
        # SONI operator logic.
        if cfg.enforce_all_n1_with_groups:
            for _ in range(int(cfg.max_group_actions)):
                if planned_k >= 0:
                    base_ratio = np.abs(f) / limits
                    mon = int(np.argmax(base_ratio)); out = None; ratio = float(base_ratio[mon])
                else:
                    ratio, mon, out = _all_n1_worst(f, limits, lodf, valid_outage)
                if ratio <= 1.0 + 1e-8:
                    break
                sf = _state_flow(f, lodf, mon, out)
                candidates = []
                for gid in (1, 2, 3, 5):
                    mask = np.asarray(masks.get(gid, np.zeros(len(dispatch), dtype=bool)), bool)
                    available = float(dispatch[mask].sum())
                    if available <= 1e-8:
                        continue
                    weights = dispatch[mask] / available
                    sens_line = -(transfer_asset[mask].T @ weights)
                    if planned_k >= 0 and np.isfinite(lodf[:, planned_k]).any():
                        sens_line = sens_line + lodf[:, planned_k] * sens_line[planned_k]
                        sens_line[planned_k] = 0.0
                    ss = _state_sensitivity(sens_line, lodf, mon, out)
                    req = _required_reduction(sf, limits[mon], ss)
                    if np.isfinite(req) and req > 1e-9:
                        candidates.append((float(req), gid, available, weights, sens_line, mask))
                if not candidates:
                    break
                candidates.sort(key=lambda x: (x[0], x[1]))
                required, gid, available, weights, sens_line, mask = candidates[0]
                delta = min(required * 1.000001 + 1e-7, available)
                if delta <= 1e-8:
                    break
                f += delta * sens_line
                ffull[line_idx] = f
                dispatch[mask] -= delta * weights
                dispatch[dispatch < 1e-10] = 0.0
                seq.append(gid); curts.append(delta)

        # Group 4 boundary proxy: north-south export beyond the configured safe bound.
        # Normally scenario generation clips to this limit, but this keeps the policy explicit
        # for future historical/market-driven scenario inputs.
        ns_export = max(0.0, -float(scenarios.north_south_mw[i]))
        if ns_export > float(cfg.north_south_export_limit_mw) + 1e-9 and 4 in masks:
            mask = np.asarray(masks[4], bool)
            available = float(dispatch[mask].sum())
            delta = min(ns_export - float(cfg.north_south_export_limit_mw), available)
            if delta > 0 and available > 0:
                weights = dispatch[mask] / available
                sens_line = -(transfer_asset[mask].T @ weights)
                f += delta * sens_line
                ffull[line_idx] = f
                dispatch[mask] -= delta * weights
                seq.append(4); curts.append(delta)

        network_dd = float(sum(curts))
        network_dd_total += network_dd
        dd_by_group: dict[int, float] = {}
        for gid, mw in zip(seq, curts):
            dd_by_group[gid] = dd_by_group.get(gid, 0.0) + float(mw)
        for gid, mw in dd_by_group.items():
            group_any[gid] += 1
            group_dd[gid] += mw
        if seq:
            group_first[seq[0]] += 1

        # Current published-state security check.
        represented_secure = True
        for gid in (1, 2, 3, 5):
            for monitor, outage, _note in group_states.get(gid, []):
                effective_outage = None if planned_k >= 0 else outage
                sf = _state_flow(f, lodf, monitor, effective_outage)
                if np.isfinite(sf) and abs(sf) > limits[monitor] + 1e-6:
                    represented_secure = False
        if represented_secure:
            security_pass += 1

        # Broader N-1 diagnostic (not all such constraints map to one of the five published groups).
        n1_ratio = 0.0; n1_m = -1; n1_k = -1
        if cfg.include_n1_diagnostic:
            if planned_k >= 0:
                _base = np.abs(f) / limits
                n1_m = int(np.argmax(_base)); n1_k = planned_k; n1_ratio = float(_base[n1_m])
            else:
                n1_ratio, n1_m, n1_k = _all_n1_worst(f, limits, lodf, valid_outage)
            if n1_ratio <= 1.0 + 1e-6:
                n1_pass += 1
        else:
            n1_ratio = float(np.max(np.abs(f) / limits))
            if n1_ratio <= 1.0 + 1e-6:
                n1_pass += 1

        accepted = float(dispatch.sum())
        accepted_renew_total += accepted
        losses = float(np.sum((f ** 2) * loss_factor))
        losses_total += losses
        total_gen = float(accepted + scenarios.initial_thermal_dispatch_mw[i].sum() + max(0.0, scenarios.moyle_mw[i]) + max(0.0, scenarios.north_south_mw[i]))
        generated_total += max(total_gen, 1e-6)

        row = {
            "run": i + 1,
            "demand_mw": float(scenarios.demand_mw[i]),
            "renewable_potential_mw": float(scenarios.renewable_potential_mw[i].sum()),
            "pre_network_dispatch_down_mw": float(scenarios.pre_network_dispatch_down_mw[i]),
            "network_group_dispatch_down_mw": network_dd,
            "total_dispatch_down_mw": float(scenarios.pre_network_dispatch_down_mw[i] + network_dd),
            "renewable_accepted_mw": accepted,
            "first_group": seq[0] if seq else np.nan,
            "activation_sequence": " → ".join(f"G{x}" for x in seq) if seq else "None",
            "represented_security_pass": represented_secure,
            "all_line_n1_pass": bool(n1_ratio <= 1.0 + 1e-6),
            "worst_base_loading_pct": float(np.max(100.0 * np.abs(f) / limits)),
            "worst_n1_loading_pct": float(100.0 * n1_ratio),
            "worst_n1_monitored_line": line_ids[n1_m] if n1_m >= 0 else "—",
            "worst_n1_outage": line_ids[n1_k] if n1_k >= 0 else "—",
            "planned_outage": str(scenarios.planned_outage_branch[i]) if hasattr(scenarios, "planned_outage_branch") else "Intact",
            "estimated_line_losses_mw": losses,
        }
        rows.append(row)
        top_cases.append({**row, "initial_injections_mw": scenarios.initial_injections_mw[i].astype(np.float32, copy=True)})
        top_cases.sort(key=lambda c: (c["worst_n1_loading_pct"], c["total_dispatch_down_mw"]), reverse=True)
        if len(top_cases) > top_n:
            del top_cases[top_n:]

        if progress is not None and ((i + 1) % max(25, int(cfg.batch_size)) == 0 or i == runs - 1):
            frac = (i + 1) / runs
            progress(progress_start + progress_span * frac, f"Policy evaluation {i+1:,}/{runs:,}")

    run_table = pd.DataFrame(rows)
    total_dd = pre_dd_total + network_dd_total
    dispatch_down_pct = 100.0 * total_dd / potential_total if potential_total > 0 else 0.0
    renewable_util = 100.0 * accepted_renew_total / potential_total if potential_total > 0 else 100.0
    electrical_eff = 100.0 * max(0.0, 1.0 - losses_total / generated_total) if generated_total > 0 else 100.0

    group_rows = []
    for gid in (1, 2, 3, 4, 5):
        group_rows.append({
            "group": gid,
            "name": SONI_GROUPS[gid]["name"],
            "members": int(np.asarray(masks[gid], bool).sum()),
            "first_activation_pct": 100.0 * group_first.get(gid, 0) / runs,
            "activation_pct": 100.0 * group_any.get(gid, 0) / runs,
            "mean_network_dd_when_activated_mw": group_dd.get(gid, 0.0) / max(group_any.get(gid, 0), 1),
        })

    line_rows = []
    for li, lid in enumerate(line_ids):
        line_rows.append({
            "line": lid,
            "overload_pct_before_group_actions": 100.0 * line_over[li] / runs,
            "worst_base_line_pct_of_runs": 100.0 * line_worst[li] / runs,
            "mean_initial_loading_pct": line_load_sum[li] / runs,
            "max_initial_loading_pct": line_max[li],
            "thermal_limit_mw": limits[li],
        })

    metrics = {
        "runs": runs,
        "dispatch_down_pct": dispatch_down_pct,
        "renewable_utilisation_pct": renewable_util,
        "electrical_efficiency_pct": electrical_eff,
        "mean_total_dispatch_down_mw": total_dd / runs,
        "mean_network_group_dispatch_down_mw": network_dd_total / runs,
        "mean_pre_network_dispatch_down_mw": pre_dd_total / runs,
        "sns_dispatch_down_mw_total": float(scenarios.pre_network_dd_sns_mw.sum()),
        "minimum_sync_dispatch_down_mw_total": float(scenarios.pre_network_dd_min_sync_mw.sum()),
        "surplus_dispatch_down_mw_total": float(scenarios.pre_network_dd_surplus_mw.sum()),
        "network_group_dispatch_down_mw_total": network_dd_total,
        "represented_security_pass_pct": 100.0 * security_pass / runs,
        "all_line_n1_pass_pct": 100.0 * n1_pass / runs,
        "potential_renewable_mw_total": potential_total,
        "accepted_renewable_mw_total": accepted_renew_total,
        "estimated_line_losses_mw_total": losses_total,
        "elapsed_seconds": time.perf_counter() - start,
    }
    return PolicyResult(
        run_table=run_table,
        metrics=metrics,
        group_summary=pd.DataFrame(group_rows),
        line_summary=pd.DataFrame(line_rows).sort_values(["overload_pct_before_group_actions", "max_initial_loading_pct"], ascending=False),
        top_cases=top_cases,
    )


def _objective(candidate: PolicyResult, baseline: PolicyResult, masks: dict[int, np.ndarray], baseline_masks: dict[int, np.ndarray], size_weight: float) -> float:
    """Annealing objective with case-by-case security non-degradation.

    A smaller candidate is not allowed to improve its score by trading one secure
    baseline case for a different secure case.  Each operating state that is secure
    with the current groups is therefore protected individually.
    """
    b = baseline.run_table
    c = candidate.run_table
    if len(b) != len(c):
        raise ValueError("Baseline and candidate must use identical frozen operating cases.")
    lost_repr_cases = int((b["represented_security_pass"].astype(bool) & ~c["represented_security_pass"].astype(bool)).sum())
    if "all_line_n1_pass" in b.columns and "all_line_n1_pass" in c.columns:
        lost_n1_cases = int((b["all_line_n1_pass"].astype(bool) & ~c["all_line_n1_pass"].astype(bool)).sum())
    else:
        lost_n1_cases = 0
    runs = max(len(b), 1)
    lost_security_pct = 100.0 * (lost_repr_cases + lost_n1_cases) / runs
    dd = candidate.metrics["dispatch_down_pct"]
    cur_members = sum(int(np.asarray(baseline_masks[g]).sum()) for g in (1,2,3,5))
    cand_members = sum(int(np.asarray(masks[g]).sum()) for g in (1,2,3,5))
    size_ratio = cand_members / max(cur_members, 1)
    # A single newly-insecure case should dominate plausible dispatch-down savings.
    return float(dd + 10_000.0 * lost_security_pct + float(size_weight) * size_ratio)


def _no_new_security_failures(candidate: PolicyResult, baseline: PolicyResult) -> bool:
    b = baseline.run_table
    c = candidate.run_table
    lost_repr = bool((b["represented_security_pass"].astype(bool) & ~c["represented_security_pass"].astype(bool)).any())
    lost_n1 = bool((b["all_line_n1_pass"].astype(bool) & ~c["all_line_n1_pass"].astype(bool)).any())
    return not (lost_repr or lost_n1)


def anneal_smaller_groups(
    network: NetworkData,
    model: DCGridModel,
    scenarios: OperatingScenarioSet,
    cfg: OperationalConfig,
    iterations: int = 250,
    training_runs: int = 500,
    initial_temperature: float = 1.0,
    cooling: float = 0.985,
    size_weight: float = 0.20,
    progress: Optional[Callable[[float, str], None]] = None,
) -> AnnealResult:
    """Find smaller subsets of the current groups using simulated annealing.

    Search is intentionally limited to removing/re-adding renewable assets already
    in each published group. This is conservative: it asks whether a broad current
    group can be made smaller without losing represented security capability.
    """
    rng = np.random.default_rng(int(cfg.seed) + 99173)
    base_masks = current_group_masks(scenarios.renewable_assets)
    full_current = evaluate_policy(network, model, scenarios, cfg, base_masks, progress=progress, progress_start=0.0, progress_span=0.20)

    ntrain = int(np.clip(training_runs, 50, len(scenarios.demand_mw)))
    # Deterministic high-value training subset: severe current network-DD cases first,
    # then fill from seeded random cases so rare and ordinary conditions are retained.
    severe = full_current.run_table.sort_values(["network_group_dispatch_down_mw", "worst_n1_loading_pct"], ascending=False).index.to_numpy()
    keep = list(severe[: min(ntrain // 2, len(severe))])
    remaining = np.setdiff1d(np.arange(len(severe)), np.array(keep, dtype=int), assume_unique=False)
    if len(keep) < ntrain and len(remaining):
        keep.extend(rng.choice(remaining, size=min(ntrain-len(keep), len(remaining)), replace=False).tolist())
    keep = np.array(sorted(set(keep[:ntrain])), dtype=int)

    def subset_scen(s: OperatingScenarioSet, ix: np.ndarray) -> OperatingScenarioSet:
        return OperatingScenarioSet(
            renewable_assets=s.renewable_assets,
            thermal_assets=s.thermal_assets,
            demand_mw=s.demand_mw[ix],
            renewable_potential_mw=s.renewable_potential_mw[ix],
            demand_dispatch_mw=s.demand_dispatch_mw[ix],
            moyle_mw=s.moyle_mw[ix],
            north_south_mw=s.north_south_mw[ix],
            initial_thermal_dispatch_mw=s.initial_thermal_dispatch_mw[ix],
            pre_network_renewable_dispatch_mw=s.pre_network_renewable_dispatch_mw[ix],
            pre_network_dispatch_down_mw=s.pre_network_dispatch_down_mw[ix],
            pre_network_dd_sns_mw=s.pre_network_dd_sns_mw[ix],
            pre_network_dd_min_sync_mw=s.pre_network_dd_min_sync_mw[ix],
            pre_network_dd_surplus_mw=s.pre_network_dd_surplus_mw[ix],
            initial_injections_mw=s.initial_injections_mw[ix],
            initial_flows_mw=s.initial_flows_mw[ix],
            planned_outage_line_index=s.planned_outage_line_index[ix],
            planned_outage_branch=s.planned_outage_branch[ix],
            metadata={**s.metadata, "runs": len(ix)},
        )

    train = subset_scen(scenarios, keep)
    base_train = evaluate_policy(network, model, train, cfg, base_masks)
    current_masks = {g: m.copy() for g, m in base_masks.items()}
    current_res = base_train
    current_score = _objective(current_res, base_train, current_masks, base_masks, size_weight)
    best_masks = {g: m.copy() for g, m in current_masks.items()}
    best_res = current_res
    best_score = current_score
    T = max(float(initial_temperature), 1e-6)
    hist = []

    mutable = []
    for g in (1,2,3,5):
        for idx in np.flatnonzero(base_masks[g]):
            mutable.append((g, int(idx)))
    if not mutable:
        raise ValueError("No renewable assets are mapped into Groups 1, 2, 3 or 5, so there is nothing to anneal.")

    iterations = int(np.clip(iterations, 1, 5000))
    for it in range(iterations):
        cand_masks = {g: m.copy() for g, m in current_masks.items()}
        gid, idx = mutable[int(rng.integers(0, len(mutable)))]
        # Candidate remains a subset of the published group. Do not empty a group completely.
        if cand_masks[gid][idx] and cand_masks[gid].sum() <= 1:
            continue
        cand_masks[gid][idx] = ~cand_masks[gid][idx]
        # Never add an asset outside the original membership.
        cand_masks[gid] &= base_masks[gid]
        cand = evaluate_policy(network, model, train, cfg, cand_masks)
        score = _objective(cand, base_train, cand_masks, base_masks, size_weight)
        delta = score - current_score
        accept = delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9))
        if accept:
            current_masks = cand_masks; current_res = cand; current_score = score
        # The final design must be a real improvement candidate on the training set:
        # no newly-insecure baseline case and no increase in renewable dispatch-down.
        final_feasible = (
            _no_new_security_failures(cand, base_train)
            and cand.metrics["dispatch_down_pct"] <= base_train.metrics["dispatch_down_pct"] + 1e-9
        )
        if final_feasible and score < best_score:
            best_masks = {g: m.copy() for g, m in cand_masks.items()}; best_res = cand; best_score = score
        T *= float(cooling)
        hist.append({
            "iteration": it + 1,
            "temperature": T,
            "current_score": current_score,
            "best_score": best_score,
            "best_dispatch_down_pct_training": best_res.metrics["dispatch_down_pct"],
            "best_security_pass_pct_training": best_res.metrics["represented_security_pass_pct"],
            "best_members": sum(int(best_masks[g].sum()) for g in (1,2,3,5)),
        })
        if progress is not None and ((it + 1) % 5 == 0 or it == iterations - 1):
            progress(0.20 + 0.55 * (it + 1) / iterations, f"Annealing {it+1:,}/{iterations:,} · best score {best_score:.3f}")

    candidate_full = evaluate_policy(network, model, scenarios, cfg, best_masks, progress=progress, progress_start=0.75, progress_span=0.25)
    rows = []
    for g in (1,2,3,4,5):
        base = base_masks[g]
        cand = best_masks[g]
        for i, a in scenarios.renewable_assets.iterrows():
            if base[i] or cand[i]:
                rows.append({
                    "group": g,
                    "group_name": SONI_GROUPS[g]["name"],
                    "asset": a.get("point_name", str(i)),
                    "bus": str(a.get("mapped_bus", "")),
                    "mec_mw": float(a.get("max_export_capacity_mw", 0.0)),
                    "current_member": bool(base[i]),
                    "annealed_member": bool(cand[i]),
                    "change": "kept" if base[i] and cand[i] else ("removed" if base[i] and not cand[i] else "added"),
                })
    # Full-sample validation: savings count only if the smaller groups do not make
    # any operating case insecure that was secure under the current memberships.
    bfull = full_current.run_table
    cfull = candidate_full.run_table
    lost_repr_full = int((bfull["represented_security_pass"].astype(bool) & ~cfull["represented_security_pass"].astype(bool)).sum())
    lost_n1_full = int((bfull["all_line_n1_pass"].astype(bool) & ~cfull["all_line_n1_pass"].astype(bool)).sum())
    security_validated = (lost_repr_full == 0 and lost_n1_full == 0)
    dispatch_down_non_increasing = (
        candidate_full.metrics["dispatch_down_pct"] <= full_current.metrics["dispatch_down_pct"] + 1e-9
    )
    policy_accepted = bool(security_validated and dispatch_down_non_increasing)

    meta = {
        "iterations": iterations,
        "training_runs": ntrain,
        "initial_temperature": float(initial_temperature),
        "cooling": float(cooling),
        "size_weight": float(size_weight),
        "current_members_1_2_3_5": sum(int(base_masks[g].sum()) for g in (1,2,3,5)),
        "candidate_members_1_2_3_5": sum(int(best_masks[g].sum()) for g in (1,2,3,5)),
        "lost_represented_secure_cases_full": lost_repr_full,
        "lost_n1_secure_cases_full": lost_n1_full,
        "security_validated": security_validated,
        "dispatch_down_non_increasing": dispatch_down_non_increasing,
        "policy_accepted": policy_accepted,
    }
    return AnnealResult(
        current=full_current,
        candidate=candidate_full,
        membership_table=pd.DataFrame(rows),
        history=pd.DataFrame(hist),
        metadata=meta,
    )
