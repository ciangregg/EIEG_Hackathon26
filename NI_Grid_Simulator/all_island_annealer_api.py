"""Annealer-facing API for the 26-county and 32-county WP2033 grid models.

The public contract intentionally matches ``ni_annealer_api``::

    from all_island_annealer_api import make_emulator

    dispatch_down, nodes = make_emulator(scope="32", runs=10_000, seed=42)
    score_pct = dispatch_down(nodes)

The annealer edits only ``nodes['groups']`` (or the equivalent scalar ``group``
column) and repeatedly calls ``dispatch_down(candidate_nodes)``.  The network,
168 WP2033 demand/renewable snapshots, sampled case weights, security-state
screen and all other model data are frozen by ``make_emulator``.

Scope
-----
``scope='26'`` models the Republic of Ireland AC grid in the supplied WP2033
file.  Northern Ireland AC buses and the North-South AC tie-lines are removed.
External HVDC schedules whose Irish terminal remains in scope are represented as
fixed boundary injections.

``scope='32'`` models the connected Republic of Ireland + Northern Ireland AC
grid.  External HVDC schedules are again represented as fixed injections at the
in-scope terminal.  The isolated Moyle external/converter-side bus is excluded
from the AC admittance matrix and its scheduled flow is represented at the
connected NI terminal.

This is a DC thermal-security research emulator.  It does not force either
2033 case to match the NI 2024 25.5% historical dispatch-down calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import ast
import math

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ni_grid_core import DCGridModel, NetworkData, read_network_nc


ROOT = Path(__file__).resolve().parent
DEFAULT_NETWORK_FILE = ROOT / "data" / "SV2024_all-island.nc"
DEFAULT_NODES_26 = ROOT / "outputs" / "annealer_nodes_26_counties_exclusive.csv"
DEFAULT_NODES_32 = ROOT / "outputs" / "annealer_nodes_32_counties_exclusive.csv"
EPS = 1e-9


@dataclass(frozen=True)
class _ExtraData:
    bus_meta: pd.DataFrame
    generators: pd.DataFrame
    renewable_profile_ids: tuple[str, ...]
    renewable_profiles: np.ndarray
    links: pd.DataFrame


@dataclass(frozen=True)
class _FrozenCases:
    renewable_potential_mw: np.ndarray      # snapshot x node
    pre_network_dispatch_mw: np.ndarray     # snapshot x node
    pre_network_dispatch_down_mw: np.ndarray# snapshot x node
    branch_flows_mw: np.ndarray             # snapshot x branch
    case_weights: np.ndarray                # snapshot, sums to 1
    node_state_sensitivity: np.ndarray       # node x screened security state
    state_base_flows_mw: np.ndarray          # snapshot x screened state
    state_limits_mw: np.ndarray              # screened state
    state_monitor: np.ndarray                # screened state
    state_outage: np.ndarray                 # -1 for intact/base state
    state_lodf_coeff: np.ndarray             # screened state
    balance_bus: str
    shortage_mw: np.ndarray                  # snapshot


def _decode(values) -> list[str]:
    return [
        v.decode("utf-8", errors="replace") if isinstance(v, (bytes, np.bytes_)) else str(v)
        for v in values
    ]


def _scope_value(scope: str | int) -> str:
    s = str(scope).strip().lower().replace("-county", "").replace("counties", "").strip()
    aliases = {"26": "26", "roi": "26", "ie": "26", "republic": "26",
               "32": "32", "all": "32", "all-island": "32", "island": "32"}
    if s not in aliases:
        raise ValueError("scope must be '26' (Republic of Ireland) or '32' (all-island)")
    return aliases[s]


def _normalise_single_group(value: Any) -> int:
    """Accept scalar or one-item tuple/list/string and return one positive group ID."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Every renewable node must have exactly one constraint group")
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError("Every renewable node must have exactly one constraint group")
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            parsed = s
        value = parsed
    if isinstance(value, (int, np.integer)):
        g = int(value)
    elif isinstance(value, float) and float(value).is_integer():
        g = int(value)
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        vals = list(value)
        if len(vals) != 1:
            raise ValueError(f"Exclusive-group model requires exactly one group per node; got {value!r}")
        g = int(vals[0])
    else:
        # Handles strings such as "7" after failed literal parsing.
        try:
            g = int(str(value).strip())
        except Exception as exc:
            raise TypeError(f"Unsupported group value {value!r}") from exc
    if g <= 0:
        raise ValueError(f"Constraint group IDs must be positive integers; got {g}")
    return g


def _read_extra(path: Path) -> _ExtraData:
    with h5py.File(path, "r") as f:
        bus_ids = _decode(f["buses_i"][:])
        bus_meta = pd.DataFrame({
            "bus": bus_ids,
            "jurisdiction": _decode(f["buses_jurisdiction"][:]),
            "area": np.asarray(f["buses_area"][:], float),
        })
        if "buses_station" in f:
            bus_meta["station"] = _decode(f["buses_station"][:])
        else:
            bus_meta["station"] = bus_ids

        generators = pd.DataFrame({
            "generator": _decode(f["generators_i"][:]),
            "bus": _decode(f["generators_bus"][:]),
            "carrier": _decode(f["generators_carrier"][:]),
            "p_nom_mw": np.asarray(f["generators_p_nom"][:], float),
            "marginal_cost": np.asarray(f["generators_marginal_cost"][:], float),
            "p_min_pu": np.asarray(f["generators_p_min_pu"][:], float),
        })

        renewable_profile_ids = tuple(_decode(f["generators_t_p_max_pu_i"][:]))
        renewable_profiles = np.asarray(f["generators_t_p_max_pu"][:], float)

        links = pd.DataFrame({
            "name": _decode(f["links_i"][:]),
            "bus0": _decode(f["links_bus0"][:]),
            "bus1": _decode(f["links_bus1"][:]),
            "p_nom_mw": np.asarray(f["links_p_nom"][:], float),
            "p_set_mw": np.asarray(f["links_p_set_tytfs"][:], float),
            "efficiency": np.asarray(f["links_efficiency"][:], float),
        }) if "links_i" in f else pd.DataFrame(columns=["name", "bus0", "bus1", "p_nom_mw", "p_set_mw", "efficiency"])

    return _ExtraData(bus_meta, generators, renewable_profile_ids, renewable_profiles, links)


def _largest_component_buses(network: NetworkData) -> set[str]:
    buses = network.buses["bus"].astype(str).tolist()
    idx = {b: i for i, b in enumerate(buses)}
    row: list[int] = []
    col: list[int] = []
    for frame in (network.lines, network.transformers):
        for b0, b1 in zip(frame["bus0"].astype(str), frame["bus1"].astype(str)):
            if b0 in idx and b1 in idx:
                i, j = idx[b0], idx[b1]
                row.extend((i, j)); col.extend((j, i))
    graph = coo_matrix((np.ones(len(row)), (row, col)), shape=(len(buses), len(buses))).tocsr()
    ncomp, labels = connected_components(graph, directed=False)
    if ncomp <= 1:
        return set(buses)
    sizes = np.bincount(labels)
    keep_label = int(np.argmax(sizes))
    return {b for b, lab in zip(buses, labels) if int(lab) == keep_label}


def _subset_network(full: NetworkData, extra: _ExtraData, scope: str) -> tuple[NetworkData, set[str]]:
    allowed_j = {"IE"} if scope == "26" else {"IE", "NI"}
    meta = extra.bus_meta.set_index("bus")
    physical = set(meta.index[meta["jurisdiction"].isin(allowed_j)])

    # Three-winding transformer star buses have jurisdiction '--'.  Keep a star
    # only when every physical transformer leg belongs to the selected scope.
    stars: set[str] = set()
    for sb in full.buses.loc[full.buses["bus"].astype(str).str.startswith("star:"), "bus"].astype(str):
        neigh = set(full.transformers.loc[full.transformers["bus0"].eq(sb), "bus1"].astype(str))
        neigh |= set(full.transformers.loc[full.transformers["bus1"].eq(sb), "bus0"].astype(str))
        nonstar = {b for b in neigh if not b.startswith("star:")}
        if nonstar and nonstar.issubset(physical):
            stars.add(sb)

    selected = physical | stars

    def filt(selected_buses: set[str]) -> NetworkData:
        buses = full.buses[full.buses["bus"].astype(str).isin(selected_buses)].copy().reset_index(drop=True)
        buses = buses.merge(extra.bus_meta[["bus", "jurisdiction", "area"]], on="bus", how="left")
        lines = full.lines[
            full.lines["bus0"].astype(str).isin(selected_buses) &
            full.lines["bus1"].astype(str).isin(selected_buses)
        ].copy().reset_index(drop=True)
        transformers = full.transformers[
            full.transformers["bus0"].astype(str).isin(selected_buses) &
            full.transformers["bus1"].astype(str).isin(selected_buses)
        ].copy().reset_index(drop=True)
        loads = full.loads[full.loads["bus"].astype(str).isin(selected_buses)].copy().reset_index(drop=True)
        load_ids = loads["load"].astype(str).tolist()
        load_profile = full.load_profile.loc[:, [x for x in load_ids if x in full.load_profile.columns]].copy()
        native_generators = full.native_generators[
            full.native_generators["bus"].astype(str).isin(selected_buses)
        ].copy().reset_index(drop=True)
        return NetworkData(buses, lines, transformers, loads, load_profile, native_generators, full.snapshots)

    first = filt(selected)
    connected = _largest_component_buses(first)
    final = filt(connected)
    return final, connected


def _build_node_table(network: NetworkData, extra: _ExtraData) -> pd.DataFrame:
    selected = set(network.buses["bus"].astype(str))
    gm = extra.generators[
        extra.generators["bus"].astype(str).isin(selected) &
        extra.generators["carrier"].str.lower().isin(["wind", "solar"])
    ].copy()
    meta = extra.bus_meta.set_index("bus")
    rows = []
    for bus, g in gm.groupby("bus", sort=False):
        wind = float(g.loc[g["carrier"].str.lower().eq("wind"), "p_nom_mw"].sum())
        solar = float(g.loc[g["carrier"].str.lower().eq("solar"), "p_nom_mw"].sum())
        m = meta.loc[str(bus)]
        area = int(m["area"]) if np.isfinite(float(m["area"])) else 1
        station = str(m.get("station", "")).strip() or f"Bus {bus}"
        tech = "wind+solar" if wind > 0 and solar > 0 else "wind" if wind > 0 else "solar"
        brow = network.buses.loc[network.buses["bus"].astype(str).eq(str(bus))].iloc[0]
        rows.append({
            "name": station,
            "bus": str(bus),
            "groups": (area,),
            "group": area,
            "mec_mw": wind + solar,
            "technology": tech,
            "latitude": float(brow.get("lat", np.nan)),
            "longitude": float(brow.get("lon", np.nan)),
            "jurisdiction": str(m["jurisdiction"]),
            "psse_area": area,
            "v_nom_kv": float(brow.get("v_nom_kv", np.nan)),
            "wind_mw": wind,
            "solar_mw": solar,
            "generator_count": int(len(g)),
        })
    out = pd.DataFrame(rows)
    out = out.sort_values(["jurisdiction", "psse_area", "name", "bus"], kind="stable").reset_index(drop=True)
    out.insert(0, "node_id", np.arange(len(out), dtype=int))
    return out


def _node_groups(nodes: pd.DataFrame, template: pd.DataFrame) -> np.ndarray:
    if not isinstance(nodes, pd.DataFrame):
        nodes = pd.DataFrame(nodes)
    if "node_id" not in nodes.columns:
        raise ValueError("nodes must contain a 'node_id' column")
    group_col = "groups" if "groups" in nodes.columns else "group" if "group" in nodes.columns else None
    if group_col is None:
        raise ValueError("nodes must contain 'groups' (or scalar 'group')")

    ids = pd.to_numeric(nodes["node_id"], errors="raise").astype(int)
    if ids.duplicated().any():
        raise ValueError("node_id values must be unique")
    expected = set(template["node_id"].astype(int))
    got = set(ids.tolist())
    if got != expected:
        raise ValueError(f"nodes must contain every renewable node exactly once; missing={sorted(expected-got)[:10]}, extra={sorted(got-expected)[:10]}")

    ordered = nodes.assign(_node_id=ids).set_index("_node_id").loc[template["node_id"].astype(int)]
    return np.array([_normalise_single_group(v) for v in ordered[group_col]], dtype=int)


def _build_lodf(model: DCGridModel, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return branch x outage LODF matrix and valid-outage mask."""
    b = model.branches.reset_index(drop=True)
    B = len(b)
    lodf = np.full((B, B), np.nan, dtype=float)
    valid = np.zeros(B, dtype=bool)
    for k, br in b.iterrows():
        i = model.bus_index[str(br.bus0)]
        j = model.bus_index[str(br.bus1)]
        transaction = H[:, i] - H[:, j]
        denom = 1.0 - float(transaction[k])
        if abs(denom) < 1e-7:
            continue
        lodf[:, k] = transaction / denom
        lodf[k, k] = -1.0
        valid[k] = True
    return lodf, valid


def _screen_security_states(
    flows: np.ndarray,
    limits: np.ndarray,
    lodf: np.ndarray,
    valid_outage: np.ndarray,
    threshold: float,
    max_states: int,
    include_n1: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Screen intact and N-1 thermal states using the frozen baseline flows."""
    S, B = flows.shape
    records: list[tuple[float, int, int, float]] = []  # max_ratio, monitor, outage, coeff

    base_max = np.max(np.abs(flows) / limits[None, :], axis=0)
    for m in np.flatnonzero(base_max >= threshold):
        records.append((float(base_max[m]), int(m), -1, 0.0))

    if include_n1:
        for k in np.flatnonzero(valid_outage):
            coeff = lodf[:, k]
            finite = np.isfinite(coeff)
            if not finite.any():
                continue
            post = flows + flows[:, [k]] * coeff[None, :]
            ratio = np.max(np.abs(post) / limits[None, :], axis=0)
            ratio[~finite] = -np.inf
            ratio[k] = -np.inf
            for m in np.flatnonzero(ratio >= threshold):
                records.append((float(ratio[m]), int(m), int(k), float(coeff[m])))

    if not records:
        # Always retain the most-loaded intact branch so diagnostics are defined.
        m = int(np.argmax(base_max))
        records = [(float(base_max[m]), m, -1, 0.0)]

    # Most severe states first; remove exact duplicate monitor/outage pairs.
    records.sort(key=lambda r: r[0], reverse=True)
    seen: set[tuple[int, int]] = set()
    kept = []
    for rec in records:
        key = (rec[1], rec[2])
        if key in seen:
            continue
        seen.add(key); kept.append(rec)
        if len(kept) >= int(max_states):
            break

    mon = np.array([r[1] for r in kept], dtype=int)
    out = np.array([r[2] for r in kept], dtype=int)
    coeff = np.array([r[3] for r in kept], dtype=float)
    state_lim = limits[mon].astype(float)
    severity = np.array([r[0] for r in kept], dtype=float)
    return mon, out, coeff, state_lim, severity


def _build_frozen_cases(
    network: NetworkData,
    model: DCGridModel,
    extra: _ExtraData,
    nodes: pd.DataFrame,
    runs: int,
    seed: int,
    thermal_scale: float,
    include_n1: bool,
    security_screen_threshold_pct: float,
    max_security_states: int,
) -> _FrozenCases:
    n_snap = len(network.load_profile)
    n_bus = len(model.bus_ids)
    n_node = len(nodes)
    node_bus = nodes["bus"].astype(str).tolist()
    node_bus_idx = np.array([model.bus_index[b] for b in node_bus], dtype=int)

    # Renewable potential from the file's 168 p_max_pu snapshots.
    profile_index = {g: i for i, g in enumerate(extra.renewable_profile_ids)}
    node_index = {b: i for i, b in enumerate(node_bus)}
    potential = np.zeros((n_snap, n_node), dtype=float)
    selected_buses = set(model.bus_ids)
    gm = extra.generators[
        extra.generators["bus"].astype(str).isin(selected_buses) &
        extra.generators["carrier"].str.lower().isin(["wind", "solar"])
    ]
    for r in gm.itertuples(index=False):
        if str(r.generator) not in profile_index or str(r.bus) not in node_index:
            continue
        pmax = np.clip(extra.renewable_profiles[:, profile_index[str(r.generator)]], 0.0, None)
        potential[:, node_index[str(r.bus)]] += float(r.p_nom_mw) * pmax

    # Demand by bus from the supplied load snapshots.
    demand = np.zeros((n_snap, n_bus), dtype=float)
    load_bus = network.loads.set_index("load")["bus"].astype(str).to_dict()
    for load_id in network.load_profile.columns:
        b = load_bus.get(str(load_id))
        if b in model.bus_index:
            demand[:, model.bus_index[b]] += network.load_profile[str(load_id)].to_numpy(float)

    # Fixed link schedules.  If one terminal is outside the AC scope, the
    # in-scope terminal becomes a boundary injection/withdrawal.
    fixed = np.zeros((n_snap, n_bus), dtype=float)
    for lk in extra.links.itertuples(index=False):
        p = float(np.clip(lk.p_set_mw, -abs(lk.p_nom_mw), abs(lk.p_nom_mw)))
        b0, b1 = str(lk.bus0), str(lk.bus1)
        if b0 in model.bus_index:
            fixed[:, model.bus_index[b0]] -= p
        if b1 in model.bus_index:
            fixed[:, model.bus_index[b1]] += p * float(lk.efficiency)

    # Dispatchable synchronous/conventional fleet from the same WP2033 file.
    conv = extra.generators[
        extra.generators["bus"].astype(str).isin(selected_buses) &
        ~extra.generators["carrier"].str.lower().isin(["wind", "solar", "load shedding", "import", "export"]) &
        (extra.generators["p_nom_mw"] > 0)
    ].copy()
    conv = conv.sort_values(["marginal_cost", "p_nom_mw"], ascending=[True, False], kind="stable").reset_index(drop=True)
    if conv.empty:
        raise ValueError("No dispatchable generation found in selected WP2033 scope")
    pmax = conv["p_nom_mw"].to_numpy(float)
    pmin = np.clip(conv["p_min_pu"].to_numpy(float), 0.0, 1.0) * pmax
    balance_row = int(np.argmax(pmax))
    balance_bus = str(conv.iloc[balance_row]["bus"])

    injections = fixed.copy()
    injections -= demand
    for j, b in enumerate(node_bus):
        injections[:, model.bus_index[b]] += potential[:, j]
    conv_dispatch = np.repeat(pmin[None, :], n_snap, axis=0)
    for j, r in conv.iterrows():
        injections[:, model.bus_index[str(r.bus)]] += pmin[j]

    pre_dd = np.zeros_like(potential)
    shortage = np.zeros(n_snap, dtype=float)
    for s in range(n_snap):
        residual = -float(injections[s].sum())
        if residual > EPS:
            for j, r in conv.iterrows():
                headroom = max(0.0, pmax[j] - conv_dispatch[s, j])
                add = min(residual, headroom)
                if add > 0:
                    injections[s, model.bus_index[str(r.bus)]] += add
                    conv_dispatch[s, j] += add
                    residual -= add
                if residual <= EPS:
                    break
            if residual > EPS:
                # Keep the DC case balanced; record this as a supply-shortage
                # diagnostic rather than counting it as renewable dispatch-down.
                injections[s, model.bus_index[balance_bus]] += residual
                shortage[s] = residual
        elif residual < -EPS:
            surplus = -residual
            total_renew = float(potential[s].sum())
            cut = min(surplus, total_renew)
            if cut > EPS and total_renew > EPS:
                frac = cut / total_renew
                pre_dd[s] = potential[s] * frac
                for j, b in enumerate(node_bus):
                    injections[s, model.bus_index[b]] -= pre_dd[s, j]
            # If non-renewable minimum output alone exceeds load, absorb the tiny
            # remainder at the balance bus without labeling it renewable DD.
            imbalance = float(injections[s].sum())
            if abs(imbalance) > 1e-7:
                injections[s, model.bus_index[balance_bus]] -= imbalance

    pre_dispatch = np.maximum(0.0, potential - pre_dd)

    # PTDF and frozen intact flows.
    H = model.ptdf(balance_bus)
    flows = injections @ H.T
    limits = model.branches["s_nom_mva"].to_numpy(float) * float(thermal_scale)

    lodf, valid_outage = _build_lodf(model, H)
    mon, out, coeff, state_limits, _severity = _screen_security_states(
        flows, limits, lodf, valid_outage,
        threshold=float(security_screen_threshold_pct) / 100.0,
        max_states=int(max_security_states),
        include_n1=bool(include_n1),
    )

    state_base = flows[:, mon].copy()
    has_out = out >= 0
    if has_out.any():
        state_base[:, has_out] += flows[:, out[has_out]] * coeff[has_out][None, :]

    # 1 MW renewable curtailment at node, replaced at balance bus.
    branch_node_sens = (H[:, model.bus_index[balance_bus]][None, :] - H[:, node_bus_idx].T)  # node x branch
    node_state_sens = branch_node_sens[:, mon].copy()
    if has_out.any():
        node_state_sens[:, has_out] += branch_node_sens[:, out[has_out]] * coeff[has_out][None, :]

    rng = np.random.default_rng(int(seed))
    runs = max(1, int(runs))
    sampled = rng.integers(0, n_snap, size=runs)
    counts = np.bincount(sampled, minlength=n_snap).astype(float)
    case_weights = counts / counts.sum()

    return _FrozenCases(
        renewable_potential_mw=potential,
        pre_network_dispatch_mw=pre_dispatch,
        pre_network_dispatch_down_mw=pre_dd,
        branch_flows_mw=flows,
        case_weights=case_weights,
        node_state_sensitivity=node_state_sens,
        state_base_flows_mw=state_base,
        state_limits_mw=state_limits,
        state_monitor=mon,
        state_outage=out,
        state_lodf_coeff=coeff,
        balance_bus=balance_bus,
        shortage_mw=shortage,
    )


def _required_reduction(flow: float, limit: float, sensitivity: float) -> float:
    """MW of group reduction required to move one state inside its thermal limit."""
    if flow > limit + 1e-9:
        if sensitivity >= -1e-12:
            return math.inf
        return (flow - limit) / (-sensitivity)
    if flow < -limit - 1e-9:
        if sensitivity <= 1e-12:
            return math.inf
        return (-limit - flow) / sensitivity
    return 0.0


def make_emulator(
    runs: int = 10_000,
    seed: int = 76,
    scope: str | int = "32",
    network_file: str | Path = DEFAULT_NETWORK_FILE,
    thermal_scale: float = 1.0,
    include_n1: bool = True,
    security_screen_threshold_pct: float = 90.0,
    max_security_states: int = 800,
    max_group_actions: int = 24,
    security_guard: bool = True,
    # Compatibility with calls written for ni_annealer_api.  These NI-specific
    # parameters are intentionally ignored for WP2033 rather than silently
    # applying a 2024 NI calibration to a 2033 all-island model.
    target_dispatch_down_pct: float | None = None,
    demand_scale: float | None = None,
    planned_outage_exposure_pct: float | None = None,
    asset_file: str | Path | None = None,
) -> tuple[Callable[[pd.DataFrame], float], pd.DataFrame]:
    """Freeze a 26- or 32-county emulator and return ``(dispatch_down, nodes)``.

    The callable name and DataFrame contract match ``ni_annealer_api``.  Every
    node must belong to exactly one positive integer group.  Group IDs are not
    limited to 1..5.

    ``dispatch_down(candidate_nodes)`` returns physical renewable dispatch-down
    as a percentage of renewable potential across the frozen weighted WP2033
    cases.  With ``security_guard=True`` (default), a candidate that makes any
    snapshot insecure that was secure under the starting assignment returns
    ``np.inf``.  Existing baseline model-security failures are not hidden; they
    remain available through diagnostics on the callable.
    """
    del target_dispatch_down_pct, demand_scale, planned_outage_exposure_pct, asset_file
    scope = _scope_value(scope)
    path = Path(network_file)
    extra = _read_extra(path)
    full = read_network_nc(path)
    network, _selected = _subset_network(full, extra, scope)
    model = DCGridModel(network)
    template = _build_node_table(network, extra)
    cases = _build_frozen_cases(
        network, model, extra, template,
        runs=int(runs), seed=int(seed), thermal_scale=float(thermal_scale),
        include_n1=bool(include_n1),
        security_screen_threshold_pct=float(security_screen_threshold_pct),
        max_security_states=int(max_security_states),
    )

    weights = cases.case_weights
    potential_weighted = float(np.sum(weights * cases.renewable_potential_mw.sum(axis=1)))
    pre_dd_weighted = float(np.sum(weights * cases.pre_network_dispatch_down_mw.sum(axis=1)))

    def evaluate_assignment(group_ids: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        unique_groups = np.unique(group_ids)
        members = {int(g): np.flatnonzero(group_ids == g) for g in unique_groups}
        network_dd_case = np.zeros(len(weights), dtype=float)
        secure_case = np.ones(len(weights), dtype=bool)
        worst_loading_case = np.zeros(len(weights), dtype=float)
        action_count_case = np.zeros(len(weights), dtype=int)

        for s in np.flatnonzero(weights > 0):
            state_flow = cases.state_base_flows_mw[s].copy()
            dispatch = cases.pre_network_dispatch_mw[s].copy()
            used_actions = 0

            for _ in range(int(max_group_actions)):
                loading = np.abs(state_flow) / cases.state_limits_mw
                worst = int(np.argmax(loading))
                worst_ratio = float(loading[worst])
                if worst_ratio <= 1.0 + 1e-8:
                    break

                best = None
                for g in unique_groups:
                    idx = members[int(g)]
                    d = dispatch[idx]
                    available = float(d.sum())
                    if available <= EPS:
                        continue
                    w = d / available
                    sens_worst = float(w @ cases.node_state_sensitivity[idx, worst])
                    req = _required_reduction(float(state_flow[worst]), float(cases.state_limits_mw[worst]), sens_worst)
                    if not np.isfinite(req):
                        continue
                    # Prefer the least renewable reduction for the current worst state.
                    candidate = (float(req), int(g), available, idx, w)
                    if best is None or candidate[0] < best[0] - 1e-12 or (abs(candidate[0]-best[0]) <= 1e-12 and candidate[1] < best[1]):
                        best = candidate

                if best is None:
                    break

                req, g, available, idx, w = best
                delta = min(available, req * 1.000001 + 1e-7)
                if delta <= EPS:
                    break
                # Full security-state sensitivity for the selected group's
                # proportional reduction, then update all screened states at once.
                group_state_sens = w @ cases.node_state_sensitivity[idx, :]
                state_flow += delta * group_state_sens
                dispatch[idx] -= delta * w
                dispatch[idx] = np.maximum(dispatch[idx], 0.0)
                network_dd_case[s] += delta
                used_actions += 1

            loading = np.abs(state_flow) / cases.state_limits_mw
            worst_loading_case[s] = float(np.max(loading))
            secure_case[s] = bool(worst_loading_case[s] <= 1.0 + 1e-6)
            action_count_case[s] = used_actions

        total_dd = pre_dd_weighted + float(np.sum(weights * network_dd_case))
        pct = 100.0 * total_dd / max(potential_weighted, EPS)
        return float(pct), secure_case, worst_loading_case, network_dd_case

    baseline_groups = _node_groups(template, template)
    baseline_pct, baseline_secure, baseline_worst, baseline_network_dd = evaluate_assignment(baseline_groups)

    def dispatch_down(nodes: pd.DataFrame) -> float:
        """Return physical renewable dispatch-down (%) for one exclusive grouping."""
        group_ids = _node_groups(nodes, template)
        pct, secure, worst, network_dd = evaluate_assignment(group_ids)
        new_fail = baseline_secure & ~secure

        dispatch_down.last_raw_dispatch_down_pct = float(pct)
        dispatch_down.last_dispatch_down_pct = float(pct)
        dispatch_down.last_security_pass_pct = float(100.0 * np.sum(weights * secure.astype(float)))
        dispatch_down.last_new_security_failures = int(np.sum(new_fail & (weights > 0)))
        dispatch_down.last_worst_screened_loading_pct = float(100.0 * np.max(worst[weights > 0]))
        dispatch_down.last_network_dispatch_down_mw_weighted = float(np.sum(weights * network_dd))
        dispatch_down.last_group_count = int(len(np.unique(group_ids)))

        if bool(security_guard) and np.any(new_fail & (weights > 0)):
            dispatch_down.last_returned_infinity_for_security = True
            return float("inf")
        dispatch_down.last_returned_infinity_for_security = False
        return float(pct)

    # Same style of immutable metadata as ni_annealer_api.
    dispatch_down.scope = scope
    dispatch_down.runs = int(runs)
    dispatch_down.seed = int(seed)
    dispatch_down.network_file = str(path)
    dispatch_down.baseline_dispatch_down_pct = float(baseline_pct)
    dispatch_down.baseline_security_pass_pct = float(100.0 * np.sum(weights * baseline_secure.astype(float)))
    dispatch_down.baseline_worst_screened_loading_pct = float(100.0 * np.max(baseline_worst[weights > 0]))
    dispatch_down.security_state_count = int(len(cases.state_limits_mw))
    dispatch_down.renewable_node_count = int(len(template))
    dispatch_down.shortage_mw_weighted = float(np.sum(weights * cases.shortage_mw))
    dispatch_down.security_guard = bool(security_guard)
    dispatch_down.model = model
    dispatch_down.network = network
    dispatch_down.cases = cases

    return dispatch_down, template.copy(deep=True)


if __name__ == "__main__":
    for _scope in ("26", "32"):
        dispatch_down, nodes = make_emulator(scope=_scope, runs=10_000, seed=42)
        score = dispatch_down(nodes)
        print(
            f"{_scope}-county: nodes={len(nodes)}, groups={nodes['group'].nunique()}, "
            f"DD={score:.3f}%, security={dispatch_down.last_security_pass_pct:.2f}%, "
            f"states={dispatch_down.security_state_count}"
        )
