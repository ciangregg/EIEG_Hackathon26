"""Core tools for a Northern Ireland transmission-grid research simulator.

The module deliberately keeps the market layer separate from the static network.
It reads the supplied PyPSA NetCDF directly with h5py, maps spatial supply points
onto transmission buses, imports aggregate or unit-level market data, performs a
DC load-flow approximation, estimates I^2R line losses, and can redispatch
controllable injections to relieve thermal overloads.

This is a research/teaching model, not an operational security-analysis tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import math
import re
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linprog

EPS = 1e-9


def _decode(values):
    return [v.decode("utf-8", errors="replace") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values]


def _normalise_name(value: str) -> str:
    value = str(value).lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    stop = {"power", "station", "wind", "farm", "solar", "unit", "the"}
    return " ".join(x for x in value.split() if x not in stop)


def _connection_kv(value: str) -> Optional[float]:
    vals = re.findall(r"(\d+(?:\.\d+)?)\s*kV", str(value), flags=re.I)
    if not vals:
        return None
    return float(vals[-1])


def _haversine_km(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    r = 6371.0088
    p1 = np.radians(lat)
    p2 = np.radians(lats)
    dp = p2 - p1
    dl = np.radians(lons - lon)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


@dataclass
class NetworkData:
    buses: pd.DataFrame
    lines: pd.DataFrame
    transformers: pd.DataFrame
    loads: pd.DataFrame
    load_profile: pd.DataFrame
    native_generators: pd.DataFrame
    snapshots: pd.Index


@dataclass
class Scenario:
    timestamp: str
    demand_by_bus: pd.Series
    assets: pd.DataFrame
    fixed_injection_by_bus: pd.Series
    total_demand_mw: float
    interconnector_mw: float
    notes: str = ""


def read_network_nc(path: str | Path) -> NetworkData:
    path = str(path)
    with h5py.File(path, "r") as f:
        buses = pd.DataFrame({
            "bus": _decode(f["buses_i"][:]),
            "v_nom_kv": np.asarray(f["buses_v_nom"][:], float),
            "lon": np.asarray(f["buses_x"][:], float),
            "lat": np.asarray(f["buses_y"][:], float),
            "station": _decode(f["buses_station"][:]) if "buses_station" in f else "",
            "psse_name": _decode(f["buses_psse_name"][:]) if "buses_psse_name" in f else "",
        })
        lines = pd.DataFrame({
            "branch": _decode(f["lines_i"][:]),
            "bus0": _decode(f["lines_bus0"][:]),
            "bus1": _decode(f["lines_bus1"][:]),
            "x_ohm": np.asarray(f["lines_x"][:], float),
            "r_ohm": np.asarray(f["lines_r"][:], float),
            "b_siemens": np.asarray(f["lines_b"][:], float),
            "s_nom_mva": np.asarray(f["lines_s_nom"][:], float),
            "length_km": np.asarray(f["lines_length"][:], float),
            "limit_source": _decode(f["lines_s_nom_source"][:]) if "lines_s_nom_source" in f else "",
        })
        transformers = pd.DataFrame({
            "branch": _decode(f["transformers_i"][:]),
            "bus0": _decode(f["transformers_bus0"][:]),
            "bus1": _decode(f["transformers_bus1"][:]),
            "x_pu": np.asarray(f["transformers_x"][:], float),
            "r_pu": np.asarray(f["transformers_r"][:], float),
            "s_nom_mva": np.asarray(f["transformers_s_nom"][:], float),
            "tap_ratio": np.asarray(f["transformers_tap_ratio"][:], float),
            "phase_shift_deg": np.asarray(f["transformers_phase_shift"][:], float),
        })
        loads = pd.DataFrame({
            "load": _decode(f["loads_i"][:]),
            "bus": _decode(f["loads_bus"][:]),
            "base_p_mw": np.asarray(f["loads_p_set"][:], float),
            "psse_name": _decode(f["loads_psse_bus_name"][:]) if "loads_psse_bus_name" in f else "",
        })
        load_ids = _decode(f["loads_t_p_set_i"][:]) if "loads_t_p_set_i" in f else loads["load"].tolist()
        load_profile = pd.DataFrame(np.asarray(f["loads_t_p_set"][:], float), columns=load_ids)
        native_generators = pd.DataFrame({
            "generator": _decode(f["generators_i"][:]),
            "bus": _decode(f["generators_bus"][:]),
            "carrier": _decode(f["generators_carrier"][:]),
            "p_nom_mw": np.asarray(f["generators_p_nom"][:], float),
            "p_set_mw": np.asarray(f["generators_p_set_tytfs"][:], float) if "generators_p_set_tytfs" in f else np.nan,
        })
        snapshots = pd.Index(np.asarray(f["snapshots_snapshot"][:]) if "snapshots_snapshot" in f else np.arange(len(load_profile)), name="snapshot")
    return NetworkData(buses, lines, transformers, loads, load_profile, native_generators, snapshots)


def read_spatial_assets(path: str | Path) -> pd.DataFrame:
    assets = pd.read_csv(path).copy()
    for c in ["latitude", "longitude", "max_export_capacity_mw"]:
        assets[c] = pd.to_numeric(assets[c], errors="coerce")
    assets["market_bucket"] = assets["point_type"].map({
        "Wind farm": "wind",
        "Solar farm": "solar",
        "Power station unit": "thermal",
        "Biomass generation": "biomass",
        "Energy from waste": "waste",
        "HVDC interconnector": "interconnector",
    }).fillna("other")
    return assets


def map_assets_to_buses(assets: pd.DataFrame, buses: pd.DataFrame) -> pd.DataFrame:
    """Map each spatial point to a proxy transmission bus.

    Transmission-connected assets prefer a bus at the stated voltage. Distribution
    assets are mapped to the nearest 110 kV bus because the supplied network is a
    transmission model. Distance is retained so weak proxy assignments are visible.
    """
    mapped = assets.copy()
    bus_lat = buses["lat"].to_numpy(float)
    bus_lon = buses["lon"].to_numpy(float)
    bus_v = buses["v_nom_kv"].to_numpy(float)
    # Do not attach physical assets to synthetic star buses used internally for
    # three-winding transformer equivalents.
    is_physical_bus = ~buses["bus"].astype(str).str.startswith("star:").to_numpy()
    finite = np.isfinite(bus_lat) & np.isfinite(bus_lon) & is_physical_bus
    all_idx = np.flatnonzero(finite)
    rows = []
    for _, a in mapped.iterrows():
        if not np.isfinite(a.get("latitude", np.nan)) or not np.isfinite(a.get("longitude", np.nan)):
            rows.append((None, np.nan, np.nan, "unmapped"))
            continue
        d = _haversine_km(float(a.latitude), float(a.longitude), bus_lat, bus_lon)
        kv = _connection_kv(a.get("connection_level", ""))
        if kv is not None and kv >= 100:
            candidates = np.flatnonzero(finite & np.isclose(bus_v, kv))
            if len(candidates) == 0:
                candidates = all_idx
        else:
            candidates = np.flatnonzero(finite & np.isclose(bus_v, 110.0))
            if len(candidates) == 0:
                candidates = all_idx
        j = candidates[np.argmin(d[candidates])]
        dist = float(d[j])
        quality = "high" if dist <= 3 else "medium" if dist <= 10 else "low"
        rows.append((str(buses.iloc[j].bus), float(bus_v[j]), dist, quality))
    mapped[["mapped_bus", "mapped_bus_kv", "mapping_distance_km", "mapping_quality"]] = pd.DataFrame(rows, index=mapped.index)
    return mapped


def load_shares(network: NetworkData) -> pd.Series:
    totals = network.load_profile.sum(axis=0)
    if float(totals.sum()) <= EPS:
        totals = network.loads.set_index("load")["base_p_mw"]
    shares_by_load = totals / totals.sum()
    load_to_bus = network.loads.set_index("load")["bus"]
    by_bus = shares_by_load.groupby(load_to_bus).sum()
    return by_bus / by_bus.sum()


def allocate_demand(network: NetworkData, total_demand_mw: float) -> pd.Series:
    s = load_shares(network) * float(total_demand_mw)
    return s.groupby(level=0).sum()


def _allocate_bucket(assets: pd.DataFrame, bucket: str, total_mw: float) -> pd.Series:
    mask = assets["market_bucket"].eq(bucket) & assets["mapped_bus"].notna()
    subset = assets.loc[mask]
    if subset.empty or abs(float(total_mw)) < EPS:
        return pd.Series(dtype=float)
    caps = subset["max_export_capacity_mw"].fillna(0).clip(lower=0)
    if caps.sum() <= EPS:
        weights = pd.Series(1 / len(subset), index=subset.index)
    else:
        weights = caps / caps.sum()
    return pd.Series(float(total_mw) * weights.to_numpy(), index=subset.index)


def detect_market_schema(df: pd.DataFrame) -> str:
    cols = {c.lower() for c in df.columns}
    if {"timestamp", "unit_name", "dispatch_mw"}.issubset(cols):
        return "unit_long"
    if "timestamp" in cols and ("ni_demand_mw" in cols or "demand_mw" in cols):
        return "aggregate"
    raise ValueError(
        "Market CSV not recognised. Use aggregate columns timestamp, ni_demand_mw, wind_mw, "
        "solar_mw, thermal_mw, biomass_mw, waste_mw, moyle_mw; or long columns "
        "timestamp, unit_name, dispatch_mw plus ni_demand_mw/moyle_mw."
    )


def list_market_timestamps(market_csv: str | Path) -> list[str]:
    df = pd.read_csv(market_csv)
    if "timestamp" not in df.columns:
        # case-insensitive rename
        cmap = {c.lower(): c for c in df.columns}
        if "timestamp" not in cmap:
            raise ValueError("Market CSV needs a timestamp column")
        df = df.rename(columns={cmap["timestamp"]: "timestamp"})
    return df["timestamp"].astype(str).drop_duplicates().tolist()


def _casefold_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: c.strip().lower() for c in df.columns}
    return df.rename(columns=rename)


def _best_asset_match(unit_name: str, assets: pd.DataFrame) -> Tuple[Optional[int], float]:
    q = _normalise_name(unit_name)
    names = assets["point_name"].astype(str).map(_normalise_name)
    exact = names[names == q]
    if len(exact):
        return int(exact.index[0]), 1.0
    scores = names.map(lambda x: SequenceMatcher(None, q, x).ratio())
    if scores.empty:
        return None, 0.0
    i = int(scores.idxmax())
    return i, float(scores.loc[i])


def import_market_scenario(
    network: NetworkData,
    mapped_assets: pd.DataFrame,
    market_csv: str | Path,
    timestamp: str,
    fuzzy_cutoff: float = 0.72,
) -> Scenario:
    df = _casefold_columns(pd.read_csv(market_csv))
    schema = detect_market_schema(df)
    df["timestamp"] = df["timestamp"].astype(str)
    rows = df[df["timestamp"] == str(timestamp)].copy()
    if rows.empty:
        raise ValueError(f"Timestamp {timestamp!r} not found in market CSV")

    asset_dispatch = pd.Series(0.0, index=mapped_assets.index, dtype=float)
    interconnector_mw = 0.0
    notes = []

    demand_col = "ni_demand_mw" if "ni_demand_mw" in rows else "demand_mw" if "demand_mw" in rows else None
    if demand_col is None:
        raise ValueError("Market data needs ni_demand_mw (or demand_mw)")
    demand_vals = pd.to_numeric(rows[demand_col], errors="coerce").dropna()
    if demand_vals.empty:
        raise ValueError(f"No NI demand value at {timestamp}")
    total_demand = float(demand_vals.iloc[0])

    if "moyle_mw" in rows:
        vals = pd.to_numeric(rows["moyle_mw"], errors="coerce").dropna()
        if len(vals):
            interconnector_mw = float(vals.iloc[0])

    if schema == "aggregate":
        for bucket, col in [
            ("wind", "wind_mw"), ("solar", "solar_mw"), ("thermal", "thermal_mw"),
            ("biomass", "biomass_mw"), ("waste", "waste_mw")
        ]:
            if col not in rows:
                continue
            vals = pd.to_numeric(rows[col], errors="coerce").dropna()
            if not len(vals):
                continue
            alloc = _allocate_bucket(mapped_assets, bucket, float(vals.iloc[0]))
            asset_dispatch.loc[alloc.index] = alloc
        notes.append("Aggregate technology totals allocated to assets in proportion to MEC.")
    else:
        match_rows = []
        for _, r in rows.iterrows():
            unit = str(r.get("unit_name", ""))
            mw = pd.to_numeric(pd.Series([r.get("dispatch_mw")]), errors="coerce").iloc[0]
            if not np.isfinite(mw):
                continue
            i, score = _best_asset_match(unit, mapped_assets)
            if i is not None and score >= fuzzy_cutoff:
                asset_dispatch.loc[i] += float(mw)
                match_rows.append((unit, mapped_assets.loc[i, "point_name"], score))
            else:
                notes.append(f"Unmatched market unit: {unit} (best score {score:.2f})")
        if match_rows:
            notes.append(f"Matched {len(match_rows)} unit dispatch rows to spatial assets.")

    out_assets = mapped_assets.copy()
    out_assets["dispatch_mw"] = asset_dispatch
    out_assets["available_mw"] = out_assets["max_export_capacity_mw"].fillna(0).clip(lower=0)

    # Interconnector is represented separately as a signed bus injection. Positive = import into NI.
    ic = out_assets[out_assets["market_bucket"].eq("interconnector") & out_assets["mapped_bus"].notna()]
    fixed = pd.Series(dtype=float)
    if len(ic):
        ic_bus = str(ic.iloc[0]["mapped_bus"])
        fixed.loc[ic_bus] = fixed.get(ic_bus, 0.0) + interconnector_mw
    elif abs(interconnector_mw) > EPS:
        notes.append("Moyle flow supplied but interconnector point could not be mapped.")

    return Scenario(
        timestamp=str(timestamp),
        demand_by_bus=allocate_demand(network, total_demand),
        assets=out_assets,
        fixed_injection_by_bus=fixed,
        total_demand_mw=total_demand,
        interconnector_mw=interconnector_mw,
        notes=" ".join(notes),
    )


def built_in_scenario(network: NetworkData, mapped_assets: pd.DataFrame, snapshot_index: int = 0) -> Scenario:
    snapshot_index = int(np.clip(snapshot_index, 0, len(network.load_profile) - 1))
    row = network.load_profile.iloc[snapshot_index]
    total = float(row.sum())
    demand_by_load = row
    load_to_bus = network.loads.set_index("load")["bus"]
    demand_by_bus = demand_by_load.groupby(load_to_bus).sum()

    assets = mapped_assets.copy()
    assets["dispatch_mw"] = 0.0
    assets["available_mw"] = assets["max_export_capacity_mw"].fillna(0).clip(lower=0)

    # Preserve only the three ordinary generators that exist in the supplied .nc as a demo dispatch.
    ordinary = network.native_generators[network.native_generators["carrier"].ne("load shedding")].copy()
    demo_rows = []
    for _, g in ordinary.iterrows():
        p = float(g.p_set_mw) if np.isfinite(g.p_set_mw) else 0.0
        demo_rows.append({
            "point_name": f"Native {g.generator}", "point_type": "Native network generator",
            "market_bucket": "thermal", "mapped_bus": str(g.bus), "mapped_bus_kv": np.nan,
            "mapping_distance_km": 0.0, "mapping_quality": "native", "max_export_capacity_mw": float(g.p_nom_mw),
            "dispatch_mw": p, "available_mw": float(g.p_nom_mw)
        })
    if demo_rows:
        assets = pd.concat([assets, pd.DataFrame(demo_rows)], ignore_index=True, sort=False)
    return Scenario(
        timestamp=f"network snapshot {snapshot_index}", demand_by_bus=demand_by_bus,
        assets=assets, fixed_injection_by_bus=pd.Series(dtype=float), total_demand_mw=total,
        interconnector_mw=0.0,
        notes="Built-in demo: original NI load profile plus the ordinary generators embedded in the supplied network; spatial assets are zero until market data is imported."
    )



def apply_demand_shock(
    scenario: Scenario,
    shock_mw: float,
    target_bus: Optional[str] = None,
    generation_response: str = "thermal_headroom",
) -> Scenario:
    """Return a copy of ``scenario`` with an exogenous demand shock applied.

    Positive ``shock_mw`` increases load; negative values reduce load.  A system-
    wide shock is distributed according to the existing load pattern.  For the
    ``thermal_headroom`` response, mapped thermal units move first within their
    available range and the normal slack-bus balance handles any residual.

    This is a quasi-steady-state scenario rule, not a frequency-response model.
    """
    shock = float(shock_mw)
    demand = scenario.demand_by_bus.astype(float).copy()
    assets = scenario.assets.copy()
    fixed = scenario.fixed_injection_by_bus.copy()

    if abs(shock) < EPS:
        return Scenario(
            timestamp=f"{scenario.timestamp} | no demand shock",
            demand_by_bus=demand,
            assets=assets,
            fixed_injection_by_bus=fixed,
            total_demand_mw=float(demand.sum()),
            interconnector_mw=scenario.interconnector_mw,
            notes=scenario.notes,
        )

    if target_bus and str(target_bus).lower() not in {"system", "system-wide", "all"}:
        b = str(target_bus)
        demand.loc[b] = max(0.0, float(demand.get(b, 0.0)) + shock)
        actual_shock = float(demand.sum() - scenario.demand_by_bus.sum())
        shock_desc = f"{actual_shock:+.1f} MW at bus {b}"
    else:
        total = float(demand.sum())
        if total > EPS:
            shares = demand.clip(lower=0) / total
        else:
            shares = pd.Series(1.0 / max(len(demand), 1), index=demand.index)
        if shock >= 0:
            demand = demand + shares * shock
        else:
            factor = max(0.0, (total + shock) / total) if total > EPS else 0.0
            demand = demand * factor
        actual_shock = float(demand.sum() - scenario.demand_by_bus.sum())
        shock_desc = f"{actual_shock:+.1f} MW system-wide"

    response_note = "generation unchanged; slack balances the shock"
    if generation_response == "thermal_headroom" and abs(actual_shock) > EPS:
        thermal = assets[
            assets.get("market_bucket", pd.Series(index=assets.index, dtype=object)).eq("thermal")
            & assets["mapped_bus"].notna()
        ].copy()
        if len(thermal):
            dispatch = pd.to_numeric(thermal["dispatch_mw"], errors="coerce").fillna(0.0)
            caps = pd.to_numeric(
                thermal.get("available_mw", thermal.get("max_export_capacity_mw", dispatch)),
                errors="coerce",
            ).fillna(dispatch)
            if actual_shock > 0:
                room = (caps - dispatch).clip(lower=0)
                available = float(room.sum())
                response = min(actual_shock, available)
                if response > EPS and available > EPS:
                    delta = room / available * response
                    assets.loc[thermal.index, "dispatch_mw"] = dispatch + delta
                residual = actual_shock - response
                response_note = f"thermal fleet picked up {response:.1f} MW; residual {residual:.1f} MW left to slack"
            else:
                reducible = dispatch.clip(lower=0)
                available = float(reducible.sum())
                response = min(-actual_shock, available)
                if response > EPS and available > EPS:
                    delta = reducible / available * response
                    assets.loc[thermal.index, "dispatch_mw"] = dispatch - delta
                residual = (-actual_shock) - response
                response_note = f"thermal fleet reduced {response:.1f} MW; residual {residual:.1f} MW left to slack"

    notes = (scenario.notes + " " if scenario.notes else "") + f"Demand shock: {shock_desc}; {response_note}."
    return Scenario(
        timestamp=f"{scenario.timestamp} | demand shock {shock_desc}",
        demand_by_bus=demand,
        assets=assets,
        fixed_injection_by_bus=fixed,
        total_demand_mw=float(demand.sum()),
        interconnector_mw=scenario.interconnector_mw,
        notes=notes,
    )


def scenario_injections(network: NetworkData, scenario: Scenario, slack_bus: Optional[str] = None) -> Tuple[pd.Series, pd.DataFrame, str, float]:
    buses = network.buses["bus"].astype(str).tolist()
    p = pd.Series(0.0, index=buses)
    p.loc[scenario.demand_by_bus.index.intersection(p.index)] -= scenario.demand_by_bus.reindex(p.index).fillna(0)
    active = scenario.assets[scenario.assets["mapped_bus"].notna()].copy()
    active["dispatch_mw"] = pd.to_numeric(active["dispatch_mw"], errors="coerce").fillna(0.0)
    for bus, mw in active.groupby("mapped_bus")["dispatch_mw"].sum().items():
        if str(bus) in p.index:
            p.loc[str(bus)] += float(mw)
    for bus, mw in scenario.fixed_injection_by_bus.items():
        if str(bus) in p.index:
            p.loc[str(bus)] += float(mw)

    if slack_bus is None or slack_bus not in p.index:
        # Prefer the bus with the greatest ordinary/native generation capacity, otherwise highest-voltage bus.
        ordinary = network.native_generators[network.native_generators["carrier"].ne("load shedding")]
        if len(ordinary):
            slack_bus = str(ordinary.sort_values("p_nom_mw", ascending=False).iloc[0].bus)
        else:
            slack_bus = str(network.buses.sort_values("v_nom_kv", ascending=False).iloc[0].bus)
    balance = -float(p.sum())
    p.loc[slack_bus] += balance
    return p, active, slack_bus, balance


class DCGridModel:
    def __init__(self, network: NetworkData):
        self.network = network
        self.buses = network.buses.copy().reset_index(drop=True)
        self.bus_ids = self.buses["bus"].astype(str).tolist()
        self.bus_index = {b: i for i, b in enumerate(self.bus_ids)}
        self.branches = self._make_branches()
        self.B = self._make_B()

    def _make_branches(self) -> pd.DataFrame:
        bvoltage = self.buses.set_index("bus")["v_nom_kv"]
        rows = []
        for _, x in self.network.lines.iterrows():
            v = float(bvoltage.loc[x.bus0])
            reactance = float(x.x_ohm)
            susceptance = v * v / reactance if abs(reactance) > EPS else 1e9
            rows.append({
                "type": "line", "branch": str(x.branch), "bus0": str(x.bus0), "bus1": str(x.bus1),
                "susceptance_mw_per_rad": susceptance, "s_nom_mva": float(x.s_nom_mva),
                "r_ohm": float(x.r_ohm), "v_nom_kv": v, "length_km": float(x.length_km)
            })
        for _, x in self.network.transformers.iterrows():
            reactance = float(x.x_pu)
            # PyPSA transformer x is per-unit on the transformer's nominal rating.
            susceptance = float(x.s_nom_mva) / reactance if abs(reactance) > EPS else 1e9
            rows.append({
                "type": "transformer", "branch": str(x.branch), "bus0": str(x.bus0), "bus1": str(x.bus1),
                "susceptance_mw_per_rad": susceptance, "s_nom_mva": float(x.s_nom_mva),
                "r_ohm": np.nan, "v_nom_kv": np.nan, "length_km": np.nan
            })
        return pd.DataFrame(rows)

    def _make_B(self) -> np.ndarray:
        n = len(self.bus_ids)
        B = np.zeros((n, n), dtype=float)
        for _, br in self.branches.iterrows():
            i, j = self.bus_index[br.bus0], self.bus_index[br.bus1]
            b = float(br.susceptance_mw_per_rad)
            B[i, i] += b; B[j, j] += b; B[i, j] -= b; B[j, i] -= b
        return B

    def solve(self, injections_mw: pd.Series, slack_bus: str, thermal_scale: float = 1.0,
              limit_overrides: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        p = injections_mw.reindex(self.bus_ids).fillna(0).to_numpy(dtype=float, copy=True)
        # Force tiny numerical imbalance into slack.
        p[self.bus_index[slack_bus]] -= p.sum()
        slack = self.bus_index[slack_bus]
        keep = np.array([i for i in range(len(self.bus_ids)) if i != slack])
        theta = np.zeros(len(self.bus_ids))
        theta[keep] = np.linalg.solve(self.B[np.ix_(keep, keep)], p[keep])

        branches = self.branches.copy()
        f = []
        for _, br in branches.iterrows():
            flow = float(br.susceptance_mw_per_rad) * (theta[self.bus_index[br.bus0]] - theta[self.bus_index[br.bus1]])
            f.append(flow)
        branches["flow_mw"] = f
        branches["abs_flow_mw"] = branches["flow_mw"].abs()
        branches["thermal_limit_mw"] = branches["s_nom_mva"] * float(thermal_scale)
        if limit_overrides:
            for branch, lim in limit_overrides.items():
                branches.loc[branches["branch"].eq(branch), "thermal_limit_mw"] = float(lim)
        branches["loading_pct"] = 100 * branches["abs_flow_mw"] / branches["thermal_limit_mw"].replace(0, np.nan)
        branches["overload_mw"] = (branches["abs_flow_mw"] - branches["thermal_limit_mw"]).clip(lower=0)

        # Diagnostic line losses: DC flow itself remains lossless. P_loss ≈ P^2 R / V^2.
        line_mask = branches["type"].eq("line") & branches["r_ohm"].notna() & branches["v_nom_kv"].notna()
        branches["estimated_loss_mw"] = 0.0
        branches.loc[line_mask, "estimated_loss_mw"] = (
            branches.loc[line_mask, "abs_flow_mw"] ** 2 * branches.loc[line_mask, "r_ohm"] /
            (branches.loc[line_mask, "v_nom_kv"] ** 2)
        )
        denom = branches["abs_flow_mw"].replace(0, np.nan)
        branches["estimated_efficiency_pct"] = (100 * (1 - branches["estimated_loss_mw"] / denom)).clip(lower=0, upper=100).fillna(100)

        bus_results = self.buses.copy()
        bus_results["injection_mw"] = p
        bus_results["angle_rad"] = theta
        return branches, bus_results

    def ptdf(self, slack_bus: str) -> np.ndarray:
        """Return branch x bus PTDF matrix for injections balanced at slack."""
        n = len(self.bus_ids)
        slack = self.bus_index[slack_bus]
        keep = np.array([i for i in range(n) if i != slack])
        inv = np.linalg.inv(self.B[np.ix_(keep, keep)])
        # theta response to 1 MW injection at non-slack bus and -1 at slack
        T = np.zeros((n, n))
        T[np.ix_(keep, keep)] = inv
        H = np.zeros((len(self.branches), n))
        for k, br in self.branches.iterrows():
            b = float(br.susceptance_mw_per_rad)
            H[k, :] = b * (T[self.bus_index[br.bus0], :] - T[self.bus_index[br.bus1], :])
        H[:, slack] = 0.0
        return H


def relieve_thermal_constraints(
    model: DCGridModel,
    scenario: Scenario,
    base_injections: pd.Series,
    active_assets: pd.DataFrame,
    slack_bus: str,
    thermal_scale: float = 1.0,
    limit_overrides: Optional[Dict[str, float]] = None,
    allow_load_shedding: bool = True,
) -> Tuple[pd.Series, pd.DataFrame, dict]:
    """Linear redispatch to relieve branch overloads.

    Variables are generator downward/upward redispatch, balancing-slack movement,
    and (optionally) load shedding as an extremely expensive last resort.
    Wind/solar/biomass/waste are allowed downward only; thermal units can move both
    ways inside [0, MEC]. This is intentionally a transparent research redispatch,
    not a replica of SEM/SONI operational scheduling.
    """
    assets = active_assets.copy().reset_index().rename(columns={"index": "asset_index"})
    assets = assets[assets["mapped_bus"].isin(model.bus_ids)].copy().reset_index(drop=True)
    if assets.empty:
        return base_injections, active_assets, {"success": False, "message": "No controllable mapped assets."}

    H = model.ptdf(slack_bus)
    base_br, _ = model.solve(base_injections, slack_bus, thermal_scale, limit_overrides)
    limits = base_br["thermal_limit_mw"].to_numpy(dtype=float, copy=True)
    f0 = base_br["flow_mw"].to_numpy(dtype=float, copy=True)

    m = len(assets)
    demand = scenario.demand_by_bus.groupby(level=0).sum()
    load_buses = [b for b in demand.index.astype(str) if b in model.bus_ids and float(demand.loc[b]) > EPS]
    q = len(load_buses) if allow_load_shedding else 0
    # Variables: [gen_down m | gen_up m | slack_up | slack_down | load_shed q]
    nv = 2 * m + 2 + q
    c = np.zeros(nv)
    c[:m] = assets["market_bucket"].map({"wind": 1.0, "solar": 1.0, "biomass": 2.0, "waste": 2.0, "thermal": 3.0}).fillna(2.5)
    c[m:2*m] = assets["market_bucket"].map({"thermal": 2.0}).fillna(20.0)
    c[2*m:2*m+2] = 8.0
    if q:
        c[2*m+2:] = 1000.0

    bounds = []
    # If a surplus scenario has explicit unit commitment, preserve it during
    # corrective redispatch: offline thermal units cannot silently start and
    # online units cannot be dispatched below their minimum stable output.
    for _, a in assets.iterrows():
        p = max(0.0, float(a.dispatch_mw))
        bucket = str(a.market_bucket)
        if bucket == "thermal" and "committed" in assets.columns:
            committed = bool(a.get("committed", p > EPS))
            min_stable = max(0.0, float(a.get("min_stable_mw", 0.0) or 0.0))
            down = max(0.0, p - min_stable) if committed else 0.0
        else:
            down = p
        bounds.append((0.0, down))
    for _, a in assets.iterrows():
        p = max(0.0, float(a.dispatch_mw))
        capval = a.get("available_mw", a.get("max_export_capacity_mw", p))
        cap = max(p, float(capval) if pd.notna(capval) else p)
        bucket = str(a.market_bucket)
        committed = bool(a.get("committed", p > EPS)) if "committed" in assets.columns else True
        up = max(0.0, cap - p) if bucket == "thermal" and committed else 0.0
        bounds.append((0.0, up))
    envelope = max(1000.0, float(scenario.total_demand_mw) * 2)
    bounds += [(0.0, envelope), (0.0, envelope)]
    if q:
        bounds += [(0.0, float(demand.loc[b])) for b in load_buses]

    D = np.zeros((len(model.bus_ids), nv))
    for j, a in assets.iterrows():
        bi = model.bus_index[str(a.mapped_bus)]
        D[bi, j] = -1.0
        D[bi, m+j] = 1.0
    D[model.bus_index[slack_bus], 2*m] = 1.0
    D[model.bus_index[slack_bus], 2*m+1] = -1.0
    if q:
        for k, bus in enumerate(load_buses):
            D[model.bus_index[bus], 2*m+2+k] = 1.0  # shedding reduces withdrawal

    Aflow = H @ D
    Aub = np.vstack([Aflow, -Aflow])
    bub = np.concatenate([limits - f0, limits + f0])
    Aeq = D.sum(axis=0, keepdims=True)
    beq = np.array([0.0])

    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        return base_injections, active_assets, {"success": False, "message": res.message}

    x = res.x
    delta_bus = D @ x
    new_p = base_injections.reindex(model.bus_ids).fillna(0).copy()
    new_p += pd.Series(delta_bus, index=model.bus_ids)
    dispatch = assets["dispatch_mw"].to_numpy(dtype=float, copy=True) - x[:m] + x[m:2*m]
    changed = active_assets.copy()
    for j, a in assets.iterrows():
        orig_idx = a.asset_index
        changed.loc[orig_idx, "dispatch_mw"] = dispatch[j]
        changed.loc[orig_idx, "redispatch_down_mw"] = x[j]
        changed.loc[orig_idx, "redispatch_up_mw"] = x[m+j]

    load_shed = float(x[2*m+2:].sum()) if q else 0.0
    renewable_mask = assets["market_bucket"].isin(["wind", "solar"]).to_numpy()
    meta = {
        "success": True,
        "message": "Thermal constraints relieved by linear redispatch." + (" Load shedding was required." if load_shed > 1e-6 else ""),
        "objective": float(res.fun),
        "slack_up_mw": float(x[2*m]),
        "slack_down_mw": float(x[2*m+1]),
        "renewable_curtailment_mw": float(x[:m][renewable_mask].sum()),
        "total_down_mw": float(x[:m].sum()),
        "total_up_mw": float(x[m:2*m].sum()),
        "load_shed_mw": load_shed,
        "load_shed_by_bus": ({bus: float(x[2*m+2+k]) for k, bus in enumerate(load_buses) if x[2*m+2+k] > 1e-6} if q else {}),
    }
    return new_p, changed, meta

def normalised_market_template() -> pd.DataFrame:
    return pd.DataFrame([
        {"timestamp": "2024-01-01 00:00", "ni_demand_mw": 900, "wind_mw": 450, "solar_mw": 0,
         "thermal_mw": 500, "biomass_mw": 15, "waste_mw": 8, "moyle_mw": -73},
        {"timestamp": "2024-01-01 00:30", "ni_demand_mw": 880, "wind_mw": 470, "solar_mw": 0,
         "thermal_mw": 470, "biomass_mw": 15, "waste_mw": 8, "moyle_mw": -83},
    ])
