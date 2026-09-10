"""Renewable-surplus and low-demand scenario engine for the NI grid simulator.

The module is the inverse of the demand-shock engine.  It creates a sequence of
quasi-steady-state operating points in which renewable *availability* rises and/or
demand falls.  Conventional thermal dispatch is reduced first; if the system still
has excess generation, wind/solar are curtailed.  A subsequent network-constrained
redispatch can add further curtailment when transmission bottlenecks prevent the
renewable power from reaching demand.

Important terminology
---------------------
``renewable utilisation`` is the fraction of available wind/solar power that is
actually dispatched.  It is deliberately kept separate from transmission/electrical
efficiency (I^2 R losses).  Curtailment is therefore a loss of *potential renewable
energy*, not an electrical loss in the conductors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ni_grid_core import (
    DCGridModel,
    NetworkData,
    Scenario,
    apply_demand_shock,
    relieve_thermal_constraints,
    scenario_injections,
)
from ni_commitment import CommitmentConfig, apply_commitment_and_balance, commitment_table

EPS = 1e-9
RENEWABLE_BUCKETS = {"wind", "solar"}


@dataclass
class RenewableSurplusConfig:
    """Configuration for a renewable-availability / low-demand experiment."""

    wind_surge_mw: float = 350.0
    solar_surge_mw: float = 0.0
    demand_change_mw: float = -250.0
    demand_scope: str = "system"  # "system" or "bus"
    target_bus: Optional[str] = None
    steps: int = 12
    duration_minutes: float = 60.0
    thermal_scale: float = 1.0
    line_limit_overrides: Optional[Dict[str, float]] = None
    auto_redispatch: bool = True
    allow_load_shedding: bool = False
    commitment_enabled: bool = True
    max_nonsynchronous_share_pct: float = 75.0
    min_synchronous_units_online: int = 2
    min_stable_output_pct: float = 20.0
    shutdown_strategy: str = "keep_current"


@dataclass
class RenewableSurplusFrame:
    """One solved state in a renewable-surplus scenario."""

    step: int
    fraction: float
    elapsed_minutes: float
    actual_demand_change_mw: float
    requested_wind_surge_mw: float
    actual_wind_surge_mw: float
    requested_solar_surge_mw: float
    actual_solar_surge_mw: float
    scenario: Scenario
    injections_mw: pd.Series
    active_assets: pd.DataFrame
    branches: pd.DataFrame
    buses: pd.DataFrame
    slack_bus: str
    slack_balance_mw: float
    balance_curtailment_mw: float
    stability_curtailment_mw: float
    minimum_output_curtailment_mw: float
    network_curtailment_mw: float
    commitment: Optional[dict]
    renewable_available_mw: float
    renewable_accepted_mw: float
    renewable_curtailment_mw: float
    renewable_utilisation_pct: float
    thermal_change_mw: float
    redispatch: Optional[dict]

    @property
    def overloaded_lines(self) -> int:
        lines = self.branches[self.branches["type"].eq("line")]
        return int((lines["loading_pct"] > 100.0 + 1e-7).sum())

    @property
    def worst_loading_pct(self) -> float:
        lines = self.branches[self.branches["type"].eq("line")]
        return float(lines["loading_pct"].max()) if len(lines) else 0.0

    @property
    def estimated_losses_mw(self) -> float:
        lines = self.branches[self.branches["type"].eq("line")]
        return float(lines["estimated_loss_mw"].sum()) if len(lines) else 0.0


def _series_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _copy_scenario(base: Scenario, *, demand: pd.Series, assets: pd.DataFrame, note: str) -> Scenario:
    notes = (base.notes + " " if base.notes else "") + note
    return Scenario(
        timestamp=base.timestamp,
        demand_by_bus=demand.astype(float).copy(),
        assets=assets.copy(),
        fixed_injection_by_bus=base.fixed_injection_by_bus.copy(),
        total_demand_mw=float(demand.sum()),
        interconnector_mw=float(base.interconnector_mw),
        notes=notes,
    )


def _increase_renewable_potential(
    assets: pd.DataFrame,
    bucket: str,
    requested_addition_mw: float,
) -> float:
    """Increase renewable potential up to each asset's stated available/MEC cap.

    Returns the actual addition achieved; the requested amount can be larger than
    the mapped fleet's remaining headroom.
    """
    requested = max(0.0, float(requested_addition_mw))
    if requested <= EPS:
        return 0.0
    mask = assets["market_bucket"].eq(bucket) & assets["mapped_bus"].notna()
    if not mask.any():
        return 0.0
    current = _series_numeric(assets.loc[mask], "potential_mw")
    caps = _series_numeric(assets.loc[mask], "available_mw")
    if "max_export_capacity_mw" in assets.columns:
        mec = pd.to_numeric(assets.loc[mask, "max_export_capacity_mw"], errors="coerce")
        caps = caps.where(caps > 0, mec).fillna(current)
    caps = pd.concat([caps, current], axis=1).max(axis=1)
    room = (caps - current).clip(lower=0.0)
    total_room = float(room.sum())
    addition = min(requested, total_room)
    if addition <= EPS or total_room <= EPS:
        return 0.0
    delta = room / total_room * addition
    assets.loc[mask, "potential_mw"] = current + delta
    # Start by attempting to accept all newly available renewable power.  System
    # balancing and network redispatch below decide what must actually be curtailed.
    assets.loc[mask, "dispatch_mw"] = assets.loc[mask, "potential_mw"].astype(float)
    return float(addition)


def _reduce_bucket_proportionally(assets: pd.DataFrame, mask: pd.Series, amount_mw: float) -> float:
    """Reduce dispatch proportionally across a selected fleet and return MW reduced."""
    amount = max(0.0, float(amount_mw))
    if amount <= EPS or not mask.any():
        return 0.0
    dispatch = _series_numeric(assets.loc[mask], "dispatch_mw").clip(lower=0.0)
    available = float(dispatch.sum())
    reduction = min(amount, available)
    if reduction <= EPS or available <= EPS:
        return 0.0
    delta = dispatch / available * reduction
    assets.loc[mask, "dispatch_mw"] = dispatch - delta
    return float(reduction)


def _increase_thermal_for_deficit(assets: pd.DataFrame, deficit_mw: float) -> float:
    """Increase thermal output in proportion to available headroom."""
    deficit = max(0.0, float(deficit_mw))
    thermal = assets["market_bucket"].eq("thermal") & assets["mapped_bus"].notna()
    if deficit <= EPS or not thermal.any():
        return 0.0
    dispatch = _series_numeric(assets.loc[thermal], "dispatch_mw").clip(lower=0.0)
    caps = _series_numeric(assets.loc[thermal], "available_mw")
    if "max_export_capacity_mw" in assets.columns:
        mec = pd.to_numeric(assets.loc[thermal, "max_export_capacity_mw"], errors="coerce")
        caps = caps.where(caps > 0, mec).fillna(dispatch)
    caps = pd.concat([caps, dispatch], axis=1).max(axis=1)
    room = (caps - dispatch).clip(lower=0.0)
    total_room = float(room.sum())
    response = min(deficit, total_room)
    if response <= EPS or total_room <= EPS:
        return 0.0
    assets.loc[thermal, "dispatch_mw"] = dispatch + room / total_room * response
    return float(response)


def _system_balance_before_network(scenario: Scenario, assets: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Balance the system before solving network constraints.

    Excess generation is handled in this order:
      1. reduce conventional thermal output;
      2. curtail wind and solar proportionally;
      3. leave any tiny/unavoidable residual for the normal slack bus.

    A deficit is met by thermal headroom first, with any remaining deficit left to
    the slack bus.  Returns (assets, balance_curtailment_mw, thermal_change_mw).
    """
    out = assets.copy()
    fixed = float(scenario.fixed_injection_by_bus.sum()) if len(scenario.fixed_injection_by_bus) else 0.0
    generation = float(_series_numeric(out, "dispatch_mw").sum())
    demand = float(scenario.demand_by_bus.sum())
    imbalance = generation + fixed - demand  # positive = too much generation

    thermal_before = float(_series_numeric(out[out["market_bucket"].eq("thermal")], "dispatch_mw").sum())
    balance_curtailment = 0.0

    if imbalance > EPS:
        thermal_mask = out["market_bucket"].eq("thermal") & out["mapped_bus"].notna()
        thermal_down = _reduce_bucket_proportionally(out, thermal_mask, imbalance)
        imbalance -= thermal_down
        if imbalance > EPS:
            renewable_mask = out["market_bucket"].isin(RENEWABLE_BUCKETS) & out["mapped_bus"].notna()
            balance_curtailment = _reduce_bucket_proportionally(out, renewable_mask, imbalance)
            imbalance -= balance_curtailment
    elif imbalance < -EPS:
        pickup = _increase_thermal_for_deficit(out, -imbalance)
        imbalance += pickup

    thermal_after = float(_series_numeric(out[out["market_bucket"].eq("thermal")], "dispatch_mw").sum())
    return out, float(balance_curtailment), float(thermal_after - thermal_before)


def _renewable_metrics(assets: pd.DataFrame) -> tuple[float, float, float, float]:
    renew = assets[assets["market_bucket"].isin(RENEWABLE_BUCKETS)].copy()
    if renew.empty:
        return 0.0, 0.0, 0.0, 100.0
    available = float(_series_numeric(renew, "potential_mw").clip(lower=0.0).sum())
    accepted = float(_series_numeric(renew, "dispatch_mw").clip(lower=0.0).sum())
    # Never report negative curtailment if a later solver nudges a value by epsilon.
    curtailed = max(0.0, available - accepted)
    utilisation = 100.0 * accepted / available if available > EPS else 100.0
    return available, accepted, curtailed, float(np.clip(utilisation, 0.0, 100.0))


def build_renewable_surplus_frames(
    network: NetworkData,
    model: DCGridModel,
    base_scenario: Scenario,
    config: RenewableSurplusConfig,
) -> List[RenewableSurplusFrame]:
    """Build solved frames for renewable surplus and/or falling demand."""
    steps = int(np.clip(config.steps, 2, 60))
    duration = max(1.0, float(config.duration_minutes))
    target_bus = config.target_bus if str(config.demand_scope).lower() == "bus" else None
    if str(config.demand_scope).lower() == "bus" and not target_bus:
        raise ValueError("A target load bus is required for a local demand change.")
    if target_bus is not None and target_bus not in set(model.bus_ids):
        raise ValueError(f"Target bus {target_bus!r} is not in the network.")

    base_assets = base_scenario.assets.copy()
    base_assets["dispatch_mw"] = _series_numeric(base_assets, "dispatch_mw")
    base_assets["potential_mw"] = base_assets["dispatch_mw"].clip(lower=0.0)
    base_thermal = float(base_assets.loc[base_assets["market_bucket"].eq("thermal"), "dispatch_mw"].sum())
    base_demand = float(base_scenario.total_demand_mw)
    overrides = config.line_limit_overrides or None
    frames: List[RenewableSurplusFrame] = []

    for step in range(steps + 1):
        fraction = step / steps
        demand_delta = float(config.demand_change_mw) * fraction
        # Demand is altered without automatic generation response; surplus balancing
        # is handled explicitly below so curtailment is measurable.
        demand_case = apply_demand_shock(
            base_scenario,
            demand_delta,
            target_bus=target_bus,
            generation_response="slack_only",
        )
        assets = base_assets.copy()
        actual_wind_add = _increase_renewable_potential(assets, "wind", float(config.wind_surge_mw) * fraction)
        actual_solar_add = _increase_renewable_potential(assets, "solar", float(config.solar_surge_mw) * fraction)

        provisional = _copy_scenario(
            demand_case,
            demand=demand_case.demand_by_bus,
            assets=assets,
            note=(
                f"Renewable availability scenario: wind +{actual_wind_add:.1f} MW, "
                f"solar +{actual_solar_add:.1f} MW before balancing."
            ),
        )
        commitment = None
        stability_curtailment = 0.0
        minimum_output_curtailment = 0.0
        if config.commitment_enabled:
            commit_cfg = CommitmentConfig(
                enabled=True,
                max_nonsynchronous_share_pct=float(config.max_nonsynchronous_share_pct),
                min_synchronous_units_online=int(config.min_synchronous_units_online),
                min_stable_output_pct=float(config.min_stable_output_pct),
                shutdown_strategy=str(config.shutdown_strategy),
            )
            fixed_total = float(provisional.fixed_injection_by_bus.sum()) if len(provisional.fixed_injection_by_bus) else 0.0
            balanced_assets, commitment = apply_commitment_and_balance(
                assets,
                demand_mw=float(provisional.total_demand_mw),
                fixed_injection_mw=fixed_total,
                interconnector_mw=float(provisional.interconnector_mw),
                config=commit_cfg,
            )
            balance_curtailment = float(commitment.get("balance_curtailment_mw", 0.0))
            stability_curtailment = float(commitment.get("stability_curtailment_mw", 0.0))
            minimum_output_curtailment = float(commitment.get("minimum_output_curtailment_mw", 0.0))
            note = (
                f"Unit commitment enabled: {commitment.get('online_thermal_units', 0)} thermal units online; "
                f"non-synchronous share {commitment.get('nonsynchronous_share_pct', 0.0):.1f}% against "
                f"a {commitment.get('max_nonsynchronous_share_pct', 100.0):.1f}% scenario cap. "
                f"Curtailment before network solve: {balance_curtailment:.1f} MW energy-balance, "
                f"{stability_curtailment:.1f} MW stability-share, "
                f"{minimum_output_curtailment:.1f} MW minimum-output."
            )
        else:
            balanced_assets, balance_curtailment, _ = _system_balance_before_network(provisional, assets)
            note = (
                f"Continuous thermal balancing used. Energy-balance curtailment at this step: "
                f"{balance_curtailment:.1f} MW."
            )
        scenario = _copy_scenario(
            provisional,
            demand=provisional.demand_by_bus,
            assets=balanced_assets,
            note=note,
        )

        injections, active, slack, slack_balance = scenario_injections(network, scenario)
        branches, buses = model.solve(
            injections,
            slack,
            thermal_scale=float(config.thermal_scale),
            limit_overrides=overrides,
        )

        pre_network_available, pre_network_accepted, pre_network_curt, _ = _renewable_metrics(active)
        redispatch = None
        network_curtailment = 0.0
        if config.auto_redispatch and (branches["loading_pct"] > 100.0 + 1e-7).any():
            new_injections, changed_assets, redispatch = relieve_thermal_constraints(
                model,
                scenario,
                injections,
                active,
                slack,
                thermal_scale=float(config.thermal_scale),
                limit_overrides=overrides,
                allow_load_shedding=bool(config.allow_load_shedding),
            )
            if redispatch.get("success"):
                injections = new_injections
                active = changed_assets
                branches, buses = model.solve(
                    injections,
                    slack,
                    thermal_scale=float(config.thermal_scale),
                    limit_overrides=overrides,
                )
                scenario = _copy_scenario(
                    scenario,
                    demand=scenario.demand_by_bus,
                    assets=changed_assets,
                    note="Corrective transmission redispatch applied after surplus balancing.",
                )
                _, post_accepted, post_curt, _ = _renewable_metrics(active)
                network_curtailment = max(0.0, post_curt - pre_network_curt)

        available, accepted, curtailed, utilisation = _renewable_metrics(active)
        thermal_now = float(
            _series_numeric(active[active["market_bucket"].eq("thermal")], "dispatch_mw").sum()
        )
        # Refresh commitment diagnostics after any network redispatch. Commitment
        # statuses remain fixed, but thermal MW and renewable acceptance can move.
        if commitment is not None:
            thermal_rows = active[active["market_bucket"].eq("thermal")].copy()
            online_mask = thermal_rows.get(
                "committed", pd.Series(thermal_rows["dispatch_mw"].to_numpy() > EPS, index=thermal_rows.index)
            ).astype(bool)
            biomass_waste = active[active["market_bucket"].isin(["biomass", "waste"])]
            other_sync = float(_series_numeric(biomass_waste, "dispatch_mw").sum())
            hvdc_import = max(0.0, float(scenario.interconnector_mw))
            hvdc_export = max(0.0, -float(scenario.interconnector_mw))
            basis = max(EPS, float(scenario.total_demand_mw) + hvdc_export)
            commitment["thermal_generation_mw"] = thermal_now
            commitment["synchronous_generation_mw"] = thermal_now + other_sync
            commitment["online_thermal_units"] = int(online_mask.sum())
            commitment["offline_thermal_units"] = int(len(thermal_rows) - online_mask.sum())
            commitment["nonsynchronous_mw"] = accepted + hvdc_import
            commitment["nonsynchronous_share_pct"] = 100.0 * (accepted + hvdc_import) / basis
        frames.append(
            RenewableSurplusFrame(
                step=step,
                fraction=fraction,
                elapsed_minutes=duration * fraction,
                actual_demand_change_mw=float(scenario.total_demand_mw - base_demand),
                requested_wind_surge_mw=float(config.wind_surge_mw) * fraction,
                actual_wind_surge_mw=actual_wind_add,
                requested_solar_surge_mw=float(config.solar_surge_mw) * fraction,
                actual_solar_surge_mw=actual_solar_add,
                scenario=scenario,
                injections_mw=injections.copy(),
                active_assets=active.copy(),
                branches=branches.copy(),
                buses=buses.copy(),
                slack_bus=str(slack),
                slack_balance_mw=float(slack_balance),
                balance_curtailment_mw=float(balance_curtailment),
                stability_curtailment_mw=float(stability_curtailment),
                minimum_output_curtailment_mw=float(minimum_output_curtailment),
                network_curtailment_mw=float(network_curtailment),
                commitment=commitment,
                renewable_available_mw=available,
                renewable_accepted_mw=accepted,
                renewable_curtailment_mw=curtailed,
                renewable_utilisation_pct=utilisation,
                thermal_change_mw=float(thermal_now - base_thermal),
                redispatch=redispatch,
            )
        )
    return frames


def surplus_frame_metrics(frames: List[RenewableSurplusFrame]) -> pd.DataFrame:
    """Scenario progression including cumulative lost renewable energy."""
    rows = []
    cumulative_mwh = 0.0
    previous = None
    for f in frames:
        if previous is not None:
            dt_h = (f.elapsed_minutes - previous.elapsed_minutes) / 60.0
            cumulative_mwh += 0.5 * (previous.renewable_curtailment_mw + f.renewable_curtailment_mw) * dt_h
        lines = f.branches[f.branches["type"].eq("line")]
        worst = lines.sort_values("loading_pct", ascending=False).iloc[0] if len(lines) else None
        rows.append(
            {
                "step": f.step,
                "elapsed_min": f.elapsed_minutes,
                "demand_change_mw": f.actual_demand_change_mw,
                "total_demand_mw": f.scenario.total_demand_mw,
                "wind_availability_added_mw": f.actual_wind_surge_mw,
                "solar_availability_added_mw": f.actual_solar_surge_mw,
                "renewable_available_mw": f.renewable_available_mw,
                "renewable_accepted_mw": f.renewable_accepted_mw,
                "curtailment_mw": f.renewable_curtailment_mw,
                "curtailment_pct": 100.0 - f.renewable_utilisation_pct,
                "renewable_utilisation_pct": f.renewable_utilisation_pct,
                "balance_curtailment_mw": f.balance_curtailment_mw,
                "stability_curtailment_mw": f.stability_curtailment_mw,
                "minimum_output_curtailment_mw": f.minimum_output_curtailment_mw,
                "network_curtailment_mw": f.network_curtailment_mw,
                "online_thermal_units": int((f.commitment or {}).get("online_thermal_units", 0)),
                "offline_thermal_units": int((f.commitment or {}).get("offline_thermal_units", 0)),
                "nonsynchronous_share_pct": float((f.commitment or {}).get("nonsynchronous_share_pct", 0.0)),
                "synchronous_generation_mw": float((f.commitment or {}).get("synchronous_generation_mw", 0.0)),
                "cumulative_curtailed_energy_mwh": cumulative_mwh,
                "thermal_change_mw": f.thermal_change_mw,
                "slack_balance_mw": f.slack_balance_mw,
                "worst_line": str(worst.branch) if worst is not None else "",
                "worst_loading_pct": float(worst.loading_pct) if worst is not None else 0.0,
                "overloaded_lines": f.overloaded_lines,
                "estimated_line_losses_mw": f.estimated_losses_mw,
            }
        )
        previous = f
    return pd.DataFrame(rows)


def renewable_curtailment_by_asset(frame: RenewableSurplusFrame) -> pd.DataFrame:
    """Final per-asset renewable potential, accepted dispatch and curtailment."""
    a = frame.active_assets[frame.active_assets["market_bucket"].isin(RENEWABLE_BUCKETS)].copy()
    if a.empty:
        return pd.DataFrame()
    a["potential_mw"] = _series_numeric(a, "potential_mw").clip(lower=0.0)
    a["dispatch_mw"] = _series_numeric(a, "dispatch_mw").clip(lower=0.0)
    a["curtailed_mw"] = (a["potential_mw"] - a["dispatch_mw"]).clip(lower=0.0)
    a["utilisation_pct"] = np.where(
        a["potential_mw"] > EPS,
        100.0 * a["dispatch_mw"] / a["potential_mw"],
        100.0,
    )
    keep = [
        c for c in [
            "point_name", "point_type", "market_bucket", "mapped_bus", "mapping_distance_km",
            "max_export_capacity_mw", "potential_mw", "dispatch_mw", "curtailed_mw", "utilisation_pct",
        ] if c in a.columns
    ]
    return a[keep].sort_values(["curtailed_mw", "potential_mw"], ascending=[False, False]).reset_index(drop=True)


def thermal_commitment_by_asset(frame: RenewableSurplusFrame) -> pd.DataFrame:
    """Final thermal unit online/offline state for the surplus scenario."""
    return commitment_table(frame.active_assets)


def peak_line_loading_surplus(frames: List[RenewableSurplusFrame]) -> pd.DataFrame:
    rows = []
    for f in frames:
        lines = f.branches[f.branches["type"].eq("line")].copy()
        lines["step"] = f.step
        lines["elapsed_min"] = f.elapsed_minutes
        lines["curtailment_mw"] = f.renewable_curtailment_mw
        rows.append(lines)
    if not rows:
        return pd.DataFrame()
    all_lines = pd.concat(rows, ignore_index=True)
    idx = all_lines.groupby("branch")["loading_pct"].idxmax()
    keep = [
        "branch", "bus0", "bus1", "step", "elapsed_min", "curtailment_mw", "flow_mw",
        "thermal_limit_mw", "loading_pct", "estimated_loss_mw", "estimated_efficiency_pct", "overload_mw",
    ]
    return all_lines.loc[idx, keep].sort_values("loading_pct", ascending=False).reset_index(drop=True)


def curtailment_energy_mwh(frames: List[RenewableSurplusFrame]) -> float:
    if len(frames) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(frames[:-1], frames[1:]):
        dt_h = (b.elapsed_minutes - a.elapsed_minutes) / 60.0
        total += 0.5 * (a.renewable_curtailment_mw + b.renewable_curtailment_mw) * dt_h
    return float(total)
