"""Dynamic scenario engine for the Northern Ireland grid simulator.

This module turns a static market/network snapshot into a sequence of quasi-steady-
state operating points.  It is intentionally *not* an electromechanical transient
or frequency-stability model: every frame is a fresh DC load-flow solution after a
controlled change in demand and generation response.

That makes it appropriate for questions such as:
- What corridors pick up a large evening/TV-pickup-style demand spike?
- Which lines approach or exceed their thermal ratings?
- Which mapped generators respond when demand rises?
- If a thermal limit is violated, which assets are redispatched to relieve it?
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


@dataclass
class DynamicScenarioConfig:
    """Configuration for a ramped demand-shock experiment."""

    shock_mw: float = 250.0
    scope: str = "system"  # "system" or "bus"
    target_bus: Optional[str] = None
    steps: int = 12
    response_mode: str = "thermal_headroom"  # or "slack_only"
    thermal_scale: float = 1.0
    line_limit_overrides: Optional[Dict[str, float]] = None
    auto_redispatch: bool = True
    allow_load_shedding: bool = False


@dataclass
class ScenarioFrame:
    """One solved quasi-steady-state point in a dynamic scenario."""

    step: int
    fraction: float
    requested_shock_mw: float
    actual_demand_change_mw: float
    scenario: Scenario
    injections_mw: pd.Series
    active_assets: pd.DataFrame
    branches: pd.DataFrame
    buses: pd.DataFrame
    slack_bus: str
    slack_balance_mw: float
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


def _scenario_with_assets(scenario: Scenario, assets: pd.DataFrame, note: str = "") -> Scenario:
    notes = scenario.notes
    if note:
        notes = (notes + " " if notes else "") + note
    return Scenario(
        timestamp=scenario.timestamp,
        demand_by_bus=scenario.demand_by_bus.copy(),
        assets=assets.copy(),
        fixed_injection_by_bus=scenario.fixed_injection_by_bus.copy(),
        total_demand_mw=float(scenario.total_demand_mw),
        interconnector_mw=float(scenario.interconnector_mw),
        notes=notes,
    )


def build_demand_shock_frames(
    network: NetworkData,
    model: DCGridModel,
    base_scenario: Scenario,
    config: DynamicScenarioConfig,
) -> List[ScenarioFrame]:
    """Generate and solve all frames for a ramped demand shock.

    The base case is included as step 0.  At every subsequent step the demand
    shock is increased linearly from 0 to ``shock_mw`` and the network is solved
    from the same original market snapshot.  This avoids compounding rounding or
    redispatch from one frame into the next.
    """
    steps = int(np.clip(config.steps, 2, 60))
    target_bus = config.target_bus if str(config.scope).lower() == "bus" else None
    if str(config.scope).lower() == "bus" and not target_bus:
        raise ValueError("A target load bus is required for a local demand shock.")
    if target_bus is not None and target_bus not in set(model.bus_ids):
        raise ValueError(f"Target bus {target_bus!r} is not in the network.")

    overrides = config.line_limit_overrides or None
    frames: List[ScenarioFrame] = []
    base_demand = float(base_scenario.total_demand_mw)

    for step in range(steps + 1):
        fraction = step / steps
        requested = float(config.shock_mw) * fraction
        shocked = apply_demand_shock(
            base_scenario,
            requested,
            target_bus=target_bus,
            generation_response=config.response_mode,
        )
        injections, active, slack, slack_balance = scenario_injections(network, shocked)
        branches, buses = model.solve(
            injections,
            slack,
            thermal_scale=float(config.thermal_scale),
            limit_overrides=overrides,
        )

        redispatch = None
        if config.auto_redispatch and (branches["loading_pct"] > 100.0 + 1e-7).any():
            new_injections, changed_assets, redispatch = relieve_thermal_constraints(
                model,
                shocked,
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
                shocked = _scenario_with_assets(
                    shocked,
                    changed_assets,
                    note="Corrective thermal-constraint redispatch applied at this scenario step.",
                )

        frames.append(
            ScenarioFrame(
                step=step,
                fraction=fraction,
                requested_shock_mw=requested,
                actual_demand_change_mw=float(shocked.total_demand_mw - base_demand),
                scenario=shocked,
                injections_mw=injections.copy(),
                active_assets=active.copy(),
                branches=branches.copy(),
                buses=buses.copy(),
                slack_bus=str(slack),
                slack_balance_mw=float(slack_balance),
                redispatch=redispatch,
            )
        )
    return frames


def frame_metrics(frames: List[ScenarioFrame]) -> pd.DataFrame:
    """Compact table for plotting/exporting the dynamic run."""
    rows = []
    for f in frames:
        lines = f.branches[f.branches["type"].eq("line")]
        worst = lines.sort_values("loading_pct", ascending=False).iloc[0] if len(lines) else None
        rd = f.redispatch or {}
        rows.append(
            {
                "step": f.step,
                "shock_fraction_pct": round(100 * f.fraction, 1),
                "demand_change_mw": f.actual_demand_change_mw,
                "total_demand_mw": f.scenario.total_demand_mw,
                "slack_balance_mw": f.slack_balance_mw,
                "worst_line": str(worst.branch) if worst is not None else "",
                "worst_loading_pct": float(worst.loading_pct) if worst is not None else 0.0,
                "overloaded_lines": f.overloaded_lines,
                "estimated_losses_mw": f.estimated_losses_mw,
                "redispatch_down_mw": float(rd.get("total_down_mw", 0.0)),
                "redispatch_up_mw": float(rd.get("total_up_mw", 0.0)),
                "renewable_curtailment_mw": float(rd.get("renewable_curtailment_mw", 0.0)),
                "load_shed_mw": float(rd.get("load_shed_mw", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def peak_line_loading(frames: List[ScenarioFrame]) -> pd.DataFrame:
    """Return each transmission line's peak loading across the whole shock."""
    rows = []
    for f in frames:
        lines = f.branches[f.branches["type"].eq("line")].copy()
        lines["step"] = f.step
        lines["demand_change_mw"] = f.actual_demand_change_mw
        rows.append(lines)
    if not rows:
        return pd.DataFrame()
    all_lines = pd.concat(rows, ignore_index=True)
    idx = all_lines.groupby("branch")["loading_pct"].idxmax()
    cols = [
        "branch", "bus0", "bus1", "step", "demand_change_mw", "flow_mw",
        "thermal_limit_mw", "loading_pct", "estimated_loss_mw",
        "estimated_efficiency_pct", "overload_mw",
    ]
    return all_lines.loc[idx, cols].sort_values("loading_pct", ascending=False).reset_index(drop=True)


def generator_response_summary(frames: List[ScenarioFrame]) -> pd.DataFrame:
    """Compare mapped asset dispatch at the start and end of the scenario."""
    if not frames:
        return pd.DataFrame()
    first = frames[0].active_assets.copy()
    last = frames[-1].active_assets.copy()
    key = "point_name"
    keep = [key, "point_type", "market_bucket", "mapped_bus", "max_export_capacity_mw", "dispatch_mw"]
    first = first[[c for c in keep if c in first.columns]].copy().rename(columns={"dispatch_mw": "dispatch_before_mw"})
    last_cols = [key, "dispatch_mw"] + [c for c in ["redispatch_down_mw", "redispatch_up_mw"] if c in last.columns]
    last = last[last_cols].copy().rename(columns={"dispatch_mw": "dispatch_after_mw"})
    out = first.merge(last, on=key, how="outer")
    out["dispatch_before_mw"] = pd.to_numeric(out["dispatch_before_mw"], errors="coerce").fillna(0.0)
    out["dispatch_after_mw"] = pd.to_numeric(out["dispatch_after_mw"], errors="coerce").fillna(0.0)
    out["change_mw"] = out["dispatch_after_mw"] - out["dispatch_before_mw"]
    return out[out["change_mw"].abs() > 1e-6].sort_values("change_mw", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
