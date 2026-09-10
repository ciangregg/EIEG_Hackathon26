"""Calibrated Northern Ireland baseline grid emulator.

This module intentionally contains no optimisation/annealing logic.  It builds a
frozen Northern Ireland operating-scenario set, applies the published/current
constraint-group policy, and produces a calibrated dispatch-down baseline that
can be handed to a separate optimiser.

Two quantities are kept separate:
1. physical_model_dispatch_down: dispatch-down produced by the represented
   network/security/operational model;
2. calibration_residual: a fixed scenario-level residual used to reconcile the
   incomplete NI-only model with an external historical SONI benchmark.

An external optimiser should change only the group-policy/network component and
must keep the calibration residual fixed for each scenario.  This prevents an
optimiser from claiming savings in effects that the model does not explicitly
represent (for example full all-island inertia/transient/voltage behaviour or
missing cross-border topology).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import json
import zipfile

import numpy as np
import pandas as pd

from ni_constraint_stress import SONI_GROUPS
from ni_grid_core import DCGridModel, NetworkData
from ni_operational_model import (
    OperationalConfig,
    OperatingScenarioSet,
    PolicyResult,
    current_group_masks,
    evaluate_policy,
    generate_operating_scenarios,
)


SONI_2024_ALL_RENEWABLE_DD_PCT = 25.5
SONI_2024_WIND_DD_PCT = 29.6
SONI_2024_SOLAR_DD_PCT = 16.9
SONI_2024_WIND_DD_GWH = 915.0
SONI_2024_SOLAR_DD_GWH = 22.0


@dataclass
class BaselineReplicaResult:
    scenarios: OperatingScenarioSet
    physical: PolicyResult
    calibrated_run_table: pd.DataFrame
    metrics: dict
    calibration_table: pd.DataFrame
    membership_table: pd.DataFrame
    metadata: dict


def current_operational_config(
    runs: int = 10_000,
    seed: int = 42,
    demand_scale: float = 2.10,
    thermal_scale: float = 1.0,
    planned_outage_exposure_pct: float = 5.0,
) -> OperationalConfig:
    """Best available NI-only approximation to current published operating rules.

    Values that are directly represented here include a 75% SNSP ceiling,
    minimum two synchronous NI units, North-South long-term transfer limits and
    Moyle import/export limits.  Full all-island inertia/transient/voltage
    security cannot be reproduced in the supplied NI-only DC network and is left
    to the explicit calibration residual.
    """
    return OperationalConfig(
        runs=int(runs),
        seed=int(seed),
        batch_size=250,
        demand_scale=float(demand_scale),
        sns_limit_pct=75.0,
        min_sync_units=2,
        min_stable_output_pct=20.0,
        north_south_export_limit_mw=400.0,
        north_south_import_limit_mw=200.0,
        moyle_export_limit_mw=410.0,
        moyle_import_limit_mw=441.0,
        thermal_scale=float(thermal_scale),
        coolkeeragh_must_run_mw=0.0,
        planned_outage_exposure_pct=float(planned_outage_exposure_pct),
        moyle_mean_mw=0.0,
        include_n1_diagnostic=True,
        enforce_all_n1_with_groups=True,
        max_group_actions=8,
    )


def _membership_table(scenarios: OperatingScenarioSet) -> pd.DataFrame:
    renew = scenarios.renewable_assets.reset_index(drop=True)
    masks = current_group_masks(renew)
    rows = []
    for gid in (1, 2, 3, 4, 5):
        for i in np.flatnonzero(masks[gid]):
            a = renew.iloc[int(i)]
            rows.append({
                "group": gid,
                "group_name": SONI_GROUPS[gid]["name"],
                "asset_index": int(i),
                "asset": str(a.get("point_name", i)),
                "bus": str(a.get("mapped_bus", "")),
                "market_bucket": str(a.get("market_bucket", "renewable")),
                "mec_mw": float(a.get("max_export_capacity_mw", 0.0)),
                "latitude": float(a.get("latitude", np.nan)),
                "longitude": float(a.get("longitude", np.nan)),
            })
    return pd.DataFrame(rows)


def _weighted_capped_allocation(total: float, weights: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Allocate total across cases proportionally, never exceeding per-case caps."""
    total = max(0.0, float(total))
    caps = np.maximum(np.asarray(caps, float), 0.0)
    weights = np.maximum(np.asarray(weights, float), 0.0)
    out = np.zeros_like(caps)
    remaining = min(total, float(caps.sum()))
    active = caps > 1e-12
    for _ in range(100):
        if remaining <= 1e-9 or not np.any(active):
            break
        w = weights.copy()
        w[~active] = 0.0
        if w.sum() <= 1e-15:
            w = active.astype(float)
        proposal = remaining * w / w.sum()
        room = caps - out
        add = np.minimum(proposal, room)
        out += add
        used = float(add.sum())
        remaining -= used
        active = room - add > 1e-10
        if used <= 1e-12:
            break
    return out


def _calibration_residual(
    scenarios: OperatingScenarioSet,
    physical: PolicyResult,
    target_dispatch_down_pct: float,
) -> tuple[np.ndarray, dict]:
    """Create a fixed per-scenario residual whose aggregate matches the benchmark.

    The aggregate benchmark is authoritative; the distribution across synthetic
    scenarios is only a risk-weighted calibration heuristic.  It is therefore
    exported and frozen for any downstream counterfactual/optimisation run.
    """
    rt = physical.run_table.reset_index(drop=True)
    potential = rt["renewable_potential_mw"].to_numpy(float)
    physical_dd = rt["total_dispatch_down_mw"].to_numpy(float)
    potential_total = float(potential.sum())
    physical_total = float(physical_dd.sum())
    target_total = float(target_dispatch_down_pct) / 100.0 * potential_total
    residual_required = max(0.0, target_total - physical_total)
    caps = np.maximum(potential - physical_dd, 0.0)

    demand = np.maximum(rt["demand_mw"].to_numpy(float), 1.0)
    renewable_ratio = potential / demand
    rr_scale = np.percentile(renewable_ratio, 95) if len(renewable_ratio) else 1.0
    rr_norm = np.clip(renewable_ratio / max(rr_scale, 1e-9), 0.0, 2.0)
    moyle_import = np.clip(np.asarray(scenarios.moyle_mw, float), 0.0, None) / 441.0
    outage = (rt["planned_outage"].astype(str).to_numpy() != "Intact").astype(float)
    dmax, dmin = float(np.max(demand)), float(np.min(demand))
    low_demand = (dmax - demand) / max(dmax - dmin, 1.0)

    # Risk weighting does not alter the aggregate target.  It simply places more
    # of the fixed residual into high-renewable, import-heavy, outage/low-demand
    # cases, which are qualitatively aligned with documented dispatch-down drivers.
    risk = potential * (0.35 + 1.10 * rr_norm + 0.45 * moyle_import + 0.65 * outage + 0.35 * low_demand)
    residual = _weighted_capped_allocation(residual_required, risk, caps)
    achieved_total = physical_total + float(residual.sum())
    achieved_pct = 100.0 * achieved_total / potential_total if potential_total > 0 else 0.0
    return residual, {
        "target_dispatch_down_pct": float(target_dispatch_down_pct),
        "physical_dispatch_down_pct": 100.0 * physical_total / potential_total if potential_total > 0 else 0.0,
        "calibration_residual_pct": 100.0 * float(residual.sum()) / potential_total if potential_total > 0 else 0.0,
        "calibrated_dispatch_down_pct": achieved_pct,
        "calibration_exact": abs(achieved_pct - float(target_dispatch_down_pct)) < 1e-6,
        "target_dispatch_down_mw_equivalent": target_total,
        "calibration_residual_mw_total": float(residual.sum()),
    }


def run_baseline_replica(
    network: NetworkData,
    model: DCGridModel,
    mapped_assets: pd.DataFrame,
    cfg: OperationalConfig,
    target_dispatch_down_pct: float = SONI_2024_ALL_RENEWABLE_DD_PCT,
    calibrate: bool = True,
    progress: Optional[Callable[[float, str], None]] = None,
) -> BaselineReplicaResult:
    if progress:
        progress(0.01, "Generating frozen NI operating cases")
    scenarios = generate_operating_scenarios(network, model, mapped_assets, cfg)
    masks = current_group_masks(scenarios.renewable_assets)
    if progress:
        progress(0.12, "Applying current published constraint-group policy")
    physical = evaluate_policy(
        network, model, scenarios, cfg, masks,
        progress=progress, progress_start=0.12, progress_span=0.70,
    )

    rt = physical.run_table.copy().reset_index(drop=True)
    if calibrate:
        residual, cal = _calibration_residual(scenarios, physical, float(target_dispatch_down_pct))
    else:
        residual = np.zeros(len(rt), dtype=float)
        p = max(float(physical.metrics["potential_renewable_mw_total"]), 1e-9)
        cal = {
            "target_dispatch_down_pct": float(target_dispatch_down_pct),
            "physical_dispatch_down_pct": float(physical.metrics["dispatch_down_pct"]),
            "calibration_residual_pct": 0.0,
            "calibrated_dispatch_down_pct": float(physical.metrics["dispatch_down_pct"]),
            "calibration_exact": False,
            "target_dispatch_down_mw_equivalent": float(target_dispatch_down_pct) / 100.0 * p,
            "calibration_residual_mw_total": 0.0,
        }

    rt["calibration_residual_mw"] = residual
    rt["calibrated_total_dispatch_down_mw"] = rt["total_dispatch_down_mw"].to_numpy(float) + residual
    rt["calibrated_renewable_accepted_mw"] = np.maximum(
        rt["renewable_potential_mw"].to_numpy(float) - rt["calibrated_total_dispatch_down_mw"].to_numpy(float), 0.0
    )
    rt["physical_dispatch_down_pct_of_case"] = np.where(
        rt["renewable_potential_mw"] > 0,
        100.0 * rt["total_dispatch_down_mw"] / rt["renewable_potential_mw"],
        0.0,
    )
    rt["calibrated_dispatch_down_pct_of_case"] = np.where(
        rt["renewable_potential_mw"] > 0,
        100.0 * rt["calibrated_total_dispatch_down_mw"] / rt["renewable_potential_mw"],
        0.0,
    )

    potential_total = max(float(physical.metrics["potential_renewable_mw_total"]), 1e-9)
    network_pct = 100.0 * float(physical.metrics["network_group_dispatch_down_mw_total"]) / potential_total
    sns_pct = 100.0 * float(physical.metrics["sns_dispatch_down_mw_total"]) / potential_total
    min_sync_pct = 100.0 * float(physical.metrics["minimum_sync_dispatch_down_mw_total"]) / potential_total
    surplus_pct = 100.0 * float(physical.metrics["surplus_dispatch_down_mw_total"]) / potential_total

    metrics = {
        **cal,
        "calibrated_renewable_utilisation_pct": 100.0 - float(cal["calibrated_dispatch_down_pct"]),
        "physical_renewable_utilisation_pct": float(physical.metrics["renewable_utilisation_pct"]),
        "electrical_efficiency_pct": float(physical.metrics["electrical_efficiency_pct"]),
        "represented_security_pass_pct": float(physical.metrics["represented_security_pass_pct"]),
        "all_line_n1_pass_pct": float(physical.metrics["all_line_n1_pass_pct"]),
        "network_group_dispatch_down_pct": network_pct,
        "sns_dispatch_down_pct": sns_pct,
        "minimum_sync_dispatch_down_pct": min_sync_pct,
        "surplus_dispatch_down_pct": surplus_pct,
        "runs": int(physical.metrics["runs"]),
    }

    calibration_table = pd.DataFrame([
        {
            "metric": "Physical NI-only model dispatch-down",
            "dispatch_down_pct": metrics["physical_dispatch_down_pct"],
            "role": "Explicitly represented operational + network constraints",
        },
        {
            "metric": "Fixed calibration residual",
            "dispatch_down_pct": metrics["calibration_residual_pct"],
            "role": "Missing/unmodelled all-island, voltage/transient/inertia/outage/topology effects; frozen for downstream optimisation",
        },
        {
            "metric": "Calibrated baseline total",
            "dispatch_down_pct": metrics["calibrated_dispatch_down_pct"],
            "role": "Baseline metric supplied to downstream counterfactual work",
        },
        {
            "metric": "SONI NI 2024 actual — all renewables",
            "dispatch_down_pct": SONI_2024_ALL_RENEWABLE_DD_PCT,
            "role": "External aggregate validation benchmark",
        },
        {
            "metric": "SONI NI 2024 wind only",
            "dispatch_down_pct": SONI_2024_WIND_DD_PCT,
            "role": "Context: wind-only historical benchmark",
        },
        {
            "metric": "SONI NI 2024 solar only",
            "dispatch_down_pct": SONI_2024_SOLAR_DD_PCT,
            "role": "Context: solar-only historical benchmark",
        },
    ])

    metadata = {
        "model_purpose": "Northern Ireland current-policy baseline / handoff model; no optimiser included",
        "topology_basis": "Supplied SV2024_northern_ireland.nc NI transmission model",
        "group_policy": "Published SONI NI WDT groups 1-5; Group 4 represented via North-South boundary proxy",
        "calibration_target_pct": float(target_dispatch_down_pct),
        "calibration_rule": "Fixed residual calibrated once and exported per scenario; downstream optimisers must not modify it",
        "sns_limit_pct": float(cfg.sns_limit_pct),
        "min_sync_units": int(cfg.min_sync_units),
        "north_south_export_limit_mw": float(cfg.north_south_export_limit_mw),
        "north_south_import_limit_mw": float(cfg.north_south_import_limit_mw),
        "moyle_import_limit_mw": float(cfg.moyle_import_limit_mw),
        "moyle_export_limit_mw": float(cfg.moyle_export_limit_mw),
        "thermal_scale": float(cfg.thermal_scale),
        "planned_outage_exposure_pct": float(cfg.planned_outage_exposure_pct),
        "seed": int(cfg.seed),
        "runs": int(cfg.runs),
        "known_model_limitations": [
            "NI-only DC load flow; no explicit Republic of Ireland AC network / Louth tie-line",
            "No full AC voltage/reactive-power solution",
            "No transient stability or explicit all-island inertia/RoCoF dynamics",
            "Generator commitment/availability and outage chronology are simplified",
            "Renewable weather and scheduled boundary flows are synthetic unless replaced by historical inputs",
            "Aggregate calibration benchmark is historical 2024 because this is the latest annual NI renewable dispatch-down report exposed on SONI's current data page",
        ],
    }
    if progress:
        progress(0.90, "Preparing calibrated baseline and handoff data")
    return BaselineReplicaResult(
        scenarios=scenarios,
        physical=physical,
        calibrated_run_table=rt,
        metrics=metrics,
        calibration_table=calibration_table,
        membership_table=_membership_table(scenarios),
        metadata=metadata,
    )


def export_handoff_pack(result: BaselineReplicaResult, output_dir: str | Path, stem: str = "ni_baseline_handoff") -> str:
    """Export a stable scenario pack for an external group optimiser."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    root = outdir / stem
    root.mkdir(parents=True, exist_ok=True)

    result.calibrated_run_table.to_csv(root / "baseline_runs.csv", index=False)
    result.membership_table.to_csv(root / "current_group_membership.csv", index=False)
    result.physical.group_summary.to_csv(root / "current_group_summary.csv", index=False)
    result.physical.line_summary.to_csv(root / "line_summary.csv", index=False)

    s = result.scenarios
    np.savez_compressed(
        root / "frozen_scenarios.npz",
        demand_mw=s.demand_mw,
        renewable_potential_mw=s.renewable_potential_mw,
        demand_dispatch_mw=s.demand_dispatch_mw,
        moyle_mw=s.moyle_mw,
        north_south_mw=s.north_south_mw,
        initial_thermal_dispatch_mw=s.initial_thermal_dispatch_mw,
        pre_network_renewable_dispatch_mw=s.pre_network_renewable_dispatch_mw,
        pre_network_dispatch_down_mw=s.pre_network_dispatch_down_mw,
        initial_injections_mw=s.initial_injections_mw,
        initial_flows_mw=s.initial_flows_mw,
        planned_outage_line_index=s.planned_outage_line_index,
        calibration_residual_mw=result.calibrated_run_table["calibration_residual_mw"].to_numpy(float),
    )
    with open(root / "baseline_metadata.json", "w", encoding="utf-8") as f:
        json.dump({**result.metadata, "metrics": result.metrics}, f, indent=2)

    readme = f"""# Northern Ireland baseline handoff\n\nThis pack contains a frozen {result.metrics['runs']:,}-scenario baseline.\n\n## Contract for downstream optimisation\n- Keep every frozen operating scenario unchanged.\n- Keep `calibration_residual_mw` unchanged for each scenario.\n- Change only the constraint-group policy/membership and the resulting network-group dispatch-down.\n- A candidate is not an improvement if it makes a scenario insecure that is secure under the baseline.\n- Report physical network/group dispatch-down separately from the calibrated total dispatch-down.\n\n## Baseline headline metrics\n- Physical model dispatch-down: {result.metrics['physical_dispatch_down_pct']:.4f}%\n- Fixed calibration residual: {result.metrics['calibration_residual_pct']:.4f}%\n- Calibrated total dispatch-down: {result.metrics['calibrated_dispatch_down_pct']:.4f}%\n- Calibrated renewable utilisation: {result.metrics['calibrated_renewable_utilisation_pct']:.4f}%\n- Estimated transmission electrical efficiency: {result.metrics['electrical_efficiency_pct']:.4f}%\n\nThe calibration residual is deliberately not an optimiser variable.\n"""
    (root / "README_HANDOFF.md").write_text(readme, encoding="utf-8")

    zip_path = outdir / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in root.rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(root.parent))
    return str(zip_path)
