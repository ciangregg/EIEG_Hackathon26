"""Simple synchronous-generator commitment model for NI renewable-surplus studies.

This is deliberately a transparent research approximation rather than a SONI unit-
commitment replica.  It adds three behaviours that the earlier continuous redispatch
model did not have:

* conventional thermal units can be explicitly online or offline;
* online units have a configurable minimum stable output;
* a configurable maximum non-synchronous penetration share can force wind/solar
  curtailment when demand is low or renewable availability is high.

The default 75% non-synchronous share is an illustrative scenario setting.  It is
not hard-coded as a statement of the current operational limit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-9


@dataclass
class CommitmentConfig:
    enabled: bool = True
    max_nonsynchronous_share_pct: float = 75.0
    min_synchronous_units_online: int = 2
    min_stable_output_pct: float = 20.0
    shutdown_strategy: str = "keep_current"  # keep_current or renewable_friendly
    count_positive_hvdc_import_as_nonsynchronous: bool = True


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _capacity(thermal: pd.DataFrame) -> pd.Series:
    caps = _num(thermal, "available_mw")
    if "max_export_capacity_mw" in thermal.columns:
        mec = pd.to_numeric(thermal["max_export_capacity_mw"], errors="coerce").fillna(0.0)
        caps = caps.where(caps > 0, mec)
    dispatch = _num(thermal, "dispatch_mw").clip(lower=0.0)
    return pd.concat([caps, dispatch], axis=1).max(axis=1).clip(lower=0.0)


def _renewable_reduce(assets: pd.DataFrame, amount_mw: float) -> float:
    """Curtail wind/solar proportionally and return the achieved reduction."""
    amount = max(0.0, float(amount_mw))
    mask = assets["market_bucket"].isin(["wind", "solar"]) & assets["mapped_bus"].notna()
    if amount <= EPS or not mask.any():
        return 0.0
    p = _num(assets.loc[mask], "dispatch_mw").clip(lower=0.0)
    total = float(p.sum())
    cut = min(amount, total)
    if cut <= EPS or total <= EPS:
        return 0.0
    assets.loc[mask, "dispatch_mw"] = p - (p / total * cut)
    return float(cut)


def _choose_committed_units(
    thermal: pd.DataFrame,
    target_mw: float,
    min_units: int,
    min_stable_fraction: float,
    strategy: str,
) -> List[int]:
    """Choose thermal units that remain committed as demand changes.

    ``keep_current`` intentionally keeps as many of the most heavily dispatched /
    largest units online as possible while their combined minimum-stable output still
    fits under the target.  This produces visible, stepwise shutdowns as demand falls
    rather than collapsing immediately to the minimum number of units.

    ``renewable_friendly`` keeps only the minimum set needed for capacity/security,
    preferring smaller minimum-stable blocks so more renewable generation can be used.
    """
    if thermal.empty:
        return []
    work = thermal.copy()
    work["_cap"] = _capacity(work)
    work["_base"] = _num(work, "dispatch_mw").clip(lower=0.0)
    work["_min"] = work["_cap"] * float(min_stable_fraction)
    work = work[work["_cap"] > EPS].copy()
    if work.empty:
        return []

    target = max(0.0, float(target_mw))
    min_units = int(np.clip(min_units, 0, len(work)))
    strategy = str(strategy or "keep_current").lower()
    if target <= EPS and min_units == 0:
        return []

    if strategy == "renewable_friendly":
        ranked = work.sort_values(["_min", "_cap", "_base"], ascending=[True, True, False])
        chosen: List[int] = []
        cap_sum = 0.0
        for idx, row in ranked.iterrows():
            if len(chosen) >= min_units and cap_sum + EPS >= target:
                break
            chosen.append(idx)
            cap_sum += float(row["_cap"])
        return chosen

    # Default: preserve the currently important units, and keep additional units online
    # until their aggregate minimum-stable output would exceed the required thermal MW.
    ranked = work.sort_values(["_base", "_cap"], ascending=[False, False])
    chosen = []
    cap_sum = 0.0
    min_sum = 0.0

    # First satisfy minimum-unit and capacity requirements.
    for idx, row in ranked.iterrows():
        need_unit_floor = len(chosen) < min_units
        need_capacity = cap_sum + EPS < target
        if not (need_unit_floor or need_capacity):
            break
        chosen.append(idx)
        cap_sum += float(row["_cap"])
        min_sum += float(row["_min"])

    # Then retain as many additional units as can remain above minimum stable output
    # without forcing thermal generation above the target.
    for idx, row in ranked.iterrows():
        if idx in chosen:
            continue
        candidate_min = min_sum + float(row["_min"])
        if candidate_min <= target + EPS:
            chosen.append(idx)
            cap_sum += float(row["_cap"])
            min_sum = candidate_min

    return chosen


def _dispatch_committed_units(
    thermal: pd.DataFrame,
    committed_indices: List[int],
    target_mw: float,
    min_stable_fraction: float,
) -> Tuple[pd.Series, float]:
    """Allocate a target thermal total across committed units.

    Returns (dispatch_by_index, forced_minimum_mw_above_requested_target).
    """
    dispatch = pd.Series(0.0, index=thermal.index, dtype=float)
    if not committed_indices:
        return dispatch, 0.0

    selected = thermal.loc[committed_indices].copy()
    caps = _capacity(selected)
    mins = caps * float(min_stable_fraction)
    minimum_total = float(mins.sum())
    cap_total = float(caps.sum())
    requested = max(0.0, float(target_mw))
    actual_target = min(max(requested, minimum_total), cap_total)
    forced = max(0.0, actual_target - requested)

    p = mins.copy()
    remaining = actual_target - minimum_total
    if remaining > EPS:
        headroom = (caps - mins).clip(lower=0.0)
        room = float(headroom.sum())
        if room > EPS:
            p += headroom / room * min(remaining, room)
    dispatch.loc[selected.index] = p
    return dispatch, float(forced)


def apply_commitment_and_balance(
    assets: pd.DataFrame,
    demand_mw: float,
    fixed_injection_mw: float,
    interconnector_mw: float,
    config: CommitmentConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Commit thermal units, balance energy, and enforce a simple non-sync cap.

    Wind and solar are treated as non-synchronous. Biomass/waste are left at their
    supplied dispatch and counted as synchronous for the diagnostic share. Positive
    HVDC import can optionally count as non-synchronous; negative HVDC flow is export
    and therefore increases the system basis.
    """
    out = assets.copy()
    out["dispatch_mw"] = _num(out, "dispatch_mw")
    if not bool(config.enabled):
        out["committed"] = np.where(out["market_bucket"].eq("thermal"), out["dispatch_mw"] > EPS, True)
        out["min_stable_mw"] = 0.0
        return out, {
            "enabled": False,
            "stability_curtailment_mw": 0.0,
            "minimum_output_curtailment_mw": 0.0,
            "balance_curtailment_mw": 0.0,
            "online_thermal_units": int((out["market_bucket"].eq("thermal") & (out["dispatch_mw"] > EPS)).sum()),
            "offline_thermal_units": int((out["market_bucket"].eq("thermal") & (out["dispatch_mw"] <= EPS)).sum()),
            "shutdown_units": [],
        }

    max_ns = float(np.clip(config.max_nonsynchronous_share_pct, 0.0, 100.0)) / 100.0
    min_stable_fraction = float(np.clip(config.min_stable_output_pct, 0.0, 100.0)) / 100.0

    # Standardised simplified SNSP-style basis for this prototype: demand plus HVDC
    # export in the denominator; positive HVDC import may count in the numerator.
    hvdc_import = max(0.0, float(interconnector_mw)) if config.count_positive_hvdc_import_as_nonsynchronous else 0.0
    hvdc_export = max(0.0, -float(interconnector_mw))
    nonsync_basis_mw = max(EPS, float(demand_mw) + hvdc_export)
    max_nonsync_mw = max_ns * nonsync_basis_mw

    renew_mask = out["market_bucket"].isin(["wind", "solar"])
    renewable_before = float(_num(out.loc[renew_mask], "dispatch_mw").clip(lower=0.0).sum())

    # 1) Stability/non-synchronous cap. Curtail only as much renewable as needed.
    max_renewable_from_share = max(0.0, max_nonsync_mw - hvdc_import)
    stability_cut_request = max(0.0, renewable_before - max_renewable_from_share)
    stability_cut = _renewable_reduce(out, stability_cut_request)

    # Other generation that is not dispatchable thermal in this commitment model.
    thermal_mask = out["market_bucket"].eq("thermal") & out["mapped_bus"].notna()
    other_mask = ~out["market_bucket"].eq("thermal") & ~out["market_bucket"].isin(["wind", "solar", "interconnector"])
    other_generation = float(_num(out.loc[other_mask], "dispatch_mw").sum())
    renewable_now = float(_num(out.loc[renew_mask], "dispatch_mw").sum())

    # Positive fixed injection (e.g. import) reduces local generation needed; negative
    # injection (export) increases it.
    local_asset_generation_required = float(demand_mw) - float(fixed_injection_mw)
    target_thermal = local_asset_generation_required - other_generation - renewable_now

    # If renewable + other generation already exceeds what the system can absorb even
    # with all thermal off, this is pure energy-balance curtailment.
    balance_cut = 0.0
    if target_thermal < -EPS:
        balance_cut = _renewable_reduce(out, -target_thermal)
        renewable_now = float(_num(out.loc[renew_mask], "dispatch_mw").sum())
        target_thermal = local_asset_generation_required - other_generation - renewable_now
    target_thermal = max(0.0, target_thermal)

    thermal = out.loc[thermal_mask].copy()
    original_thermal_dispatch = _num(thermal, "dispatch_mw").copy()
    chosen = _choose_committed_units(
        thermal,
        target_thermal,
        int(config.min_synchronous_units_online),
        min_stable_fraction,
        config.shutdown_strategy,
    )
    committed_dispatch, forced_minimum = _dispatch_committed_units(
        thermal,
        chosen,
        target_thermal,
        min_stable_fraction,
    )

    # If minimum stable outputs force more synchronous generation than requested,
    # curtail renewable power by the same amount to preserve energy balance.
    minimum_output_cut = 0.0
    if forced_minimum > EPS:
        minimum_output_cut = _renewable_reduce(out, forced_minimum)

    # After renewable curtailment, the required thermal total rises by exactly the
    # curtailed amount. Re-dispatch selected units to the updated target.
    renewable_now = float(_num(out.loc[renew_mask], "dispatch_mw").sum())
    updated_target_thermal = max(0.0, local_asset_generation_required - other_generation - renewable_now)
    chosen = _choose_committed_units(
        thermal,
        updated_target_thermal,
        int(config.min_synchronous_units_online),
        min_stable_fraction,
        config.shutdown_strategy,
    )
    committed_dispatch, _ = _dispatch_committed_units(
        thermal,
        chosen,
        updated_target_thermal,
        min_stable_fraction,
    )

    # Apply commitment fields to all assets.
    out["committed"] = True
    out["min_stable_mw"] = 0.0
    out["commitment_action"] = ""
    if len(thermal):
        out.loc[thermal.index, "dispatch_mw"] = committed_dispatch.reindex(thermal.index).fillna(0.0)
        caps = _capacity(thermal)
        out.loc[thermal.index, "min_stable_mw"] = caps * min_stable_fraction
        committed_set = set(chosen)
        out.loc[thermal.index, "committed"] = [idx in committed_set for idx in thermal.index]
        for idx in thermal.index:
            was_on = float(original_thermal_dispatch.loc[idx]) > EPS
            is_on = idx in committed_set
            if was_on and not is_on:
                out.loc[idx, "commitment_action"] = "shutdown"
            elif not was_on and is_on:
                out.loc[idx, "commitment_action"] = "startup"
            elif is_on:
                out.loc[idx, "commitment_action"] = "online"
            else:
                out.loc[idx, "commitment_action"] = "offline"

    thermal_generation = float(_num(out.loc[thermal_mask], "dispatch_mw").sum())
    biomass_waste_mask = out["market_bucket"].isin(["biomass", "waste"])
    other_sync_generation = float(_num(out.loc[biomass_waste_mask], "dispatch_mw").sum())
    renewable_accepted = float(_num(out.loc[renew_mask], "dispatch_mw").sum())
    nonsync_mw = renewable_accepted + hvdc_import
    nonsync_share_pct = 100.0 * nonsync_mw / nonsync_basis_mw if nonsync_basis_mw > EPS else 0.0
    synchronous_generation = thermal_generation + other_sync_generation

    shutdown_names = []
    if len(thermal):
        names = thermal.get("point_name", pd.Series(thermal.index.astype(str), index=thermal.index)).astype(str)
        shutdown_names = names[[idx not in set(chosen) for idx in thermal.index]].tolist()

    metrics: Dict[str, object] = {
        "enabled": True,
        "max_nonsynchronous_share_pct": 100.0 * max_ns,
        "nonsynchronous_basis_mw": nonsync_basis_mw,
        "max_nonsynchronous_mw": max_nonsync_mw,
        "nonsynchronous_mw": nonsync_mw,
        "nonsynchronous_share_pct": nonsync_share_pct,
        "synchronous_generation_mw": synchronous_generation,
        "thermal_generation_mw": thermal_generation,
        "online_thermal_units": int(len(chosen)),
        "offline_thermal_units": int(max(0, len(thermal) - len(chosen))),
        "min_synchronous_units_online": int(config.min_synchronous_units_online),
        "min_stable_output_pct": float(config.min_stable_output_pct),
        "stability_curtailment_mw": float(stability_cut),
        "minimum_output_curtailment_mw": float(minimum_output_cut),
        "balance_curtailment_mw": float(balance_cut),
        "shutdown_units": shutdown_names,
    }
    return out, metrics


def commitment_table(assets: pd.DataFrame) -> pd.DataFrame:
    """Human-readable thermal unit status table."""
    thermal = assets[assets["market_bucket"].eq("thermal")].copy()
    if thermal.empty:
        return pd.DataFrame()
    thermal["dispatch_mw"] = _num(thermal, "dispatch_mw")
    thermal["capacity_mw"] = _capacity(thermal)
    if "committed" not in thermal.columns:
        thermal["committed"] = thermal["dispatch_mw"] > EPS
    if "min_stable_mw" not in thermal.columns:
        thermal["min_stable_mw"] = 0.0
    if "commitment_action" not in thermal.columns:
        thermal["commitment_action"] = np.where(thermal["committed"], "online", "offline")
    thermal["status"] = np.where(thermal["committed"].astype(bool), "ONLINE", "OFFLINE")
    keep = [
        c for c in [
            "point_name", "mapped_bus", "status", "commitment_action", "dispatch_mw",
            "min_stable_mw", "capacity_mw", "mapping_quality",
        ] if c in thermal.columns
    ]
    return thermal[keep].sort_values(["status", "dispatch_mw"], ascending=[False, False]).reset_index(drop=True)
