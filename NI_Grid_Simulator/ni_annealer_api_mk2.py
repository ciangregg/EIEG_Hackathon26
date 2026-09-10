"""Minimal annealer-facing API for the calibrated Northern Ireland grid emulator.

Usage
-----
    from ni_annealer_api import make_emulator

    dispatch_down, nodes = make_emulator(runs=10_000, seed=42)

    # `nodes` is the annealer input. Edit only the `groups` column.
    score_pct = dispatch_down(nodes)

The operating scenarios, topology, operational rules and calibration residual are
frozen when ``make_emulator`` is called. Repeated calls therefore compare group
assignments on exactly the same Northern Ireland operating cases.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Any
import ast

import numpy as np
import pandas as pd

from ni_grid_core import DCGridModel, map_assets_to_buses, read_network_nc, read_spatial_assets
from ni_operational_model import current_group_masks, evaluate_policy
from ni_baseline_replica import current_operational_config, run_baseline_replica


ROOT = Path(__file__).resolve().parent
DEFAULT_NETWORK_FILE = ROOT / "data" / "SV2024_northern_ireland.nc"
DEFAULT_ASSET_FILE = ROOT / "data" / "northern_ireland_spatial_supply_and_interconnector_points.csv"
_GROUP_IDS = (1, 2, 3, 4, 5)


def _normalise_groups(value: Any) -> tuple[int, ...]:
    """Convert common annealer representations into a sorted tuple of group IDs."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return tuple()
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return tuple()
        # Accept "1,4,5", "[1, 4, 5]", "(1,4,5)" and "1 4 5".
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (int, np.integer)):
                value = [int(parsed)]
            elif isinstance(parsed, (list, tuple, set)):
                value = parsed
            else:
                value = [int(x) for x in s.replace(";", ",").replace(" ", ",").split(",") if x]
        except Exception:
            value = [int(x) for x in s.replace(";", ",").replace(" ", ",").split(",") if x]
    elif isinstance(value, (int, np.integer)):
        value = [int(value)]
    elif not isinstance(value, (list, tuple, set, np.ndarray)):
        raise TypeError(f"Unsupported groups value: {value!r}")

    groups = tuple(sorted({int(g) for g in value}))
    invalid = [g for g in groups if g not in _GROUP_IDS]
    if invalid:
        raise ValueError(f"Constraint groups must be in 1..5; got {invalid}")
    return groups


def _build_node_table(renewables: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical annealer input table, one row per renewable node."""
    masks = current_group_masks(renewables)
    rows = []
    for i, a in renewables.reset_index(drop=True).iterrows():
        groups = tuple(g for g in _GROUP_IDS if bool(masks[g][i]))
        rows.append({
            "node_id": int(i),
            "name": str(a.get("point_name", i)),
            "bus": str(a.get("mapped_bus", "")),
            "groups": groups,
            "mec_mw": float(a.get("max_export_capacity_mw", 0.0)),
            "technology": str(a.get("market_bucket", "renewable")),
            "latitude": float(a.get("latitude", np.nan)),
            "longitude": float(a.get("longitude", np.nan)),
        })
    return pd.DataFrame(rows)


def _validated_node_ids(nodes: pd.DataFrame, n_nodes: int) -> np.ndarray:
    """Validate the annealer node table once and return row -> node_id mapping."""
    if not isinstance(nodes, pd.DataFrame):
        nodes = pd.DataFrame(nodes)

    required = {"node_id", "groups"}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"nodes must contain columns {sorted(required)}; missing {sorted(missing)}")

    ids = pd.to_numeric(nodes["node_id"], errors="raise").astype(int)
    if ids.duplicated().any():
        dup = ids[ids.duplicated()].tolist()
        raise ValueError(f"Duplicate node_id values: {dup[:10]}")

    ids_array = ids.to_numpy(dtype=np.int64, copy=False)
    if len(ids_array) != n_nodes:
        got = set(ids_array.tolist())
        expected = set(range(n_nodes))
        missing_ids = sorted(expected - got)
        extra_ids = sorted(got - expected)
        raise ValueError(
            f"nodes must contain every renewable node exactly once. "
            f"Missing={missing_ids[:10]}, extra={extra_ids[:10]}"
        )

    # For exactly n_nodes unique integer IDs, bounds 0..n_nodes-1 are equivalent
    # to the original set equality check but avoid constructing two Python sets.
    if ids_array.size and (ids_array.min() < 0 or ids_array.max() >= n_nodes):
        got = set(ids_array.tolist())
        expected = set(range(n_nodes))
        missing_ids = sorted(expected - got)
        extra_ids = sorted(got - expected)
        raise ValueError(
            f"nodes must contain every renewable node exactly once. "
            f"Missing={missing_ids[:10]}, extra={extra_ids[:10]}"
        )

    return ids_array


def _masks_from_exclusive_grouping(
    grouping: np.ndarray,
    node_ids: np.ndarray,
    n_nodes: int,
) -> dict[int, np.ndarray]:
    """Fast path for the annealer's one-integer-group-per-row representation."""
    groups = np.asarray(grouping)
    if groups.ndim != 1 or len(groups) != n_nodes:
        raise ValueError(f"grouping must be one-dimensional with length {n_nodes}")

    # This mirrors grouping_to_nodes(), which applies int(group) before passing
    # the candidate to dispatch_down().
    groups = groups.astype(np.int64, copy=False)
    invalid = groups[(groups < 1) | (groups > 5)]
    if invalid.size:
        raise ValueError(f"Constraint groups must be in 1..5; got {np.unique(invalid).tolist()}")

    masks = {g: np.zeros(n_nodes, dtype=bool) for g in _GROUP_IDS}
    for g in _GROUP_IDS:
        masks[g][node_ids] = groups == g
    return masks


def _masks_from_nodes(nodes: pd.DataFrame, n_nodes: int) -> dict[int, np.ndarray]:
    """Convert the node/group assignment table into emulator group masks."""
    if not isinstance(nodes, pd.DataFrame):
        nodes = pd.DataFrame(nodes)

    ids = _validated_node_ids(nodes, n_nodes)
    raw_groups = nodes["groups"].to_numpy(dtype=object, copy=False)

    # Common annealer fast path: every row is exactly one integer tuple, e.g. (4,).
    # It preserves arbitrary row order by assigning through node_id.
    exclusive = np.empty(n_nodes, dtype=np.int64)
    fast_path = True
    for i, value in enumerate(raw_groups):
        if (
            isinstance(value, tuple)
            and len(value) == 1
            and isinstance(value[0], (int, np.integer))
        ):
            exclusive[i] = int(value[0])
        else:
            fast_path = False
            break

    if fast_path:
        return _masks_from_exclusive_grouping(exclusive, ids, n_nodes)

    # Generic API path: preserve all original accepted group representations,
    # including multi-group membership and strings.
    masks = {g: np.zeros(n_nodes, dtype=bool) for g in _GROUP_IDS}
    for node_id, groups in zip(ids, raw_groups):
        for g in _normalise_groups(groups):
            masks[g][int(node_id)] = True
    return masks



def make_emulator(
    runs: int = 10_000,
    seed: int = 42,
    target_dispatch_down_pct: float = 25.5,
    demand_scale: float = 2.10,
    thermal_scale: float = 1.0,
    planned_outage_exposure_pct: float = 5.0,
    network_file: str | Path = DEFAULT_NETWORK_FILE,
    asset_file: str | Path = DEFAULT_ASSET_FILE,
) -> tuple[Callable[[pd.DataFrame], float], pd.DataFrame]:
    """Freeze the NI emulator and return ``(dispatch_down, nodes)``.

    Parameters are used once, when the emulator is built.  The returned
    ``dispatch_down(nodes)`` function then evaluates only the supplied group
    membership table on those same frozen operating cases.

    Returns
    -------
    dispatch_down:
        Function taking the node table and returning calibrated renewable
        dispatch-down as a percentage of available renewable generation.
    nodes:
        DataFrame with one row per renewable node.  The annealer should change
        only ``groups``; all other columns are identifiers/metadata.

    Notes
    -----
    The calibrated residual is frozen from the current-group baseline.  Therefore
    later changes in the score are caused by the represented group/network policy,
    not by re-fitting the historical benchmark for every candidate.
    """
    network = read_network_nc(Path(network_file))
    mapped_assets = map_assets_to_buses(read_spatial_assets(Path(asset_file)), network.buses)
    model = DCGridModel(network)
    cfg = current_operational_config(
        runs=int(runs),
        seed=int(seed),
        demand_scale=float(demand_scale),
        thermal_scale=float(thermal_scale),
        planned_outage_exposure_pct=float(planned_outage_exposure_pct),
    )

    baseline = run_baseline_replica(
        network,
        model,
        mapped_assets,
        cfg,
        target_dispatch_down_pct=float(target_dispatch_down_pct),
        calibrate=True,
    )

    scenarios = baseline.scenarios
    node_template = _build_node_table(scenarios.renewable_assets)
    n_nodes = len(node_template)
    fixed_residual_pct = float(baseline.metrics["calibration_residual_pct"])

    # Baseline-secure cases are frozen too. This is exposed on the callable for
    # downstream code that wants to inspect validation, but it does not alter the
    # scalar return contract requested for an annealer objective.
    baseline_repr_secure = baseline.physical.run_table[
        "represented_security_pass"
    ].to_numpy(dtype=bool, copy=False)
    baseline_n1_secure = baseline.physical.run_table[
        "all_line_n1_pass"
    ].to_numpy(dtype=bool, copy=False)


    def _evaluate_masks(masks: dict[int, np.ndarray]) -> float:
        """Evaluate masks and update the same diagnostics as dispatch_down()."""
        result = evaluate_policy(network, model, scenarios, cfg, masks)

        # The calibration residual belongs to the frozen environment, not to the
        # candidate group policy, so it is added unchanged to every candidate.
        score = float(result.metrics["dispatch_down_pct"]) + fixed_residual_pct

        # Attach the most recent diagnostics for optional annealer validation,
        # while keeping the function return value as one scalar dispatch-down %.
        candidate_repr = result.run_table[
            "represented_security_pass"
        ].to_numpy(dtype=bool, copy=False)
        candidate_n1 = result.run_table[
            "all_line_n1_pass"
        ].to_numpy(dtype=bool, copy=False)
        dispatch_down.last_physical_dispatch_down_pct = float(result.metrics["dispatch_down_pct"])
        dispatch_down.last_calibrated_dispatch_down_pct = score
        dispatch_down.last_represented_security_pass_pct = float(
            result.metrics["represented_security_pass_pct"]
        )
        dispatch_down.last_all_line_n1_pass_pct = float(result.metrics["all_line_n1_pass_pct"])
        dispatch_down.last_new_represented_security_failures = int(
            np.count_nonzero(baseline_repr_secure & ~candidate_repr)
        )
        dispatch_down.last_new_n1_security_failures = int(
            np.count_nonzero(baseline_n1_secure & ~candidate_n1)
        )
        dispatch_down.last_result = result
        return score

    def dispatch_down(nodes: pd.DataFrame) -> float:
        """Return calibrated NI renewable dispatch-down (%) for a node/group assignment."""
        masks = _masks_from_nodes(nodes, n_nodes)
        return _evaluate_masks(masks)

    def prepare_grouping_evaluator(nodes: pd.DataFrame) -> Callable[[np.ndarray], float]:
        """Prepare a fast integer-group evaluator for this fixed annealer node table.

        Validation and row-to-node mapping are done once.  Each subsequent call
        converts the integer grouping directly to policy masks, avoiding a
        DataFrame copy and tuple parsing for every proposal.
        """
        if not isinstance(nodes, pd.DataFrame):
            nodes = pd.DataFrame(nodes)
        node_ids = _validated_node_ids(nodes, n_nodes).copy()

        def evaluate_grouping(grouping: np.ndarray) -> float:
            masks = _masks_from_exclusive_grouping(grouping, node_ids, n_nodes)
            return _evaluate_masks(masks)

        return evaluate_grouping

    # Useful immutable metadata without changing the simple callable API.
    dispatch_down.baseline_dispatch_down_pct = float(baseline.metrics["calibrated_dispatch_down_pct"])
    dispatch_down.baseline_physical_dispatch_down_pct = float(baseline.metrics["physical_dispatch_down_pct"])
    dispatch_down.fixed_calibration_residual_pct = fixed_residual_pct
    dispatch_down.runs = int(runs)
    dispatch_down.seed = int(seed)
    dispatch_down.baseline = baseline

    # Optional fast path used by the tandem annealer. Existing callers can ignore it.
    dispatch_down.prepare_grouping_evaluator = prepare_grouping_evaluator

    return dispatch_down, node_template.copy(deep=True)


if __name__ == "__main__":
    dispatch_down, nodes = make_emulator(runs=1_000, seed=42)
    print(nodes[["node_id", "name", "bus", "groups"]].head(10).to_string(index=False))
    print(f"Dispatch-down: {dispatch_down(nodes):.3f}%")
