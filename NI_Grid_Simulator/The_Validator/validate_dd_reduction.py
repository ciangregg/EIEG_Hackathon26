#!/usr/bin/env python3
"""Robust validation harness for the SV2024 26/32-county DD grouping project.

Purpose
-------
This program is deliberately a *stress test*, not just another optimiser.  It:

1. measures the published/exclusive WDT-derived starting grouping;
2. runs substantially larger simulated-annealing searches under N-1 security;
3. repeats optimisation with different renewable/weather and annealer seeds;
4. cross-validates each candidate on weather seeds it was not optimised on;
5. selects a robust candidate per scope;
6. tests that candidate against intact, N-1, tighter/looser thermal limits,
   low/high demand, wind-heavy, solar-heavy, and a stricter N-1 screen;
7. writes machine-readable CSVs and a short Markdown report.

The electrical model remains a DC thermal-security research model.  It does not
constitute an AC/dynamic-security or operator validation of the Irish system.

Run examples
------------
    python3 validate_dd_reduction.py --preset quick
    python3 validate_dd_reduction.py --preset standard
    python3 validate_dd_reduction.py --preset standard --scopes 32
    python3 validate_dd_reduction.py --preset standard --anneal-evals 5000
    python3 validate_dd_reduction.py --preset deep --scopes 26 32

Outputs are written below outputs/validation_<timestamp>/.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from all_island_annealer_api_validation import make_emulator


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
EPS = 1e-12


PRESETS = {
    "quick": {
        "weather_seeds": [42, 77],
        "anneal_seeds": [1],
        "anneal_evals": 120,
        "temperature_probes": 20,
        "runs": 2_000,
        "stress_level": "core",
    },
    "standard": {
        "weather_seeds": [42, 77, 123, 202, 314],
        "anneal_seeds": [1, 2, 3],
        "anneal_evals": 1_500,
        "temperature_probes": 40,
        "runs": 10_000,
        "stress_level": "full",
    },
    "deep": {
        "weather_seeds": [11, 22, 33, 44, 55, 66, 77, 88, 99, 111],
        "anneal_seeds": [1, 2, 3, 4, 5],
        "anneal_evals": 7_500,
        "temperature_probes": 80,
        "runs": 50_000,
        "stress_level": "full",
    },
}


@dataclass(frozen=True)
class Scenario:
    name: str
    include_n1: bool = True
    thermal_scale: float = 1.0
    demand_scale: float = 1.0
    wind_scale: float = 1.0
    solar_scale: float = 1.0
    security_threshold_pct: float = 90.0
    max_security_states: int = 800


CORE_SCENARIOS = [
    Scenario("intact_nominal", include_n1=False),
    Scenario("n1_nominal", include_n1=True),
    Scenario(
        "n1_strict_screen",
        include_n1=True,
        security_threshold_pct=70.0,
        max_security_states=3000,
    ),
]

FULL_SCENARIOS = CORE_SCENARIOS + [
    Scenario("n1_tight_limits", include_n1=True, thermal_scale=0.95),
    Scenario("n1_relaxed_limits", include_n1=True, thermal_scale=1.05),
    Scenario("n1_low_demand", include_n1=True, demand_scale=0.90),
    Scenario("n1_high_demand", include_n1=True, demand_scale=1.10),
    Scenario("n1_wind_heavy", include_n1=True, wind_scale=1.15),
    Scenario("n1_solar_heavy", include_n1=True, solar_scale=1.25),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stress-test DD reductions from 26/32-county constraint-group optimisation."
    )
    p.add_argument("--preset", choices=sorted(PRESETS), default="standard")
    p.add_argument("--scopes", nargs="+", choices=["26", "32"], default=["26", "32"])
    p.add_argument("--weather-seeds", nargs="+", type=int, default=None)
    p.add_argument("--anneal-seeds", nargs="+", type=int, default=None)
    p.add_argument("--anneal-evals", type=int, default=None,
                   help="Candidate evaluations per annealing run, excluding temperature probes.")
    p.add_argument("--temperature-probes", type=int, default=None)
    p.add_argument("--runs", type=int, default=None,
                   help="Snapshot resampling count inside make_emulator.")
    p.add_argument("--optimisation-security-threshold", type=float, default=90.0)
    p.add_argument("--optimisation-max-states", type=int, default=800)
    p.add_argument("--skip-stress", action="store_true",
                   help="Only anneal and cross-validate nominal N-1; skip the wider stress matrix.")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def one_group(value) -> int:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            value = value.strip()
    if isinstance(value, (tuple, list, set, np.ndarray)):
        value = list(value)
        if len(value) != 1:
            raise ValueError(f"Expected exclusive one-group assignment, got {value!r}")
        value = value[0]
    return int(value)


def grouping_from_nodes(nodes: pd.DataFrame) -> np.ndarray:
    column = "groups" if "groups" in nodes.columns else "group"
    return np.asarray([one_group(v) for v in nodes[column]], dtype=np.int64)


def nodes_with_grouping(nodes: pd.DataFrame, grouping: np.ndarray) -> pd.DataFrame:
    out = nodes.copy(deep=True)
    grouping = np.asarray(grouping, dtype=np.int64)
    out["group"] = grouping
    out["groups"] = [(int(g),) for g in grouping]
    return out


def propose_swap(grouping: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(grouping)
    for _ in range(1000):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i != j and grouping[i] != grouping[j]:
            out = grouping.copy()
            out[i], out[j] = out[j], out[i]
            return out
    raise RuntimeError("Could not find two nodes from different groups")


def prepare_energy(dispatch_down: Callable, nodes: pd.DataFrame) -> Callable[[np.ndarray], float]:
    prepare = getattr(dispatch_down, "prepare_grouping_evaluator", None)
    if callable(prepare):
        return prepare(nodes)

    def fallback(grouping: np.ndarray) -> float:
        return float(dispatch_down(nodes_with_grouping(nodes, grouping)))
    return fallback


def estimate_temperatures(
    initial_grouping: np.ndarray,
    energy: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    initial_energy: float,
    n_probes: int,
) -> tuple[float, float, int]:
    grouping = initial_grouping.copy()
    current = float(initial_energy)
    deltas: list[float] = []
    rejected = 0

    for _ in range(max(1, int(n_probes))):
        cand = propose_swap(grouping, rng)
        e = float(energy(cand))
        if not np.isfinite(e):
            rejected += 1
            continue
        d = abs(e - current)
        if d > EPS:
            deltas.append(d)
        grouping = cand
        current = e

    if not deltas:
        return 1e-3, 1e-6, rejected

    arr = np.asarray(deltas, dtype=float)
    typical = float(np.percentile(arr, 50))
    small = float(np.percentile(arr, 5))
    t_start = max(typical / -math.log(0.5), 1e-8)
    t_end = max(small / -math.log(1e-3), 1e-10)
    if t_end >= t_start:
        t_end = max(t_start * 1e-3, 1e-10)
    return t_start, t_end, rejected


def anneal(
    nodes: pd.DataFrame,
    dispatch_down: Callable,
    anneal_seed: int,
    eval_budget: int,
    temperature_probes: int,
    progress_every: int = 250,
) -> tuple[np.ndarray, float, dict, pd.DataFrame]:
    rng = np.random.default_rng(int(anneal_seed))
    initial = grouping_from_nodes(nodes)
    energy = prepare_energy(dispatch_down, nodes)
    e0 = float(energy(initial))
    if not np.isfinite(e0):
        raise RuntimeError("Baseline grouping unexpectedly failed its own security guard")

    t0, t1, rejected_probe = estimate_temperatures(
        initial, energy, rng, e0, temperature_probes
    )

    current_grouping = initial.copy()
    current_energy = e0
    best_grouping = initial.copy()
    best_energy = e0
    accepted = 0
    security_rejected = rejected_probe
    improvements = 0
    history = []

    n = max(1, int(eval_budget))
    for k in range(n):
        if n == 1:
            temp = t1
        else:
            frac = k / (n - 1)
            temp = t0 * ((t1 / t0) ** frac)

        cand = propose_swap(current_grouping, rng)
        e_new = float(energy(cand))
        if not np.isfinite(e_new):
            security_rejected += 1
        else:
            d_e = e_new - current_energy
            accept = d_e <= 0 or rng.random() < math.exp(-d_e / max(temp, 1e-15))
            if accept:
                current_grouping = cand
                current_energy = e_new
                accepted += 1
                if current_energy < best_energy - 1e-12:
                    best_grouping = current_grouping.copy()
                    best_energy = current_energy
                    improvements += 1

        if k == 0 or (k + 1) % max(1, progress_every) == 0 or k + 1 == n:
            history.append({
                "evaluation": k + 1,
                "temperature": temp,
                "current_dd_pct": current_energy,
                "best_dd_pct": best_energy,
                "accepted": accepted,
                "security_rejected": security_rejected,
                "improvements": improvements,
            })

    meta = {
        "initial_dd_pct": e0,
        "best_dd_pct": best_energy,
        "reduction_pp": e0 - best_energy,
        "relative_reduction_pct": 100.0 * (e0 - best_energy) / max(e0, EPS),
        "temperature_start": t0,
        "temperature_end": t1,
        "eval_budget": n,
        "temperature_probes": int(temperature_probes),
        "accepted": accepted,
        "security_rejected": security_rejected,
        "improvements": improvements,
    }
    return best_grouping, best_energy, meta, pd.DataFrame(history)


def build_emulator(
    scope: str,
    weather_seed: int,
    runs: int,
    scenario: Scenario,
    security_guard: bool = True,
):
    return make_emulator(
        scope=scope,
        runs=runs,
        seed=weather_seed,
        thermal_scale=scenario.thermal_scale,
        include_n1=scenario.include_n1,
        security_screen_threshold_pct=scenario.security_threshold_pct,
        max_security_states=scenario.max_security_states,
        security_guard=security_guard,
        demand_scale=scenario.demand_scale,
        wind_scale=scenario.wind_scale,
        solar_scale=scenario.solar_scale,
    )


def diagnostic_row(
    dispatch_down: Callable,
    nodes: pd.DataFrame,
    grouping: np.ndarray,
    scope: str,
    weather_seed: int,
    scenario: Scenario,
    candidate_id: str,
) -> dict:
    candidate_nodes = nodes_with_grouping(nodes, grouping)
    returned = float(dispatch_down(candidate_nodes))
    raw = float(getattr(dispatch_down, "last_raw_dispatch_down_pct", returned))
    return {
        "scope": scope,
        "weather_seed": int(weather_seed),
        "scenario": scenario.name,
        "candidate_id": candidate_id,
        "returned_dd_pct": returned,
        "raw_dd_pct": raw,
        "security_pass_pct": float(getattr(dispatch_down, "last_security_pass_pct", np.nan)),
        "new_security_failures": int(getattr(dispatch_down, "last_new_security_failures", 0)),
        "worst_screened_loading_pct": float(getattr(dispatch_down, "last_worst_screened_loading_pct", np.nan)),
        "network_dd_mw_weighted": float(getattr(dispatch_down, "last_network_dispatch_down_mw_weighted", np.nan)),
        "security_state_count": int(getattr(dispatch_down, "security_state_count", 0)),
        "shortage_mw_weighted": float(getattr(dispatch_down, "shortage_mw_weighted", np.nan)),
        "safe_under_guard": bool(np.isfinite(returned)),
    }


def enrich_reduction(candidate: dict, baseline: dict) -> dict:
    row = dict(candidate)
    row["baseline_dd_pct"] = baseline["raw_dd_pct"]
    row["reduction_pp"] = baseline["raw_dd_pct"] - candidate["raw_dd_pct"]
    row["relative_reduction_pct"] = (
        100.0 * row["reduction_pp"] / max(baseline["raw_dd_pct"], EPS)
    )
    row["baseline_security_pass_pct"] = baseline["security_pass_pct"]
    row["security_pass_change_pp"] = candidate["security_pass_pct"] - baseline["security_pass_pct"]
    row["baseline_worst_screened_loading_pct"] = baseline["worst_screened_loading_pct"]
    row["worst_loading_change_pp"] = candidate["worst_screened_loading_pct"] - baseline["worst_screened_loading_pct"]
    row["absolute_model_secure"] = bool(candidate["security_pass_pct"] >= 100.0 - 1e-9)
    return row


def scenario_dict(s: Scenario) -> dict:
    d = asdict(s)
    d["scenario"] = d.pop("name")
    return d


def write_markdown_report(
    out_dir: Path,
    preset: str,
    run_summary: pd.DataFrame,
    stress_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    config: dict,
) -> None:
    lines = [
        "# Dispatch-down reduction validation report",
        "",
        f"Preset: **{preset}**",
        "",
        "This report compares the published/WDT-derived **exclusive baseline used by the research emulator** with annealed groupings. It is not a claim about the exact historical overlapping WDT implementation.",
        "",
        "## Robust candidate selection",
        "",
    ]

    if selection_df.empty:
        lines.append("No robust candidate selection was completed.")
    else:
        for r in selection_df.itertuples(index=False):
            lines += [
                f"### {r.scope}-county",
                "",
                f"Selected `{r.candidate_id}`. Cross-seed mean reduction: **{r.mean_reduction_pp:.3f} percentage points** ({r.mean_relative_reduction_pct:.1f}% relative). Safe on **{r.safe_cases}/{r.total_cases}** nominal N-1 seed checks.",
                "",
            ]

    lines += ["## Stress-test summary", ""]
    if not run_summary.empty:
        for r in run_summary.itertuples(index=False):
            lines += [
                f"### {r.scope}-county",
                "",
                f"Across {r.test_cases} stress cases, mean DD reduction was **{r.mean_reduction_pp:.3f} pp**, median **{r.median_reduction_pp:.3f} pp**, and worst observed reduction **{r.worst_reduction_pp:.3f} pp**. The candidate improved DD in **{r.improved_fraction_pct:.1f}%** of cases and introduced no new baseline-relative security failures in **{r.safe_fraction_pct:.1f}%** of cases. Absolute 100% security in the modelled screened states occurred in **{r.absolute_model_secure_fraction_pct:.1f}%** of cases; the minimum candidate security-pass rate was **{r.candidate_min_security_pass_pct:.2f}%**.",
                "",
            ]

    lines += [
        "## Interpretation rules",
        "",
        "A promising result is one where the reduction remains positive across different weather seeds and stress scenarios, particularly `n1_strict_screen`, with zero new security failures. A large training-seed reduction that disappears or becomes unsafe on held-out seeds should be treated as overfitting rather than a grid improvement.",
        "",
        "The model is DC/thermal. Passing these tests is necessary evidence for the research hypothesis, but not sufficient for real-world deployment; AC voltage/reactive, dynamic stability, reserve/inertia and operator studies remain outside this harness.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
    ]
    (out_dir / "validation_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    preset = dict(PRESETS[args.preset])
    weather_seeds = args.weather_seeds or preset["weather_seeds"]
    anneal_seeds = args.anneal_seeds or preset["anneal_seeds"]
    anneal_evals = args.anneal_evals or preset["anneal_evals"]
    probes = args.temperature_probes or preset["temperature_probes"]
    runs = args.runs or preset["runs"]

    scenarios = CORE_SCENARIOS if preset["stress_level"] == "core" else FULL_SCENARIOS
    if args.skip_stress:
        scenarios = [Scenario("n1_nominal")]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (OUTPUT_ROOT / f"validation_{timestamp}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = out_dir / "candidates"
    history_dir = out_dir / "histories"
    candidate_dir.mkdir(exist_ok=True)
    history_dir.mkdir(exist_ok=True)

    config = {
        "preset": args.preset,
        "scopes": args.scopes,
        "weather_seeds": weather_seeds,
        "anneal_seeds": anneal_seeds,
        "anneal_evals": anneal_evals,
        "temperature_probes": probes,
        "runs": runs,
        "optimisation_security_threshold_pct": args.optimisation_security_threshold,
        "optimisation_max_states": args.optimisation_max_states,
        "scenarios": [scenario_dict(s) for s in scenarios],
    }
    (out_dir / "configuration.json").write_text(json.dumps(config, indent=2))

    print("=" * 78)
    print("ROBUST DISPATCH-DOWN REDUCTION VALIDATION")
    print("=" * 78)
    print(f"Output:          {out_dir}")
    print(f"Scopes:          {args.scopes}")
    print(f"Weather seeds:   {weather_seeds}")
    print(f"Annealer seeds:  {anneal_seeds}")
    print(f"Anneal evals:    {anneal_evals:,} per run")
    print(f"N-1 optimisation: ON")
    print()

    optimisation_scenario = Scenario(
        "n1_optimisation",
        include_n1=True,
        security_threshold_pct=float(args.optimisation_security_threshold),
        max_security_states=int(args.optimisation_max_states),
    )

    anneal_rows: list[dict] = []
    candidates: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # PHASE 1: Large, repeated N-1 annealing runs.
    # ------------------------------------------------------------------
    for scope in args.scopes:
        print(f"\n{'='*30} {scope}-COUNTY ANNEALING {'='*30}")
        for weather_seed in weather_seeds:
            print(f"\nBuilding N-1 emulator: scope={scope}, weather_seed={weather_seed}")
            build_t0 = time.perf_counter()
            dispatch_down, nodes = build_emulator(
                scope, weather_seed, runs, optimisation_scenario, security_guard=True
            )
            build_s = time.perf_counter() - build_t0
            baseline_grouping = grouping_from_nodes(nodes)
            baseline_row = diagnostic_row(
                dispatch_down, nodes, baseline_grouping, scope, weather_seed,
                optimisation_scenario, "baseline"
            )
            print(
                f"  baseline DD={baseline_row['raw_dd_pct']:.4f}% | "
                f"security={baseline_row['security_pass_pct']:.2f}% | "
                f"states={baseline_row['security_state_count']} | build={build_s:.2f}s"
            )

            for anneal_seed in anneal_seeds:
                candidate_id = f"{scope}_w{weather_seed}_a{anneal_seed}"
                print(f"  Annealing {candidate_id} ({anneal_evals:,} evals)...")
                t0 = time.perf_counter()
                best_grouping, best_dd, meta, hist = anneal(
                    nodes=nodes,
                    dispatch_down=dispatch_down,
                    anneal_seed=anneal_seed,
                    eval_budget=anneal_evals,
                    temperature_probes=probes,
                    progress_every=max(100, anneal_evals // 5),
                )
                elapsed = time.perf_counter() - t0
                final_diag = diagnostic_row(
                    dispatch_down, nodes, best_grouping, scope, weather_seed,
                    optimisation_scenario, candidate_id
                )
                reduction_pp = baseline_row["raw_dd_pct"] - final_diag["raw_dd_pct"]
                rel = 100.0 * reduction_pp / max(baseline_row["raw_dd_pct"], EPS)

                row = {
                    "candidate_id": candidate_id,
                    "scope": scope,
                    "training_weather_seed": weather_seed,
                    "anneal_seed": anneal_seed,
                    "baseline_dd_pct": baseline_row["raw_dd_pct"],
                    "best_dd_pct": final_diag["raw_dd_pct"],
                    "reduction_pp": reduction_pp,
                    "relative_reduction_pct": rel,
                    "security_pass_pct": final_diag["security_pass_pct"],
                    "new_security_failures": final_diag["new_security_failures"],
                    "worst_screened_loading_pct": final_diag["worst_screened_loading_pct"],
                    "safe_under_guard": final_diag["safe_under_guard"],
                    "security_states": final_diag["security_state_count"],
                    "anneal_seconds": elapsed,
                    **{f"anneal_{k}": v for k, v in meta.items()},
                }
                anneal_rows.append(row)
                candidates[candidate_id] = {
                    "scope": scope,
                    "training_weather_seed": weather_seed,
                    "anneal_seed": anneal_seed,
                    "grouping": best_grouping.copy(),
                    "nodes_template": nodes.copy(deep=True),
                }

                nodes_with_grouping(nodes, best_grouping).to_csv(
                    candidate_dir / f"candidate_{candidate_id}.csv", index=False
                )
                hist.assign(
                    scope=scope,
                    training_weather_seed=weather_seed,
                    anneal_seed=anneal_seed,
                    candidate_id=candidate_id,
                ).to_csv(history_dir / f"history_{candidate_id}.csv", index=False)
                print(
                    f"    best={final_diag['raw_dd_pct']:.4f}% | "
                    f"reduction={reduction_pp:.4f} pp ({rel:.1f}%) | "
                    f"new failures={final_diag['new_security_failures']} | {elapsed:.1f}s"
                )

    anneal_df = pd.DataFrame(anneal_rows)
    anneal_df.to_csv(out_dir / "anneal_runs.csv", index=False)

    # ------------------------------------------------------------------
    # PHASE 2: Cross-seed validation, no re-optimisation.
    # This is the key overfitting test.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("CROSS-SEED N-1 VALIDATION (NO RE-OPTIMISATION)")
    print("=" * 78)
    cross_rows: list[dict] = []
    selection_rows: list[dict] = []
    selected_candidates: dict[str, str] = {}

    nominal_validation = Scenario(
        "n1_nominal",
        include_n1=True,
        security_threshold_pct=float(args.optimisation_security_threshold),
        max_security_states=int(args.optimisation_max_states),
    )

    for scope in args.scopes:
        scope_ids = [cid for cid, c in candidates.items() if c["scope"] == scope]
        for weather_seed in weather_seeds:
            dispatch_down, nodes = build_emulator(
                scope, weather_seed, runs, nominal_validation, security_guard=True
            )
            baseline_grouping = grouping_from_nodes(nodes)
            baseline = diagnostic_row(
                dispatch_down, nodes, baseline_grouping, scope, weather_seed,
                nominal_validation, "baseline"
            )
            for cid in scope_ids:
                cand = candidates[cid]
                diag = diagnostic_row(
                    dispatch_down, nodes, cand["grouping"], scope, weather_seed,
                    nominal_validation, cid
                )
                row = enrich_reduction(diag, baseline)
                row["training_weather_seed"] = cand["training_weather_seed"]
                row["anneal_seed"] = cand["anneal_seed"]
                row["held_out_weather"] = bool(weather_seed != cand["training_weather_seed"])
                cross_rows.append(row)

        scope_cross = pd.DataFrame([r for r in cross_rows if r["scope"] == scope])
        ranked = []
        for cid, g in scope_cross.groupby("candidate_id", sort=False):
            safe_cases = int(g["safe_under_guard"].sum())
            total_cases = int(len(g))
            unsafe_cases = total_cases - safe_cases
            ranked.append({
                "candidate_id": cid,
                "scope": scope,
                "safe_cases": safe_cases,
                "total_cases": total_cases,
                "unsafe_cases": unsafe_cases,
                "mean_raw_dd_pct": float(g["raw_dd_pct"].mean()),
                "mean_reduction_pp": float(g["reduction_pp"].mean()),
                "mean_relative_reduction_pct": float(g["relative_reduction_pct"].mean()),
                "worst_reduction_pp": float(g["reduction_pp"].min()),
                "total_new_security_failures": int(g["new_security_failures"].sum()),
            })
        rank_df = pd.DataFrame(ranked).sort_values(
            ["unsafe_cases", "mean_raw_dd_pct", "worst_reduction_pp"],
            ascending=[True, True, False],
            kind="stable",
        )
        best = rank_df.iloc[0].to_dict()
        selected_candidates[scope] = str(best["candidate_id"])
        selection_rows.append(best)
        print(
            f"{scope}-county selected {best['candidate_id']}: "
            f"mean reduction={best['mean_reduction_pp']:.4f} pp, "
            f"safe={int(best['safe_cases'])}/{int(best['total_cases'])}, "
            f"worst reduction={best['worst_reduction_pp']:.4f} pp"
        )

        selected = candidates[str(best["candidate_id"])]
        nodes_with_grouping(selected["nodes_template"], selected["grouping"]).to_csv(
            out_dir / f"robust_candidate_{scope}_counties.csv", index=False
        )

    cross_df = pd.DataFrame(cross_rows)
    selection_df = pd.DataFrame(selection_rows)
    cross_df.to_csv(out_dir / "cross_seed_validation.csv", index=False)
    selection_df.to_csv(out_dir / "candidate_selection.csv", index=False)

    # ------------------------------------------------------------------
    # PHASE 3: Stress selected grouping.  No re-optimisation.
    # ------------------------------------------------------------------
    stress_rows: list[dict] = []
    if not args.skip_stress:
        print("\n" + "=" * 78)
        print("STRESS VALIDATION OF SELECTED GROUPING")
        print("=" * 78)

        for scope in args.scopes:
            cid = selected_candidates[scope]
            grouping = candidates[cid]["grouping"]
            for scenario in scenarios:
                print(f"{scope}-county | {scenario.name}")
                for weather_seed in weather_seeds:
                    dispatch_down, nodes = build_emulator(
                        scope, weather_seed, runs, scenario, security_guard=True
                    )
                    baseline_grouping = grouping_from_nodes(nodes)
                    baseline = diagnostic_row(
                        dispatch_down, nodes, baseline_grouping, scope, weather_seed,
                        scenario, "baseline"
                    )
                    candidate = diagnostic_row(
                        dispatch_down, nodes, grouping, scope, weather_seed,
                        scenario, cid
                    )
                    stress_rows.append(enrich_reduction(candidate, baseline))

    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(out_dir / "stress_validation.csv", index=False)

    # ------------------------------------------------------------------
    # PHASE 4: Summary / pass-fail style indicators.
    # ------------------------------------------------------------------
    summary_rows = []
    source_for_summary = stress_df if not stress_df.empty else cross_df
    for scope in args.scopes:
        g = source_for_summary[source_for_summary["scope"] == scope]
        if g.empty:
            continue
        summary_rows.append({
            "scope": scope,
            "test_cases": int(len(g)),
            "mean_reduction_pp": float(g["reduction_pp"].mean()),
            "median_reduction_pp": float(g["reduction_pp"].median()),
            "worst_reduction_pp": float(g["reduction_pp"].min()),
            "best_reduction_pp": float(g["reduction_pp"].max()),
            "mean_relative_reduction_pct": float(g["relative_reduction_pct"].mean()),
            "improved_fraction_pct": float(100.0 * (g["reduction_pp"] > 0).mean()),
            "safe_fraction_pct": float(100.0 * g["safe_under_guard"].mean()),
            "absolute_model_secure_fraction_pct": float(100.0 * g["absolute_model_secure"].mean()),
            "candidate_min_security_pass_pct": float(g["security_pass_pct"].min()),
            "baseline_min_security_pass_pct": float(g["baseline_security_pass_pct"].min()),
            "total_new_security_failures": int(g["new_security_failures"].sum()),
            "strict_n1_safe": bool(
                g.loc[g["scenario"] == "n1_strict_screen", "safe_under_guard"].all()
                if (g["scenario"] == "n1_strict_screen").any() else True
            ),
            "strict_n1_min_reduction_pp": float(
                g.loc[g["scenario"] == "n1_strict_screen", "reduction_pp"].min()
                if (g["scenario"] == "n1_strict_screen").any() else np.nan
            ),
            "strict_n1_min_security_pass_pct": float(
                g.loc[g["scenario"] == "n1_strict_screen", "security_pass_pct"].min()
                if (g["scenario"] == "n1_strict_screen").any() else np.nan
            ),
            "strict_n1_absolute_model_secure": bool(
                g.loc[g["scenario"] == "n1_strict_screen", "absolute_model_secure"].all()
                if (g["scenario"] == "n1_strict_screen").any() else False
            ),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "scope_summary.csv", index=False)

    write_markdown_report(
        out_dir=out_dir,
        preset=args.preset,
        run_summary=summary_df,
        stress_df=stress_df,
        selection_df=selection_df,
        config=config,
    )

    print("\n" + "=" * 78)
    print("VALIDATION COMPLETE")
    print("=" * 78)
    if not summary_df.empty:
        cols = [
            "scope", "test_cases", "mean_reduction_pp", "worst_reduction_pp",
            "improved_fraction_pct", "safe_fraction_pct",
            "absolute_model_secure_fraction_pct", "strict_n1_safe",
        ]
        print(summary_df[cols].to_string(index=False))
    print(f"\nResults saved to: {out_dir}")
    print("Read validation_report.md first, then scope_summary.csv and stress_validation.csv.")


if __name__ == "__main__":
    main()
