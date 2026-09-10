SV2024 DISPATCH-DOWN REDUCTION VALIDATION SUITE
===============================================

WHAT THIS IS
------------
A stress-test harness for the 26-county and 32-county renewable constraint-group
project. It is designed to answer a harder question than "can the annealer find
a lower score?":

    Does the dispatch-down reduction survive N-1 contingency testing,
    different renewable/weather seeds, larger annealing searches, and
    reasonable operating stresses without introducing new modeled failures?

MAIN COMMANDS
-------------
From this folder:

    python3 validate_dd_reduction.py --preset quick
    python3 validate_dd_reduction.py --preset standard
    python3 validate_dd_reduction.py --preset deep

Only one scope:

    python3 validate_dd_reduction.py --preset standard --scopes 32
    python3 validate_dd_reduction.py --preset standard --scopes 26

Larger anneal than the preset:

    python3 validate_dd_reduction.py --preset standard --anneal-evals 5000

Custom seeds:

    python3 validate_dd_reduction.py --preset standard \
        --weather-seeds 42 77 123 202 314 \
        --anneal-seeds 1 2 3 4 5 \
        --anneal-evals 5000

PRESETS
-------
quick:
  2 weather seeds, 1 annealer seed, 120 candidate evaluations per anneal,
  2,000 snapshot-resampling draws, core stress scenarios only.
  Intended only to check that the code runs.

standard:
  5 weather seeds, 3 annealer seeds, 1,500 candidate evaluations per anneal,
  10,000 snapshot-resampling draws, full stress matrix.
  This is the recommended first serious run.

deep:
  10 weather seeds, 5 annealer seeds, 7,500 candidate evaluations per anneal,
  50,000 snapshot-resampling draws, full stress matrix.
  This is intentionally computationally heavy.

WHAT THE PROGRAM DOES
---------------------
PHASE 1 - N-1 optimisation
  For every requested 26/32 scope and weather seed, build an SV2024 emulator
  with N-1 enabled. Run independent simulated anneals using multiple annealer
  seeds. Moves preserve group sizes by swapping two nodes from different groups.

PHASE 2 - Cross-seed validation
  Each candidate is tested on every weather seed WITHOUT re-optimising it.
  This is the main test for overfitting to one synthetic renewable year.
  A robust candidate is chosen by:
      1) fewest unsafe seed tests,
      2) lowest mean DD,
      3) best worst-case DD reduction.

PHASE 3 - Stress validation
  The selected grouping is compared against the exact same published/exclusive
  starting grouping under:
      - intact grid, nominal limits
      - N-1, nominal limits
      - stricter N-1 security-state screening
      - 5% tighter thermal limits
      - 5% relaxed thermal limits
      - 10% lower demand
      - 10% higher demand
      - wind-heavy renewable conditions
      - solar-heavy renewable conditions

PHASE 4 - Reporting
  Results are written to outputs/validation_<timestamp>/:
      anneal_runs.csv
      cross_seed_validation.csv
      candidate_selection.csv
      stress_validation.csv
      scope_summary.csv
      robust_candidate_26_counties.csv / robust_candidate_32_counties.csv
      candidates/
      histories/
      validation_report.md
      configuration.json

HOW TO READ THE RESULT
----------------------
The strongest evidence is NOT the single best training run. Look at:

  scope_summary.csv
    mean_reduction_pp
    worst_reduction_pp
    improved_fraction_pct
    safe_fraction_pct
    absolute_model_secure_fraction_pct
    strict_n1_safe
    strict_n1_min_reduction_pp
    strict_n1_min_security_pass_pct

  stress_validation.csv
    Every baseline/candidate comparison by scenario and weather seed.

A result is much more convincing if:
  - reduction_pp stays positive on held-out weather seeds;
  - reduction_pp stays positive in n1_strict_screen;
  - new_security_failures remains 0;
  - candidate security pass does not deteriorate;
  - the effect is reproduced by multiple independent annealer seeds;
  - the magnitude is not dominated by one weather seed.

SECURITY WARNING
----------------
"safe_under_guard" means the grouping introduced no NEW modeled security
failure in a snapshot that was secure under the published/exclusive baseline.
It does NOT mean the baseline or candidate is absolutely secure.

The report therefore separately records:
  security_pass_pct
  absolute_model_secure
  absolute_model_secure_fraction_pct

If the baseline model itself has a low security-pass percentage, a candidate
can be "safe under guard" while the simplified model still contains existing
insecure cases. Do not describe such a case as having proven real-grid security.

MODEL LIMITATIONS
-----------------
- Network topology/demand are from SV2024.
- Renewable locations/capacities are from the 2024 asset/node work in this project.
- Renewable availability is deterministic synthetic data because the SV2024 .nc
  lacks renewable p_max_pu time series.
- The starting CSV is an EXCLUSIVE approximation of the published overlapping
  WDT constraint memberships, created for the current annealer formulation.
- Power flow/security is DC + thermal/PTDF/LODF based.
- The harness tests single branch contingencies represented by the N-1 screen.
- It does NOT validate AC voltage/reactive limits, transient/dynamic stability,
  inertia/reserve, protection, market rules, SCADA, or operator procedures.
- It does NOT yet explicitly rebuild the network around a planned outage and
  then test a second contingency (N-1-1).

FILES
-----
validate_dd_reduction.py
    Main program.

all_island_annealer_api_validation.py
    Validation copy of the working SV2024 API. Adds:
      - active demand_scale
      - wind_scale / solar_scale
      - a fast integer-group evaluator for large anneals
    Your original working API is not overwritten.

ni_grid_core.py
    Existing grid core used by the emulator.

data/SV2024_all-island.nc
    SV2024 network.

outputs/annealer_nodes_26_counties_sv2024_wdt_exclusive.csv
outputs/annealer_nodes_32_counties_sv2024_wdt_exclusive.csv
    Starting renewable node/group tables.
