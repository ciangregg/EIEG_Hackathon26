# Dispatch-down reduction validation report

Preset: **quick**

This report compares the published/WDT-derived **exclusive baseline used by the research emulator** with annealed groupings. It is not a claim about the exact historical overlapping WDT implementation.

## Robust candidate selection

### 26-county

Selected `26_w77_a1`. Cross-seed mean reduction: **0.290 percentage points** (4.3% relative). Safe on **2/2** nominal N-1 seed checks.

### 32-county

Selected `32_w77_a1`. Cross-seed mean reduction: **0.006 percentage points** (0.0% relative). Safe on **2/2** nominal N-1 seed checks.

## Stress-test summary

### 26-county

Across 6 stress cases, mean DD reduction was **0.193 pp**, median **0.281 pp**, and worst observed reduction **0.000 pp**. The candidate improved DD in **66.7%** of cases and introduced no new baseline-relative security failures in **100.0%** of cases. Absolute 100% security in the modelled screened states occurred in **33.3%** of cases; the minimum candidate security-pass rate was **43.90%**.

### 32-county

Across 6 stress cases, mean DD reduction was **0.004 pp**, median **0.000 pp**, and worst observed reduction **-0.010 pp**. The candidate improved DD in **33.3%** of cases and introduced no new baseline-relative security failures in **100.0%** of cases. Absolute 100% security in the modelled screened states occurred in **33.3%** of cases; the minimum candidate security-pass rate was **58.25%**.

## Interpretation rules

A promising result is one where the reduction remains positive across different weather seeds and stress scenarios, particularly `n1_strict_screen`, with zero new security failures. A large training-seed reduction that disappears or becomes unsafe on held-out seeds should be treated as overfitting rather than a grid improvement.

The model is DC/thermal. Passing these tests is necessary evidence for the research hypothesis, but not sufficient for real-world deployment; AC voltage/reactive, dynamic stability, reserve/inertia and operator studies remain outside this harness.

## Configuration

```json
{
  "preset": "quick",
  "scopes": [
    "26",
    "32"
  ],
  "weather_seeds": [
    42,
    77
  ],
  "anneal_seeds": [
    1
  ],
  "anneal_evals": 120,
  "temperature_probes": 20,
  "runs": 2000,
  "optimisation_security_threshold_pct": 90.0,
  "optimisation_max_states": 800,
  "scenarios": [
    {
      "include_n1": false,
      "thermal_scale": 1.0,
      "demand_scale": 1.0,
      "wind_scale": 1.0,
      "solar_scale": 1.0,
      "security_threshold_pct": 90.0,
      "max_security_states": 800,
      "scenario": "intact_nominal"
    },
    {
      "include_n1": true,
      "thermal_scale": 1.0,
      "demand_scale": 1.0,
      "wind_scale": 1.0,
      "solar_scale": 1.0,
      "security_threshold_pct": 90.0,
      "max_security_states": 800,
      "scenario": "n1_nominal"
    },
    {
      "include_n1": true,
      "thermal_scale": 1.0,
      "demand_scale": 1.0,
      "wind_scale": 1.0,
      "solar_scale": 1.0,
      "security_threshold_pct": 70.0,
      "max_security_states": 3000,
      "scenario": "n1_strict_screen"
    }
  ]
}
```
