# Dispatch-down reduction validation report

Preset: **standard**

This report compares the published/WDT-derived **exclusive baseline used by the research emulator** with annealed groupings. It is not a claim about the exact historical overlapping WDT implementation.

## Robust candidate selection

### 26-county

Selected `26_w123_a2`. Cross-seed mean reduction: **0.454 percentage points** (7.3% relative). Safe on **5/5** nominal N-1 seed checks.

### 32-county

Selected `32_w77_a1`. Cross-seed mean reduction: **0.176 percentage points** (1.9% relative). Safe on **5/5** nominal N-1 seed checks.

## Stress-test summary

### 26-county

Across 45 stress cases, mean DD reduction was **0.380 pp**, median **0.439 pp**, and worst observed reduction **0.000 pp**. The candidate improved DD in **88.9%** of cases and introduced no new baseline-relative security failures in **82.2%** of cases. Absolute 100% security in the modelled screened states occurred in **11.1%** of cases; the minimum candidate security-pass rate was **28.50%**.

### 32-county

Across 45 stress cases, mean DD reduction was **0.134 pp**, median **0.129 pp**, and worst observed reduction **0.000 pp**. The candidate improved DD in **88.9%** of cases and introduced no new baseline-relative security failures in **91.1%** of cases. Absolute 100% security in the modelled screened states occurred in **11.1%** of cases; the minimum candidate security-pass rate was **38.82%**.

## Interpretation rules

A promising result is one where the reduction remains positive across different weather seeds and stress scenarios, particularly `n1_strict_screen`, with zero new security failures. A large training-seed reduction that disappears or becomes unsafe on held-out seeds should be treated as overfitting rather than a grid improvement.

The model is DC/thermal. Passing these tests is necessary evidence for the research hypothesis, but not sufficient for real-world deployment; AC voltage/reactive, dynamic stability, reserve/inertia and operator studies remain outside this harness.

## Configuration

```json
{
  "preset": "standard",
  "scopes": [
    "26",
    "32"
  ],
  "weather_seeds": [
    42,
    77,
    123,
    202,
    314
  ],
  "anneal_seeds": [
    1,
    2,
    3
  ],
  "anneal_evals": 5000,
  "temperature_probes": 40,
  "runs": 10000,
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
    },
    {
      "include_n1": true,
      "thermal_scale": 0.95,
      "demand_scale": 1.0,
      "wind_scale": 1.0,
      "solar_scale": 1.0,
      "security_threshold_pct": 90.0,
      "max_security_states": 800,
      "scenario": "n1_tight_limits"
    },
    {
      "include_n1": true,
      "thermal_scale": 1.05,
      "demand_scale": 1.0,
      "wind_scale": 1.0,
      "solar_scale": 1.0,
      "security_threshold_pct": 90.0,
      "max_security_states": 800,
      "scenario": "n1_relaxed_limits"
    },
    {
      "include_n1": true,
      "thermal_scale": 1.0,
      "demand_scale": 0.9,
      "wind_scale": 1.0,
      "solar_scale": 1.0,
      "security_threshold_pct": 90.0,
      "max_security_states": 800,
      "scenario": "n1_low_demand"
    },
    {
      "include_n1": true,
      "thermal_scale": 1.0,
      "demand_scale": 1.1,
      "wind_scale": 1.0,
      "solar_scale": 1.0,
      "security_threshold_pct": 90.0,
      "max_security_states": 800,
      "scenario": "n1_high_demand"
    },
    {
      "include_n1": true,
      "thermal_scale": 1.0,
      "demand_scale": 1.0,
      "wind_scale": 1.15,
      "solar_scale": 1.0,
      "security_threshold_pct": 90.0,
      "max_security_states": 800,
      "scenario": "n1_wind_heavy"
    },
    {
      "include_n1": true,
      "thermal_scale": 1.0,
      "demand_scale": 1.0,
      "wind_scale": 1.0,
      "solar_scale": 1.25,
      "security_threshold_pct": 90.0,
      "max_security_states": 800,
      "scenario": "n1_solar_heavy"
    }
  ]
}
```
