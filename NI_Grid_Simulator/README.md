# Northern Ireland grid constraint simulator

This research prototype combines the supplied NI transmission network with the spatial generator/interconnector register and a market time series.

## What it does

- maps each spatial generator/interconnector point to a proxy transmission bus;
- uses the existing network load distribution to spatially allocate total NI demand;
- accepts **aggregate market data** or **unit-level dispatch**;
- models Moyle as a signed injection at its mapped NI terminal (positive = import, negative = export);
- runs a DC load flow across lines + transformers;
- evaluates thermal loading against each branch's `s_nom` rating;
- estimates line current and resistive `I²R` losses from line resistance and voltage;
- lets you derate all branch limits or impose a custom MW limit on one line;
- optionally performs a transparent linear redispatch to relieve overloads;
- reports renewable curtailment and any last-resort load shedding.

## Run it

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python ni_market_importer.py
python ni_grid_app.py
```

Gradio will print a local URL. Open it in your browser.

## Market CSV option A: aggregate

Use `data/market_aggregate_template.csv`:

```text
timestamp,ni_demand_mw,wind_mw,solar_mw,thermal_mw,biomass_mw,waste_mw,moyle_mw
```

Technology totals are allocated among mapped assets in proportion to their maximum export capacities. This is a practical starting approximation when unit dispatch is unavailable.

## Market CSV option B: unit-level

Use `data/market_unit_template.csv`:

```text
timestamp,unit_name,dispatch_mw,ni_demand_mw,moyle_mw
```

Names are matched to the spatial register. Review `outputs/asset_bus_mapping.csv` and low-distance-quality mappings before treating results as final.

## Thermal rating vs efficiency

These are different concepts:

1. **Thermal capacity / ampacity:** the simulator uses the network `s_nom` (MVA) as the branch thermal rating. Under the DC approximation, `|P| ≈ |S|`, so `|P| / s_nom` is used as the thermal loading percentage.
2. **Resistive efficiency:** for ordinary AC lines the simulator estimates loss as

   `P_loss ≈ P² R / V²`

   and reports `100 × (1 - P_loss / |P|)` as a diagnostic transfer efficiency.

The DC flow equations themselves are lossless, so this loss calculation is a **diagnostic approximation**, not a full AC loss model. Transformers are included in power transfer and thermal constraints, but transformer loss is not estimated here because the supplied transformer resistance is in per-unit and the three-winding equivalents deserve a more careful AC treatment.

## Important modelling limitations

- Distribution-connected farms are proxied to the nearest 110 kV transmission bus. The mapping distance is retained; several sites are more than 10 km from their proxy bus.
- The supplied NI network contains only a few ordinary generator objects, so the spatial CSV is used to construct the richer generation layer.
- Aggregate technology allocation by MEC is not the same as actual unit availability or dispatch.
- DC load flow ignores reactive power, voltage constraints, contingencies, dynamic stability and the explicit effect of losses on power balance.
- Redispatch is an explanatory linear programme, not a recreation of SONI/SEMO dispatch rules.

## Recommended next extension for constraint groups

For every stressed line and timestamp, calculate the generator PTDF vector and record which generators the redispatch actually curtails. Those vectors can then feed hierarchical clustering to form smaller, electrically coherent candidate constraint groups. Compare them against a broad-group baseline using total curtailed MWh, overload violations and redispatch cost.
