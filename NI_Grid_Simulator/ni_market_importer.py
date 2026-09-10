"""CLI importer for NI market/time-series data.

Examples
--------
Create the bus mapping + example templates:
    python ni_market_importer.py

Import a market CSV and export normalised nodal/asset time series:
    python ni_market_importer.py --market my_market_data.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from ni_grid_core import (
    read_network_nc, read_spatial_assets, map_assets_to_buses,
    list_market_timestamps, import_market_scenario, scenario_injections,
    normalised_market_template,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_NETWORK = ROOT / "data" / "SV2024_northern_ireland.nc"
DEFAULT_ASSETS = ROOT / "data" / "northern_ireland_spatial_supply_and_interconnector_points.csv"


def main():
    ap = argparse.ArgumentParser(description="Map NI market data onto the supplied transmission network")
    ap.add_argument("--network", default=str(DEFAULT_NETWORK))
    ap.add_argument("--assets", default=str(DEFAULT_ASSETS))
    ap.add_argument("--market", default=None, help="Aggregate or unit-level market CSV")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    network = read_network_nc(args.network)
    assets = map_assets_to_buses(read_spatial_assets(args.assets), network.buses)

    mapping_cols = [
        "point_name", "point_type", "connection_level", "max_export_capacity_mw",
        "latitude", "longitude", "mapped_bus", "mapped_bus_kv",
        "mapping_distance_km", "mapping_quality", "market_bucket",
        "semo_dispatch_join_candidate",
    ]
    mapping_path = out / "asset_bus_mapping.csv"
    assets[mapping_cols].to_csv(mapping_path, index=False)
    print(f"Wrote {mapping_path}")

    agg_template = ROOT / "data" / "market_aggregate_template.csv"
    normalised_market_template().to_csv(agg_template, index=False)
    unit_template = ROOT / "data" / "market_unit_template.csv"
    pd.DataFrame([
        {"timestamp": "2024-01-01 00:00", "unit_name": "Slieve Kirk Wind Farm", "dispatch_mw": 55.0, "ni_demand_mw": 900, "moyle_mw": -50},
        {"timestamp": "2024-01-01 00:00", "unit_name": "Coolkeeragh Power Station GT", "dispatch_mw": 220.0, "ni_demand_mw": 900, "moyle_mw": -50},
    ]).to_csv(unit_template, index=False)
    print(f"Wrote {agg_template}")
    print(f"Wrote {unit_template}")

    if not args.market:
        print("No --market file supplied; templates and mapping are ready.")
        return

    nodal_rows, asset_rows = [], []
    for ts in list_market_timestamps(args.market):
        scenario = import_market_scenario(network, assets, args.market, ts)
        injections, active, slack, slack_balance = scenario_injections(network, scenario)
        demand = scenario.demand_by_bus.groupby(level=0).sum()
        generation = active.groupby("mapped_bus")["dispatch_mw"].sum()
        fixed = scenario.fixed_injection_by_bus.groupby(level=0).sum() if len(scenario.fixed_injection_by_bus) else pd.Series(dtype=float)
        for bus in network.buses["bus"].astype(str):
            nodal_rows.append({
                "timestamp": ts, "bus": bus,
                "demand_mw": float(demand.get(bus, 0.0)),
                "generation_mw": float(generation.get(bus, 0.0)),
                "fixed_interconnector_mw": float(fixed.get(bus, 0.0)),
                "net_injection_mw_before_slack": float(injections.get(bus, 0.0) - (slack_balance if bus == slack else 0.0)),
                "balancing_slack_mw": float(slack_balance if bus == slack else 0.0),
                "net_injection_mw": float(injections.get(bus, 0.0)),
            })
        tmp = active[["point_name", "point_type", "market_bucket", "mapped_bus", "dispatch_mw", "max_export_capacity_mw"]].copy()
        tmp.insert(0, "timestamp", ts)
        asset_rows.append(tmp)

    nodal_path = out / "market_nodal_injections.csv"
    pd.DataFrame(nodal_rows).to_csv(nodal_path, index=False)
    asset_path = out / "market_asset_dispatch.csv"
    pd.concat(asset_rows, ignore_index=True).to_csv(asset_path, index=False)
    print(f"Wrote {nodal_path}")
    print(f"Wrote {asset_path}")


if __name__ == "__main__":
    main()
