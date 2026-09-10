"""Interactive NI transmission flow, constraint and dynamic demand-shock simulator."""
from __future__ import annotations

from pathlib import Path
import time

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ni_grid_core import (
    DCGridModel,
    built_in_scenario,
    import_market_scenario,
    list_market_timestamps,
    map_assets_to_buses,
    read_network_nc,
    read_spatial_assets,
    relieve_thermal_constraints,
    scenario_injections,
)
from ni_scenarios import (
    DynamicScenarioConfig,
    build_demand_shock_frames,
    frame_metrics,
    generator_response_summary,
    peak_line_loading,
)
from ni_animation import make_power_flow_animation, make_power_flow_frame
from ni_surplus import (
    RenewableSurplusConfig,
    build_renewable_surplus_frames,
    curtailment_energy_mwh,
    peak_line_loading_surplus,
    renewable_curtailment_by_asset,
    surplus_frame_metrics,
    thermal_commitment_by_asset,
)
from ni_constraint_stress import (
    StressTestConfig,
    SONI_GROUPS,
    group_catalogue,
    run_stress_test,
)
from ni_baseline_replica import (
    SONI_2024_ALL_RENEWABLE_DD_PCT,
    current_operational_config,
    export_handoff_pack,
    run_baseline_replica,
)

ROOT = Path(__file__).resolve().parent
NETWORK_FILE = ROOT / "data" / "SV2024_northern_ireland.nc"
ASSET_FILE = ROOT / "data" / "northern_ireland_spatial_supply_and_interconnector_points.csv"

NETWORK = read_network_nc(NETWORK_FILE)
ASSETS = map_assets_to_buses(read_spatial_assets(ASSET_FILE), NETWORK.buses)
MODEL = DCGridModel(NETWORK)
GROUP_CATALOGUE = group_catalogue(ASSETS, MODEL)
LINE_CHOICES = NETWORK.lines["branch"].astype(str).tolist()
BUS_COORD = NETWORK.buses.set_index("bus")[["lon", "lat"]]

# Restrict local shock targets to actual load buses.  Include available names to
# make the dropdown a little easier to interpret than a bare numeric bus ID.
_bus_meta = NETWORK.buses.set_index("bus")
_load_bus_ids = sorted(set(NETWORK.loads["bus"].astype(str)))
LOAD_BUS_LABEL_TO_ID = {}
for bus in _load_bus_ids:
    label = bus
    if bus in _bus_meta.index:
        name = str(_bus_meta.loc[bus].get("psse_name", "")).strip()
        station = str(_bus_meta.loc[bus].get("station", "")).strip()
        extra = name or station
        if extra and extra.lower() not in {"nan", "none"}:
            label = f"{bus} — {extra}"
    LOAD_BUS_LABEL_TO_ID[label] = bus
LOAD_BUS_CHOICES = list(LOAD_BUS_LABEL_TO_ID.keys())


def _market_path(file_value):
    if file_value is None:
        return None
    if isinstance(file_value, (str, Path)):
        return str(file_value)
    if hasattr(file_value, "name"):
        return str(file_value.name)
    return str(file_value)


def refresh_timestamps(market_file):
    p = _market_path(market_file)
    if not p:
        choices = [str(i) for i in range(len(NETWORK.load_profile))]
        return (
            gr.update(choices=choices, value=choices[0]),
            "Using the built-in 168-snapshot load profile. Spatial generators remain at zero unless market data is uploaded.",
        )
    try:
        choices = list_market_timestamps(p)
        return gr.update(choices=choices, value=choices[0] if choices else None), f"Loaded {len(choices)} market timestamp(s)."
    except Exception as e:
        return gr.update(choices=[], value=None), f"Market CSV error: {e}"


def _scenario(market_file, timestamp):
    p = _market_path(market_file)
    if p:
        return import_market_scenario(NETWORK, ASSETS, p, str(timestamp))
    idx = int(timestamp or 0)
    return built_in_scenario(NETWORK, ASSETS, idx)


def _line_style(loading):
    if loading > 100:
        return "#b91c1c", 5.0
    if loading > 90:
        return "#ea580c", 4.0
    if loading > 70:
        return "#ca8a04", 3.3
    return "#64748b", 2.0


def make_grid_figure(branches: pd.DataFrame, buses: pd.DataFrame, title: str):
    fig = go.Figure()
    line_results = branches[branches["type"].eq("line")].copy()
    mid_lon, mid_lat, mid_text, mid_color, mid_size = [], [], [], [], []
    for _, r in line_results.iterrows():
        if r.bus0 not in BUS_COORD.index or r.bus1 not in BUS_COORD.index:
            continue
        a, b = BUS_COORD.loc[r.bus0], BUS_COORD.loc[r.bus1]
        if not np.all(np.isfinite([a.lon, a.lat, b.lon, b.lat])):
            continue
        color, width = _line_style(float(r.loading_pct))
        direction = f"{r.bus0} → {r.bus1}" if r.flow_mw >= 0 else f"{r.bus1} → {r.bus0}"
        hover = (
            f"<b>{r.branch}</b><br>{direction}<br>Flow: {abs(r.flow_mw):.1f} MW"
            f"<br>Thermal limit: {r.thermal_limit_mw:.1f} MW"
            f"<br>Loading: {r.loading_pct:.1f}%"
            f"<br>Est. I²R loss: {r.estimated_loss_mw:.3f} MW"
            f"<br>Est. efficiency: {r.estimated_efficiency_pct:.2f}%"
        )
        fig.add_trace(
            go.Scatter(
                x=[a.lon, b.lon],
                y=[a.lat, b.lat],
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        mid_lon.append((a.lon + b.lon) / 2)
        mid_lat.append((a.lat + b.lat) / 2)
        mid_text.append(hover)
        mid_color.append(color)
        mid_size.append(7 if r.loading_pct <= 100 else 11)
    fig.add_trace(
        go.Scatter(
            x=mid_lon,
            y=mid_lat,
            mode="markers",
            text=mid_text,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=mid_size, color=mid_color, symbol="triangle-up", line=dict(width=0)),
            showlegend=False,
        )
    )

    plot_buses = buses[np.isfinite(buses["lon"]) & np.isfinite(buses["lat"])].copy()
    mag = plot_buses["injection_mw"].abs()
    sizes = np.clip(5 + np.sqrt(mag.clip(lower=0)) * 0.9, 5, 22)
    colors = np.where(plot_buses["injection_mw"] >= 0, "#2563eb", "#111827")
    text = [
        f"<b>Bus {r.bus}</b><br>Injection: {r.injection_mw:.1f} MW<br>Voltage: {r.v_nom_kv:.0f} kV"
        for _, r in plot_buses.iterrows()
    ]
    fig.add_trace(
        go.Scatter(
            x=plot_buses.lon,
            y=plot_buses.lat,
            mode="markers",
            text=text,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=sizes, color=colors, opacity=0.85, line=dict(width=0.5, color="white")),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=690,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(title="Longitude", showgrid=False, zeroline=False),
        yaxis=dict(title="Latitude", showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        hovermode="closest",
    )
    return fig


def _summary(branches, scenario, slack_bus, slack_balance, redispatch=None):
    lines = branches[branches.type.eq("line")]
    overloaded = lines[lines.loading_pct > 100]
    worst = lines.sort_values("loading_pct", ascending=False).iloc[0]
    loss = lines.estimated_loss_mw.sum()
    s = [
        f"**Scenario:** {scenario.timestamp}",
        f"**NI demand:** {scenario.total_demand_mw:.1f} MW | **Moyle:** {scenario.interconnector_mw:+.1f} MW",
        f"**Balancing slack:** {slack_bus} ({slack_balance:+.1f} MW)",
        f"**Worst line:** `{worst.branch}` at **{worst.loading_pct:.1f}%** | overloaded lines: **{len(overloaded)}**",
        f"**Estimated line I²R losses:** {loss:.2f} MW (diagnostic; DC power flow itself is lossless)",
    ]
    if redispatch:
        s.append(
            f"**Redispatch:** {redispatch.get('total_down_mw', 0):.1f} MW down / {redispatch.get('total_up_mw', 0):.1f} MW up; "
            f"renewable curtailment {redispatch.get('renewable_curtailment_mw', 0):.1f} MW; "
            f"load shed {redispatch.get('load_shed_mw', 0):.1f} MW."
        )
        s.append(f"**Solver:** {redispatch.get('message', '')}")
    if scenario.notes:
        s.append(f"_Importer note:_ {scenario.notes}")
    return "  \n".join(s)


def run_simulation(market_file, timestamp, thermal_scale, override_line, override_limit, auto_redispatch, allow_load_shed):
    try:
        scenario = _scenario(market_file, timestamp)
        injections, active, slack, slack_balance = scenario_injections(NETWORK, scenario)
        base_br, base_bus = MODEL.solve(injections, slack, thermal_scale=1.0)
        base_fig = make_grid_figure(base_br, base_bus, "Baseline — nominal thermal ratings")

        overrides = None
        if override_line and override_limit and float(override_limit) > 0:
            overrides = {str(override_line): float(override_limit)}
        constrained_br, constrained_bus = MODEL.solve(injections, slack, float(thermal_scale), overrides)
        redispatch = None
        changed = active.copy()
        if auto_redispatch and (constrained_br.loading_pct > 100 + 1e-7).any():
            new_p, changed, redispatch = relieve_thermal_constraints(
                MODEL, scenario, injections, active, slack, float(thermal_scale), overrides, bool(allow_load_shed)
            )
            if redispatch.get("success"):
                constrained_br, constrained_bus = MODEL.solve(new_p, slack, float(thermal_scale), overrides)
        constrained_fig = make_grid_figure(constrained_br, constrained_bus, "Constrained case — after redispatch where feasible")

        line_cols = [
            "branch", "bus0", "bus1", "flow_mw", "thermal_limit_mw", "loading_pct",
            "estimated_loss_mw", "estimated_efficiency_pct", "overload_mw",
        ]
        line_table = constrained_br[constrained_br.type.eq("line")][line_cols].sort_values("loading_pct", ascending=False).round(3)
        dispatch_cols = [
            c for c in [
                "point_name", "point_type", "market_bucket", "mapped_bus", "mapping_distance_km",
                "dispatch_mw", "redispatch_down_mw", "redispatch_up_mw", "max_export_capacity_mw",
            ] if c in changed.columns
        ]
        dispatch = changed[changed.get("dispatch_mw", pd.Series(0, index=changed.index)).abs() > 1e-9][dispatch_cols].copy()
        if "redispatch_down_mw" in dispatch:
            dispatch["net_redispatch_mw"] = dispatch.get("redispatch_up_mw", 0).fillna(0) - dispatch.get("redispatch_down_mw", 0).fillna(0)
            dispatch = dispatch.sort_values("net_redispatch_mw", key=lambda x: x.abs(), ascending=False)
        return base_fig, constrained_fig, _summary(constrained_br, scenario, slack, slack_balance, redispatch), line_table, dispatch.round(3)
    except Exception as e:
        blank = go.Figure().update_layout(template="plotly_white", title=f"Simulation error: {e}")
        return blank, blank, f"**Simulation error:** {e}", pd.DataFrame(), pd.DataFrame()


def apply_dynamic_preset(preset):
    presets = {
        "TV tea-time spike": (250.0, "System-wide", 12),
        "Very large NI-wide spike": (500.0, "System-wide", 16),
        "Local city-scale spike": (100.0, "Single load bus", 12),
        "Demand collapse": (-250.0, "System-wide", 12),
        "Custom": (250.0, "System-wide", 12),
    }
    shock, scope, steps = presets.get(str(preset), presets["Custom"])
    return shock, scope, steps


def _dynamic_summary(frames, response_mode):
    if not frames:
        return "No dynamic scenario frames were created."
    first, last = frames[0], frames[-1]
    metrics = frame_metrics(frames)
    peak = metrics.loc[metrics["worst_loading_pct"].idxmax()]
    delta_loss = last.estimated_losses_mw - first.estimated_losses_mw
    thermal_pickup = 0.0
    if len(first.active_assets) and len(last.active_assets):
        a = first.active_assets.groupby("market_bucket")["dispatch_mw"].sum()
        b = last.active_assets.groupby("market_bucket")["dispatch_mw"].sum()
        thermal_pickup = float(b.get("thermal", 0.0) - a.get("thermal", 0.0))
    text = [
        f"**Demand:** {first.scenario.total_demand_mw:.1f} → **{last.scenario.total_demand_mw:.1f} MW** "
        f"({last.actual_demand_change_mw:+.1f} MW)",
        f"**Generation response mode:** `{response_mode}` | thermal fleet change: **{thermal_pickup:+.1f} MW**",
        f"**Peak network stress:** `{peak.worst_line}` at **{peak.worst_loading_pct:.1f}%** "
        f"with **{int(peak.overloaded_lines)}** overloaded line(s) at that step",
        f"**Estimated line-loss change:** {first.estimated_losses_mw:.2f} → **{last.estimated_losses_mw:.2f} MW** "
        f"({delta_loss:+.2f} MW)",
    ]
    if last.redispatch:
        rd = last.redispatch
        text.append(
            f"**Final corrective redispatch:** {rd.get('total_down_mw', 0):.1f} MW down / "
            f"{rd.get('total_up_mw', 0):.1f} MW up; renewable curtailment "
            f"{rd.get('renewable_curtailment_mw', 0):.1f} MW; load shed {rd.get('load_shed_mw', 0):.1f} MW."
        )
    text.append(
        "_This animation is a sequence of quasi-steady-state DC power-flow solves, not a frequency/transient-stability simulation. "
        "The particle motion visualises solved MW direction; it is not literal electron travel._"
    )
    return "  \n".join(text)


def run_dynamic_scenario(
    market_file,
    timestamp,
    preset,
    shock_mw,
    scope,
    target_bus_label,
    steps,
    response_mode,
    thermal_scale,
    override_line,
    override_limit,
    auto_redispatch,
    allow_load_shed,
    particle_smoothness,
    particles_per_line,
    animation_speed_ms,
):
    try:
        base = _scenario(market_file, timestamp)
        target_bus = None
        if str(scope) == "Single load bus":
            target_bus = LOAD_BUS_LABEL_TO_ID.get(str(target_bus_label), str(target_bus_label) if target_bus_label else None)
        overrides = None
        if override_line and override_limit and float(override_limit) > 0:
            overrides = {str(override_line): float(override_limit)}

        cfg = DynamicScenarioConfig(
            shock_mw=float(shock_mw),
            scope="bus" if str(scope) == "Single load bus" else "system",
            target_bus=target_bus,
            steps=int(steps),
            response_mode=str(response_mode),
            thermal_scale=float(thermal_scale),
            line_limit_overrides=overrides,
            auto_redispatch=bool(auto_redispatch),
            allow_load_shedding=bool(allow_load_shed),
        )
        frames = build_demand_shock_frames(NETWORK, MODEL, base, cfg)
        animation = make_power_flow_animation(
            frames,
            BUS_COORD,
            title=f"Dynamic response — {preset}",
            particle_phases=int(particle_smoothness),
            max_particles_per_line=int(particles_per_line),
            frame_duration_ms=int(animation_speed_ms),
        )
        metrics = frame_metrics(frames).round(3)
        peak_lines = peak_line_loading(frames).head(25).round(3)
        gen_response = generator_response_summary(frames).round(3)
        # Return the solved frames as Gradio state. The visible plot uses an
        # ordinary static figure; Play streams successive phases explicitly.
        initial_plot = make_power_flow_frame(
            frames[0], BUS_COORD, phase=0.0,
            title=f"Dynamic response — {preset}",
            max_particles_per_line=int(particles_per_line),
        )
        return initial_plot, _dynamic_summary(frames, response_mode), metrics, peak_lines, gen_response, frames
    except Exception as e:
        blank = go.Figure().update_layout(template="plotly_white", title=f"Dynamic scenario error: {e}")
        return blank, f"**Dynamic scenario error:** {e}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []


def apply_surplus_preset(preset):
    """Synthetic, editable renewable-surplus presets for stress testing."""
    presets = {
        "Windy low-demand night": (350.0, 0.0, -250.0, 12, 60.0),
        "Extreme wind surplus": (600.0, 0.0, -150.0, 16, 90.0),
        "Solar-rich low-demand noon": (100.0, 80.0, -120.0, 12, 60.0),
        "Wind + solar glut": (450.0, 80.0, -200.0, 16, 90.0),
        "Low demand only": (0.0, 0.0, -350.0, 12, 60.0),
        "Custom": (350.0, 0.0, -250.0, 12, 60.0),
    }
    return presets.get(str(preset), presets["Custom"])


def _surplus_summary(frames):
    if not frames:
        return "No renewable-surplus frames were created."
    first, last = frames[0], frames[-1]
    lost_mwh = curtailment_energy_mwh(frames)
    metrics = surplus_frame_metrics(frames)
    peak = metrics.loc[metrics["worst_loading_pct"].idxmax()]
    curtailment_reason = []
    if last.balance_curtailment_mw > 1e-6:
        curtailment_reason.append(f"{last.balance_curtailment_mw:.1f} MW energy-balance")
    if last.stability_curtailment_mw > 1e-6:
        curtailment_reason.append(f"{last.stability_curtailment_mw:.1f} MW stability / non-synchronous-share")
    if last.minimum_output_curtailment_mw > 1e-6:
        curtailment_reason.append(f"{last.minimum_output_curtailment_mw:.1f} MW minimum-stable-output")
    if last.network_curtailment_mw > 1e-6:
        curtailment_reason.append(f"{last.network_curtailment_mw:.1f} MW transmission-constrained")
    reason = ", ".join(curtailment_reason) if curtailment_reason else "none at the final frame"
    text = [
        f"**Demand:** {first.scenario.total_demand_mw:.1f} → **{last.scenario.total_demand_mw:.1f} MW** "
        f"({last.actual_demand_change_mw:+.1f} MW)",
        f"**Added renewable availability:** wind **+{last.actual_wind_surge_mw:.1f} MW**, "
        f"solar **+{last.actual_solar_surge_mw:.1f} MW**",
        f"**Renewable potential:** {last.renewable_available_mw:.1f} MW | accepted **{last.renewable_accepted_mw:.1f} MW** | "
        f"curtailed **{last.renewable_curtailment_mw:.1f} MW**",
        f"**Renewable utilisation:** **{last.renewable_utilisation_pct:.1f}%** "
        f"(curtailment rate {100-last.renewable_utilisation_pct:.1f}%)",
        f"**Curtailment cause at final frame:** {reason}",
        f"**Conventional thermal change:** **{last.thermal_change_mw:+.1f} MW** relative to the starting snapshot",
        (
            f"**Synchronous commitment:** **{int((last.commitment or {}).get('online_thermal_units', 0))} thermal units online**, "
            f"{int((last.commitment or {}).get('offline_thermal_units', 0))} offline; "
            f"non-synchronous share **{float((last.commitment or {}).get('nonsynchronous_share_pct', 0.0)):.1f}%** "
            f"against a **{float((last.commitment or {}).get('max_nonsynchronous_share_pct', 100.0)):.1f}%** scenario cap"
        ) if last.commitment else "**Synchronous commitment:** disabled",
        f"**Potential renewable energy lost during the {last.elapsed_minutes:.0f}-minute ramp:** **{lost_mwh:.1f} MWh**",
        f"**Peak network stress:** `{peak.worst_line}` at **{peak.worst_loading_pct:.1f}%**; "
        f"estimated line losses at the final state **{last.estimated_losses_mw:.2f} MW**",
    ]
    if abs(last.slack_balance_mw) > 1e-3:
        text.append(
            f"**Residual balancing/slack:** {last.slack_balance_mw:+.1f} MW. A large residual means the simplified fleet/export assumptions "
            "could not fully absorb or supply the requested scenario."
        )
    text.append(
        "_Renewable utilisation is not the same as conductor efficiency: curtailed MWh are usable generation that was available but not accepted; "
        "I²R line losses are reported separately. Baseline renewable potential is assumed equal to baseline dispatch because the market template does not provide a separate availability series._"
    )
    return "  \n".join(text)




def _renewable_benchmark_table(frames):
    """Compare the final simulated renewable utilisation with historical SONI NI references.

    The 2024 reference is all renewables; the 2023 reference is wind-only, so the
    latter is shown as contextual rather than strictly like-for-like.
    """
    if not frames:
        return pd.DataFrame()
    last = frames[-1]
    model_util = float(last.renewable_utilisation_pct)
    model_dd = 100.0 - model_util
    refs = [
        {
            "benchmark": "Current simulation (final frame)",
            "scope": "Mapped wind + solar in model",
            "dispatch_down_pct": model_dd,
            "renewable_utilisation_pct": model_util,
            "gap_vs_model_pp": 0.0,
        },
        {
            "benchmark": "SONI NI 2024",
            "scope": "All renewables (historical annual)",
            "dispatch_down_pct": 25.5,
            "renewable_utilisation_pct": 74.5,
            "gap_vs_model_pp": model_util - 74.5,
        },
        {
            "benchmark": "SONI NI 2023",
            "scope": "Wind only (historical annual)",
            "dispatch_down_pct": 10.8,
            "renewable_utilisation_pct": 89.2,
            "gap_vs_model_pp": model_util - 89.2,
        },
    ]
    return pd.DataFrame(refs).round(2)


def run_surplus_scenario(
    market_file,
    timestamp,
    preset,
    wind_surge_mw,
    solar_surge_mw,
    demand_change_mw,
    demand_scope,
    target_bus_label,
    steps,
    duration_minutes,
    thermal_scale,
    override_line,
    override_limit,
    auto_redispatch,
    commitment_enabled,
    max_nonsynchronous_share_pct,
    min_synchronous_units_online,
    min_stable_output_pct,
    shutdown_strategy,
    particle_smoothness,
    particles_per_line,
    animation_speed_ms,
):
    try:
        base = _scenario(market_file, timestamp)
        target_bus = None
        if str(demand_scope) == "Single load bus":
            target_bus = LOAD_BUS_LABEL_TO_ID.get(
                str(target_bus_label), str(target_bus_label) if target_bus_label else None
            )
        overrides = None
        if override_line and override_limit and float(override_limit) > 0:
            overrides = {str(override_line): float(override_limit)}
        cfg = RenewableSurplusConfig(
            wind_surge_mw=max(0.0, float(wind_surge_mw)),
            solar_surge_mw=max(0.0, float(solar_surge_mw)),
            demand_change_mw=float(demand_change_mw),
            demand_scope="bus" if str(demand_scope) == "Single load bus" else "system",
            target_bus=target_bus,
            steps=int(steps),
            duration_minutes=float(duration_minutes),
            thermal_scale=float(thermal_scale),
            line_limit_overrides=overrides,
            auto_redispatch=bool(auto_redispatch),
            allow_load_shedding=False,
            commitment_enabled=bool(commitment_enabled),
            max_nonsynchronous_share_pct=float(max_nonsynchronous_share_pct),
            min_synchronous_units_online=int(min_synchronous_units_online),
            min_stable_output_pct=float(min_stable_output_pct),
            shutdown_strategy=str(shutdown_strategy),
        )
        frames = build_renewable_surplus_frames(NETWORK, MODEL, base, cfg)
        labels = [
            f"Curt {f.renewable_curtailment_mw:.0f} MW"
            for f in frames
        ]
        animation = make_power_flow_animation(
            frames,
            BUS_COORD,
            title=f"Renewable surplus / low demand — {preset}",
            particle_phases=int(particle_smoothness),
            max_particles_per_line=int(particles_per_line),
            frame_duration_ms=int(animation_speed_ms),
            slider_labels=labels,
            slider_prefix="Renewable state: ",
            annotation_text=(
                "Moving dots follow solved MW flow. Orange diamonds = online thermal units; grey × = offline units. "
                "Line: grey <70% · amber 70–90% · orange 90–100% · red >100%. "
                "Curtailment means available wind/solar that the system does not accept; line I²R losses are separate."
            ),
        )
        metrics = surplus_frame_metrics(frames).round(3)
        peak_lines = peak_line_loading_surplus(frames).head(25).round(3)
        assets = renewable_curtailment_by_asset(frames[-1]).round(3)
        commitment_units = thermal_commitment_by_asset(frames[-1]).round(3)
        initial_plot = make_power_flow_frame(
            frames[0], BUS_COORD, phase=0.0,
            title=f"Renewable surplus / low demand — {preset}",
            max_particles_per_line=int(particles_per_line),
            annotation_text=(
                "Moving dots follow solved MW flow. Orange diamonds = online thermal units; grey × = offline units. "
                "Line: grey <70% · amber 70–90% · orange 90–100% · red >100%. "
                "Curtailment means available wind/solar that the system does not accept; line I²R losses are separate."
            ),
        )
        return initial_plot, _surplus_summary(frames), metrics, _renewable_benchmark_table(frames), peak_lines, assets, commitment_units, frames
    except Exception as e:
        import traceback
        traceback.print_exc()
        blank = go.Figure().update_layout(template="plotly_white", title=f"Renewable surplus error: {e}")
        return blank, f"**Renewable surplus error:** {e}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []



def _stress_group_figure(group_summary: pd.DataFrame):
    fig = go.Figure()
    if group_summary is None or group_summary.empty:
        return fig.update_layout(template="plotly_white", title="No stress-test results yet")
    d = group_summary.sort_values("group")
    labels = [f"G{int(g)} · {n}" for g, n in zip(d["group"], d["name"])]
    fig.add_trace(go.Bar(x=labels, y=d["first_activation_pct"], name="First activation %"))
    fig.add_trace(go.Bar(x=labels, y=d["activation_pct"], name="Activated at any point %"))
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        title="Constraint-group activation frequency",
        yaxis_title="Share of Monte-Carlo runs (%)",
        xaxis_title="SONI NI WDT group",
        height=440,
        margin=dict(l=55, r=20, t=55, b=90),
    )
    return fig


def _stress_line_figure(line_summary: pd.DataFrame):
    fig = go.Figure()
    if line_summary is None or line_summary.empty:
        return fig.update_layout(template="plotly_white", title="No stress-test results yet")
    d = line_summary.head(15).iloc[::-1]
    fig.add_trace(go.Bar(y=d["line"], x=d["overload_pct_of_runs"], orientation="h", name="Overloaded runs %"))
    fig.add_trace(go.Bar(y=d["line"], x=d["worst_line_pct_of_runs"], orientation="h", name="Worst line runs %"))
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        title="Most repeatedly stressed transmission lines",
        xaxis_title="Share of Monte-Carlo runs (%)",
        yaxis_title="Line",
        height=560,
        margin=dict(l=130, r=20, t=55, b=50),
    )
    return fig




def _stress_frequency_style(overload_pct: float):
    if overload_pct > 20.0:
        return "#b91c1c", 5.2
    if overload_pct > 10.0:
        return "#ea580c", 4.2
    if overload_pct > 3.0:
        return "#ca8a04", 3.2
    return "#94a3b8", 2.0


def _stress_network_map(line_summary: pd.DataFrame):
    fig = go.Figure()
    if line_summary is None or line_summary.empty:
        return fig.update_layout(template="plotly_white", title="No stress-map results yet")
    meta = NETWORK.lines.merge(line_summary, left_on="branch", right_on="line", how="left")
    mid_lon, mid_lat, mid_text, mid_color, mid_size = [], [], [], [], []
    for _, r in meta.iterrows():
        if r.bus0 not in BUS_COORD.index or r.bus1 not in BUS_COORD.index:
            continue
        a, b = BUS_COORD.loc[r.bus0], BUS_COORD.loc[r.bus1]
        if not np.all(np.isfinite([a.lon, a.lat, b.lon, b.lat])):
            continue
        overload_pct = float(pd.to_numeric(pd.Series([r.get("overload_pct_of_runs", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        p95 = float(pd.to_numeric(pd.Series([r.get("p95_loading_pct", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        worst_pct = float(pd.to_numeric(pd.Series([r.get("worst_line_pct_of_runs", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        max_load = float(pd.to_numeric(pd.Series([r.get("max_loading_pct", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        color, width = _stress_frequency_style(overload_pct)
        fig.add_trace(go.Scatter(
            x=[a.lon, b.lon], y=[a.lat, b.lat], mode="lines",
            line=dict(color=color, width=width), hoverinfo="skip", showlegend=False,
        ))
        mid_lon.append((a.lon + b.lon) / 2)
        mid_lat.append((a.lat + b.lat) / 2)
        mid_text.append(
            f"<b>{r.branch}</b><br>Overloaded in: {overload_pct:.1f}% of runs"
            f"<br>Worst line in: {worst_pct:.1f}% of runs"
            f"<br>95th percentile loading: {p95:.1f}%"
            f"<br>Maximum loading: {max_load:.1f}%"
            f"<br>Thermal limit: {float(r.s_nom_mva):.1f} MW"
        )
        mid_color.append(color)
        mid_size.append(7 if overload_pct <= 0 else 10)
    fig.add_trace(go.Scatter(
        x=mid_lon, y=mid_lat, mode="markers", text=mid_text,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(size=mid_size, color=mid_color, symbol="triangle-up", line=dict(width=0)),
        showlegend=False,
    ))
    plot_buses = NETWORK.buses[np.isfinite(NETWORK.buses["lon"]) & np.isfinite(NETWORK.buses["lat"])].copy()
    fig.add_trace(go.Scatter(
        x=plot_buses.lon, y=plot_buses.lat, mode="markers",
        text=[f"Bus {b}" for b in plot_buses.bus], hovertemplate="%{text}<extra></extra>",
        marker=dict(size=5, color="#334155", opacity=0.7, line=dict(width=0.4, color="white")),
        showlegend=False,
    ))
    fig.update_layout(
        title="Stress map — lines repeatedly hit hardest across all Monte-Carlo runs",
        template="plotly_white", height=720, margin=dict(l=10, r=10, t=60, b=30),
        xaxis=dict(title="Longitude", showgrid=False, zeroline=False),
        yaxis=dict(title="Latitude", showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        hovermode="closest",
        annotations=[{
            "text": "Colour shows overload frequency across all runs: grey 0–3% · amber 3–10% · orange 10–20% · red >20%. Hover any line for p95/max loading.",
            "xref": "paper", "yref": "paper", "x": 0, "y": 1.04, "showarrow": False, "align": "left",
            "font": {"size": 11, "color": "#475569"},
        }],
    )
    return fig


def _stress_worst_case_map(result):
    fig = go.Figure()
    if result is None or not getattr(result, "top_cases", None):
        return fig.update_layout(template="plotly_white", title="No worst-case trial map yet")
    case = sorted(result.top_cases, key=lambda c: (c["initial_worst_loading_pct"], c["dispatch_down_mw"]), reverse=True)[0]
    injections = pd.Series(case["initial_injections_mw"], index=MODEL.bus_ids, dtype=float)
    branches, buses = MODEL.solve(injections, str(case["slack_bus"]), float(case.get("thermal_scale", 1.0)))
    title = (
        f"Worst single trial map — run {case['run']}<br><sup>Initial worst line {case['initial_worst_line']} at "
        f"{case['initial_worst_loading_pct']:.1f}% · dispatch-down {case['dispatch_down_mw']:.1f} MW · {case['activation_sequence']}</sup>"
    )
    return make_grid_figure(branches, buses, title)

def _stress_dispatch_benchmark(result):
    if result is None:
        return pd.DataFrame()
    sim = float(result.metadata.get("aggregate_dispatch_down_pct", 0.0))
    rows = [
        {
            "benchmark": "Current stress simulation",
            "dispatch_down_pct": sim,
            "renewable_accommodated_pct": 100.0 - sim,
            "gap_vs_simulation_percentage_points": 0.0,
            "scope": "Thermal-only Monte-Carlo model",
        },
        {
            "benchmark": "SONI NI 2024 actual",
            "dispatch_down_pct": 25.5,
            "renewable_accommodated_pct": 74.5,
            "gap_vs_simulation_percentage_points": sim - 25.5,
            "scope": "All renewable sources; historical actual",
        },
        {
            "benchmark": "SONI NI 2023 wind-only context",
            "dispatch_down_pct": 10.8,
            "renewable_accommodated_pct": 89.2,
            "gap_vs_simulation_percentage_points": sim - 10.8,
            "scope": "Wind-only; contextual, not exactly like-for-like",
        },
    ]
    return pd.DataFrame(rows)


def _stress_summary(result):
    m = result.metadata
    runs = int(m["runs"])
    over = int(m["any_initial_overload_runs"])
    unresolved = int(m["unresolved_runs"])
    first = result.group_summary.sort_values("first_activation_count", ascending=False).iloc[0]
    worst = result.line_summary.iloc[0]
    return "  \n".join([
        f"**Completed:** **{runs:,} runs** in **{m['elapsed_seconds']:.2f} s** "
        f"(**{m['runs_per_second']:,.0f} runs/s**) using random seed `{m['seed']}`.",
        f"**Simulated thermal-only dispatch-down:** **{m.get('aggregate_dispatch_down_pct', 0.0):.2f}%** "
        f"of pre-dispatch-down renewable generation across all trials. "
        f"SONI NI 2024 actual all-renewables dispatch-down: **25.5%**; model gap: "
        f"**{m.get('aggregate_dispatch_down_pct', 0.0) - 25.5:+.2f} percentage points**.",
        f"**Thermal stress occurred:** **{over:,}/{runs:,} runs ({100*over/runs:.1f}%)** had at least one initial line overload.",
        f"**Most common first represented group:** **G{int(first.group)} — {first['name']}** "
        f"at **{first.first_activation_pct:.1f}%** of all runs.",
        f"**Most repeatedly overloaded line:** `{worst.line}` — overloaded in **{worst.overload_pct_of_runs:.1f}%** of runs; "
        f"95th-percentile loading **{worst.p95_loading_pct:.1f}%**.",
        f"**Runs with an overload left after the represented group actions:** **{unresolved:,} ({100*unresolved/runs:.1f}%)**. "
        "This includes overloads on lines not assigned to these five WDT group proxies, so it is a diagnostic rather than a solver failure.",
        "**Experiment isolation:** renewable availability is intentionally unbounded; mapped MEC is used only as a spatial weighting prior. "
        "Total generation equals total demand in every trial. SNSP/inertia, low-demand surplus and economic curtailment are disabled, so group dispatch-down is triggered only by thermal line loading.",
        "**Group 4 caveat:** All-NI membership is included, but the supplied NI-only network does not contain the Tandragee–Louth tie-lines, "
        "so Group 4 has no physical thermal trigger in this experiment yet.",
    ])


def run_constraint_group_stress(
    runs,
    seed,
    min_stress_pct,
    max_stress_pct,
    stress_thermal_scale,
    generation_concentration,
    demand_concentration,
    batch_size,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        runs = int(runs)
        lo = float(min_stress_pct)
        hi = float(max_stress_pct)
        if hi < lo:
            raise ValueError("Maximum generation stress must be greater than or equal to the minimum.")
        progress(0.0, desc=f"Preparing {runs:,} thermal stress trials")

        def _progress(frac, desc):
            progress(float(frac), desc=str(desc))

        cfg = StressTestConfig(
            runs=runs,
            seed=int(seed),
            min_generation_pct_of_mec=lo,
            max_generation_pct_of_mec=hi,
            thermal_scale=float(stress_thermal_scale),
            generation_concentration=float(generation_concentration),
            demand_concentration=float(demand_concentration),
            batch_size=int(batch_size),
            max_group_actions=5,
        )
        result = run_stress_test(NETWORK, MODEL, ASSETS, cfg, progress=_progress)
        progress(1.0, desc=f"Completed {runs:,}/{runs:,} runs")
        worst_runs = result.run_table.sort_values(
            ["initial_worst_loading_pct", "dispatch_down_mw"], ascending=False
        ).head(200).round(3)
        return (
            _stress_summary(result),
            _stress_group_figure(result.group_summary),
            _stress_line_figure(result.line_summary),
            _stress_network_map(result.line_summary),
            _stress_worst_case_map(result),
            _stress_dispatch_benchmark(result).round(3),
            result.group_catalogue.round(2),
            result.group_summary.round(3),
            result.line_summary.head(40).round(3),
            result.sequence_summary.head(25).round(3),
            worst_runs,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        blank = go.Figure().update_layout(template="plotly_white", title=f"Stress-test error: {e}")
        return (
            f"**Stress-test error:** {e}", blank, blank, blank, blank, pd.DataFrame(), GROUP_CATALOGUE.round(2),
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        )






_GROUP_COLORS = {1: "#ef4444", 2: "#3b82f6", 3: "#22c55e", 4: "#8b5cf6", 5: "#f59e0b"}
_GROUP_SYMBOLS = {1: "circle", 2: "square", 3: "diamond", 4: "circle-open", 5: "triangle-up"}
_GROUP_JITTER = {1: (-0.010, 0.006), 2: (0.010, 0.006), 3: (0.010, -0.006), 4: (0.0, 0.0), 5: (-0.010, -0.006)}


def _base_network_backdrop(fig):
    for _, r in NETWORK.lines.iterrows():
        if r.bus0 not in BUS_COORD.index or r.bus1 not in BUS_COORD.index:
            continue
        a, b = BUS_COORD.loc[r.bus0], BUS_COORD.loc[r.bus1]
        if not np.all(np.isfinite([a.lon, a.lat, b.lon, b.lat])):
            continue
        fig.add_trace(go.Scatter(
            x=[a.lon, b.lon], y=[a.lat, b.lat], mode="lines",
            line=dict(color="#d1d5db", width=1.3), hoverinfo="skip", showlegend=False,
        ))
    plot_buses = NETWORK.buses[np.isfinite(NETWORK.buses["lon"]) & np.isfinite(NETWORK.buses["lat"])].copy()
    fig.add_trace(go.Scatter(
        x=plot_buses.lon, y=plot_buses.lat, mode="markers",
        text=[f"Bus {b}" for b in plot_buses.bus], hovertemplate="%{text}<extra></extra>",
        marker=dict(size=4, color="#94a3b8", opacity=0.55, line=dict(width=0.3, color="white")),
        showlegend=False,
    ))


def _baseline_group_map(result):
    fig = go.Figure()
    if result is None or result.membership_table is None or result.membership_table.empty:
        return fig.update_layout(template="plotly_white", title="Run the NI baseline first")
    _base_network_backdrop(fig)
    renew = ASSETS[ASSETS["market_bucket"].isin(["wind", "solar"])].copy()
    renew = renew[np.isfinite(pd.to_numeric(renew["latitude"], errors="coerce")) & np.isfinite(pd.to_numeric(renew["longitude"], errors="coerce"))]
    fig.add_trace(go.Scatter(
        x=renew["longitude"], y=renew["latitude"], mode="markers",
        text=[f"{r.point_name}<br>{str(r.market_bucket).title()} · MEC {float(r.max_export_capacity_mw):.1f} MW" for r in renew.itertuples()],
        hovertemplate="%{text}<extra></extra>",
        marker=dict(size=7, color="#e5e7eb", opacity=0.55, line=dict(width=0.3, color="#9ca3af")),
        name="All mapped renewables",
    ))
    for gid in [1, 2, 3, 5, 4]:
        sub = result.membership_table[result.membership_table["group"].eq(gid)].copy()
        sub = sub[np.isfinite(sub["latitude"]) & np.isfinite(sub["longitude"])]
        if sub.empty:
            continue
        dx, dy = _GROUP_JITTER.get(gid, (0.0, 0.0))
        fig.add_trace(go.Scatter(
            x=sub["longitude"] + dx,
            y=sub["latitude"] + dy,
            mode="markers",
            text=[
                f"<b>{r.asset}</b><br>G{gid} — {r.group_name}<br>Bus {r.bus}<br>MEC {float(r.mec_mw):.1f} MW"
                for r in sub.itertuples()
            ],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=12 if gid != 4 else 10,
                color=_GROUP_COLORS[gid],
                opacity=0.88 if gid != 4 else 0.65,
                symbol=_GROUP_SYMBOLS[gid],
                line=dict(width=1.0 if gid != 4 else 1.8, color=_GROUP_COLORS[gid]),
            ),
            name=f"G{gid} — {SONI_GROUPS[gid]['name']}",
        ))
    fig.update_layout(
        title="Current SONI NI renewable constraint groups",
        template="plotly_white", height=720,
        margin=dict(l=10, r=10, t=65, b=35),
        xaxis=dict(title="Longitude", showgrid=False, zeroline=False),
        yaxis=dict(title="Latitude", showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
        hovermode="closest",
        annotations=[{
            "text": "Group 4 is All NI and overlaps all controllable renewable assets; it is shown as purple open circles.",
            "xref": "paper", "yref": "paper", "x": 0, "y": 1.06,
            "showarrow": False, "align": "left", "font": {"size": 11, "color": "#475569"},
        }],
    )
    return fig


def _baseline_line_map(result):
    fig = go.Figure()
    if result is None or result.physical.line_summary is None or result.physical.line_summary.empty:
        return fig.update_layout(template="plotly_white", title="Run the NI baseline first")
    ls = result.physical.line_summary.rename(columns={
        "line": "line",
        "overload_pct_before_group_actions": "overload_pct_of_runs",
        "max_initial_loading_pct": "max_loading_pct",
        "mean_initial_loading_pct": "p95_loading_pct",
    }).copy()
    # Reuse the stress-map renderer.  Here the hover's "p95" slot is labelled by
    # the renderer but contains mean initial loading, so create a more exact map below.
    meta = NETWORK.lines.merge(result.physical.line_summary, left_on="branch", right_on="line", how="left")
    mids=[]
    for _, r in meta.iterrows():
        if r.bus0 not in BUS_COORD.index or r.bus1 not in BUS_COORD.index:
            continue
        a,b=BUS_COORD.loc[r.bus0],BUS_COORD.loc[r.bus1]
        if not np.all(np.isfinite([a.lon,a.lat,b.lon,b.lat])):
            continue
        ov=float(pd.to_numeric(pd.Series([r.get("overload_pct_before_group_actions",0)]),errors="coerce").fillna(0).iloc[0])
        mx=float(pd.to_numeric(pd.Series([r.get("max_initial_loading_pct",0)]),errors="coerce").fillna(0).iloc[0])
        mn=float(pd.to_numeric(pd.Series([r.get("mean_initial_loading_pct",0)]),errors="coerce").fillna(0).iloc[0])
        color,width=_stress_frequency_style(ov)
        fig.add_trace(go.Scatter(x=[a.lon,b.lon],y=[a.lat,b.lat],mode="lines",line=dict(color=color,width=width),hoverinfo="skip",showlegend=False))
        mids.append(((a.lon+b.lon)/2,(a.lat+b.lat)/2,str(r.branch),ov,mn,mx,color))
    if mids:
        fig.add_trace(go.Scatter(
            x=[x[0] for x in mids], y=[x[1] for x in mids], mode="markers",
            text=[f"<b>{x[2]}</b><br>Initially overloaded in {x[3]:.1f}% of cases<br>Mean initial loading {x[4]:.1f}%<br>Maximum initial loading {x[5]:.1f}%" for x in mids],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=[10 if x[3]>0 else 6 for x in mids], color=[x[6] for x in mids], symbol="triangle-up", line=dict(width=0)),
            showlegend=False,
        ))
    fig.update_layout(
        title="Current-grid stress map before constraint-group actions",
        template="plotly_white", height=720, margin=dict(l=10,r=10,t=60,b=35),
        xaxis=dict(title="Longitude",showgrid=False,zeroline=False),
        yaxis=dict(title="Latitude",showgrid=False,zeroline=False,scaleanchor="x",scaleratio=1), hovermode="closest",
    )
    return fig


def _baseline_metric_table(result):
    m=result.metrics
    return pd.DataFrame([
        {"metric":"Physical model dispatch-down", "value_pct":m["physical_dispatch_down_pct"], "meaning":"Explicit NI-only operational/network model"},
        {"metric":"Fixed calibration residual", "value_pct":m["calibration_residual_pct"], "meaning":"Unmodelled effects; frozen for downstream work"},
        {"metric":"Calibrated total dispatch-down", "value_pct":m["calibrated_dispatch_down_pct"], "meaning":"Baseline headline dispatch-down"},
        {"metric":"Calibrated renewable utilisation", "value_pct":m["calibrated_renewable_utilisation_pct"], "meaning":"Available renewable energy accommodated"},
        {"metric":"Estimated transmission electrical efficiency", "value_pct":m["electrical_efficiency_pct"], "meaning":"Diagnostic I²R line-loss efficiency; separate from dispatch-down"},
        {"metric":"Represented security pass rate", "value_pct":m["represented_security_pass_pct"], "meaning":"Published corridor/security-state checks"},
        {"metric":"All-line N-1 pass rate", "value_pct":m["all_line_n1_pass_pct"], "meaning":"DC N-1 thermal diagnostic"},
    ])


def _baseline_dd_figure(result):
    m=result.metrics
    fig=go.Figure()
    fig.add_trace(go.Bar(
        x=["Physical model", "Calibration residual", "Calibrated baseline", "SONI 2024 actual"],
        y=[m["physical_dispatch_down_pct"], m["calibration_residual_pct"], m["calibrated_dispatch_down_pct"], SONI_2024_ALL_RENEWABLE_DD_PCT],
        name="Dispatch-down %",
    ))
    fig.update_layout(template="plotly_white", title="NI renewable dispatch-down baseline calibration", yaxis_title="Dispatch-down (% of available renewable energy)", height=430, margin=dict(l=55,r=20,t=55,b=50))
    return fig


def _baseline_summary(result):
    m=result.metrics
    cal_note=("matched" if m.get("calibration_exact") else "could not fully match")
    return "  \n".join([
        f"**Frozen NI operating cases:** **{int(m['runs']):,}**.",
        f"**Physical NI-only model:** dispatch-down **{m['physical_dispatch_down_pct']:.2f}%**; renewable utilisation **{m['physical_renewable_utilisation_pct']:.2f}%**.",
        f"**Calibration residual:** **{m['calibration_residual_pct']:.2f}%**. This is fixed per scenario and represents effects the supplied NI-only DC model cannot explicitly reproduce.",
        f"**Calibrated baseline:** dispatch-down **{m['calibrated_dispatch_down_pct']:.2f}%** and renewable utilisation **{m['calibrated_renewable_utilisation_pct']:.2f}%** — {cal_note} to the selected historical target.",
        f"**Network/group component actually represented by the model:** **{m['network_group_dispatch_down_pct']:.2f}%** of available renewable energy.",
        f"**Estimated electrical transmission efficiency:** **{m['electrical_efficiency_pct']:.2f}%**. This is an I²R diagnostic and is not the same thing as renewable utilisation.",
        f"**Security diagnostics:** represented published states **{m['represented_security_pass_pct']:.1f}%** secure; all-line N-1 thermal diagnostic **{m['all_line_n1_pass_pct']:.1f}%** secure after current-group actions.",
        "**Handoff rule:** a future annealer should keep the frozen operating cases and `calibration_residual_mw` unchanged, and alter only the group-policy/network dispatch-down component. This makes any claimed saving attributable to changed constraint groups rather than to re-calibration.",
    ])


def run_current_grid_baseline(
    base_runs, base_seed, base_demand_scale, base_thermal_scale, base_outage_exposure,
    base_target_dd, base_calibrate, progress=gr.Progress(track_tqdm=False),
):
    try:
        cfg=current_operational_config(
            runs=int(base_runs), seed=int(base_seed), demand_scale=float(base_demand_scale),
            thermal_scale=float(base_thermal_scale), planned_outage_exposure_pct=float(base_outage_exposure),
        )
        def _progress(frac, desc):
            progress(float(np.clip(frac,0,1)), desc=str(desc))
        result=run_baseline_replica(
            NETWORK, MODEL, ASSETS, cfg,
            target_dispatch_down_pct=float(base_target_dd), calibrate=bool(base_calibrate), progress=_progress,
        )
        progress(0.94, desc="Exporting frozen baseline handoff pack")
        handoff=export_handoff_pack(result, ROOT / "outputs", stem=f"ni_baseline_handoff_seed{int(base_seed)}_runs{int(base_runs)}")
        progress(1.0, desc="NI current-grid baseline complete")
        worst=result.calibrated_run_table.sort_values(["calibrated_dispatch_down_pct_of_case","worst_n1_loading_pct"],ascending=False).head(200).round(3)
        return (
            _baseline_summary(result), _baseline_dd_figure(result), _baseline_group_map(result), _baseline_line_map(result),
            _baseline_metric_table(result).round(4), result.calibration_table.round(4),
            result.physical.group_summary.round(4), result.physical.line_summary.head(40).round(3), worst, handoff,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        blank=go.Figure().update_layout(template="plotly_white",title=f"Baseline error: {e}")
        return (f"**Baseline error:** {e}", blank, blank, blank, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None)

def _stream_flow_frames(frames, particle_smoothness, particles_per_line, animation_speed_ms, title, annotation_text=None):
    """Yield ordinary gr.Plot figures so animation works reliably in Gradio."""
    if not frames:
        yield go.Figure().update_layout(template="plotly_white", title="Run a scenario first, then press Play flow")
        return
    phases = int(np.clip(particle_smoothness, 2, 30))
    particles = int(np.clip(particles_per_line, 1, 12))
    # Browser/server round-trips below ~60 ms are counterproductive; clamp gently.
    delay = max(0.06, float(animation_speed_ms) / 1000.0)
    for sf in frames:
        for p in range(phases):
            yield make_power_flow_frame(
                sf, BUS_COORD, phase=p / phases, title=title,
                max_particles_per_line=particles, annotation_text=annotation_text,
            )
            time.sleep(delay)


def play_dynamic_flow(frames, particle_smoothness, particles_per_line, animation_speed_ms, preset):
    yield from _stream_flow_frames(
        frames, particle_smoothness, particles_per_line, animation_speed_ms,
        f"Dynamic response — {preset}",
    )


def play_surplus_flow(frames, particle_smoothness, particles_per_line, animation_speed_ms, preset):
    yield from _stream_flow_frames(
        frames, particle_smoothness, particles_per_line, animation_speed_ms,
        f"Renewable surplus / low demand — {preset}",
        annotation_text=(
            "Moving dots follow solved MW flow. Orange diamonds = online thermal units; grey × = offline units. "
            "Line: grey <70% · amber 70–90% · orange 90–100% · red >100%. "
            "Curtailment means available wind/solar that the system does not accept; line I²R losses are separate."
        ),
    )


with gr.Blocks(title="Northern Ireland Grid Constraint Simulator") as demo:
    gr.Markdown(
        "## Northern Ireland grid flow, constraint & scenario simulator\n"
        "Use one market snapshot for static constraints, demand spikes, or renewable-surplus / low-demand experiments. "
        "Positive Moyle MW means import into NI; negative means export."
    )
    with gr.Row():
        market_file = gr.File(label="Optional market CSV", file_types=[".csv"], type="filepath")
        timestamp = gr.Dropdown(
            choices=[str(i) for i in range(len(NETWORK.load_profile))],
            value="0",
            label="Timestamp / built-in snapshot",
        )
    market_status = gr.Markdown("Using the built-in network load profile.")

    with gr.Tabs():
        with gr.Tab("Static constraint experiment"):
            with gr.Row():
                thermal_scale = gr.Slider(0.4, 1.2, value=1.0, step=0.01, label="All-branch thermal rating multiplier")
                override_line = gr.Dropdown(choices=[""] + LINE_CHOICES, value="", label="Optional single-line constraint")
                override_limit = gr.Number(value=0, label="Selected line limit (MW; 0 = normal rating)")
            with gr.Row():
                auto_redispatch = gr.Checkbox(value=True, label="Automatically redispatch to relieve overloads")
                allow_load_shed = gr.Checkbox(
                    value=True,
                    label="Allow high-penalty load shedding if redispatch cannot solve a radial constraint",
                )
                run_btn = gr.Button("Run static simulation", variant="primary")
            summary = gr.Markdown()
            with gr.Row():
                base_plot = gr.Plot(label="Baseline")
                constrained_plot = gr.Plot(label="Constrained")
            gr.Markdown("### Constrained-line results")
            line_table = gr.Dataframe(interactive=False)
            gr.Markdown("### Active generation / redispatch")
            dispatch_table = gr.Dataframe(interactive=False)

        with gr.Tab("SONI constraint-group stress test"):
            gr.Markdown(
                "### Monte-Carlo thermal stress test of the five NI WDT groups\n"
                "Runs thousands of balanced generation/demand states using the published SONI Northern Ireland constraint-group structure. "
                "This tab deliberately disables surplus, SNSP/inertia and economic curtailment: **thermal line ratings are the only dispatch-down trigger**. "
                "Generation availability is unbounded for stress testing; mapped MEC is used only to shape where generation tends to appear."
            )
            with gr.Row():
                stress_runs = gr.Slider(100, 50000, value=10000, step=100, label="Monte-Carlo runs")
                stress_seed = gr.Number(value=42, precision=0, label="Random seed (repeatability)")
                stress_batch = gr.Dropdown(choices=[100, 250, 500, 1000, 2000], value=500, label="Batch size (progress update interval)")
            with gr.Row():
                stress_min_pct = gr.Slider(10, 400, value=40, step=5, label="Minimum generation stress (% of mapped renewable MEC)")
                stress_max_pct = gr.Slider(20, 1000, value=120, step=5, label="Maximum generation stress (% of mapped renewable MEC)")
                stress_thermal_scale = gr.Slider(0.5, 1.5, value=1.0, step=0.01, label="Thermal rating multiplier")
            with gr.Accordion("Spatial-randomness controls", open=False):
                gr.Markdown(
                    "Lower concentration = more extreme geographic clustering from run to run. Higher concentration = patterns stay closer to the mapped MEC/load-share distribution."
                )
                with gr.Row():
                    stress_gen_conc = gr.Slider(2, 150, value=35, step=1, label="Generation spatial concentration")
                    stress_dem_conc = gr.Slider(5, 400, value=120, step=5, label="Demand spatial concentration")
            with gr.Row():
                stress_btn = gr.Button("Run thermal stress test", variant="primary")
                stress_stop = gr.Button("Stop")
            stress_summary = gr.Markdown(
                "Choose the number of runs and press **Run thermal stress test**. The Gradio progress indicator shows completed runs, throughput and ETA."
            )
            with gr.Row():
                stress_group_plot = gr.Plot(label="Constraint-group activation frequency")
                stress_line_plot = gr.Plot(label="Most stressed lines")
            with gr.Row():
                stress_map_plot = gr.Plot(label="Stress map across all runs")
                stress_worst_map = gr.Plot(label="Worst individual trial map")
            gr.Markdown(
                "### Dispatch-down percentage versus SONI history\n"
                "The model percentage is total renewable MW dispatched down divided by total pre-dispatch-down renewable MW across all Monte-Carlo trials. "
                "SONI 2024 is the main real-world comparison; 2023 is wind-only context."
            )
            stress_dispatch_benchmark = gr.Dataframe(interactive=False)
            gr.Markdown(
                "### Five SONI Northern Ireland groups represented in the experiment\n"
                "Group 4 membership is present, but its Tandragee–Louth thermal trigger cannot be solved physically until the cross-border tie-line is added to this NI-only network."
            )
            stress_catalogue = gr.Dataframe(value=GROUP_CATALOGUE.round(2), interactive=False)
            gr.Markdown("### Which group activates first / most often")
            stress_group_table = gr.Dataframe(interactive=False)
            gr.Markdown("### Lines hit hardest across repeated runs")
            stress_line_table = gr.Dataframe(interactive=False)
            gr.Markdown("### Most common group activation sequences")
            stress_sequence_table = gr.Dataframe(interactive=False)
            gr.Markdown("### 200 most severe individual trials")
            stress_worst_runs = gr.Dataframe(interactive=False)

        with gr.Tab("NI current-grid baseline"):
            gr.Markdown(
                "### Northern Ireland current-policy baseline emulator\n"
                "This is the handoff model: **no annealing or group optimisation is performed here**. It uses the five current/published NI renewable constraint groups, "
                "DC transmission flows, N-1 thermal security, a 75% SNSP-style ceiling, a two-machine NI synchronous floor, North-South and Moyle limits, "
                "minimum stable thermal output and planned-outage exposure. The aggregate dispatch-down result can be calibrated to SONI's latest published annual NI benchmark."
            )
            gr.Markdown(
                "**Important model boundary:** the supplied transmission NetCDF is an NI-only 2024 network and does not contain the full Republic of Ireland AC system or the physical Louth tie-line. "
                "The calibration residual is therefore kept explicit rather than silently weakening line ratings to force a match."
            )
            with gr.Row():
                base_runs=gr.Slider(500,20000,value=10000,step=500,label="Frozen operating cases")
                base_seed=gr.Number(value=42,precision=0,label="Random seed")
                base_demand_scale=gr.Slider(1.5,2.5,value=2.10,step=0.01,label="Annual demand scale")
            with gr.Row():
                base_thermal_scale=gr.Slider(0.8,1.2,value=1.0,step=0.01,label="Transmission thermal-rating multiplier")
                base_outage_exposure=gr.Slider(0,40,value=5,step=1,label="Key-corridor planned-outage exposure (% cases)")
                base_target_dd=gr.Dropdown(
                    choices=[25.5,29.6,16.9], value=25.5,
                    label="Aggregate calibration target (%)",
                    info="25.5 = SONI NI 2024 all renewables; 29.6 = NI wind; 16.9 = NI solar",
                )
            base_calibrate=gr.Checkbox(value=True,label="Apply explicit fixed residual so aggregate dispatch-down matches the selected SONI benchmark")
            with gr.Row():
                base_run_btn=gr.Button("Run NI baseline replica",variant="primary")
                base_stop_btn=gr.Button("Stop")
            base_summary=gr.Markdown("Press **Run NI baseline replica**. The same frozen scenario pack can then be handed to an external optimiser.")
            with gr.Row():
                base_dd_plot=gr.Plot(label="Dispatch-down calibration")
                base_group_map=gr.Plot(label="Current constraint groups")
            base_line_map=gr.Plot(label="Network stress map")
            gr.Markdown("### Baseline headline metrics")
            base_metrics_table=gr.Dataframe(interactive=False)
            gr.Markdown("### Calibration audit trail")
            base_calibration_table=gr.Dataframe(interactive=False)
            gr.Markdown("### Current SONI group activation / dispatch-down behaviour")
            base_group_table=gr.Dataframe(interactive=False)
            gr.Markdown("### Most stressed represented transmission lines")
            base_line_table=gr.Dataframe(interactive=False)
            gr.Markdown("### 200 highest-dispatch-down / highest-N-1-stress cases")
            base_worst_table=gr.Dataframe(interactive=False)
            base_handoff_file=gr.File(label="Frozen baseline handoff pack for external optimisation")

        with gr.Tab("Dynamic demand shock"):
            gr.Markdown(
                "### TV-pickup / tea-time demand spike\n"
                "Each point in the animation is a newly solved steady-state network. Start with the **TV tea-time spike** preset, "
                "then change the MW shock or apply it to one load bus to see which corridors react."
            )
            with gr.Row():
                preset = gr.Dropdown(
                    choices=["TV tea-time spike", "Very large NI-wide spike", "Local city-scale spike", "Demand collapse", "Custom"],
                    value="TV tea-time spike",
                    label="Scenario preset",
                )
                shock_mw = gr.Number(value=250.0, label="Final demand change (MW; negative = drop)")
                shock_scope = gr.Dropdown(
                    choices=["System-wide", "Single load bus"],
                    value="System-wide",
                    label="Shock location",
                )
            with gr.Row():
                target_bus = gr.Dropdown(
                    choices=LOAD_BUS_CHOICES,
                    value=LOAD_BUS_CHOICES[0] if LOAD_BUS_CHOICES else None,
                    label="Target load bus (used only for local shock)",
                )
                dynamic_steps = gr.Slider(4, 30, value=12, step=1, label="Solved ramp steps")
                response_mode = gr.Dropdown(
                    choices=["thermal_headroom", "slack_only"],
                    value="thermal_headroom",
                    label="Primary generation response",
                )
            with gr.Row():
                dynamic_thermal_scale = gr.Slider(0.5, 1.2, value=1.0, step=0.01, label="Thermal rating multiplier")
                dynamic_line = gr.Dropdown(choices=[""] + LINE_CHOICES, value="", label="Optional line to constrain during spike")
                dynamic_limit = gr.Number(value=0, label="Optional selected-line limit (MW)")
            with gr.Row():
                dynamic_redispatch = gr.Checkbox(value=True, label="Correct thermal overloads with redispatch")
                dynamic_load_shed = gr.Checkbox(value=False, label="Allow emergency load shedding")
                particle_smoothness = gr.Slider(3, 24, value=12, step=1, label="Particle smoothness (visual frames per solved state)")
                particles_per_line = gr.Slider(1, 10, value=5, step=1, label="Particles on busy lines")
                animation_speed = gr.Slider(30, 250, value=70, step=10, label="Visual frame duration (ms; lower = faster)")
                dynamic_btn = gr.Button("Run animated demand shock", variant="primary")
            dynamic_summary = gr.Markdown()
            dynamic_frames_state = gr.State([])
            dynamic_plot = gr.Plot(label="Animated power flow")
            with gr.Row():
                dynamic_play = gr.Button("▶ Play flow", variant="primary")
                dynamic_pause = gr.Button("⏸ Pause")
            gr.Markdown("### Scenario progression")
            dynamic_metrics = gr.Dataframe(interactive=False)
            gr.Markdown("### Lines with the highest peak loading during the spike")
            dynamic_peak_lines = gr.Dataframe(interactive=False)
            gr.Markdown("### Generator response from start to finish")
            dynamic_generators = gr.Dataframe(interactive=False)


        with gr.Tab("Renewable surplus / low demand"):
            gr.Markdown(
                "### Wind/solar surplus and curtailment\n"
                "Increase renewable availability, reduce demand, or combine both. With unit commitment enabled, whole synchronous thermal "
                "units can shut down as demand falls. The model keeps a configurable stability floor, then curtails wind/solar when further "
                "thermal shutdown is not allowed; transmission overloads can cause additional locational curtailment."
            )
            with gr.Row():
                surplus_preset = gr.Dropdown(
                    choices=[
                        "Windy low-demand night", "Extreme wind surplus", "Solar-rich low-demand noon",
                        "Wind + solar glut", "Low demand only", "Custom"
                    ],
                    value="Windy low-demand night",
                    label="Scenario preset",
                )
                wind_surge = gr.Number(value=350.0, label="Extra wind availability (MW)")
                solar_surge = gr.Number(value=0.0, label="Extra solar availability (MW)")
            with gr.Row():
                surplus_demand_change = gr.Number(value=-250.0, label="Demand change (MW; negative = lower demand)")
                surplus_scope = gr.Dropdown(
                    choices=["System-wide", "Single load bus"], value="System-wide", label="Demand-change location"
                )
                surplus_target_bus = gr.Dropdown(
                    choices=LOAD_BUS_CHOICES,
                    value=LOAD_BUS_CHOICES[0] if LOAD_BUS_CHOICES else None,
                    label="Target load bus (only for local demand change)",
                )
            with gr.Row():
                surplus_steps = gr.Slider(4, 30, value=12, step=1, label="Solved ramp steps")
                surplus_duration = gr.Slider(10, 240, value=60, step=5, label="Scenario ramp duration (minutes)")
                surplus_smoothness = gr.Slider(3, 24, value=12, step=1, label="Particle smoothness (visual frames per solved state)")
                surplus_particles = gr.Slider(1, 10, value=5, step=1, label="Particles on busy lines")
                surplus_speed = gr.Slider(30, 250, value=70, step=10, label="Visual frame duration (ms; lower = faster)")
            with gr.Row():
                commitment_enabled = gr.Checkbox(value=True, label="Switch synchronous thermal units on/off")
                max_nonsynchronous_share = gr.Slider(40, 100, value=75, step=1, label="Maximum non-synchronous share (%)")
                min_sync_units = gr.Slider(0, 8, value=2, step=1, label="Minimum synchronous thermal units online")
            with gr.Row():
                min_stable_output = gr.Slider(0, 50, value=20, step=1, label="Minimum stable output of an online thermal unit (% capacity)")
                shutdown_strategy = gr.Dropdown(
                    choices=["keep_current", "renewable_friendly"], value="keep_current",
                    label="Unit shutdown strategy",
                )
                gr.Markdown(
                    "**75% means:** wind/solar (plus positive HVDC import in this simplified calculation) can supply up to 75% "
                    "of the demand-plus-export basis. It is a configurable research assumption, not a hard-coded claim about current SONI rules."
                )
            with gr.Row():
                surplus_thermal_scale = gr.Slider(0.5, 1.2, value=1.0, step=0.01, label="Thermal rating multiplier")
                surplus_line = gr.Dropdown(choices=[""] + LINE_CHOICES, value="", label="Optional line constraint")
                surplus_limit = gr.Number(value=0, label="Optional selected-line limit (MW)")
            with gr.Row():
                surplus_redispatch = gr.Checkbox(
                    value=True,
                    label="Correct transmission overloads (may cause additional renewable curtailment)",
                )
                surplus_btn = gr.Button("Run animated renewable-surplus scenario", variant="primary")
            surplus_summary = gr.Markdown()
            surplus_frames_state = gr.State([])
            surplus_plot = gr.Plot(label="Animated renewable-surplus power flow")
            with gr.Row():
                surplus_play = gr.Button("▶ Play flow", variant="primary")
                surplus_pause = gr.Button("⏸ Pause")
            gr.Markdown("### Renewable utilisation / curtailment progression")
            surplus_metrics = gr.Dataframe(interactive=False)
            gr.Markdown(
                "### Historical SONI benchmark comparison\n"
                "The model's final renewable utilisation is compared with NI historical dispatch-down references. "
                "2024 is an all-renewables reference; 2023 is wind-only, so treat the latter as contextual rather than exactly like-for-like."
            )
            surplus_benchmark = gr.Dataframe(interactive=False)
            gr.Markdown("### Lines with highest loading during the surplus event")
            surplus_peak_lines = gr.Dataframe(interactive=False)
            gr.Markdown("### Final wind/solar potential versus accepted generation")
            surplus_assets = gr.Dataframe(interactive=False)
            gr.Markdown("### Final synchronous thermal unit commitment")
            surplus_commitment = gr.Dataframe(interactive=False)

    market_file.change(refresh_timestamps, inputs=[market_file], outputs=[timestamp, market_status])
    run_btn.click(
        run_simulation,
        inputs=[market_file, timestamp, thermal_scale, override_line, override_limit, auto_redispatch, allow_load_shed],
        outputs=[base_plot, constrained_plot, summary, line_table, dispatch_table],
    )
    stress_event = stress_btn.click(
        run_constraint_group_stress,
        inputs=[
            stress_runs, stress_seed, stress_min_pct, stress_max_pct, stress_thermal_scale,
            stress_gen_conc, stress_dem_conc, stress_batch,
        ],
        outputs=[
            stress_summary, stress_group_plot, stress_line_plot, stress_map_plot, stress_worst_map, stress_dispatch_benchmark, stress_catalogue,
            stress_group_table, stress_line_table, stress_sequence_table, stress_worst_runs,
        ],
        show_progress="full",
    )
    stress_stop.click(fn=None, cancels=[stress_event])
    base_event=base_run_btn.click(
        run_current_grid_baseline,
        inputs=[base_runs,base_seed,base_demand_scale,base_thermal_scale,base_outage_exposure,base_target_dd,base_calibrate],
        outputs=[
            base_summary,base_dd_plot,base_group_map,base_line_map,base_metrics_table,base_calibration_table,
            base_group_table,base_line_table,base_worst_table,base_handoff_file,
        ],
        show_progress="full",
    )
    base_stop_btn.click(fn=None,cancels=[base_event])
    preset.change(
        apply_dynamic_preset,
        inputs=[preset],
        outputs=[shock_mw, shock_scope, dynamic_steps],
    )
    dynamic_btn.click(
        run_dynamic_scenario,
        inputs=[
            market_file, timestamp, preset, shock_mw, shock_scope, target_bus, dynamic_steps,
            response_mode, dynamic_thermal_scale, dynamic_line, dynamic_limit,
            dynamic_redispatch, dynamic_load_shed, particle_smoothness, particles_per_line, animation_speed,
        ],
        outputs=[dynamic_plot, dynamic_summary, dynamic_metrics, dynamic_peak_lines, dynamic_generators, dynamic_frames_state],
    )

    dynamic_play_event = dynamic_play.click(
        play_dynamic_flow,
        inputs=[dynamic_frames_state, particle_smoothness, particles_per_line, animation_speed, preset],
        outputs=[dynamic_plot],
        show_progress="hidden",
        stream_every=0.05,
    )
    dynamic_pause.click(fn=None, cancels=[dynamic_play_event])

    surplus_preset.change(
        apply_surplus_preset,
        inputs=[surplus_preset],
        outputs=[wind_surge, solar_surge, surplus_demand_change, surplus_steps, surplus_duration],
    )
    surplus_btn.click(
        run_surplus_scenario,
        inputs=[
            market_file, timestamp, surplus_preset, wind_surge, solar_surge, surplus_demand_change,
            surplus_scope, surplus_target_bus, surplus_steps, surplus_duration, surplus_thermal_scale,
            surplus_line, surplus_limit, surplus_redispatch, commitment_enabled, max_nonsynchronous_share,
            min_sync_units, min_stable_output, shutdown_strategy, surplus_smoothness, surplus_particles, surplus_speed,
        ],
        outputs=[surplus_plot, surplus_summary, surplus_metrics, surplus_benchmark, surplus_peak_lines, surplus_assets, surplus_commitment, surplus_frames_state],
    )

    surplus_play_event = surplus_play.click(
        play_surplus_flow,
        inputs=[surplus_frames_state, surplus_smoothness, surplus_particles, surplus_speed, surplus_preset],
        outputs=[surplus_plot],
        show_progress="hidden",
        stream_every=0.05,
    )
    surplus_pause.click(fn=None, cancels=[surplus_play_event])


if __name__ == "__main__":
    demo.launch()
