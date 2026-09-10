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

ROOT = Path(__file__).resolve().parent
NETWORK_FILE = ROOT / "data" / "SV2024_northern_ireland.nc"
ASSET_FILE = ROOT / "data" / "northern_ireland_spatial_supply_and_interconnector_points.csv"

NETWORK = read_network_nc(NETWORK_FILE)
ASSETS = map_assets_to_buses(read_spatial_assets(ASSET_FILE), NETWORK.buses)
MODEL = DCGridModel(NETWORK)
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
        return initial_plot, _surplus_summary(frames), metrics, peak_lines, assets, commitment_units, frames
    except Exception as e:
        import traceback
        traceback.print_exc()
        blank = go.Figure().update_layout(template="plotly_white", title=f"Renewable surplus error: {e}")
        return blank, f"**Renewable surplus error:** {e}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []


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
        outputs=[surplus_plot, surplus_summary, surplus_metrics, surplus_peak_lines, surplus_assets, surplus_commitment, surplus_frames_state],
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
