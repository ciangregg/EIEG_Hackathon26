"""Plotly animation helpers for solved NI grid scenario frames."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ni_scenarios import ScenarioFrame


BAND_STYLES = {
    "light": ("#64748b", 1.8),
    "medium": ("#ca8a04", 2.8),
    "high": ("#ea580c", 3.8),
    "overload": ("#b91c1c", 5.0),
}


def _band(loading: float) -> str:
    if loading > 100.0:
        return "overload"
    if loading > 90.0:
        return "high"
    if loading > 70.0:
        return "medium"
    return "light"


def _segments_for_band(lines: pd.DataFrame, coords: pd.DataFrame, band: str) -> Tuple[list, list]:
    xs, ys = [], []
    for _, r in lines.iterrows():
        if _band(float(r.loading_pct)) != band:
            continue
        if r.bus0 not in coords.index or r.bus1 not in coords.index:
            continue
        a, b = coords.loc[r.bus0], coords.loc[r.bus1]
        if not np.all(np.isfinite([a.lon, a.lat, b.lon, b.lat])):
            continue
        xs.extend([float(a.lon), float(b.lon), None])
        ys.extend([float(a.lat), float(b.lat), None])
    return xs, ys


def _midpoints(lines: pd.DataFrame, coords: pd.DataFrame):
    xs, ys, text, colors, sizes = [], [], [], [], []
    for _, r in lines.iterrows():
        if r.bus0 not in coords.index or r.bus1 not in coords.index:
            continue
        a, b = coords.loc[r.bus0], coords.loc[r.bus1]
        if not np.all(np.isfinite([a.lon, a.lat, b.lon, b.lat])):
            continue
        direction = f"{r.bus0} → {r.bus1}" if r.flow_mw >= 0 else f"{r.bus1} → {r.bus0}"
        xs.append((float(a.lon) + float(b.lon)) / 2)
        ys.append((float(a.lat) + float(b.lat)) / 2)
        text.append(
            f"<b>{r.branch}</b><br>{direction}<br>Flow: {abs(float(r.flow_mw)):.1f} MW"
            f"<br>Limit: {float(r.thermal_limit_mw):.1f} MW"
            f"<br>Loading: {float(r.loading_pct):.1f}%"
            f"<br>Est. loss: {float(r.estimated_loss_mw):.3f} MW"
        )
        colors.append(BAND_STYLES[_band(float(r.loading_pct))][0])
        sizes.append(7 if r.loading_pct <= 100 else 11)
    return xs, ys, text, colors, sizes


def _bus_trace(frame: ScenarioFrame):
    buses = frame.buses[np.isfinite(frame.buses["lon"]) & np.isfinite(frame.buses["lat"])].copy()
    injection = buses["injection_mw"].astype(float)
    sizes = np.clip(5 + np.sqrt(injection.abs().clip(lower=0)) * 0.9, 5, 24)
    # Sources blue, sinks black, almost-balanced buses grey.
    colors = np.where(injection > 1e-6, "#2563eb", np.where(injection < -1e-6, "#111827", "#94a3b8"))
    text = [
        f"<b>Bus {r.bus}</b><br>Net injection: {r.injection_mw:+.1f} MW"
        f"<br>Voltage: {r.v_nom_kv:.0f} kV"
        for _, r in buses.iterrows()
    ]
    return go.Scatter(
        x=buses.lon,
        y=buses.lat,
        mode="markers",
        text=text,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(size=sizes, color=colors, opacity=0.9, line=dict(width=0.6, color="white")),
        name="Sources / sinks",
        showlegend=False,
    )


def _thermal_unit_trace(frame: ScenarioFrame, coords: pd.DataFrame):
    """Show individual committed/offline thermal units when commitment data exists."""
    assets = getattr(frame, "active_assets", None)
    if assets is None or not isinstance(assets, pd.DataFrame) or "committed" not in assets.columns:
        return go.Scatter(x=[], y=[], mode="markers", showlegend=False, hoverinfo="skip")
    thermal = assets[assets["market_bucket"].eq("thermal") & assets["mapped_bus"].notna()].copy()
    if thermal.empty:
        return go.Scatter(x=[], y=[], mode="markers", showlegend=False, hoverinfo="skip")

    thermal["_name"] = thermal.get("point_name", pd.Series(thermal.index.astype(str), index=thermal.index)).astype(str)
    thermal = thermal.sort_values(["mapped_bus", "_name"]).copy()
    xs, ys, text, colors, symbols, sizes = [], [], [], [], [], []
    for bus, group in thermal.groupby("mapped_bus", sort=True):
        bus = str(bus)
        if bus not in coords.index:
            continue
        c = coords.loc[bus]
        if not np.all(np.isfinite([c.lon, c.lat])):
            continue
        n = len(group)
        for k, (_, a) in enumerate(group.iterrows()):
            # Small deterministic fan-out prevents colocated station units from hiding each other.
            angle = 2.0 * np.pi * (k / max(1, n))
            radius = 0.012 if n > 1 else 0.0
            x = float(c.lon) + radius * np.cos(angle)
            y = float(c.lat) + radius * np.sin(angle)
            online = bool(a.get("committed", float(a.get("dispatch_mw", 0.0)) > 1e-6))
            dispatch = float(pd.to_numeric(pd.Series([a.get("dispatch_mw", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            min_stable = float(pd.to_numeric(pd.Series([a.get("min_stable_mw", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            action = str(a.get("commitment_action", "online" if online else "offline"))
            xs.append(x); ys.append(y)
            colors.append("#f97316" if online else "#9ca3af")
            symbols.append("diamond" if online else "x")
            sizes.append(11 if online else 8)
            text.append(
                f"<b>{a.get('_name', '')}</b><br>{'ONLINE' if online else 'OFFLINE'} ({action})"
                f"<br>Dispatch: {dispatch:.1f} MW<br>Minimum stable: {min_stable:.1f} MW<br>Bus: {bus}"
            )
    return go.Scatter(
        x=xs, y=ys, mode="markers", text=text, hovertemplate="%{text}<extra></extra>",
        marker=dict(size=sizes, color=colors, symbol=symbols, opacity=0.95, line=dict(width=0.7, color="white")),
        name="Thermal unit commitment", showlegend=False,
    )


def _particle_trace(lines: pd.DataFrame, coords: pd.DataFrame, phase: float, max_particles_per_line: int = 5):
    """Moving dots along each line, following the sign of solved MW flow."""
    xs, ys, sizes, text = [], [], [], []
    visible = lines[lines["abs_flow_mw"] > 2.0]
    if visible.empty:
        return go.Scatter(x=[], y=[], mode="markers", showlegend=False, hoverinfo="skip")
    scale = max(float(visible["abs_flow_mw"].quantile(0.9)), 1.0)
    for _, r in visible.iterrows():
        if r.bus0 not in coords.index or r.bus1 not in coords.index:
            continue
        a, b = coords.loc[r.bus0], coords.loc[r.bus1]
        if not np.all(np.isfinite([a.lon, a.lat, b.lon, b.lat])):
            continue
        # Reverse coordinate order when the solved flow is negative.
        x0, y0, x1, y1 = float(a.lon), float(a.lat), float(b.lon), float(b.lat)
        if float(r.flow_mw) < 0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        max_particles = int(np.clip(max_particles_per_line, 1, 12))
        flow_ratio = abs(float(r.flow_mw)) / scale
        n = int(np.clip(np.ceil(1.0 + flow_ratio * (max_particles - 1)), 1, max_particles))
        for k in range(n):
            t = (phase + k / n) % 1.0
            xs.append(x0 + (x1 - x0) * t)
            ys.append(y0 + (y1 - y0) * t)
            sizes.append(float(np.clip(4.0 + 5.0 * abs(float(r.flow_mw)) / scale, 4.0, 9.0)))
            text.append(f"{r.branch}: {abs(float(r.flow_mw)):.1f} MW")
    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        text=text,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(size=sizes, color="#0ea5e9", opacity=0.78, line=dict(width=0)),
        name="Power flow",
        showlegend=False,
    )


def _traces(frame: ScenarioFrame, coords: pd.DataFrame, phase: float, max_particles_per_line: int = 5):
    lines = frame.branches[frame.branches["type"].eq("line")].copy()
    traces = []
    for band in ["light", "medium", "high", "overload"]:
        xs, ys = _segments_for_band(lines, coords, band)
        color, width = BAND_STYLES[band]
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                showlegend=False,
                name=band,
            )
        )
    mx, my, mt, mc, ms = _midpoints(lines, coords)
    traces.append(
        go.Scatter(
            x=mx,
            y=my,
            mode="markers",
            text=mt,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=ms, color=mc, symbol="triangle-up", line=dict(width=0)),
            showlegend=False,
            name="Line details",
        )
    )
    traces.append(_bus_trace(frame))
    traces.append(_thermal_unit_trace(frame, coords))
    traces.append(_particle_trace(lines, coords, phase, max_particles_per_line=max_particles_per_line))
    return traces



def make_power_flow_frame(
    frame: ScenarioFrame,
    bus_coords: pd.DataFrame,
    phase: float = 0.0,
    title: str = "NI grid power flow",
    max_particles_per_line: int = 5,
    annotation_text: Optional[str] = None,
) -> go.Figure:
    """Build one ordinary Plotly figure for a single solved state/particle phase.

    This is intentionally frame-free so it works reliably inside Gradio's
    ``gr.Plot`` component. Animation is produced by streaming successive
    figures from the Gradio callback rather than by Plotly's internal frame
    engine.
    """
    phase = float(phase) % 1.0
    max_particles = int(np.clip(max_particles_per_line, 1, 12))
    fig = go.Figure(data=_traces(frame, bus_coords, phase, max_particles_per_line=max_particles))
    scope_note = frame.scenario.timestamp
    fig.update_layout(
        title=f"{title}<br><sup>{scope_note} · solved step {frame.step}</sup>",
        template="plotly_white",
        height=720,
        margin=dict(l=10, r=10, t=75, b=45),
        xaxis=dict(title="Longitude", showgrid=False, zeroline=False),
        yaxis=dict(title="Latitude", showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        hovermode="closest",
        uirevision="keep-view",
        annotations=[
            {
                "text": annotation_text or (
                    "Line: grey <70% · amber 70–90% · orange 90–100% · red >100%. "
                    "Blue buses inject; dark buses consume. Moving dots follow solved MW flow direction. "
                    "When commitment is enabled, orange diamonds are online thermal units and grey × markers are offline units."
                ),
                "xref": "paper", "yref": "paper", "x": 0, "y": 1.04,
                "showarrow": False, "align": "left",
                "font": {"size": 11, "color": "#475569"},
            }
        ],
    )
    return fig

def make_power_flow_animation(
    frames: List[ScenarioFrame],
    bus_coords: pd.DataFrame,
    title: str = "Dynamic NI grid response",
    particle_phases: int = 12,
    max_particles_per_line: int = 5,
    frame_duration_ms: int = 70,
    slider_labels: Optional[List[str]] = None,
    slider_prefix: str = "Demand change: ",
    annotation_text: Optional[str] = None,
) -> go.Figure:
    """Build a compact Plotly animation with solved flow direction and loading.

    ``slider_labels`` allows non-demand scenarios (for example renewable-surplus
    experiments) to reuse the same physically solved directional-flow animation.
    """
    if not frames:
        return go.Figure().update_layout(title="No scenario frames")
    phases = int(np.clip(particle_phases, 1, 30))
    max_particles = int(np.clip(max_particles_per_line, 1, 12))
    duration = int(np.clip(frame_duration_ms, 30, 1500))

    plotly_frames = []
    for sf in frames:
        for p in range(phases):
            phase = p / phases
            name = f"s{sf.step:02d}-p{p:02d}"
            plotly_frames.append(go.Frame(name=name, data=_traces(sf, bus_coords, phase, max_particles_per_line=max_particles)))

    initial = _traces(frames[0], bus_coords, 0.0, max_particles_per_line=max_particles)
    fig = go.Figure(data=initial, frames=plotly_frames)

    # One slider marker per physical scenario step, jumping to phase 0.
    slider_steps = []
    for i, sf in enumerate(frames):
        if slider_labels is not None and i < len(slider_labels):
            label = str(slider_labels[i])
        else:
            label = f"{sf.actual_demand_change_mw:+.0f} MW"
        slider_steps.append(
            {
                "args": [[f"s{sf.step:02d}-p00"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": label,
                "method": "animate",
            }
        )

    final = frames[-1]
    scope_note = final.scenario.timestamp
    fig.update_layout(
        title=f"{title}<br><sup>{scope_note}</sup>",
        template="plotly_white",
        height=720,
        margin=dict(l=10, r=10, t=75, b=80),
        xaxis=dict(title="Longitude", showgrid=False, zeroline=False),
        yaxis=dict(title="Latitude", showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        hovermode="closest",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": -0.07,
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": duration, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.22,
                "len": 0.75,
                "y": -0.055,
                "currentvalue": {"prefix": slider_prefix},
                "steps": slider_steps,
            }
        ],
        annotations=[
            {
                "text": annotation_text or "Line: grey <70% · amber 70–90% · orange 90–100% · red >100%. Blue buses inject; dark buses consume. Moving dots follow solved flow direction.",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.04,
                "showarrow": False,
                "align": "left",
                "font": {"size": 11, "color": "#475569"},
            }
        ],
    )
    return fig
