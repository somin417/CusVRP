#!/usr/bin/env python3
"""
Generate a single Folium map with radio-button (layer control) to toggle
baseline vs best (ALSM) routes.

Outputs:
  outputs/map_compare.html
"""

import json
from pathlib import Path
import folium


def load_json(path: Path):
    return json.loads(path.read_text())


def add_solution_layer(m, name, solution, stops, depots, color):
    fg = folium.FeatureGroup(name=name, overlay=False, control=True, show=False)

    depot_lookup = {d.get("id", "dc"): (d["lat"], d["lon"]) for d in depots}
    # depots
    for d in depots:
        folium.Marker(
            (d["lat"], d["lon"]),
            icon=folium.Icon(color="blue", icon="home"),
            tooltip=f"{name} - {d.get('id', 'dc')}",
        ).add_to(fg)

    routes_by_dc = solution.get("routes_by_dc", {})
    colors = ["red", "green", "purple", "orange", "darkblue", "darkred"]

    for dc_idx, (dc_id, routes) in enumerate(routes_by_dc.items()):
        route_color = colors[dc_idx % len(colors)] if color is None else color
        depot_coord = depot_lookup.get(dc_id) or next(iter(depot_lookup.values()), None)
        for r in routes:
            coords = []
            if depot_coord:
                coords.append(depot_coord)
            for sid in r.get("ordered_stop_ids", []):
                s = stops.get(sid)
                if s:
                    coords.append((s["lat"], s["lon"]))
            if coords:
                folium.PolyLine(
                    coords,
                    color=route_color,
                    weight=3,
                    opacity=0.8,
                    tooltip=f"{name} - {dc_id} - {r.get('vehicle_id', 'veh')}",
                ).add_to(fg)
            for sid in r.get("ordered_stop_ids", []):
                s = stops.get(sid)
                if s:
                    folium.CircleMarker(
                        (s["lat"], s["lon"]),
                        radius=3,
                        color=route_color,
                        fill=True,
                        fill_opacity=0.7,
                        tooltip=f"{name}: {sid}",
                    ).add_to(fg)

    fg.add_to(m)


def main():
    base_path = Path(__file__).parent.parent
    out = base_path / "outputs"

    baseline = load_json(out / "baseline.json")
    best = load_json(out / "best_solutions/seed42_n50_daejeon_best.json")["solution"]

    stops = baseline.get("stops_dict", {})
    depots = baseline.get("depots", [])

    # center
    coords = [(s["lat"], s["lon"]) for s in stops.values() if "lat" in s and "lon" in s]
    center = (
        sum(x for x, _ in coords) / len(coords),
        sum(y for _, y in coords) / len(coords),
    ) if coords else (36.35, 127.39)

    m = folium.Map(location=center, zoom_start=12)

    add_solution_layer(m, "Baseline", baseline, stops, depots, color="gray")
    add_solution_layer(m, "Best (ALSM)", best, stops, depots, color="red")

    folium.LayerControl(collapsed=False).add_to(m)

    out_file = out / "map_compare.html"
    m.save(out_file)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
