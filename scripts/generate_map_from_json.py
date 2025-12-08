#!/usr/bin/env python3
"""
Generate interactive map from baseline/improved solution JSON files.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vrp_fairness.map_folium import save_route_map_html


def main():
    parser = argparse.ArgumentParser(description="Generate map from solution JSON files")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline solution JSON file")
    parser.add_argument("--improved", type=str, default=None, help="Improved solution JSON file (optional)")
    parser.add_argument("--output", type=str, default=None, help="Output HTML file (default: auto-generated)")
    parser.add_argument("--tiles", type=str, default="OpenStreetMap", help="Map tile provider")
    parser.add_argument("--toggle-mode", type=str, default="radio", choices=["radio", "checkbox"], help="Toggle mode")
    
    args = parser.parse_args()
    
    # Load baseline solution
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"Error: Baseline file not found: {baseline_path}")
        sys.exit(1)
    
    print(f"Loading baseline solution from: {baseline_path}")
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    # Load improved solution if provided
    improved = None
    if args.improved:
        improved_path = Path(args.improved)
        if not improved_path.exists():
            print(f"Warning: Improved file not found: {improved_path}")
        else:
            print(f"Loading improved solution from: {improved_path}")
            with open(improved_path, 'r') as f:
                improved = json.load(f)
    
    # Extract depots from solution
    depots = baseline.get("depots", [])
    if not depots:
        # Try to infer from routes_by_dc keys and stops_dict
        routes_by_dc = baseline.get("routes_by_dc", {})
        stops_dict = baseline.get("stops_dict", {})
        
        if routes_by_dc:
            print("Warning: No depots in solution, inferring from routes and stops...")
            depots = []
            for dc_id in routes_by_dc.keys():
                # Try to find coordinates from first route's geometry or first stop
                routes = routes_by_dc[dc_id]
                lat, lon = 36.35, 127.38  # Default Daejeon
                
                if routes:
                    route = routes[0]
                    # Try to get from geometry if available
                    if "geometry" in route:
                        # Geometry might be encoded polyline - skip for now
                        pass
                    
                    # Try to get from first stop
                    stop_ids = route.get("ordered_stop_ids", [])
                    if stop_ids and stop_ids[0] in stops_dict:
                        first_stop = stops_dict[stop_ids[0]]
                        # Use first stop as approximate depot location
                        lat = first_stop.get("lat", 36.35)
                        lon = first_stop.get("lon", 127.38)
                        print(f"  Using first stop location for {dc_id}: ({lat}, {lon})")
                
                depots.append({
                    "id": dc_id,
                    "lat": lat,
                    "lon": lon
                })
    
    if not depots:
        print("Error: Could not determine depot locations")
        sys.exit(1)
    
    # Convert depots to DC format for map
    class DC:
        def __init__(self, dc_id, lat, lon):
            self.id = dc_id
            self.lat = lat
            self.lon = lon
    
    dcs = [DC(d["id"], d["lat"], d["lon"]) for d in depots]
    
    # Get stops_by_id from solution
    stops_by_id = baseline.get("stops_dict", {})
    if not stops_by_id:
        print("Warning: No stops_dict in baseline, trying to extract from routes...")
        # Try to extract from routes
        stops_by_id = {}
        routes_by_dc = baseline.get("routes_by_dc", {})
        for dc_id, routes in routes_by_dc.items():
            for route in routes:
                for stop_id in route.get("ordered_stop_ids", []):
                    if stop_id not in stops_by_id:
                        # Create placeholder stop (we don't have coordinates)
                        stops_by_id[stop_id] = {
                            "lat": 36.35,  # Default
                            "lon": 127.38,
                            "demand": 1
                        }
    
    if not stops_by_id:
        print("Error: Could not determine stop locations")
        sys.exit(1)
    
    print(f"Found {len(dcs)} depots and {len(stops_by_id)} stops")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate from baseline filename
        baseline_name = baseline_path.stem
        output_dir = baseline_path.parent / "maps"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{baseline_name}_map.html"
    
    print(f"Generating map: {output_path}")
    
    # Generate map
    try:
        save_route_map_html(
            baseline_solution=baseline,
            improved_solution=improved,
            dcs=dcs,
            stops_by_id=stops_by_id,
            out_html=str(output_path),
            tiles=args.tiles,
            toggle_mode=args.toggle_mode
        )
        print(f"\n✓ Map generated successfully!")
        print(f"  File: {output_path.absolute()}")
        print(f"  Access: http://localhost:8080/{output_path.name}")
    except Exception as e:
        print(f"\n✗ Error generating map: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

