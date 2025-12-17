"""
Main experiment runner for VRP fairness improvement.
"""

import argparse
import json
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

from .config import ExperimentConfig, get_city_bounds, get_city_config
from .data import generate_random_stops, load_stops_from_csv, load_stops_from_gpkg, Stop
from .inavi import iNaviCache, PolylineCache, get_leg_route, RouteLeg
from .vroom_vrp import solve_multi_depot, solve_single_depot
from .assignment import assign_stops_to_depots, create_osrm_time_provider
from .metrics import calculate_solution_metrics, metrics_to_dict
from .local_search import FairnessLocalSearch
from .plotting import plot_routes_baseline_vs_improved, plot_waiting_time_histograms, extract_waiting_times
from .map_folium import save_route_map_html, save_multi_solution_map_html

# Optional plotly for interactive plots
try:
    import plotly.graph_objects as go
except ImportError:
    go = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def attach_wait_and_travel(solution: Dict[str, Any], stops_dict: Dict[str, Any], time_provider, distance_provider) -> Dict[str, Any]:
    """
    Compute waiting map and per-leg travel times for a solution and attach to the solution dict.
    Stores:
      - waiting_times: {stop_id: arrival_time}
      - travel_times: list of {"from": prev_id, "to": stop_id, "time": travel_time, "dc": dc_id, "vehicle": vehicle_id}
    """
    from .objectives import compute_waiting_times, compute_Z1, compute_Z2, compute_Z3_MAD

    waiting = compute_waiting_times(solution, stops_dict, time_provider)
    travel_records = []

    routes = []
    if "routes_by_dc" in solution:
        for dc_id, dc_routes in solution["routes_by_dc"].items():
            for r in dc_routes:
                routes.append((dc_id, r))
    elif "routes" in solution:
        routes = [(None, r) for r in solution["routes"]]

    # depot lookup
    depots = solution.get("depots", [])
    depot_lookup = {d.get("id", "depot"): d for d in depots} if isinstance(depots, list) else {}

    for dc_id, route in routes:
        stop_ids = route.get("ordered_stop_ids", [])
        if not stop_ids:
            continue
        depot_id = route.get("dc_id") or route.get("depot_id") or dc_id or (list(depot_lookup.keys())[0] if depot_lookup else "depot")
        prev_id = depot_id
        for sid in stop_ids:
            if sid == depot_id:
                continue
            t = time_provider(prev_id, sid)
            travel_records.append({
                "from": prev_id,
                "to": sid,
                "time": t,
                "dc": dc_id,
                "vehicle": route.get("vehicle_id")
            })
            prev_id = sid

    solution["waiting_times"] = waiting
    solution["travel_times"] = travel_records
    # Also store objectives for convenience
    solution["objectives"] = {
        "Z1": compute_Z1(waiting, stops_dict),
        "Z2": compute_Z2(solution, distance_provider, time_provider, False),
        "Z3_MAD": compute_Z3_MAD(waiting, stops_dict),
    }
    return solution


def build_time_matrix_dict(
    locations: List[Dict[str, Any]],
    cache: iNaviCache
) -> Dict[tuple, int]:
    """
    Build time matrix as dictionary keyed by (stop1_id, stop2_id).
    
    REFACTORED: Now uses get_leg_route() to ensure all distances/times come from iNavi road routing.
    
    Args:
        locations: List of location dicts with 'name', 'lat'/'lon' or 'coordinates'
        cache: iNaviCache instance
    
    Returns:
        Dictionary mapping (stop1_id, stop2_id) to travel time (from iNavi road routing)
    """
    time_matrix = {}
    n = len(locations)
    total_pairs = n * (n - 1)
    completed = 0
    
    for i in range(n):
        for j in range(n):
            if i != j:
                loc1 = locations[i]
                loc2 = locations[j]
                
                # Handle both 'lat'/'lon' and 'coordinates' formats
                if 'lat' in loc1 and 'lon' in loc1:
                    lat1, lon1 = loc1['lat'], loc1['lon']
                elif 'coordinates' in loc1:
                    lat1, lon1 = loc1['coordinates']
                else:
                    continue
                
                if 'lat' in loc2 and 'lon' in loc2:
                    lat2, lon2 = loc2['lat'], loc2['lon']
                elif 'coordinates' in loc2:
                    lat2, lon2 = loc2['coordinates']
                else:
                    continue
                
                # Use get_leg_route() to ensure iNavi road routing
                leg = get_leg_route(
                    origin=(lat1, lon1),
                    dest=(lat2, lon2),
                    origin_id=loc1['name'],
                    dest_id=loc2['name'],
                    cache=cache
                )
                time_matrix[(loc1['name'], loc2['name'])] = int(leg.travel_time_s)
                
                completed += 1
                if completed % 100 == 0:
                    logger.info(f"  Progress: {completed}/{total_pairs} pairs ({100*completed/total_pairs:.1f}%)")
    
    logger.info(f"  Completed: {completed}/{total_pairs} pairs")
    return time_matrix


def run_baseline(
    config: ExperimentConfig,
    stops_by_dc: Dict[str, List[Stop]],
    cache: iNaviCache
) -> Dict[str, Any]:
    """
    Run baseline VRP for each DC.
    
    Returns:
        Dictionary with routes_by_dc and overall metrics
    """
    all_routes = []
    all_stops_dict = {}
    all_time_matrix = {}
    
    for dc in config.dc_list:
        logger.info(f"Processing {dc.id} ({dc.name or dc.id})")
        
        stops = stops_by_dc.get(dc.id, [])
        if not stops:
            logger.warning(f"No stops assigned to {dc.id}")
            continue
        
        # Build locations list
        depot = {
            "name": dc.id,
            "coordinates": [dc.lat, dc.lon],
            "demand": 0
        }
        
        locations = [depot] + [
            {
                "name": stop.id,
                "lat": stop.lat,
                "lon": stop.lon,
                "coordinates": [stop.lat, stop.lon],
                "demand": stop.demand
            }
            for stop in stops
        ]
        
        # Build time matrix for this DC
        dc_time_matrix = build_time_matrix_dict(locations, cache)
        all_time_matrix.update(dc_time_matrix)
        
        # Store stops in global dict
        for stop in stops:
            all_stops_dict[stop.id] = {
                "lat": stop.lat,
                "lon": stop.lon,
                "service_time": stop.service_time,
                "demand": stop.demand,
                "households": stop.demand  # Use demand as households (A26 value from GPKG)
            }
        
        # Call /vrp
        vehicles = [{"capacity": 100} for _ in range(config.vehicles_per_dc)]
        
        # Build distance and time matrices
        distance_matrix, time_matrix = cache.build_distance_matrix(locations)
        
        response = call_vrp(
            depot=depot,
            stops=[loc for loc in locations[1:]],  # Exclude depot
            vehicles=vehicles,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            time_limit_seconds=30.0
        )
        
        parsed = parse_vrp_response(response)
        
        # Add DC prefix to routes and build polylines using get_leg_route()
        # REFACTORED: All route legs now use get_leg_route() to ensure iNavi road routing
        for route in parsed["routes"]:
            route["dc_id"] = dc.id
            
            # Build polylines for route legs using get_leg_route() (ensures iNavi road routing)
            if "legs" not in route and "polyline" not in route:
                route["legs"] = []
                stop_ids = route.get("ordered_stop_ids", route.get("stops", []))
                
                if not stop_ids:
                    continue
                
                # Add DC -> first stop leg
                first_stop_id = stop_ids[0]
                first_stop = next((s for s in stops if s.id == first_stop_id), None)
                if first_stop:
                    leg = get_leg_route(
                        origin=(dc.lat, dc.lon),
                        dest=(first_stop.lat, first_stop.lon),
                        origin_id=dc.id,
                        dest_id=first_stop_id,
                        cache=cache
                    )
                    route["legs"].append({
                        "from": leg.origin_id,
                        "to": leg.dest_id,
                        "polyline": leg.polyline,
                        "travel_time_s": leg.travel_time_s,
                        "travel_distance_m": leg.travel_distance_m
                    })
                
                # Add stop -> stop legs
                for i in range(len(stop_ids) - 1):
                    from_id = stop_ids[i]
                    to_id = stop_ids[i + 1]
                    
                    from_stop = next((s for s in stops if s.id == from_id), None)
                    to_stop = next((s for s in stops if s.id == to_id), None)
                    
                    if not from_stop or not to_stop:
                        continue
                    
                    # Use get_leg_route() to get iNavi road routing with polyline
                    leg = get_leg_route(
                        origin=(from_stop.lat, from_stop.lon),
                        dest=(to_stop.lat, to_stop.lon),
                        origin_id=from_id,
                        dest_id=to_id,
                        cache=cache
                    )
                    route["legs"].append({
                        "from": leg.origin_id,
                        "to": leg.dest_id,
                        "polyline": leg.polyline,
                        "travel_time_s": leg.travel_time_s,
                        "travel_distance_m": leg.travel_distance_m
                    })
                
                # Add last stop -> DC leg
                last_stop_id = stop_ids[-1]
                last_stop = next((s for s in stops if s.id == last_stop_id), None)
                if last_stop:
                    leg = get_leg_route(
                        origin=(last_stop.lat, last_stop.lon),
                        dest=(dc.lat, dc.lon),
                        origin_id=last_stop_id,
                        dest_id=dc.id,
                        cache=cache
                    )
                    route["legs"].append({
                        "from": leg.origin_id,
                        "to": leg.dest_id,
                        "polyline": leg.polyline,
                        "travel_time_s": leg.travel_time_s,
                        "travel_distance_m": leg.travel_distance_m
                    })
            
            all_routes.append(route)
        
        logger.info(f"  Generated {len(parsed['routes'])} routes for {dc.id}")
    
    # Calculate overall metrics
    metrics = calculate_solution_metrics(
        all_routes,
        all_stops_dict,
        all_time_matrix,
        depot_id=config.dc_list[0].id if config.dc_list else "Depot"
    )
    
    # Group routes by DC
    routes_by_dc = {}
    for route in all_routes:
        dc_id = route.get("dc_id", "Unknown")
        if dc_id not in routes_by_dc:
            routes_by_dc[dc_id] = []
        routes_by_dc[dc_id].append(route)
    
    return {
        "routes_by_dc": routes_by_dc,
        "routes": all_routes,
        "metrics": metrics_to_dict(metrics),
        "stops_dict": all_stops_dict,
        "time_matrix": {f"{k[0]},{k[1]}": v for k, v in all_time_matrix.items()}  # JSON serializable
    }


def run_improvement(
    baseline_solution: Dict[str, Any],
    config: ExperimentConfig,
    cache: iNaviCache
) -> Dict[str, Any]:
    """
    Run fairness improvement algorithm.
    
    Returns:
        Improved solution with metrics
    """
    logger.info("Running fairness improvement algorithm...")
    
    # Reconstruct time matrix from saved format
    time_matrix = {}
    for k, v in baseline_solution["time_matrix"].items():
        # Parse key format "stop1,stop2"
        parts = k.split(",")
        if len(parts) == 2:
            time_matrix[(parts[0], parts[1])] = v
    
    # Run improvement for each DC
    improved_routes = []
    all_stops_dict = baseline_solution["stops_dict"]
    
    for dc_id, routes in baseline_solution["routes_by_dc"].items():
        logger.info(f"Improving routes for {dc_id}...")
        
        ls = FairnessLocalSearch(
            routes=routes,
            stops_dict=all_stops_dict,
            time_matrix=time_matrix,
            baseline_cost=baseline_solution["metrics"]["total_cost"],
            eps=config.eps,
            lambda_balance=config.lambda_balance,
            depot_id=dc_id
        )
        
        improved_dc_routes, metrics, iterations = ls.improve(max_iters=config.max_iters)
        
        for route in improved_dc_routes:
            route["dc_id"] = dc_id
            # Remove geometry from improved routes - it's from baseline and incorrect
            # The map will recalculate from stop order
            if "geometry" in route:
                del route["geometry"]
            if "legs" in route:
                del route["legs"]
            improved_routes.append(route)
        
        logger.info(f"  Improved {dc_id}: {iterations} iterations, W_max={metrics.W_max:.1f}s")
    
    # Reconstruct full time matrix for final metrics
    full_time_matrix = {}
    for k, v in baseline_solution["time_matrix"].items():
        parts = k.split(",")
        if len(parts) == 2:
            full_time_matrix[(parts[0], parts[1])] = v
    
    # Calculate overall improved metrics
    improved_metrics = calculate_solution_metrics(
        improved_routes,
        all_stops_dict,
        full_time_matrix,
        depot_id=config.dc_list[0].id if config.dc_list else "Depot"
    )
    
    # Group by DC
    routes_by_dc = {}
    for route in improved_routes:
        dc_id = route.get("dc_id", "Unknown")
        if dc_id not in routes_by_dc:
            routes_by_dc[dc_id] = []
        routes_by_dc[dc_id].append(route)
    
    # Include stops_dict and depots from baseline for consistency
    depots = baseline_solution.get("depots", [])
    
    return {
        "routes_by_dc": routes_by_dc,
        "routes": improved_routes,
        "metrics": metrics_to_dict(improved_metrics),
        "stops_dict": all_stops_dict,  # Include stops_dict for scoring/plotting
        "depots": depots  # Include depots for scoring/plotting
    }


def save_results(
    baseline: Dict[str, Any],
    improved: Optional[Dict[str, Any]],
    config: ExperimentConfig,
    method: str = "local",
    operator_mode: str = "fixed",
    force: bool = False
) -> None:
    """Save results to JSON and CSV files in organized directories."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create organized subdirectories
    solutions_dir = output_dir / "solutions"
    data_dir = output_dir / "data"
    solutions_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    # Save baseline
    baseline_file = solutions_dir / "baseline.json"
    if baseline_file.exists() and not force:
        logger.warning(f"Baseline file already exists: {baseline_file}")
        logger.warning("Use --force to overwrite, or the file will be skipped")
    else:
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        logger.info(f"Saved baseline to {baseline_file}")
    
    # Save baseline metrics CSV
    metrics_file = data_dir / "baseline_metrics.csv"
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in baseline["metrics"].items():
            writer.writerow([key, value])
    logger.info(f"Saved baseline metrics to {metrics_file}")
    
    # Save improved if available
    if improved:
        # Save local results to local.json, CTS results to cts_solution.json, other proposed to ALNS_MAD.json
        if method == "local":
            improved_file = solutions_dir / "local.json"
            comparison_file = data_dir / "baseline_vs_local_comparison.csv"
        elif method == "proposed" and operator_mode == "cts":
            improved_file = solutions_dir / "cts_solution.json"
            comparison_file = data_dir / "baseline_vs_cts_comparison.csv"
        else:
            improved_file = solutions_dir / "ALNS_MAD.json"
            comparison_file = data_dir / "baseline_vs_alns_mad_comparison.csv"
        if improved_file.exists() and not force:
            logger.warning(f"Improved solution file already exists: {improved_file}")
            logger.warning("Use --force to overwrite, or the file will be skipped")
        else:
            with open(improved_file, 'w') as f:
                json.dump(improved, f, indent=2)
            logger.info(f"Saved improved solution to {improved_file}")
        
        # Save comparison CSV
        with open(comparison_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Baseline", "Improved", "Change %"])
            
            baseline_metrics = baseline["metrics"]
            improved_metrics = improved["metrics"]
            
            for key in baseline_metrics:
                baseline_val = baseline_metrics[key]
                improved_val = improved_metrics.get(key, 0)
                
                if isinstance(baseline_val, (int, float)) and baseline_val != 0:
                    change_pct = ((improved_val - baseline_val) / baseline_val) * 100
                else:
                    change_pct = 0.0
                
                writer.writerow([key, baseline_val, improved_val, f"{change_pct:.2f}%"])
        
        logger.info(f"Saved comparison to {comparison_file}")


def _save_wait_hist_interactive(
    baseline_waits: List[float],
    improved_waits: Optional[List[float]],
    city_name: str,
    output_path: str
) -> None:
    """Save interactive waiting-time histogram (baseline vs improved) as HTML."""
    if go is None:
        logger.info("plotly not installed; skipping interactive wait histogram")
        return
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=baseline_waits,
        name="Baseline",
        opacity=0.6,
        marker_color="#1f77b4",
        nbinsx=40
    ))
    if improved_waits is not None:
        fig.add_trace(go.Histogram(
            x=improved_waits,
            name="Improved",
            opacity=0.6,
            marker_color="#ff7f0e",
            nbinsx=40
        ))
    
    fig.update_layout(
        barmode="overlay",
        title=f"Waiting Time Distribution (weighted by demand) - {city_name}",
        xaxis_title="Weighted waiting time (seconds)",
        yaxis_title="Count",
        legend_title="Solution",
        template="plotly_white"
    )
    
    fig.write_html(output_path, include_plotlyjs="cdn")
    logger.info(f"Generated interactive waiting histogram: {output_path}")


def generate_plots(
    baseline: Dict[str, Any],
    improved: Optional[Dict[str, Any]],
    config: ExperimentConfig,
    plot_format: str = "png",
    method: str = "local"
) -> List[str]:
    """
    Generate visualization plots.
    
    Args:
        baseline: Baseline solution
        improved: Improved solution (None if no improvement)
        config: Experiment configuration
        plot_format: Plot format ("png" or "pdf")
        method: Method name for labeling (default: "local", can be "proposed" -> "ALNS")
    
    Returns:
        List of generated plot file paths
    """
    plot_paths = []
    plots_dir = Path(config.output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate run_id from seed, n, city
    run_id = f"seed{config.seed}_n{config.n_stops}_{config.city}"
    
    # Check if time_matrix exists (not available for proposed method)
    has_time_matrix = "time_matrix" in baseline and baseline.get("time_matrix")
    
    # Reconstruct time matrix for waiting time extraction (if available)
    time_matrix = {}
    if has_time_matrix:
        for k, v in baseline["time_matrix"].items():
            parts = k.split(",")
            if len(parts) == 2:
                time_matrix[(parts[0], parts[1])] = v
    
    # Plot 1: Route comparison (doesn't require time_matrix)
    routes_plot_path = plots_dir / f"{run_id}_routes.{plot_format}"
    plot_routes_baseline_vs_improved(
        baseline_solution=baseline,
        improved_solution=improved,
        dcs=config.dc_list,
        city_name=config.city,
        output_path=str(routes_plot_path),
        plot_format=plot_format
    )
    plot_paths.append(str(routes_plot_path))
    logger.info(f"Generated route plot: {routes_plot_path}")
    
    # Plot 2: Waiting time histograms (weighted by demand) + interactive HTML
    # Always attempt to produce waiting hist; fallback to OSRM if no time_matrix
    baseline_waits: Optional[List[float]] = None
    improved_waits: Optional[List[float]] = None
    
    def _build_waits_from_matrix():
        bw = []
        iw = [] if improved else None
        for dc in config.dc_list:
            dc_baseline_routes = baseline["routes_by_dc"].get(dc.id, [])
            bw.extend(
                extract_waiting_times(dc_baseline_routes, baseline["stops_dict"], time_matrix, dc.id)
            )
            if improved:
                dc_improved_routes = improved["routes_by_dc"].get(dc.id, [])
                iw.extend(
                    extract_waiting_times(dc_improved_routes, baseline["stops_dict"], time_matrix, dc.id)
                )
        return bw, iw
    
    def _build_waits_from_osrm():
        from .osrm_provider import create_osrm_providers
        from .objectives import compute_waiting_times
        from .inavi import iNaviCache
        
        cache = iNaviCache(approx_mode=False)
        depots = [{"id": dc.id, "lat": dc.lat, "lon": dc.lon} for dc in config.dc_list]
        stops_by_id = baseline.get("stops_dict", {})
        if not depots or not stops_by_id:
            raise RuntimeError("Missing depots or stops_dict for waiting-time histogram fallback")
        
        time_provider, _ = create_osrm_providers(depots, stops_by_id, cache)
        
        bw = []
        baseline_waiting = compute_waiting_times(baseline, stops_by_id, time_provider)
        for stop_id, w in baseline_waiting.items():
            demand = stops_by_id.get(stop_id, {}).get("demand", 1)
            bw.append(w * demand)
        
        iw = None
        if improved:
            iw = []
            improved_waiting = compute_waiting_times(improved, stops_by_id, time_provider)
            for stop_id, w in improved_waiting.items():
                demand = stops_by_id.get(stop_id, {}).get("demand", 1)
                iw.append(w * demand)
        return bw, iw
    
    try:
        if has_time_matrix:
            baseline_waits, improved_waits = _build_waits_from_matrix()
        else:
            baseline_waits, improved_waits = _build_waits_from_osrm()
    except Exception as e:
        logger.info(f"Skipping waiting time histogram (all fallbacks failed: {e})")
        baseline_waits, improved_waits = None, None
    
    if baseline_waits is not None:
        # Determine improved label based on method
        improved_label = "ALNS" if method == "proposed" else method.title()
        
        # Matplotlib histogram
        wait_hist_plot_path = plots_dir / f"{run_id}_wait_hist.{plot_format}"
        plot_waiting_time_histograms(
            baseline_waits=baseline_waits,
            improved_waits=improved_waits,
            city_name=config.city,
            output_path=str(wait_hist_plot_path),
            plot_format=plot_format,
            improved_label=improved_label
        )
        plot_paths.append(str(wait_hist_plot_path))
        logger.info(f"Generated waiting time histogram: {wait_hist_plot_path}")
        
        # Interactive Plotly histogram
        _save_wait_hist_interactive(
            baseline_waits=baseline_waits,
            improved_waits=improved_waits,
            city_name=config.city,
            output_path=str(plots_dir / f"{run_id}_wait_hist.html")
        )
    
    return plot_paths


def print_comparison(baseline: Dict[str, Any], improved: Optional[Dict[str, Any]]) -> None:
    """Print comparison table to console."""
    print("\n" + "=" * 80)
    print("BASELINE vs IMPROVED COMPARISON")
    print("=" * 80)
    
    baseline_metrics = baseline["metrics"]
    
    if improved:
        improved_metrics = improved["metrics"]
        
        print(f"\n{'Metric':<25} {'Baseline':<15} {'Improved':<15} {'Change %':<15}")
        print("-" * 80)
        
        for key in ["W_max", "W_mean", "W_p95", "W_top10_mean", "total_cost", "driver_balance"]:
            if key in baseline_metrics:
                baseline_val = baseline_metrics[key]
                improved_val = improved_metrics.get(key, 0)
                
                if isinstance(baseline_val, (int, float)) and baseline_val != 0:
                    change_pct = ((improved_val - baseline_val) / baseline_val) * 100
                    change_str = f"{change_pct:+.2f}%"
                else:
                    change_str = "N/A"
                
                print(f"{key:<25} {baseline_val:<15.2f} {improved_val:<15.2f} {change_str:<15}")
    else:
        print("\nBaseline Metrics:")
        for key, value in baseline_metrics.items():
            print(f"  {key}: {value}")
    
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="VRP Fairness Improvement Experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n", type=int, default=60, help="Number of stops")
    parser.add_argument("--city", type=str, default="daejeon", help="City name")
    parser.add_argument("--dcs", nargs="+", default=None, help="DC coordinates as 'lat,lon' (or use --num-dcs to generate randomly)")
    parser.add_argument("--num-dcs", type=int, default=None, help="Number of DCs to generate randomly within city bounds")
    parser.add_argument("--eps", type=float, default=0.10, help="Cost budget tolerance")
    parser.add_argument("--iters", type=int, default=300, help="Max local search iterations")
    parser.add_argument("--stops-file", type=str, default=None, help="CSV file for stops")
    parser.add_argument("--gpkg", type=str, default="data/yuseong_housing_3__point.gpkg", help="GeoPackage file path")
    parser.add_argument("--layer", type=str, default="yuseong_housing_2__point", help="GPKG layer/table name")
    parser.add_argument("--sample-n", type=int, default=None, help="Number of stops to sample from GPKG")
    parser.add_argument("--housing-type", type=str, default=None, help="Filter by housing type (공동주택/단독주택)")
    parser.add_argument("--demand-field", type=str, default="A26", help="Column to use for demand (e.g., A26)")
    parser.add_argument("--approx", action="store_true", help="Use haversine instead of iNavi")
    parser.add_argument("--baseline-only", action="store_true", help="Run baseline only")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--plots-only", action="store_true", help="Only generate plots (assumes outputs exist)")
    parser.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf"], help="Plot format")
    parser.add_argument("--map", action="store_true", help="Generate interactive HTML map")
    parser.add_argument("--map-tiles", type=str, default="OpenStreetMap", help="Map tile provider (default: OpenStreetMap)")
    parser.add_argument("--method", type=str, default="local", choices=["baseline", "local", "proposed"], help="Solution method")
    parser.add_argument("--alpha", type=float, default=0.5, help="Z1 weight for proposed algorithm")
    parser.add_argument("--beta", type=float, default=0.3, help="Z2 weight for proposed algorithm")
    parser.add_argument("--gamma", type=float, default=0.2, help="Z3 weight for proposed algorithm")
    parser.add_argument("--normalize", type=str, default="baseline", choices=["baseline", "best_known"], help="Normalization method")
    parser.add_argument("--use-distance-objective", action="store_true", help="Use distance for Z2 (else duration)")
    parser.add_argument("--enforce-capacity", action="store_true", help="Enforce capacity constraints")
    parser.add_argument("--operator-mode", type=str, default="fixed", choices=["fixed", "cts"], help="ALNS operator selection mode: fixed or cts (Contextual Thompson Sampling)")
    parser.add_argument("--use-mad", action="store_true", help="Use MAD (Mean Absolute Deviation) for Z3 instead of variance")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing solution files")
    
    args = parser.parse_args()
    
    # Generate random DCs if --num-dcs is specified
    if args.num_dcs is not None:
        if args.dcs:
            logger.warning("Both --dcs and --num-dcs specified. Using --num-dcs and ignoring --dcs")
        
        # Use predefined DCs for daejeon if num_dcs == 3
        if args.num_dcs == 3 and args.city == "daejeon":
            dcs_list = [
                "36.3800587,127.3777765,DC_Logen",
                "36.3711833,127.4050933,DC_Hanjin",
                "36.449416,127.4070349,DC_CJ"
            ]
            logger.info(f"Using predefined DCs for Daejeon")
            for dc_str in dcs_list:
                parts = dc_str.split(",")
                if len(parts) >= 3:
                    logger.info(f"  {parts[2]}: {parts[0]}, {parts[1]}")
                else:
                    logger.info(f"  {dc_str}")
        else:
            import random
            random.seed(args.seed)
            city_config = get_city_config(args.city)
            dcs_list = []
            for i in range(args.num_dcs):
                lat = random.uniform(city_config.lat_min, city_config.lat_max)
                lon = random.uniform(city_config.lon_min, city_config.lon_max)
                dcs_list.append(f"{lat:.6f},{lon:.6f}")
            logger.info(f"Generated {args.num_dcs} random DCs in {args.city}")
            for i, dc_str in enumerate(dcs_list, 1):
                logger.info(f"  DC{i}: {dc_str}")
        args.dcs = dcs_list
    elif not args.dcs:
        logger.error("Either --dcs or --num-dcs must be specified")
        sys.exit(1)
    
    # Handle plots-only mode
    if args.plots_only:
        logger.info("Plots-only mode: Loading existing solutions...")
        baseline_file = Path(args.output_dir) / "solutions" / "baseline.json"
        if not baseline_file.exists():
            baseline_file = Path(args.output_dir) / "baseline.json"
        improved_file = Path(args.output_dir) / "solutions" / "ALNS_MAD.json"
        if not improved_file.exists():
            improved_file = Path(args.output_dir) / "solutions" / "ALNS_MAD.json"
        if not improved_file.exists():
            improved_file = Path(args.output_dir) / "solutions" / "improved.json"  # backward compatibility
        if not improved_file.exists():
            improved_file = Path(args.output_dir) / "improved.json"  # backward compatibility
        
        if not baseline_file.exists():
            logger.error(f"Baseline file not found: {baseline_file}")
            sys.exit(1)
        
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        improved = None
        if improved_file.exists():
            with open(improved_file, 'r') as f:
                improved = json.load(f)
        
        # Reconstruct config from baseline (minimal)
        config = ExperimentConfig(
            seed=args.seed if args.seed else 0,
            n_stops=args.n if args.n else 60,
            city=args.city if args.city else "daejeon",
            dcs=args.dcs if args.dcs else ["36.35,127.38"],
            output_dir=args.output_dir
        )
        
        # Generate plots
        plot_format = args.plot_format.lower()
        if plot_format not in ["png", "pdf"]:
            plot_format = "png"
        
        plot_paths = generate_plots(baseline, improved, config, plot_format=plot_format, method=args.method)
        print("\n" + "=" * 80)
        print("GENERATED PLOTS")
        print("=" * 80)
        for path in plot_paths:
            print(f"  {path}")
        print("=" * 80 + "\n")
        sys.exit(0)
    
    # Create config
    config = ExperimentConfig(
        seed=args.seed,
        n_stops=args.n,
        city=args.city,
        dcs=args.dcs,
        eps=args.eps,
        max_iters=args.iters,
        stops_file=args.stops_file,
        approx_mode=args.approx,
        output_dir=args.output_dir
    )
    
    # Initialize cache
    cache = iNaviCache(
        cache_file=f"{config.output_dir}/cache_inavi.json",
        approx_mode=config.approx_mode
    )
    
    # Generate or load stops
    if args.gpkg:
        logger.info(f"Loading stops from GPKG: {args.gpkg} (layer: {args.layer})")
        stops = load_stops_from_gpkg(
            gpkg_path=args.gpkg,
            layer=args.layer,
            n=args.sample_n,
            seed=config.seed,
            housing_type=args.housing_type,
            demand_field=args.demand_field,
            service_time_s=300
        )
        logger.info(f"Loaded {len(stops)} stops from GPKG")
        logger.info(f"Starting experiment: seed={config.seed}, n={len(stops)}, city={config.city}")
    elif config.stops_file:
        logger.info(f"Loading stops from {config.stops_file}")
        stops = load_stops_from_csv(config.stops_file)
        logger.info(f"Starting experiment: seed={config.seed}, n={len(stops)}, city={config.city}")
    else:
        logger.info(f"Generating {config.n_stops} random stops in {config.city}")
        stops = generate_random_stops(config.n_stops, config.city, config.seed)
        logger.info(f"Starting experiment: seed={config.seed}, n={len(stops)}, city={config.city}")
    
    # Convert DCs to depot format
    depots = [
        {"id": dc.id, "lat": dc.lat, "lon": dc.lon, "name": dc.id}
        for dc in config.dc_list
    ]
    
    # Create vehicles per depot (default: 3 vehicles per depot)
    vehicles_by_depot = {}
    for depot in depots:
        vehicles_by_depot[depot["id"]] = [
            {"id": f"{depot['id']}_v{i+1}", "capacity": 100, "depot_id": depot["id"]}
            for i in range(3)  # Default 3 vehicles per depot
        ]
    
    logger.info(f"Setup: {len(depots)} depots")
    for depot_id, vehicles in vehicles_by_depot.items():
        logger.info(f"  Depot {depot_id}: {len(vehicles)} vehicles")
    
    # Step 1: Assign stops to depots using OSRM travel time
    logger.info("Step 1: Assigning stops to depots using OSRM travel time...")
    time_provider = create_osrm_time_provider(cache)
    stops_by_depot = assign_stops_to_depots(stops, depots, time_provider)
    
    logger.info("Stop assignments:")
    for depot_id, assigned_stops in stops_by_depot.items():
        logger.info(f"  Depot {depot_id}: {len(assigned_stops)} stops")
    
    # Step 2: Solve VRP for each depot
    logger.info("Step 2: Solving VRP for each depot...")
    
    # Convert stops to dict format for solve_multi_depot
    stops_by_depot_dict = {}
    for depot_id, assigned_stops in stops_by_depot.items():
        stops_by_depot_dict[depot_id] = [
            {
                "id": stop.id,
                "lat": stop.lat,
                "lon": stop.lon,
                "demand": stop.demand,
                "service_time_s": stop.service_time
            }
            for stop in assigned_stops
        ]
    
    baseline = solve_multi_depot(
        depots=depots,
        vehicles_by_depot=vehicles_by_depot,
        stops_by_depot=stops_by_depot_dict,
        request_geometry=True
    )
    
    # Build time_matrix only if needed for local search (not for proposed algorithm)
    if args.method != "proposed":
        n_locations = len(depots) + sum(len(stops) for stops in stops_by_depot.values())
        n_calls = n_locations * (n_locations - 1)
        logger.info(f"Building time matrix for {n_locations} locations ({n_calls} OSRM calls)...")
        logger.info("This may take a few minutes. Use --method proposed to skip this step.")
        
        all_locations = []
        # Add depots
        for depot in depots:
            all_locations.append({
                "name": depot["id"],
                "lat": depot["lat"],
                "lon": depot["lon"],
                "coordinates": [depot["lat"], depot["lon"]]
            })
        # Add stops
        for depot_id, assigned_stops in stops_by_depot.items():
            for stop in assigned_stops:
                all_locations.append({
                    "name": stop.id,
                    "lat": stop.lat,
                    "lon": stop.lon,
                    "coordinates": [stop.lat, stop.lon]
                })
        
        all_time_matrix = build_time_matrix_dict(all_locations, cache)
        baseline["time_matrix"] = {f"{k[0]},{k[1]}": v for k, v in all_time_matrix.items()}  # JSON serializable
        
        # Calculate full solution metrics (including total_cost for improvement algorithm)
        logger.info("Calculating solution metrics...")
        all_routes = []
        for depot_id, routes in baseline["routes_by_dc"].items():
            for route in routes:
                route["dc_id"] = depot_id
                all_routes.append(route)
        
        from .metrics import calculate_solution_metrics, metrics_to_dict
        solution_metrics = calculate_solution_metrics(
            all_routes,
            baseline["stops_dict"],
            all_time_matrix,
            depot_id=depots[0]["id"] if depots else "Depot"
        )
        # Merge with existing metrics, adding total_cost
        baseline["metrics"].update(metrics_to_dict(solution_metrics))
    else:
        # For proposed algorithm, we don't need full time matrix
        # Just ensure stops_dict is populated
        logger.info("Skipping time matrix build (not needed for proposed algorithm)")
        if "stops_dict" not in baseline or not baseline["stops_dict"]:
            baseline["stops_dict"] = {}
            for depot_id, assigned_stops in stops_by_depot.items():
                for stop in assigned_stops:
                    # stop is a Stop object, not a dict
                    baseline["stops_dict"][stop.id] = {
                        "lat": stop.lat,
                        "lon": stop.lon,
                        "demand": stop.demand,
                        "service_time_s": stop.service_time,
                        "households": stop.demand  # Use demand as households (A26 value from GPKG)
                    }
    
    # Log baseline metrics per depot and overall
    logger.info("Baseline metrics:")
    logger.info(f"  Overall: duration={baseline['metrics']['total_duration']}s, "
                f"distance={baseline['metrics']['total_distance']:.0f}m, "
                f"vehicles={baseline['metrics']['num_vehicles_used']}")
    for depot_metrics in baseline['metrics'].get('per_depot', []):
        depot_id = depot_metrics.get('depot_id', 'unknown')
        logger.info(f"  Depot {depot_id}: duration={depot_metrics.get('total_duration', 0)}s, "
                    f"distance={depot_metrics.get('total_distance', 0):.0f}m, "
                    f"vehicles={depot_metrics.get('num_vehicles_used', 0)}")
    
    # Run improvement or proposed algorithm
    improved = None
    proposed_debug = None
    if not args.baseline_only:
        if args.method == "proposed":
            from .proposed_algorithm import proposed_algorithm
            from .osrm_provider import create_osrm_providers
            
            logger.info("Running proposed algorithm...")
            # Ensure stops_dict has all stops
            if "stops_dict" not in baseline or not baseline["stops_dict"]:
                baseline["stops_dict"] = {}
                for depot_id, assigned_stops in stops_by_depot.items():
                    for stop in assigned_stops:
                        # stop is a Stop object, not a dict
                        baseline["stops_dict"][stop.id] = {
                            "lat": stop.lat,
                            "lon": stop.lon,
                            "demand": stop.demand,
                            "service_time_s": stop.service_time,
                            "households": stop.demand  # Use demand as households (A26 value from GPKG)
                        }
            
            time_provider, distance_provider = create_osrm_providers(depots, baseline["stops_dict"], cache)
            
            # Patch compute_Z3 to use MAD if --use-mad is set
            import src.vrp_fairness.proposed_algorithm as pa_module
            import src.vrp_fairness.objectives as obj_module
            original_compute_Z3 = None
            if args.use_mad:
                original_compute_Z3 = obj_module.compute_Z3
                from .objectives import compute_Z3_MAD
                
                def compute_Z3_wrapper(waiting, stops_by_id):
                    return compute_Z3_MAD(waiting, stops_by_id)
                
                # Patch both modules (proposed_algorithm imports compute_Z3 directly)
                obj_module.compute_Z3 = compute_Z3_wrapper
                pa_module.compute_Z3 = compute_Z3_wrapper
                logger.info("Patched compute_Z3 to use MAD for Z3 calculation")
            
            try:
                improved, debug = proposed_algorithm(
                depots=depots,
                vehicles=vehicles_by_depot,
                stops_by_depot=stops_by_depot_dict,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                eps=config.eps,
                iters=config.max_iters,
                seed=config.seed,
                time_provider=time_provider,
                distance_provider=distance_provider,
                normalize=args.normalize,
                use_distance_objective=args.use_distance_objective,
                enforce_capacity=args.enforce_capacity,
                operator_mode=args.operator_mode
            )
            
                # Store debug info for later use in Z score calculation
                proposed_debug = debug
            finally:
                # Restore original compute_Z3 if patched
                if args.use_mad and original_compute_Z3 is not None:
                    obj_module.compute_Z3 = original_compute_Z3
                    pa_module.compute_Z3 = original_compute_Z3
                    logger.info("Restored original compute_Z3")
            
            # Save trace
            import csv
            trace_dir = Path(config.output_dir) / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            run_id = f"seed{config.seed}_n{len(stops)}_daejeon"
            # Use different filenames for CTS vs ALNS to avoid conflicts
            if args.operator_mode == "cts":
                trace_file = trace_dir / f"{run_id}_cts.csv"
            else:
                trace_file = trace_dir / f"{run_id}_alns.csv"
            # Trace fields depend on operator mode
            fieldnames = ["iter", "Z", "Z1", "Z2", "Z3", "accepted", "k_removed"]
            if args.operator_mode == "cts":
                fieldnames.extend(["chosen_arm", "reward"])
            
            with open(trace_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(debug["trace"])
            logger.info(f"Trace saved: {trace_file}")
            
            # Save best solution backup if available
            # CTS and ALNS both save best solutions, but with different filenames
            best_backup = debug.get("best_solution_backup")
            if best_backup:
                backup_dir = Path(config.output_dir) / "solutions"
                backup_dir.mkdir(parents=True, exist_ok=True)
                # Use different filenames for CTS vs ALNS to avoid conflicts
                if args.operator_mode == "cts":
                    backup_file = backup_dir / f"{run_id}_cts_best.json"
                else:
                    backup_file = backup_dir / f"{run_id}_alns_best.json"
                # Save solution with metadata
                backup_data = {
                    "solution": best_backup["solution"],
                    "waiting": best_backup.get("waiting"),
                    "Z": best_backup["Z"],
                    "Z1": best_backup["Z1"],
                    "Z2": best_backup["Z2"],
                    "Z3": best_backup["Z3"],
                    "iteration": best_backup["iteration"],
                    "run_id": run_id,
                    "operator_mode": args.operator_mode
                }
                with open(backup_file, 'w') as f:
                    json.dump(backup_data, f, indent=2)
                logger.info(f"Best solution backup saved: {backup_file} (iter {best_backup['iteration']}, Z={best_backup['Z']:.6f}, mode={args.operator_mode})")
        else:
            improved = run_improvement(baseline, config, cache)
            proposed_debug = None
    
    # Attach waiting/travel times (baseline and improved) before saving
    from .osrm_provider import create_osrm_providers
    try:
        tp, dp = create_osrm_providers(depots, baseline.get("stops_dict", {}), cache)
        baseline = attach_wait_and_travel(baseline, baseline.get("stops_dict", {}), tp, dp)
        if improved:
            # If we have best_solution_backup from debug, use its waiting times (already computed)
            if proposed_debug and proposed_debug.get("best_solution_backup"):
                best_backup = proposed_debug["best_solution_backup"]
                # Use waiting times from backup if available (more accurate)
                if best_backup.get("waiting"):
                    improved["waiting_times"] = best_backup["waiting"]
                    # Also attach travel times
                    improved = attach_wait_and_travel(improved, baseline.get("stops_dict", {}), tp, dp)
                    # Overwrite waiting_times with backup (more accurate)
                    improved["waiting_times"] = best_backup["waiting"]
                else:
                    improved = attach_wait_and_travel(improved, baseline.get("stops_dict", {}), tp, dp)
            else:
                improved = attach_wait_and_travel(improved, baseline.get("stops_dict", {}), tp, dp)
    except Exception as e:
        logger.warning(f"Failed to attach waiting/travel times: {e}")
    
    # Save results
    save_results(baseline, improved, config, method=args.method, operator_mode=args.operator_mode, force=args.force)
    
    # Save proposed_debug if available (for comparison scripts)
    # Use different filenames for CTS vs ALNS to avoid conflicts
    if proposed_debug is not None:
        debug_dir = Path(config.output_dir) / "debug"
        debug_dir.mkdir(exist_ok=True)
        if args.operator_mode == "cts":
            debug_file = debug_dir / "cts_debug.json"
        else:
            debug_file = debug_dir / "alns_mad_debug.json"
        with open(debug_file, 'w') as f:
            json.dump(proposed_debug, f, indent=2)
        logger.info(f"Saved proposed debug info: {debug_file}")
    
    # Save proposed solution separately if method is proposed
    # Use different filenames for CTS vs ALNS to avoid conflicts
    if args.method == "proposed" and improved:
        solution_dir = Path(config.output_dir) / "solutions"
        solution_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"seed{config.seed}_n{len(stops)}_daejeon"
        if args.operator_mode == "cts":
            solution_file = solution_dir / f"{run_id}_cts.json"
        else:
            solution_file = solution_dir / f"{run_id}_alns.json"
        with open(solution_file, 'w') as f:
            json.dump(improved, f, indent=2)
        logger.info(f"Proposed solution saved: {solution_file} (mode={args.operator_mode})")
    
    # Print Z scores comparison for all methods (skip if baseline-only or improved is None)
    if improved and args.method != "baseline":
        from .objectives import compute_waiting_times, compute_Z1, compute_Z2, compute_Z3, compute_Z3_MAD, compute_combined_Z
        from .osrm_provider import create_osrm_providers
        
        # Use same provider instance for both baseline and improved to ensure consistency
        time_provider, distance_provider = create_osrm_providers(depots, baseline["stops_dict"], cache)
        
        # Choose Z3 function based on --use-mad flag
        compute_Z3_func = compute_Z3_MAD if args.use_mad else compute_Z3
        
        # For proposed method, use normalizers and baseline objectives from debug info if available
        if args.method == "proposed" and proposed_debug:
            normalizers = proposed_debug.get("normalizers", {})
            baseline_objectives = proposed_debug.get("baseline_objectives", {})
            
            if baseline_objectives:
                # Use exact baseline objectives from proposed_algorithm (computed with same provider)
                Z1_baseline = baseline_objectives.get("Z1")
                Z2_baseline = baseline_objectives.get("Z2")
                Z3_baseline = baseline_objectives.get("Z3")
                logger.info(f"Using baseline objectives from proposed_algorithm: Z1={Z1_baseline:.1f}, Z2={Z2_baseline:.1f}, Z3={Z3_baseline:.1f}")
            else:
                # Fallback: recompute baseline
                waiting_baseline = compute_waiting_times(baseline, baseline["stops_dict"], time_provider)
                Z1_baseline = compute_Z1(waiting_baseline, baseline["stops_dict"])
                Z2_baseline = compute_Z2(baseline, distance_provider, time_provider, args.use_distance_objective)
                Z3_baseline = compute_Z3_func(waiting_baseline, baseline["stops_dict"])
            
            Z1_star = normalizers.get("Z1_star", Z1_baseline)
            Z2_star = normalizers.get("Z2_star", Z2_baseline)
            Z3_star = normalizers.get("Z3_star", Z3_baseline)
        else:
            # For local method: recompute baseline with same provider for consistency
            # This ensures baseline and improved use the exact same provider
            waiting_baseline = compute_waiting_times(baseline, baseline["stops_dict"], time_provider)
            Z1_baseline = compute_Z1(waiting_baseline, baseline["stops_dict"])
            Z2_baseline = compute_Z2(baseline, distance_provider, time_provider, args.use_distance_objective)
            Z3_baseline = compute_Z3_func(waiting_baseline, baseline["stops_dict"])
            
            # Use baseline values as normalizers
            Z1_star, Z2_star, Z3_star = Z1_baseline, Z2_baseline, Z3_baseline
        
        # Baseline Z should be exactly 1.0 (using its own values as normalizers)
        Z_baseline = compute_combined_Z(Z1_baseline, Z2_baseline, Z3_baseline,
                                        Z1_star, Z2_star, Z3_star,
                                        args.alpha, args.beta, args.gamma)
        
        # Verify baseline Z is 1.0
        if abs(Z_baseline - 1.0) > 0.001:
            logger.warning(f"Baseline Z is not 1.0! Z_baseline={Z_baseline:.6f}, expected 1.0")
            logger.warning(f"  Z1={Z1_baseline:.1f}, Z2={Z2_baseline:.1f}, Z3={Z3_baseline:.1f}")
            logger.warning(f"  Z1*={Z1_star:.1f}, Z2*={Z2_star:.1f}, Z3*={Z3_star:.1f}")
            logger.warning(f"  alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
        
        # Compute improved Z-scores using same provider
        # Always use baseline stops_dict for consistency (stops_dict contains metadata that shouldn't change)
        waiting_improved = compute_waiting_times(improved, baseline["stops_dict"], time_provider)
        Z1_improved = compute_Z1(waiting_improved, baseline["stops_dict"])
        Z2_improved = compute_Z2(improved, distance_provider, time_provider, args.use_distance_objective)
        Z3_improved = compute_Z3_func(waiting_improved, baseline["stops_dict"])
        Z_improved = compute_combined_Z(Z1_improved, Z2_improved, Z3_improved,
                                       Z1_star, Z2_star, Z3_star,
                                       args.alpha, args.beta, args.gamma)
        
        method_name = "ALNS" if args.method == "proposed" else args.method.title()
        print("\n" + "=" * 80)
        print(f"BASELINE vs {method_name.upper()}")
        print("=" * 80)
        print(f"{'Metric':<10} {'Baseline':<15} {method_name:<15} {'Change %':<15}")
        print("-" * 80)
        
        for metric, base_val, imp_val in [
            ("Z", Z_baseline, Z_improved),
            ("Z1", Z1_baseline, Z1_improved),
            ("Z2", Z2_baseline, Z2_improved),
            ("Z3", Z3_baseline, Z3_improved)
        ]:
            if base_val > 0:
                change_pct = ((base_val - imp_val) / base_val) * 100
            else:
                change_pct = 0.0
            print(f"{metric:<10} {base_val:<15.3f} {imp_val:<15.3f} {change_pct:+.2f}%")
        
        print("=" * 80 + "\n")
    
    # Generate plots (unless disabled)
    plot_paths = []
    if not args.no_plots:
        plot_format = args.plot_format.lower()
        if plot_format not in ["png", "pdf"]:
            logger.warning(f"Invalid plot format '{plot_format}', using 'png'")
            plot_format = "png"
        
        plot_paths = generate_plots(baseline, improved, config, plot_format=plot_format, method=args.method)
        print("\n" + "=" * 80)
        print("GENERATED PLOTS")
        print("=" * 80)
        for path in plot_paths:
            print(f"  {path}")
        print("=" * 80 + "\n")
    
    # Print comparison
    print_comparison(baseline, improved)
    
    # Generate map (if requested)
    if args.map:
        logger.info("Generating interactive map...")
        
        # Build stops_by_id mapping from baseline (ensures all stops are included)
        # Use baseline['stops_dict'] which has all stops that were processed
        stops_by_id = baseline.get("stops_dict", {})
        
        # If stops_dict is not available, fallback to building from stops_by_depot
        if not stops_by_id:
            stops_by_id = {}
            for depot_id, assigned_stops in stops_by_depot.items():
                for stop in assigned_stops:
                    stops_by_id[stop.id] = {
                        "lat": stop.lat,
                        "lon": stop.lon,
                        "demand": stop.demand,
                        "service_time": stop.service_time,
                        "households": stop.demand  # Use demand as households (A26 value from GPKG)
                    }
        
        # Generate run ID for filename
        run_id = f"seed{config.seed}_n{len(stops)}_daejeon"
        map_file = Path(config.output_dir) / "maps" / f"{run_id}_routes.html"
        
        try:
            # Convert depots to DC format for map (compatibility)
            dcs_for_map = [
                type('DC', (), {"id": d["id"], "lat": d["lat"], "lon": d["lon"]})()
                for d in depots
            ]
            
            # Collect all available solutions
            solutions = {"Baseline": baseline}
            if improved:
                solutions["Improved"] = improved
            
            # Try to load local solution if it exists
            local_file = Path(config.output_dir) / "solutions" / "local.json"
            if local_file.exists():
                try:
                    with open(local_file, 'r') as f:
                        local = json.load(f)
                        solutions["Local"] = local
                except Exception as e:
                    logger.debug(f"Could not load local solution: {e}")
            
            # Use multi-solution generator (works with 1+ solutions)
            save_multi_solution_map_html(
                solutions=solutions,
                dcs=dcs_for_map,
                stops_by_id=stops_by_id,
                out_html=str(map_file),
                tiles=args.map_tiles
            )
            print("\n" + "=" * 80)
            print("GENERATED MAP")
            print("=" * 80)
            print(f"  {map_file.absolute()}")
            print(f"  Access: http://localhost:8080/{map_file.name}")
            print("=" * 80 + "\n")
        except Exception as e:
            logger.error(f"Failed to generate map: {e}")
    
    logger.info("Experiment completed!")


if __name__ == "__main__":
    main()

