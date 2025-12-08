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
from .map_folium import save_route_map_html

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
                "demand": stop.demand
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
    
    return {
        "routes_by_dc": routes_by_dc,
        "routes": improved_routes,
        "metrics": metrics_to_dict(improved_metrics)
    }


def save_results(
    baseline: Dict[str, Any],
    improved: Optional[Dict[str, Any]],
    config: ExperimentConfig
) -> None:
    """Save results to JSON and CSV files."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save baseline
    baseline_file = output_dir / "baseline.json"
    with open(baseline_file, 'w') as f:
        json.dump(baseline, f, indent=2)
    logger.info(f"Saved baseline to {baseline_file}")
    
    # Save baseline metrics CSV
    metrics_file = output_dir / "baseline_metrics.csv"
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in baseline["metrics"].items():
            writer.writerow([key, value])
    logger.info(f"Saved baseline metrics to {metrics_file}")
    
    # Save improved if available
    if improved:
        improved_file = output_dir / "improved.json"
        with open(improved_file, 'w') as f:
            json.dump(improved, f, indent=2)
        logger.info(f"Saved improved solution to {improved_file}")
        
        # Save comparison CSV
        comparison_file = output_dir / "comparison.csv"
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


def generate_plots(
    baseline: Dict[str, Any],
    improved: Optional[Dict[str, Any]],
    config: ExperimentConfig,
    plot_format: str = "png"
) -> List[str]:
    """
    Generate visualization plots.
    
    Args:
        baseline: Baseline solution
        improved: Improved solution (None if no improvement)
        config: Experiment configuration
        plot_format: Plot format ("png" or "pdf")
    
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
    
    # Plot 2: Waiting time histograms (requires time_matrix)
    if has_time_matrix:
        # Extract waiting times for all DCs
        baseline_waits = []
        improved_waits = [] if improved else None
        
        for dc in config.dc_list:
            dc_baseline_routes = baseline["routes_by_dc"].get(dc.id, [])
            baseline_waits.extend(
                extract_waiting_times(dc_baseline_routes, baseline["stops_dict"], time_matrix, dc.id)
            )
            
            if improved:
                dc_improved_routes = improved["routes_by_dc"].get(dc.id, [])
                improved_waits.extend(
                    extract_waiting_times(dc_improved_routes, baseline["stops_dict"], time_matrix, dc.id)
                )
        
        wait_hist_plot_path = plots_dir / f"{run_id}_wait_hist.{plot_format}"
        plot_waiting_time_histograms(
            baseline_waits=baseline_waits,
            improved_waits=improved_waits,
            city_name=config.city,
            output_path=str(wait_hist_plot_path),
            plot_format=plot_format
        )
        plot_paths.append(str(wait_hist_plot_path))
        logger.info(f"Generated waiting time histogram: {wait_hist_plot_path}")
    else:
        logger.info("Skipping waiting time histogram (time_matrix not available - use --method local for timetable plots)")
    
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
    
    args = parser.parse_args()
    
    # Generate random DCs if --num-dcs is specified
    if args.num_dcs is not None:
        if args.dcs:
            logger.warning("Both --dcs and --num-dcs specified. Using --num-dcs and ignoring --dcs")
        import random
        random.seed(args.seed)
        city_config = get_city_config(args.city)
        dcs_list = []
        for i in range(args.num_dcs):
            lat = random.uniform(city_config.lat_min, city_config.lat_max)
            lon = random.uniform(city_config.lon_min, city_config.lon_max)
            dcs_list.append(f"{lat:.6f},{lon:.6f}")
        args.dcs = dcs_list
        logger.info(f"Generated {args.num_dcs} random DCs in {args.city}")
        for i, dc_str in enumerate(args.dcs, 1):
            logger.info(f"  DC{i}: {dc_str}")
    elif not args.dcs:
        logger.error("Either --dcs or --num-dcs must be specified")
        sys.exit(1)
    
    # Handle plots-only mode
    if args.plots_only:
        logger.info("Plots-only mode: Loading existing solutions...")
        baseline_file = Path(args.output_dir) / "baseline.json"
        improved_file = Path(args.output_dir) / "improved.json"
        
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
        
        plot_paths = generate_plots(baseline, improved, config, plot_format=plot_format)
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
                    baseline["stops_dict"][stop["id"]] = {
                        "lat": stop["lat"],
                        "lon": stop["lon"],
                        "demand": stop.get("demand", 1),
                        "service_time_s": stop.get("service_time_s", 300),
                        "households": stop.get("households", 1)
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
                        baseline["stops_dict"][stop["id"]] = {
                            "lat": stop["lat"],
                            "lon": stop["lon"],
                            "demand": stop.get("demand", 1),
                            "service_time_s": stop.get("service_time_s", 300),
                            "households": stop.get("households", 1)
                        }
            
            time_provider, distance_provider = create_osrm_providers(depots, baseline["stops_dict"], cache)
            
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
                enforce_capacity=args.enforce_capacity
            )
            
            # Save trace
            import csv
            trace_dir = Path(config.output_dir) / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            run_id = f"seed{config.seed}_n{len(stops)}_daejeon"
            trace_file = trace_dir / f"{run_id}_proposed.csv"
            with open(trace_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["iter", "Z", "Z1", "Z2", "Z3", "accepted", "k_removed"])
                writer.writeheader()
                writer.writerows(debug["trace"])
            logger.info(f"Trace saved: {trace_file}")
        else:
            improved = run_improvement(baseline, config, cache)
    
    # Save results
    save_results(baseline, improved, config)
    
    # Save proposed solution separately if method is proposed
    if args.method == "proposed" and improved:
        import json
        solution_dir = Path(config.output_dir) / "solutions"
        solution_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"seed{config.seed}_n{len(stops)}_daejeon"
        solution_file = solution_dir / f"{run_id}_proposed.json"
        with open(solution_file, 'w') as f:
            json.dump(improved, f, indent=2)
        logger.info(f"Proposed solution saved: {solution_file}")
        
        # Print comparison
        from .objectives import compute_waiting_times, compute_Z1, compute_Z2, compute_Z3, compute_combined_Z
        from .osrm_provider import create_osrm_providers
        time_provider, distance_provider = create_osrm_providers(depots, baseline["stops_dict"], cache)
        
        waiting_baseline = compute_waiting_times(baseline, baseline["stops_dict"], time_provider)
        Z1_baseline = compute_Z1(waiting_baseline, baseline["stops_dict"])
        Z2_baseline = compute_Z2(baseline, distance_provider, time_provider, args.use_distance_objective)
        Z3_baseline = compute_Z3(waiting_baseline, baseline["stops_dict"])
        Z_baseline = compute_combined_Z(Z1_baseline, Z2_baseline, Z3_baseline,
                                        Z1_baseline, Z2_baseline, Z3_baseline,
                                        args.alpha, args.beta, args.gamma)
        
        waiting_proposed = compute_waiting_times(improved, improved["stops_dict"], time_provider)
        Z1_proposed = compute_Z1(waiting_proposed, improved["stops_dict"])
        Z2_proposed = compute_Z2(improved, distance_provider, time_provider, args.use_distance_objective)
        Z3_proposed = compute_Z3(waiting_proposed, improved["stops_dict"])
        Z_proposed = compute_combined_Z(Z1_proposed, Z2_proposed, Z3_proposed,
                                       Z1_baseline, Z2_baseline, Z3_baseline,
                                       args.alpha, args.beta, args.gamma)
        
        print("\n" + "=" * 80)
        print("BASELINE vs PROPOSED")
        print("=" * 80)
        print(f"Z:  {Z_baseline:.3f} -> {Z_proposed:.3f} ({((Z_baseline-Z_proposed)/Z_baseline*100):.1f}% improvement)")
        print(f"Z1: {Z1_baseline:.1f} -> {Z1_proposed:.1f} ({((Z1_baseline-Z1_proposed)/Z1_baseline*100):.1f}% improvement)")
        print(f"Z2: {Z2_baseline:.1f} -> {Z2_proposed:.1f} ({((Z2_baseline-Z2_proposed)/Z2_baseline*100):.1f}% improvement)")
        print(f"Z3: {Z3_baseline:.1f} -> {Z3_proposed:.1f} ({((Z3_baseline-Z3_proposed)/Z3_baseline*100):.1f}% improvement)")
        print("=" * 80 + "\n")
    
    # Generate plots (unless disabled)
    plot_paths = []
    if not args.no_plots:
        plot_format = args.plot_format.lower()
        if plot_format not in ["png", "pdf"]:
            logger.warning(f"Invalid plot format '{plot_format}', using 'png'")
            plot_format = "png"
        
        plot_paths = generate_plots(baseline, improved, config, plot_format=plot_format)
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
                        "service_time": stop.service_time
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
            
            save_route_map_html(
                baseline_solution=baseline,
                improved_solution=improved,
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

