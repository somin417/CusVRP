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

from .config import ExperimentConfig
from .data import generate_random_stops, load_stops_from_csv, assign_stops_to_dcs, Stop
from .inavi import iNaviCache, PolylineCache, get_leg_route, RouteLeg
from .vroom_vrp import call_vrp, parse_vrp_response
from .metrics import calculate_solution_metrics, metrics_to_dict
from .local_search import FairnessLocalSearch
from .plotting import plot_routes_baseline_vs_improved, plot_waiting_time_histograms, extract_waiting_times

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
                
                for i in range(len(stop_ids) - 1):
                    from_id = stop_ids[i]
                    to_id = stop_ids[i + 1]
                    
                    # Get coordinates
                    if from_id == dc.id:
                        from_lat, from_lon = dc.lat, dc.lon
                    else:
                        from_stop = next((s for s in stops if s.id == from_id), None)
                        if from_stop:
                            from_lat, from_lon = from_stop.lat, from_stop.lon
                        else:
                            continue
                    
                    if to_id == dc.id:
                        to_lat, to_lon = dc.lat, dc.lon
                    else:
                        to_stop = next((s for s in stops if s.id == to_id), None)
                        if to_stop:
                            to_lat, to_lon = to_stop.lat, to_stop.lon
                        else:
                            continue
                    
                    # Use get_leg_route() to get iNavi road routing with polyline
                    leg = get_leg_route(
                        origin=(from_lat, from_lon),
                        dest=(to_lat, to_lon),
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
    
    # Reconstruct time matrix for waiting time extraction
    time_matrix = {}
    for k, v in baseline["time_matrix"].items():
        parts = k.split(",")
        if len(parts) == 2:
            time_matrix[(parts[0], parts[1])] = v
    
    # Plot 1: Route comparison
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
    
    # Plot 2: Waiting time histograms
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
    parser.add_argument("--dcs", nargs="+", required=True, help="DC coordinates as 'lat,lon'")
    parser.add_argument("--eps", type=float, default=0.10, help="Cost budget tolerance")
    parser.add_argument("--iters", type=int, default=300, help="Max local search iterations")
    parser.add_argument("--stops-file", type=str, default=None, help="CSV file for stops")
    parser.add_argument("--approx", action="store_true", help="Use haversine instead of iNavi")
    parser.add_argument("--baseline-only", action="store_true", help="Run baseline only")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--plots-only", action="store_true", help="Only generate plots (assumes outputs exist)")
    parser.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf"], help="Plot format")
    
    args = parser.parse_args()
    
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
    
    logger.info(f"Starting experiment: seed={config.seed}, n={config.n_stops}, city={config.city}")
    
    # Initialize cache
    cache = iNaviCache(
        cache_file=f"{config.output_dir}/cache_inavi.json",
        approx_mode=config.approx_mode
    )
    
    # Generate or load stops
    if config.stops_file:
        logger.info(f"Loading stops from {config.stops_file}")
        stops = load_stops_from_csv(config.stops_file)
    else:
        logger.info(f"Generating {config.n_stops} random stops in {config.city}")
        stops = generate_random_stops(config.n_stops, config.city, config.seed)
    
    # Assign stops to DCs using iNavi road routing
    def distance_func(lat1, lon1, lat2, lon2):
        # Use get_leg_route() to ensure iNavi road routing for DC assignment
        leg = get_leg_route(origin=(lat1, lon1), dest=(lat2, lon2), cache=cache)
        return leg.travel_distance_m
    
    stops_by_dc = assign_stops_to_dcs(stops, config.dc_list, distance_func)
    
    logger.info("Stop assignments:")
    for dc_id, dc_stops in stops_by_dc.items():
        logger.info(f"  {dc_id}: {len(dc_stops)} stops")
    
    # Run baseline
    logger.info("Running baseline VRP...")
    baseline = run_baseline(config, stops_by_dc, cache)
    
    # Run improvement
    improved = None
    if not args.baseline_only:
        improved = run_improvement(baseline, config, cache)
    
    # Save results
    save_results(baseline, improved, config)
    
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
    
    logger.info("Experiment completed!")


if __name__ == "__main__":
    main()

