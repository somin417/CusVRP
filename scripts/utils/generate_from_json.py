#!/usr/bin/env python3
"""
Generate plots and maps from JSON solution files.

This script provides three main functions:
1. Waiting plot: Raw waiting time histogram (household-weighted frequency)
2. Weighted waiting plot: Weighted waiting time histogram (frequency count)
3. Map with radio buttons: Interactive map with multiple solutions

Usage:
    python scripts/generate_from_json.py --baseline outputs/baseline.json --improved outputs/improved.json [--local outputs/local.json] [--cts outputs/cts_solution.json]
    
Options:
    --waiting-plot: Generate waiting time plot
    --weighted-plot: Generate weighted waiting time plot
    --map: Generate interactive map with radio buttons
    (If none specified, generates all)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.utils import load_json, extract_waiting_times_with_weights, calculate_weighted_waiting_times
from scripts.utils.plotting_utils import plot_waiting_histogram, plot_weighted_waiting_histogram
from src.vrp_fairness.objectives import compute_waiting_times
from src.vrp_fairness.osrm_provider import create_osrm_providers
from src.vrp_fairness.inavi import iNaviCache
from src.vrp_fairness.map_folium import save_multi_solution_map_html

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_solutions(
    baseline_path: Path,
    improved_path: Optional[Path] = None,
    local_path: Optional[Path] = None,
    cts_path: Optional[Path] = None,
    solution_paths: Optional[List[tuple]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load solutions from JSON files.
    
    Args:
        baseline_path: Path to baseline.json (required)
        improved_path: Optional path to improved.json
        local_path: Optional path to local.json
        cts_path: Optional path to cts_solution.json
        solution_paths: Optional list of (name, path) tuples for additional solutions
    
    Returns:
        Dict mapping solution name -> solution dict
    """
    solutions = {}
    
    # Load baseline (required)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    solutions["Baseline"] = load_json(baseline_path)
    logger.info(f"Loaded baseline from: {baseline_path}")
    
    # Load improved
    if improved_path and improved_path.exists():
        solutions["Improved"] = load_json(improved_path)
        logger.info(f"Loaded improved from: {improved_path}")
    
    # Load local
    if local_path and local_path.exists():
        solutions["Local"] = load_json(local_path)
        logger.info(f"Loaded local from: {local_path}")
    
    # Load CTS
    if cts_path and cts_path.exists():
        solutions["CTS"] = load_json(cts_path)
        logger.info(f"Loaded CTS from: {cts_path}")
    
    # Load additional solutions
    if solution_paths:
        for name, path in solution_paths:
            sol_path = Path(path)
            if sol_path.exists():
                sol = load_json(sol_path)
                # If it's a best_solution backup, extract the solution
                if "solution" in sol:
                    sol = sol["solution"]
                solutions[name] = sol
                logger.info(f"Loaded {name} from: {sol_path}")
    
    return solutions


def generate_waiting_plot(
    solutions: Dict[str, Dict[str, Any]],
    depots: List[Dict[str, Any]],
    stops_by_id: Dict[str, Dict[str, Any]],
    output_path: Path,
    city_name: str = "daejeon"
) -> None:
    """
    Generate waiting time plot (raw waiting times, household-weighted frequency).
    
    Args:
        solutions: Dict mapping solution name -> solution dict
        depots: List of depot dicts
        stops_by_id: Dict mapping stop_id -> stop data
        output_path: Output file path (without extension)
        city_name: City name for title
    """
    logger.info("Generating waiting time plot...")
    
    # Create OSRM providers
    cache = iNaviCache(approx_mode=False)
    time_provider, distance_provider = create_osrm_providers(depots, stops_by_id, cache)
    
    # Compute waiting times for each solution
    waiting_by_method = {}
    weights_by_method = {}
    
    for method_name, solution in solutions.items():
        waiting = compute_waiting_times(solution, stops_by_id, time_provider)
        waits, weights = extract_waiting_times_with_weights(waiting, stops_by_id)
        waiting_by_method[method_name] = waits
        weights_by_method[method_name] = weights
        logger.info(f"  {method_name}: {len(waits)} stops")
    
    # Plot
    plot_waiting_histogram(
        waiting_by_method=waiting_by_method,
        output_path=output_path,
        weights_by_method=weights_by_method,
        city_name=city_name,
        title_suffix="Waiting Time Distribution"
    )
    
    logger.info(f"✓ Waiting plot saved: {output_path}.png")


def generate_weighted_waiting_plot(
    solutions: Dict[str, Dict[str, Any]],
    depots: List[Dict[str, Any]],
    stops_by_id: Dict[str, Dict[str, Any]],
    output_path: Path,
    city_name: str = "daejeon"
) -> None:
    """
    Generate weighted waiting time plot (weighted waiting times, frequency count).
    
    Args:
        solutions: Dict mapping solution name -> solution dict
        depots: List of depot dicts
        stops_by_id: Dict mapping stop_id -> stop data
        output_path: Output file path (without extension)
        city_name: City name for title
    """
    logger.info("Generating weighted waiting time plot...")
    
    # Create OSRM providers
    cache = iNaviCache(approx_mode=False)
    time_provider, distance_provider = create_osrm_providers(depots, stops_by_id, cache)
    
    # Compute weighted waiting times for each solution
    weighted_by_method = {}
    
    for method_name, solution in solutions.items():
        waiting = compute_waiting_times(solution, stops_by_id, time_provider)
        weighted = calculate_weighted_waiting_times(waiting, stops_by_id)
        weighted_by_method[method_name] = weighted
        logger.info(f"  {method_name}: {len(weighted)} stops, range: [{min(weighted):.1f}, {max(weighted):.1f}]")
    
    # Plot
    plot_weighted_waiting_histogram(
        weighted_by_method=weighted_by_method,
        output_path=output_path,
        city_name=city_name
    )
    
    logger.info(f"✓ Weighted waiting plot saved: {output_path}.png")


def generate_map(
    solutions: Dict[str, Dict[str, Any]],
    depots: List[Dict[str, Any]],
    stops_by_id: Dict[str, Dict[str, Any]],
    output_path: Path,
    tiles: str = "OpenStreetMap"
) -> None:
    """
    Generate interactive map with radio buttons for each solution.
    
    Args:
        solutions: Dict mapping solution name -> solution dict
        depots: List of depot dicts
        stops_by_id: Dict mapping stop_id -> stop data
        output_path: Output HTML file path
        tiles: Map tile provider
    """
    logger.info("Generating interactive map with radio buttons...")
    
    # Convert depots to DC format (required by map_folium)
    class DC:
        def __init__(self, dc_id, lat, lon):
            self.id = dc_id
            self.lat = lat
            self.lon = lon
    
    dcs = [DC(d.get("id", f"DC{i}"), d["lat"], d["lon"]) for i, d in enumerate(depots)]
    
    # Generate map
    save_multi_solution_map_html(
        solutions=solutions,
        dcs=dcs,
        stops_by_id=stops_by_id,
        out_html=str(output_path),
        tiles=tiles
    )
    
    logger.info(f"✓ Map saved: {output_path}")
    logger.info(f"  Access: http://localhost:8080/{output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots and maps from JSON solution files"
    )
    parser.add_argument("--baseline", type=str, required=True,
                       help="Path to baseline.json (required)")
    parser.add_argument("--improved", type=str, default=None,
                       help="Path to improved.json (optional)")
    parser.add_argument("--local", type=str, default=None,
                       help="Path to local.json (optional)")
    parser.add_argument("--cts", type=str, default=None,
                       help="Path to cts_solution.json (optional)")
    parser.add_argument("--solution", type=str, nargs=2, metavar=("NAME", "PATH"), action="append",
                       help="Additional solution: --solution NAME PATH (can be used multiple times)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory (default: outputs)")
    parser.add_argument("--city", type=str, default="daejeon",
                       help="City name for plots (default: daejeon)")
    parser.add_argument("--tiles", type=str, default="OpenStreetMap",
                       help="Map tile provider (default: OpenStreetMap)")
    
    # Generation options
    parser.add_argument("--waiting-plot", action="store_true",
                       help="Generate waiting time plot")
    parser.add_argument("--weighted-plot", action="store_true",
                       help="Generate weighted waiting time plot")
    parser.add_argument("--map", action="store_true",
                       help="Generate interactive map")
    
    args = parser.parse_args()
    
    # If no specific option, generate all
    generate_all = not (args.waiting_plot or args.weighted_plot or args.map)
    
    # Load solutions
    baseline_path = Path(args.baseline)
    improved_path = Path(args.improved) if args.improved else None
    local_path = Path(args.local) if args.local else None
    cts_path = Path(args.cts) if args.cts else None
    solution_paths = args.solution if args.solution else None
    
    try:
        solutions = load_solutions(
            baseline_path=baseline_path,
            improved_path=improved_path,
            local_path=local_path,
            cts_path=cts_path,
            solution_paths=solution_paths
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    if not solutions:
        logger.error("No solutions loaded!")
        sys.exit(1)
    
    # Extract depots and stops_dict from baseline
    baseline = solutions["Baseline"]
    depots = baseline.get("depots", [])
    stops_by_id = baseline.get("stops_dict", {})
    
    if not depots:
        logger.error("No depots found in baseline solution!")
        sys.exit(1)
    
    if not stops_by_id:
        logger.error("No stops_dict found in baseline solution!")
        sys.exit(1)
    
    logger.info(f"Loaded {len(solutions)} solutions: {', '.join(solutions.keys())}")
    logger.info(f"Found {len(depots)} depots and {len(stops_by_id)} stops")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots and maps
    if generate_all or args.waiting_plot:
        waiting_output = output_dir / "waiting_plot"
        generate_waiting_plot(
            solutions=solutions,
            depots=depots,
            stops_by_id=stops_by_id,
            output_path=waiting_output,
            city_name=args.city
        )
    
    if generate_all or args.weighted_plot:
        weighted_output = output_dir / "weighted_waiting_plot"
        generate_weighted_waiting_plot(
            solutions=solutions,
            depots=depots,
            stops_by_id=stops_by_id,
            output_path=weighted_output,
            city_name=args.city
        )
    
    if generate_all or args.map:
        map_output = output_dir / "map_compare.html"
        generate_map(
            solutions=solutions,
            depots=depots,
            stops_by_id=stops_by_id,
            output_path=map_output,
            tiles=args.tiles
        )
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    if generate_all or args.waiting_plot:
        print(f"  Waiting plot: {output_dir / 'waiting_plot.png'}")
    if generate_all or args.weighted_plot:
        print(f"  Weighted plot: {output_dir / 'weighted_waiting_plot.png'}")
    if generate_all or args.map:
        print(f"  Map: {map_output}")
    print("=" * 80)


if __name__ == "__main__":
    main()

