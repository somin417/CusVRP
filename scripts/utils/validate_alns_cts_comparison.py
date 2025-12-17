#!/usr/bin/env python3
"""
Comprehensive comparison validation for ALNS (Fixed) vs CTS.

This script compares ALNS and CTS results on the same input with detailed metrics:
- Objective values (Z, Z1, Z2, Z3)
- Waiting time distribution statistics
- Fairness metrics
- Route structure comparison
- Budget constraint analysis
- Convergence analysis (from trace/debug)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from src.vrp_fairness.objectives import (
    compute_waiting_times,
    compute_Z1,
    compute_Z2,
    compute_Z3_MAD,
    compute_combined_Z,
)
from src.vrp_fairness.osrm_provider import create_osrm_providers
from src.vrp_fairness.inavi import iNaviCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file."""
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def calc_scores(
    solution: Dict[str, Any],
    depots: List[Dict[str, Any]],
    use_distance: bool,
    alpha: float,
    beta: float,
    gamma: float,
    Z1_star: Optional[float] = None,
    Z2_star: Optional[float] = None,
    Z3_star: Optional[float] = None,
    time_provider: Optional[Any] = None,
    distance_provider: Optional[Any] = None,
    cache: Optional[Any] = None,
    baseline_stops_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate Z-scores for a solution using MAD for Z3."""
    if cache is None:
        cache = iNaviCache(approx_mode=False)
    
    if baseline_stops_dict is None:
        baseline_stops_dict = solution.get("stops_dict", {})
    stops_by_id = baseline_stops_dict
    
    if time_provider is None or distance_provider is None:
        tp, dp = create_osrm_providers(depots, stops_by_id, cache)
        if time_provider is None:
            time_provider = tp
        if distance_provider is None:
            distance_provider = dp
    
    waiting = compute_waiting_times(solution, stops_by_id, time_provider)
    Z1 = compute_Z1(waiting, stops_by_id)
    Z2 = compute_Z2(solution, distance_provider, time_provider, use_distance)
    Z3 = compute_Z3_MAD(waiting, stops_by_id)
    
    if Z1_star is None:
        Z1_star, Z2_star, Z3_star = Z1, Z2, Z3
    
    Z = compute_combined_Z(Z1, Z2, Z3, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
    return {
        "waiting": waiting,
        "Z1": Z1,
        "Z2": Z2,
        "Z3": Z3,
        "Z": Z,
        "Z_star": (Z1_star, Z2_star, Z3_star),
    }


def compare_objectives(
    alns_scores: Dict[str, Any],
    cts_scores: Dict[str, Any],
    baseline_scores: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare objective values."""
    baseline_Z = baseline_scores["Z"]
    baseline_Z1 = baseline_scores["Z1"]
    baseline_Z2 = baseline_scores["Z2"]
    baseline_Z3 = baseline_scores["Z3"]
    
    alns_Z = alns_scores["Z"]
    alns_Z1 = alns_scores["Z1"]
    alns_Z2 = alns_scores["Z2"]
    alns_Z3 = alns_scores["Z3"]
    
    cts_Z = cts_scores["Z"]
    cts_Z1 = cts_scores["Z1"]
    cts_Z2 = cts_scores["Z2"]
    cts_Z3 = cts_scores["Z3"]
    
    def calc_pct_change(old_val, new_val):
        if old_val == 0:
            return 0.0
        return ((new_val - old_val) / old_val) * 100
    
    return {
        "baseline": {
            "Z": baseline_Z,
            "Z1": baseline_Z1,
            "Z2": baseline_Z2,
            "Z3": baseline_Z3
        },
        "alns": {
            "Z": alns_Z,
            "Z1": alns_Z1,
            "Z2": alns_Z2,
            "Z3": alns_Z3
        },
        "cts": {
            "Z": cts_Z,
            "Z1": cts_Z1,
            "Z2": cts_Z2,
            "Z3": cts_Z3
        },
        "improvements": {
            "alns_vs_baseline": {
                "Z": calc_pct_change(baseline_Z, alns_Z),
                "Z1": calc_pct_change(baseline_Z1, alns_Z1),
                "Z2": calc_pct_change(baseline_Z2, alns_Z2),
                "Z3": calc_pct_change(baseline_Z3, alns_Z3)
            },
            "cts_vs_baseline": {
                "Z": calc_pct_change(baseline_Z, cts_Z),
                "Z1": calc_pct_change(baseline_Z1, cts_Z1),
                "Z2": calc_pct_change(baseline_Z2, cts_Z2),
                "Z3": calc_pct_change(baseline_Z3, cts_Z3)
            },
            "cts_vs_alns": {
                "Z": calc_pct_change(alns_Z, cts_Z),
                "Z1": calc_pct_change(alns_Z1, cts_Z1),
                "Z2": calc_pct_change(alns_Z2, cts_Z2),
                "Z3": calc_pct_change(alns_Z3, cts_Z3)
            }
        }
    }


def compare_waiting_distributions(
    alns_waiting: Dict[str, float],
    cts_waiting: Dict[str, float],
    stops_by_id: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare waiting time distributions."""
    def compute_stats(waiting_dict):
        values = list(waiting_dict.values())
        if not values:
            return {}
        
        # Weighted statistics
        weights = []
        weighted_values = []
        for stop_id, waiting_time in waiting_dict.items():
            stop_data = stops_by_id.get(stop_id, {})
            weight = stop_data.get("households", stop_data.get("demand", 1))
            if weight is None or weight == 0:
                weight = 1
            weights.append(weight)
            weighted_values.append(waiting_time * weight)
        
        values_array = np.array(values)
        weights_array = np.array(weights)
        weighted_values_array = np.array(weighted_values)
        
        return {
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "p95": float(np.percentile(values_array, 95)),
            "p99": float(np.percentile(values_array, 99)),
            "weighted_mean": float(np.sum(weighted_values_array) / np.sum(weights_array)) if np.sum(weights_array) > 0 else 0.0,
            "count": len(values)
        }
    
    alns_stats = compute_stats(alns_waiting)
    cts_stats = compute_stats(cts_waiting)
    
    # Fairness metrics: top-10% vs overall mean
    def compute_fairness_metrics(waiting_dict):
        WW = {}
        for stop_id, w_i in waiting_dict.items():
            stop_data = stops_by_id.get(stop_id, {})
            n_i = stop_data.get("households", stop_data.get("demand", 1))
            if n_i is None or n_i == 0:
                n_i = 1
            WW[stop_id] = n_i * w_i
        
        if not WW:
            return {"tail_ratio": 1.0, "variance": 0.0}
        
        ww_values = sorted(WW.values(), reverse=True)
        n_top10 = max(1, len(ww_values) // 10)
        top10_mean = np.mean(ww_values[:n_top10])
        overall_mean = np.mean(ww_values)
        tail_ratio = top10_mean / overall_mean if overall_mean > 0 else 1.0
        
        # Weighted variance
        values = list(waiting_dict.values())
        weights = [stops_by_id.get(sid, {}).get("households", stops_by_id.get(sid, {}).get("demand", 1)) or 1 
                   for sid in waiting_dict.keys()]
        if len(values) > 1 and np.sum(weights) > 0:
            weighted_mean = np.average(values, weights=weights)
            variance = np.average((values - weighted_mean) ** 2, weights=weights)
        else:
            variance = 0.0
        
        return {
            "tail_ratio": float(tail_ratio),
            "variance": float(variance)
        }
    
    alns_fairness = compute_fairness_metrics(alns_waiting)
    cts_fairness = compute_fairness_metrics(cts_waiting)
    
    return {
        "alns": {**alns_stats, **alns_fairness},
        "cts": {**cts_stats, **cts_fairness}
    }


def compare_route_structures(
    alns_solution: Dict[str, Any],
    cts_solution: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare route structures."""
    def analyze_routes(solution):
        routes_by_dc = solution.get("routes_by_dc", {})
        total_routes = 0
        route_lengths = []
        depot_utilization = {}
        
        for dc_id, routes in routes_by_dc.items():
            depot_utilization[dc_id] = len(routes)
            total_routes += len(routes)
            for route in routes:
                stop_ids = route.get("ordered_stop_ids", [])
                route_lengths.append(len(stop_ids))
        
        return {
            "num_routes": total_routes,
            "avg_route_length": float(np.mean(route_lengths)) if route_lengths else 0.0,
            "min_route_length": int(np.min(route_lengths)) if route_lengths else 0,
            "max_route_length": int(np.max(route_lengths)) if route_lengths else 0,
            "depot_utilization": depot_utilization
        }
    
    alns_routes = analyze_routes(alns_solution)
    cts_routes = analyze_routes(cts_solution)
    
    return {
        "alns": alns_routes,
        "cts": cts_routes
    }


def compare_convergence(
    alns_debug: Optional[Dict[str, Any]],
    cts_debug: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare convergence behavior."""
    def extract_convergence(debug):
        if debug is None:
            return {"best_iteration": None, "total_improvements": None}
        
        trace = debug.get("trace", [])
        if not trace:
            return {"best_iteration": None, "total_improvements": None}
        
        # Find best iteration
        best_iter = None
        best_Z = float("inf")
        improvements = 0
        
        for entry in trace:
            iter_num = entry.get("iter", 0)
            Z = entry.get("Z", float("inf"))
            accepted = entry.get("accepted", False)
            
            if Z < best_Z:
                best_Z = Z
                best_iter = iter_num
            
            if accepted:
                improvements += 1
        
        return {
            "best_iteration": best_iter,
            "total_improvements": improvements,
            "best_Z": best_Z if best_Z != float("inf") else None
        }
    
    alns_conv = extract_convergence(alns_debug)
    cts_conv = extract_convergence(cts_debug)
    
    return {
        "alns": alns_conv,
        "cts": cts_conv
    }


def generate_comparison_report(all_comparisons: Dict[str, Any]) -> str:
    """Generate human-readable comparison report."""
    lines = [
        "=" * 70,
        "ALNS vs CTS COMPREHENSIVE COMPARISON",
        "=" * 70,
        "",
    ]
    
    # Objectives
    obj = all_comparisons["objectives"]
    lines.extend([
        "OBJECTIVES:",
        f"  Baseline: Z={obj['baseline']['Z']:.4f}, Z1={obj['baseline']['Z1']:.1f}, "
        f"Z2={obj['baseline']['Z2']:.1f}, Z3={obj['baseline']['Z3']:.1f}",
        f"  ALNS:     Z={obj['alns']['Z']:.4f}, Z1={obj['alns']['Z1']:.1f}, "
        f"Z2={obj['alns']['Z2']:.1f}, Z3={obj['alns']['Z3']:.1f}",
        f"  CTS:      Z={obj['cts']['Z']:.4f}, Z1={obj['cts']['Z1']:.1f}, "
        f"Z2={obj['cts']['Z2']:.1f}, Z3={obj['cts']['Z3']:.1f}",
        "",
        "IMPROVEMENTS vs Baseline:",
        f"  ALNS: Z={obj['improvements']['alns_vs_baseline']['Z']:+.2f}%, "
        f"Z1={obj['improvements']['alns_vs_baseline']['Z1']:+.2f}%, "
        f"Z2={obj['improvements']['alns_vs_baseline']['Z2']:+.2f}%, "
        f"Z3={obj['improvements']['alns_vs_baseline']['Z3']:+.2f}%",
        f"  CTS:  Z={obj['improvements']['cts_vs_baseline']['Z']:+.2f}%, "
        f"Z1={obj['improvements']['cts_vs_baseline']['Z1']:+.2f}%, "
        f"Z2={obj['improvements']['cts_vs_baseline']['Z2']:+.2f}%, "
        f"Z3={obj['improvements']['cts_vs_baseline']['Z3']:+.2f}%",
        "",
        "CTS vs ALNS:",
        f"  Z={obj['improvements']['cts_vs_alns']['Z']:+.2f}%, "
        f"Z1={obj['improvements']['cts_vs_alns']['Z1']:+.2f}%, "
        f"Z2={obj['improvements']['cts_vs_alns']['Z2']:+.2f}%, "
        f"Z3={obj['improvements']['cts_vs_alns']['Z3']:+.2f}%",
        "",
    ])
    
    # Waiting distribution
    wait = all_comparisons["waiting"]
    lines.extend([
        "WAITING TIME DISTRIBUTION:",
        f"  ALNS: mean={wait['alns']['mean']:.1f}s, median={wait['alns']['median']:.1f}s, "
        f"max={wait['alns']['max']:.1f}s, p95={wait['alns']['p95']:.1f}s",
        f"  CTS:  mean={wait['cts']['mean']:.1f}s, median={wait['cts']['median']:.1f}s, "
        f"max={wait['cts']['max']:.1f}s, p95={wait['cts']['p95']:.1f}s",
        "",
        "FAIRNESS METRICS:",
        f"  ALNS: tail_ratio={wait['alns']['tail_ratio']:.2f}, variance={wait['alns']['variance']:.1f}",
        f"  CTS:  tail_ratio={wait['cts']['tail_ratio']:.2f}, variance={wait['cts']['variance']:.1f}",
        "",
    ])
    
    # Route structure
    routes = all_comparisons["routes"]
    lines.extend([
        "ROUTE STRUCTURE:",
        f"  ALNS: {routes['alns']['num_routes']} routes, "
        f"avg_length={routes['alns']['avg_route_length']:.1f} stops",
        f"  CTS:  {routes['cts']['num_routes']} routes, "
        f"avg_length={routes['cts']['avg_route_length']:.1f} stops",
        "",
    ])
    
    # Convergence
    conv = all_comparisons.get("convergence", {})
    if conv.get("alns", {}).get("best_iteration") is not None:
        lines.extend([
            "CONVERGENCE:",
            f"  ALNS: best at iter {conv['alns']['best_iteration']}, "
            f"{conv['alns']['total_improvements']} improvements",
            f"  CTS:  best at iter {conv['cts']['best_iteration']}, "
            f"{conv['cts']['total_improvements']} improvements",
            "",
        ])
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare ALNS and CTS solutions comprehensively")
    parser.add_argument("--baseline", type=str, default="outputs/solutions/baseline.json", help="Baseline solution JSON")
    parser.add_argument("--alns", type=str, default="outputs/solutions/ALNS_MAD.json", help="ALNS solution JSON")
    parser.add_argument("--cts", type=str, default="outputs/solutions/cts_solution.json", help="CTS solution JSON")
    parser.add_argument("--alns-debug", type=str, default="outputs/debug/alns_mad_debug.json", help="ALNS debug JSON")
    parser.add_argument("--cts-debug", type=str, default="outputs/debug/cts_debug.json", help="CTS debug JSON (optional)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.5, help="Z1 weight")
    parser.add_argument("--beta", type=float, default=0.3, help="Z2 weight")
    parser.add_argument("--gamma", type=float, default=0.2, help="Z3 weight")
    parser.add_argument("--use-distance-objective", action="store_true", help="Use distance for Z2")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load solutions
    logger.info("Loading solutions...")
    baseline = load_json(Path(args.baseline))
    alns_solution = load_json(Path(args.alns))
    cts_solution = load_json(Path(args.cts))
    alns_debug = load_json(Path(args.alns_debug))
    cts_debug = load_json(Path(args.cts_debug))
    
    if baseline is None or alns_solution is None or cts_solution is None:
        logger.error("Missing required solution files")
        sys.exit(1)
    
    # Get depots and stops
    depots = baseline.get("depots", [])
    stops_by_id = baseline.get("stops_dict", {})
    
    # Create providers
    cache = iNaviCache(approx_mode=False)
    time_provider, distance_provider = create_osrm_providers(depots, stops_by_id, cache)
    
    # Compute scores
    logger.info("Computing scores...")
    baseline_scores = calc_scores(
        baseline, depots, args.use_distance_objective,
        args.alpha, args.beta, args.gamma,
        time_provider=time_provider, distance_provider=distance_provider,
        cache=cache, baseline_stops_dict=stops_by_id
    )
    
    alns_scores = calc_scores(
        alns_solution, depots, args.use_distance_objective,
        args.alpha, args.beta, args.gamma,
        Z1_star=baseline_scores["Z_star"][0],
        Z2_star=baseline_scores["Z_star"][1],
        Z3_star=baseline_scores["Z_star"][2],
        time_provider=time_provider, distance_provider=distance_provider,
        cache=cache, baseline_stops_dict=stops_by_id
    )
    
    cts_scores = calc_scores(
        cts_solution, depots, args.use_distance_objective,
        args.alpha, args.beta, args.gamma,
        Z1_star=baseline_scores["Z_star"][0],
        Z2_star=baseline_scores["Z_star"][1],
        Z3_star=baseline_scores["Z_star"][2],
        time_provider=time_provider, distance_provider=distance_provider,
        cache=cache, baseline_stops_dict=stops_by_id
    )
    
    # Compare all aspects
    logger.info("Comparing objectives...")
    objectives_comparison = compare_objectives(alns_scores, cts_scores, baseline_scores)
    
    logger.info("Comparing waiting distributions...")
    waiting_comparison = compare_waiting_distributions(
        alns_scores["waiting"], cts_scores["waiting"], stops_by_id
    )
    
    logger.info("Comparing route structures...")
    route_comparison = compare_route_structures(alns_solution, cts_solution)
    
    logger.info("Comparing convergence...")
    convergence_comparison = compare_convergence(alns_debug, cts_debug)
    
    # Compile report
    report = {
        "objectives": objectives_comparison,
        "waiting": waiting_comparison,
        "routes": route_comparison,
        "convergence": convergence_comparison
    }
    
    # Save JSON report
    report_file = output_dir / "validation_comparison_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved comparison report: {report_file}")
    
    # Generate summary
    summary = generate_comparison_report(report)
    summary_file = output_dir / "validation_comparison_summary.txt"
    with open(summary_file, "w") as f:
        f.write(summary)
    logger.info(f"Saved comparison summary: {summary_file}")
    
    # Save CSV metrics
    csv_file = output_dir / "validation_comparison_metrics.csv"
    with open(csv_file, "w") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["Metric", "Baseline", "ALNS", "CTS", "ALNS vs Baseline %", "CTS vs Baseline %", "CTS vs ALNS %"])
        
        obj = objectives_comparison
        w.writerow(["Z", obj["baseline"]["Z"], obj["alns"]["Z"], obj["cts"]["Z"],
                   obj["improvements"]["alns_vs_baseline"]["Z"],
                   obj["improvements"]["cts_vs_baseline"]["Z"],
                   obj["improvements"]["cts_vs_alns"]["Z"]])
        w.writerow(["Z1", obj["baseline"]["Z1"], obj["alns"]["Z1"], obj["cts"]["Z1"],
                   obj["improvements"]["alns_vs_baseline"]["Z1"],
                   obj["improvements"]["cts_vs_baseline"]["Z1"],
                   obj["improvements"]["cts_vs_alns"]["Z1"]])
        w.writerow(["Z2", obj["baseline"]["Z2"], obj["alns"]["Z2"], obj["cts"]["Z2"],
                   obj["improvements"]["alns_vs_baseline"]["Z2"],
                   obj["improvements"]["cts_vs_baseline"]["Z2"],
                   obj["improvements"]["cts_vs_alns"]["Z2"]])
        w.writerow(["Z3", obj["baseline"]["Z3"], obj["alns"]["Z3"], obj["cts"]["Z3"],
                   obj["improvements"]["alns_vs_baseline"]["Z3"],
                   obj["improvements"]["cts_vs_baseline"]["Z3"],
                   obj["improvements"]["cts_vs_alns"]["Z3"]])
        
        wait = waiting_comparison
        w.writerow(["Waiting Mean", "", wait["alns"]["mean"], wait["cts"]["mean"], "", "", ""])
        w.writerow(["Waiting Max", "", wait["alns"]["max"], wait["cts"]["max"], "", "", ""])
        w.writerow(["Tail Ratio", "", wait["alns"]["tail_ratio"], wait["cts"]["tail_ratio"], "", "", ""])
        
        routes = route_comparison
        w.writerow(["Num Routes", "", routes["alns"]["num_routes"], routes["cts"]["num_routes"], "", "", ""])
        w.writerow(["Avg Route Length", "", routes["alns"]["avg_route_length"], routes["cts"]["avg_route_length"], "", "", ""])
    logger.info(f"Saved comparison metrics CSV: {csv_file}")
    
    # Print summary to console
    print("\n" + summary)


if __name__ == "__main__":
    main()
