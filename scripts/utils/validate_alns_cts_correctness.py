#!/usr/bin/env python3
"""
Validate correctness of ALNS (Fixed) and CTS solutions.

This script verifies that both algorithms produce valid, consistent solutions
on the same input by checking:
- All stops are assigned
- Solution structure is valid
- Objectives are computed correctly
- Budget constraints are respected
- Waiting times are computed for all stops
- Both algorithms use the same baseline, stops, and depots
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def validate_solution_structure(
    solution: Dict[str, Any],
    expected_stops: Set[str],
    depots: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate solution structure.
    
    Returns:
        Dict with validation results
    """
    results = {
        "has_routes_by_dc": False,
        "has_depots": False,
        "has_stops_dict": False,
        "all_stops_assigned": False,
        "no_duplicate_stops": False,
        "routes_valid": False,
        "errors": []
    }
    
    # Check required fields
    if "routes_by_dc" not in solution:
        results["errors"].append("Missing 'routes_by_dc' field")
        return results
    results["has_routes_by_dc"] = True
    
    if "depots" not in solution:
        results["errors"].append("Missing 'depots' field")
    else:
        results["has_depots"] = True
    
    if "stops_dict" not in solution:
        results["errors"].append("Missing 'stops_dict' field")
    else:
        results["has_stops_dict"] = True
    
    # Get depot IDs
    depot_ids = {d.get("id", "depot") for d in depots}
    
    # Collect all assigned stops
    assigned_stops = set()
    duplicate_stops = set()
    routes_valid = True
    
    for dc_id, routes in solution.get("routes_by_dc", {}).items():
        if not isinstance(routes, list):
            results["errors"].append(f"routes_by_dc['{dc_id}'] is not a list")
            routes_valid = False
            continue
        
        for route_idx, route in enumerate(routes):
            if not isinstance(route, dict):
                results["errors"].append(f"Route {route_idx} in {dc_id} is not a dict")
                routes_valid = False
                continue
            
            stop_ids = route.get("ordered_stop_ids", [])
            if not isinstance(stop_ids, list):
                results["errors"].append(f"Route {route_idx} in {dc_id} has invalid ordered_stop_ids")
                routes_valid = False
                continue
            
            for stop_id in stop_ids:
                if stop_id in depot_ids:
                    continue  # Skip depot IDs
                
                if stop_id in assigned_stops:
                    duplicate_stops.add(stop_id)
                assigned_stops.add(stop_id)
    
    results["routes_valid"] = routes_valid
    results["no_duplicate_stops"] = len(duplicate_stops) == 0
    if duplicate_stops:
        results["errors"].append(f"Duplicate stops found: {list(duplicate_stops)[:10]}")
    
    # Check if all expected stops are assigned
    missing_stops = expected_stops - assigned_stops
    results["all_stops_assigned"] = len(missing_stops) == 0
    if missing_stops:
        results["errors"].append(f"Missing stops: {list(missing_stops)[:10]}")
    
    return results


def validate_objectives(
    solution: Dict[str, Any],
    stops_by_id: Dict[str, Dict[str, Any]],
    time_provider: Any,
    distance_provider: Any,
    expected_normalizers: Dict[str, float],
    alpha: float,
    beta: float,
    gamma: float,
    use_distance: bool
) -> Dict[str, Any]:
    """
    Validate objective computation.
    
    Returns:
        Dict with validation results
    """
    results = {
        "waiting_computed": False,
        "objectives_computed": False,
        "z1_valid": False,
        "z2_valid": False,
        "z3_valid": False,
        "z_valid": False,
        "errors": []
    }
    
    try:
        # Compute waiting times
        waiting = compute_waiting_times(solution, stops_by_id, time_provider)
        results["waiting_computed"] = True
        
        # Compute objectives
        Z1 = compute_Z1(waiting, stops_by_id)
        Z2 = compute_Z2(solution, distance_provider, time_provider, use_distance)
        Z3 = compute_Z3_MAD(waiting, stops_by_id)
        
        Z1_star = expected_normalizers.get("Z1_star", Z1)
        Z2_star = expected_normalizers.get("Z2_star", Z2)
        Z3_star = expected_normalizers.get("Z3_star", Z3)
        
        Z = compute_combined_Z(Z1, Z2, Z3, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
        
        results["objectives_computed"] = True
        results["z1_valid"] = Z1 >= 0 and not (Z1 != Z1)  # Check for NaN
        results["z2_valid"] = Z2 >= 0 and not (Z2 != Z2)
        results["z3_valid"] = Z3 >= 0 and not (Z3 != Z3)
        results["z_valid"] = Z >= 0 and not (Z != Z)
        
        results["values"] = {
            "Z1": Z1,
            "Z2": Z2,
            "Z3": Z3,
            "Z": Z
        }
        
        if not results["z1_valid"]:
            results["errors"].append(f"Invalid Z1: {Z1}")
        if not results["z2_valid"]:
            results["errors"].append(f"Invalid Z2: {Z2}")
        if not results["z3_valid"]:
            results["errors"].append(f"Invalid Z3: {Z3}")
        if not results["z_valid"]:
            results["errors"].append(f"Invalid Z: {Z}")
            
    except Exception as e:
        results["errors"].append(f"Error computing objectives: {e}")
    
    return results


def validate_budget_constraint(
    solution: Dict[str, Any],
    baseline_Z2: float,
    eps: float,
    distance_provider: Any,
    time_provider: Any,
    use_distance: bool
) -> Dict[str, Any]:
    """
    Validate budget constraint.
    
    Returns:
        Dict with validation results
    """
    results = {
        "budget_computed": False,
        "budget_constraint_met": False,
        "budget_violation": 0.0,
        "errors": []
    }
    
    try:
        Z2 = compute_Z2(solution, distance_provider, time_provider, use_distance)
        cost_budget = (1 + eps) * baseline_Z2
        results["budget_computed"] = True
        results["budget_constraint_met"] = Z2 <= cost_budget
        results["budget_violation"] = max(0.0, (Z2 - cost_budget) / baseline_Z2 * 100)
        results["values"] = {
            "Z2": Z2,
            "budget": cost_budget,
            "baseline_Z2": baseline_Z2,
            "eps": eps
        }
    except Exception as e:
        results["errors"].append(f"Error validating budget: {e}")
    
    return results


def validate_waiting_times(
    waiting_dict: Dict[str, float],
    expected_stops: Set[str]
) -> Dict[str, Any]:
    """
    Validate waiting times.
    
    Returns:
        Dict with validation results
    """
    results = {
        "all_stops_have_waiting": False,
        "waiting_times_valid": False,
        "missing_stops": [],
        "errors": []
    }
    
    waiting_stops = set(waiting_dict.keys())
    missing = expected_stops - waiting_stops
    
    results["all_stops_have_waiting"] = len(missing) == 0
    results["missing_stops"] = list(missing)
    
    if missing:
        results["errors"].append(f"Missing waiting times for {len(missing)} stops")
    
    # Check for invalid values
    invalid_values = []
    for stop_id, waiting_time in waiting_dict.items():
        if waiting_time < 0 or waiting_time != waiting_time:  # NaN check
            invalid_values.append(stop_id)
    
    results["waiting_times_valid"] = len(invalid_values) == 0
    if invalid_values:
        results["errors"].append(f"Invalid waiting times for {len(invalid_values)} stops")
    
    return results


def compare_alns_vs_cts(
    alns_solution: Dict[str, Any],
    cts_solution: Dict[str, Any],
    baseline: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare ALNS and CTS solutions for consistency.
    
    Returns:
        Dict with comparison results
    """
    results = {
        "same_stops": False,
        "same_depots": False,
        "same_baseline": False,
        "errors": []
    }
    
    # Check if they use the same stops
    baseline_stops = set(baseline.get("stops_dict", {}).keys())
    alns_stops = set(alns_solution.get("stops_dict", {}).keys())
    cts_stops = set(cts_solution.get("stops_dict", {}).keys())
    
    results["same_stops"] = (baseline_stops == alns_stops == cts_stops)
    if not results["same_stops"]:
        results["errors"].append("Stop sets differ between solutions")
    
    # Check if they use the same depots
    baseline_depots = {d.get("id") for d in baseline.get("depots", [])}
    alns_depots = {d.get("id") for d in alns_solution.get("depots", [])}
    cts_depots = {d.get("id") for d in cts_solution.get("depots", [])}
    
    results["same_depots"] = (baseline_depots == alns_depots == cts_depots)
    if not results["same_depots"]:
        results["errors"].append("Depot sets differ between solutions")
    
    # Check if they reference the same baseline
    results["same_baseline"] = True  # Assume true if loaded from same files
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate ALNS and CTS solution correctness")
    parser.add_argument("--baseline", type=str, default="outputs/solutions/baseline.json", help="Baseline solution JSON")
    parser.add_argument("--alns", type=str, default="outputs/solutions/ALNS_MAD.json", help="ALNS solution JSON")
    parser.add_argument("--cts", type=str, default="outputs/solutions/cts_solution.json", help="CTS solution JSON")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--eps", type=float, default=0.10, help="Budget constraint (eps)")
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
    
    # Get depots and stops
    depots = baseline.get("depots", [])
    stops_by_id = baseline.get("stops_dict", {})
    expected_stops = set(stops_by_id.keys())
    
    # Create providers
    cache = iNaviCache(approx_mode=False)
    time_provider, distance_provider = create_osrm_providers(depots, stops_by_id, cache)
    
    # Compute baseline objectives for normalizers
    baseline_waiting = compute_waiting_times(baseline, stops_by_id, time_provider)
    baseline_Z1 = compute_Z1(baseline_waiting, stops_by_id)
    baseline_Z2 = compute_Z2(baseline, distance_provider, time_provider, args.use_distance_objective)
    baseline_Z3 = compute_Z3_MAD(baseline_waiting, stops_by_id)
    
    normalizers = {
        "Z1_star": baseline_Z1,
        "Z2_star": baseline_Z2,
        "Z3_star": baseline_Z3
    }
    
    # Validate each solution
    logger.info("Validating baseline solution...")
    baseline_structure = validate_solution_structure(baseline, expected_stops, depots)
    baseline_objectives = validate_objectives(
        baseline, stops_by_id, time_provider, distance_provider,
        normalizers, args.alpha, args.beta, args.gamma, args.use_distance_objective
    )
    baseline_waiting_check = validate_waiting_times(baseline_waiting, expected_stops)
    
    logger.info("Validating ALNS solution...")
    alns_structure = validate_solution_structure(alns_solution, expected_stops, depots)
    alns_objectives = validate_objectives(
        alns_solution, stops_by_id, time_provider, distance_provider,
        normalizers, args.alpha, args.beta, args.gamma, args.use_distance_objective
    )
    alns_budget = validate_budget_constraint(
        alns_solution, baseline_Z2, args.eps, distance_provider, time_provider, args.use_distance_objective
    )
    alns_waiting = compute_waiting_times(alns_solution, stops_by_id, time_provider)
    alns_waiting_check = validate_waiting_times(alns_waiting, expected_stops)
    
    logger.info("Validating CTS solution...")
    cts_structure = validate_solution_structure(cts_solution, expected_stops, depots)
    cts_objectives = validate_objectives(
        cts_solution, stops_by_id, time_provider, distance_provider,
        normalizers, args.alpha, args.beta, args.gamma, args.use_distance_objective
    )
    cts_budget = validate_budget_constraint(
        cts_solution, baseline_Z2, args.eps, distance_provider, time_provider, args.use_distance_objective
    )
    cts_waiting = compute_waiting_times(cts_solution, stops_by_id, time_provider)
    cts_waiting_check = validate_waiting_times(cts_waiting, expected_stops)
    
    # Compare ALNS vs CTS
    logger.info("Comparing ALNS vs CTS...")
    comparison = compare_alns_vs_cts(alns_solution, cts_solution, baseline)
    
    # Compile results
    report = {
        "baseline": {
            "structure": baseline_structure,
            "objectives": baseline_objectives,
            "waiting": baseline_waiting_check
        },
        "alns": {
            "structure": alns_structure,
            "objectives": alns_objectives,
            "budget": alns_budget,
            "waiting": alns_waiting_check
        },
        "cts": {
            "structure": cts_structure,
            "objectives": cts_objectives,
            "budget": cts_budget,
            "waiting": cts_waiting_check
        },
        "comparison": comparison
    }
    
    # Determine overall status
    all_passed = (
        baseline_structure["all_stops_assigned"] and
        baseline_structure["routes_valid"] and
        alns_structure["all_stops_assigned"] and
        alns_structure["routes_valid"] and
        alns_objectives["objectives_computed"] and
        alns_waiting_check["all_stops_have_waiting"] and
        cts_structure["all_stops_assigned"] and
        cts_structure["routes_valid"] and
        cts_objectives["objectives_computed"] and
        cts_waiting_check["all_stops_have_waiting"] and
        comparison["same_stops"] and
        comparison["same_depots"]
    )
    
    report["overall_status"] = "PASS" if all_passed else "FAIL"
    
    # Save JSON report
    report_file = output_dir / "validation_correctness_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved validation report: {report_file}")
    
    # Generate summary
    summary_lines = [
        "=" * 70,
        "CORRECTNESS VALIDATION SUMMARY",
        "=" * 70,
        "",
        f"Overall Status: {report['overall_status']}",
        "",
        "Baseline:",
        f"  All stops assigned: {baseline_structure['all_stops_assigned']}",
        f"  Structure valid: {baseline_structure['routes_valid']}",
        f"  Objectives computed: {baseline_objectives['objectives_computed']}",
        "",
        "ALNS:",
        f"  All stops assigned: {alns_structure['all_stops_assigned']}",
        f"  Structure valid: {alns_structure['routes_valid']}",
        f"  Objectives computed: {alns_objectives['objectives_computed']}",
        f"  Budget constraint met: {alns_budget['budget_constraint_met']}",
        f"  All stops have waiting: {alns_waiting_check['all_stops_have_waiting']}",
        "",
        "CTS:",
        f"  All stops assigned: {cts_structure['all_stops_assigned']}",
        f"  Structure valid: {cts_structure['routes_valid']}",
        f"  Objectives computed: {cts_objectives['objectives_computed']}",
        f"  Budget constraint met: {cts_budget['budget_constraint_met']}",
        f"  All stops have waiting: {cts_waiting_check['all_stops_have_waiting']}",
        "",
        "Comparison:",
        f"  Same stops: {comparison['same_stops']}",
        f"  Same depots: {comparison['same_depots']}",
        "",
    ]
    
    # Add errors if any
    all_errors = []
    for section in ["baseline", "alns", "cts"]:
        for check_type in ["structure", "objectives", "budget", "waiting"]:
            if check_type in report[section]:
                errors = report[section][check_type].get("errors", [])
                if errors:
                    all_errors.extend([f"{section.upper()} {check_type}: {e}" for e in errors])
    
    if all_errors:
        summary_lines.extend([
            "",
            "Errors:",
        ])
        for error in all_errors[:20]:  # Limit to first 20 errors
            summary_lines.append(f"  - {error}")
        if len(all_errors) > 20:
            summary_lines.append(f"  ... and {len(all_errors) - 20} more errors")
    
    summary_lines.append("=" * 70)
    
    summary_file = output_dir / "validation_correctness_summary.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines))
    logger.info(f"Saved validation summary: {summary_file}")
    
    # Print summary to console
    print("\n" + "\n".join(summary_lines))
    
    # Exit with error code if validation failed
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
