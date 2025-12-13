#!/usr/bin/env python3
"""
Analyze trace CSV to find the best solution and fix the saved solution if needed.

This script:
1. Reads the trace CSV to find the iteration with the best Z score
2. Checks proposed_debug.json for best_objectives
3. Compares with the saved improved.json to identify discrepancies
4. Reports findings and suggests fixes
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_trace_csv(trace_file: Path) -> List[Dict[str, Any]]:
    """Load trace CSV file."""
    if not trace_file.exists():
        logger.error(f"Trace file not found: {trace_file}")
        return []
    
    trace = []
    with open(trace_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            try:
                row['iter'] = int(row.get('iter', 0))
                row['Z'] = float(row.get('Z', float('inf')))
                row['Z1'] = float(row.get('Z1', 0))
                row['Z2'] = float(row.get('Z2', 0))
                row['Z3'] = float(row.get('Z3', 0))
                row['accepted'] = row.get('accepted', 'False').lower() == 'true'
                trace.append(row)
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid row: {row}, error: {e}")
    
    logger.info(f"Loaded {len(trace)} iterations from {trace_file}")
    return trace


def find_best_iteration(trace: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the iteration with the best (lowest) Z score."""
    if not trace:
        return None
    
    best = min(trace, key=lambda x: x['Z'])
    logger.info(f"Best iteration found: iter={best['iter']}, Z={best['Z']:.6f}, "
               f"Z1={best['Z1']:.1f}, Z2={best['Z2']:.1f}, Z3={best['Z3']:.1f}, "
               f"accepted={best['accepted']}")
    return best


def load_debug_json(debug_file: Path) -> Optional[Dict[str, Any]]:
    """Load proposed_debug.json."""
    if not debug_file.exists():
        logger.warning(f"Debug file not found: {debug_file}")
        return None
    
    with open(debug_file, 'r') as f:
        debug = json.load(f)
    
    logger.info(f"Loaded debug info from {debug_file}")
    return debug


def load_solution_json(solution_file: Path) -> Optional[Dict[str, Any]]:
    """Load solution JSON file."""
    if not solution_file.exists():
        logger.warning(f"Solution file not found: {solution_file}")
        return None
    
    with open(solution_file, 'r') as f:
        solution = json.load(f)
    
    logger.info(f"Loaded solution from {solution_file}")
    return solution


def analyze_discrepancy(
    trace: List[Dict[str, Any]],
    debug: Optional[Dict[str, Any]],
    solution: Optional[Dict[str, Any]],
    baseline: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze discrepancies between trace, debug, and saved solution."""
    results = {
        "best_from_trace": None,
        "best_from_debug": None,
        "saved_solution_info": None,
        "discrepancy_found": False,
        "recommendations": []
    }
    
    # Find best from trace
    best_iter = find_best_iteration(trace)
    if best_iter:
        results["best_from_trace"] = {
            "iter": best_iter['iter'],
            "Z": best_iter['Z'],
            "Z1": best_iter['Z1'],
            "Z2": best_iter['Z2'],
            "Z3": best_iter['Z3'],
            "accepted": best_iter['accepted']
        }
    
    # Check debug best_objectives
    if debug:
        best_obj = debug.get("best_objectives", {})
        if best_obj:
            results["best_from_debug"] = {
                "Z1": best_obj.get("Z1"),
                "Z2": best_obj.get("Z2"),
                "Z3": best_obj.get("Z3")
            }
            
            # Compute Z from best_objectives if we have normalizers
            normalizers = debug.get("normalizers", {})
            if normalizers and best_obj:
                Z1_star = normalizers.get("Z1_star")
                Z2_star = normalizers.get("Z2_star")
                Z3_star = normalizers.get("Z3_star")
                if Z1_star and Z2_star and Z3_star:
                    # Need alpha, beta, gamma - try to infer from baseline
                    baseline_obj = debug.get("baseline_objectives", {})
                    if baseline_obj:
                        # Estimate weights (default values)
                        alpha, beta, gamma = 0.5, 0.3, 0.2
                        Z_best = (alpha * (best_obj["Z1"] / Z1_star) +
                                 beta * (best_obj["Z2"] / Z2_star) +
                                 gamma * (best_obj["Z3"] / Z3_star))
                        results["best_from_debug"]["Z"] = Z_best
    
    # Check if there's a discrepancy
    if best_iter and results["best_from_debug"]:
        trace_Z1 = best_iter['Z1']
        debug_Z1 = results["best_from_debug"].get("Z1")
        
        if debug_Z1 and abs(trace_Z1 - debug_Z1) > 0.1:
            results["discrepancy_found"] = True
            results["recommendations"].append(
                f"DISCREPANCY: Trace shows best Z1={trace_Z1:.1f} at iter={best_iter['iter']}, "
                f"but debug shows Z1={debug_Z1:.1f}"
            )
    
    # Check saved solution (if we can compute its objectives)
    if solution and baseline:
        results["saved_solution_info"] = {
            "has_routes": "routes_by_dc" in solution,
            "num_dcs": len(solution.get("routes_by_dc", {})),
            "has_stops_dict": "stops_dict" in solution
        }
        results["recommendations"].append(
            "To verify saved solution, recompute its objectives using the same providers as the experiment"
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trace to find best solution and identify discrepancies"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory containing trace and solution files"
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        default=None,
        help="Specific trace file path (if not provided, will search in output-dir/traces/)"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Find trace file
    if args.trace_file:
        trace_file = Path(args.trace_file)
    else:
        # Search for trace files
        trace_dir = output_dir / "traces"
        trace_files = list(trace_dir.glob("*_proposed.csv")) if trace_dir.exists() else []
        if not trace_files:
            logger.error(f"No trace files found in {trace_dir}")
            logger.error("Please specify --trace-file or ensure trace files exist in outputs/traces/")
            sys.exit(1)
        # Use the most recent one
        trace_file = max(trace_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using trace file: {trace_file}")
    
    # Load files
    trace = load_trace_csv(trace_file)
    if not trace:
        logger.error("No trace data loaded!")
        sys.exit(1)
    
    debug_file = output_dir / "proposed_debug.json"
    debug = load_debug_json(debug_file)
    
    solution_file = output_dir / "improved.json"
    solution = load_solution_json(solution_file)
    
    baseline_file = output_dir / "baseline.json"
    baseline = load_solution_json(baseline_file)
    
    # Analyze
    results = analyze_discrepancy(trace, debug, solution, baseline)
    
    # Print report
    print("\n" + "=" * 80)
    print("TRACE ANALYSIS REPORT")
    print("=" * 80)
    
    if results["best_from_trace"]:
        best = results["best_from_trace"]
        print(f"\nBest solution from TRACE (iteration {best['iter']}):")
        print(f"  Z  = {best['Z']:.6f}")
        print(f"  Z1 = {best['Z1']:.1f}")
        print(f"  Z2 = {best['Z2']:.1f}")
        print(f"  Z3 = {best['Z3']:.1f}")
        print(f"  Accepted = {best['accepted']}")
    
    if results["best_from_debug"]:
        best_debug = results["best_from_debug"]
        print(f"\nBest solution from DEBUG (proposed_debug.json):")
        if "Z" in best_debug:
            print(f"  Z  = {best_debug['Z']:.6f}")
        print(f"  Z1 = {best_debug.get('Z1', 'N/A')}")
        print(f"  Z2 = {best_debug.get('Z2', 'N/A')}")
        print(f"  Z3 = {best_debug.get('Z3', 'N/A')}")
    
    if results["discrepancy_found"]:
        print("\n" + "!" * 80)
        print("DISCREPANCY DETECTED!")
        print("!" * 80)
        for rec in results["recommendations"]:
            print(f"  - {rec}")
    
    # Show all iterations with Z < baseline (if we can determine baseline)
    if debug:
        baseline_obj = debug.get("baseline_objectives", {})
        normalizers = debug.get("normalizers", {})
        if baseline_obj and normalizers:
            # Estimate baseline Z
            alpha, beta, gamma = 0.5, 0.3, 0.2  # Default weights
            Z1_0 = baseline_obj.get("Z1")
            Z2_0 = baseline_obj.get("Z2")
            Z3_0 = baseline_obj.get("Z3")
            Z1_star = normalizers.get("Z1_star")
            Z2_star = normalizers.get("Z2_star")
            Z3_star = normalizers.get("Z3_star")
            
            if all(v is not None for v in [Z1_0, Z2_0, Z3_0, Z1_star, Z2_star, Z3_star]):
                baseline_Z = (alpha * (Z1_0 / Z1_star) +
                             beta * (Z2_0 / Z2_star) +
                             gamma * (Z3_0 / Z3_star))
                
                print(f"\nBaseline Z (estimated): {baseline_Z:.6f}")
                print(f"\nIterations with Z < baseline:")
                better_iters = [t for t in trace if t['Z'] < baseline_Z]
                print(f"  Found {len(better_iters)} iterations better than baseline")
                if better_iters:
                    print(f"  Best: iter={better_iters[0]['iter']}, Z={better_iters[0]['Z']:.6f}")
                    print(f"  Worst of better: iter={better_iters[-1]['iter']}, Z={better_iters[-1]['Z']:.6f}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
1. The trace shows the best Z found during iterations, but the saved solution
   might not match if there was a bug in tracking best_solution.

2. To fix this, you would need to:
   a) Re-run the experiment (if you need the actual solution structure)
   b) OR use the best_objectives from proposed_debug.json if they are correct
   c) OR check if intermediate solutions were saved

3. The discrepancy suggests that best_solution was not properly maintained
   during the ALNS search. This is a bug in proposed_algorithm.py that should
   be fixed for future runs.
    """)
    
    # Save analysis to file
    analysis_file = output_dir / "trace_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved analysis to {analysis_file}")


if __name__ == "__main__":
    main()
