#!/usr/bin/env python3
"""
Regenerate plot from existing JSON files (baseline.json, improved.json, variance solution).
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.vrp_fairness.objectives import compute_waiting_times
from src.vrp_fairness.osrm_provider import create_osrm_providers
from src.vrp_fairness.inavi import iNaviCache

# Import plot function
from scripts.compare_Z3_variance_vs_MAD import plot_Z3_comparison_histograms

def main():
    output_dir = Path("outputs")
    
    print("Loading solutions...")
    # Load solutions
    baseline_path = output_dir / "baseline.json"
    mad_path = output_dir / "improved.json"
    variance_debug_path = output_dir / "variance_vs_mad_VARIANCE_debug.json"
    
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing {baseline_path}")
    if not mad_path.exists():
        raise FileNotFoundError(f"Missing {mad_path}")
    
    baseline_solution = json.loads(baseline_path.read_text())
    mad_solution = json.loads(mad_path.read_text())
    print(f"  Loaded baseline: {len(baseline_solution.get('routes_by_dc', {}))} depots")
    print(f"  Loaded MAD solution: {len(mad_solution.get('routes_by_dc', {}))} depots")
    
    depots = baseline_solution.get("depots", [])
    stops_by_id = baseline_solution.get("stops_dict", {})
    
    # Load variance solution if available
    variance_solution = None
    if variance_debug_path.exists():
        variance_debug = json.loads(variance_debug_path.read_text())
        variance_solution = variance_debug.get("best_solution")
    
    if variance_solution is None:
        print("Warning: variance solution not found. Using baseline for variance plot.")
        variance_solution = baseline_solution
    
    # Create providers
    print("Creating OSRM providers...")
    cache = iNaviCache(approx_mode=False)
    time_provider, distance_provider = create_osrm_providers(depots, stops_by_id, cache)
    
    # Compute waiting times
    print("Computing waiting times...")
    baseline_waiting = compute_waiting_times(baseline_solution, stops_by_id, time_provider)
    mad_waiting = compute_waiting_times(mad_solution, stops_by_id, time_provider)
    variance_waiting = compute_waiting_times(variance_solution, stops_by_id, time_provider)
    print(f"  Baseline: {len(baseline_waiting)} stops")
    print(f"  MAD: {len(mad_waiting)} stops")
    print(f"  Variance: {len(variance_waiting)} stops")
    
    # Collect raw waits and their household weights
    def _raw_waits_with_weights(waiting):
        vals, wts = [], []
        for stop_id, w in waiting.items():
            wt = stops_by_id.get(stop_id, {}).get("households", stops_by_id.get(stop_id, {}).get("demand", 1))
            if wt in (None, 0):
                wt = 1
            vals.append(w)  # Raw waiting time
            wts.append(wt)  # Household weight
        return vals, wts
    
    baseline_waits, baseline_wts = _raw_waits_with_weights(baseline_waiting)
    variance_waits, variance_wts = _raw_waits_with_weights(variance_waiting)
    mad_waits, mad_wts = _raw_waits_with_weights(mad_waiting)
    
    # Plot histograms
    print("Generating plot...")
    hist_output = output_dir / "variance_vs_mad_wait_hist"
    plot_Z3_comparison_histograms(
        baseline_waits=baseline_waits,
        variance_waits=variance_waits,
        mad_waits=mad_waits,
        city_name="daejeon",
        output_path=hist_output,
        baseline_weights=baseline_wts,
        variance_weights=variance_wts,
        mad_weights=mad_wts,
    )
    
    print(f"✓ Plot regenerated: {hist_output}.png")

if __name__ == "__main__":
    main()
