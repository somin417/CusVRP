#!/usr/bin/env python3
"""
Generate waiting_times_baseline_alns_cts.png and weighted_waiting_times_baseline_alns_cts.png with baseline, ALNS, and CTS.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.generate_from_json import generate_waiting_plot, generate_weighted_waiting_plot
from scripts.utils.utils import load_json

def main():
    output_dir = Path("outputs")
    solutions_dir = output_dir / "solutions"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    baseline_path = solutions_dir / "baseline.json"
    alns_path = solutions_dir / "ALNS_MAD.json"  # ALNS solution
    cts_path = solutions_dir / "cts_solution.json"
    
    if not baseline_path.exists():
        print(f"Error: {baseline_path} not found")
        sys.exit(1)
    
    if not alns_path.exists():
        print(f"Error: {alns_path} not found")
        sys.exit(1)
    
    if not cts_path.exists():
        print(f"Error: {cts_path} not found")
        sys.exit(1)
    
    # Load baseline to get depots and stops
    baseline = load_json(baseline_path)
    depots = baseline.get("depots", [])
    stops_by_id = baseline.get("stops_dict", {})
    
    # Load solutions
    solutions = {}
    
    # Add baseline
    solutions["baseline"] = baseline
    print(f"Loaded baseline from {baseline_path}")
    
    # Add ALNS (not "improved", use "ALNS" as label)
    alns_solution = load_json(alns_path)
    solutions["ALNS"] = alns_solution
    print(f"Loaded ALNS from {alns_path}")
    
    # Add CTS
    cts_solution = load_json(cts_path)
    solutions["CTS"] = cts_solution
    print(f"Loaded CTS from {cts_path}")
    
    # Generate waiting plot
    waiting_output_path = plots_dir / "waiting_times_baseline_alns_cts"
    generate_waiting_plot(
        solutions=solutions,
        depots=depots,
        stops_by_id=stops_by_id,
        output_path=waiting_output_path,
        city_name="daejeon"
    )
    print(f"\nGenerated: {waiting_output_path}.png")
    
    # Generate weighted waiting plot
    weighted_output_path = plots_dir / "weighted_waiting_times_baseline_alns_cts"
    generate_weighted_waiting_plot(
        solutions=solutions,
        depots=depots,
        stops_by_id=stops_by_id,
        output_path=weighted_output_path,
        city_name="daejeon"
    )
    print(f"Generated: {weighted_output_path}.png")

if __name__ == "__main__":
    main()

