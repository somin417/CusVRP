#!/usr/bin/env python3
"""
Generate weighted_waiting_times_abc_balance_models.png from three ABC balance models.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.generate_from_json import generate_weighted_waiting_plot, load_solutions
from scripts.utils.utils import load_json

def main():
    output_dir = Path("outputs")
    solutions_dir = output_dir / "solutions"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    best_solutions_dir = output_dir / "solutions" / "abc_balance"
    baseline_path = solutions_dir / "baseline.json"
    
    if not baseline_path.exists():
        print(f"Error: {baseline_path} not found")
        sys.exit(1)
    
    # Load baseline to get depots and stops
    baseline = load_json(baseline_path)
    depots = baseline.get("depots", [])
    stops_by_id = baseline.get("stops_dict", {})
    
    # Load three model solutions
    solutions = {}
    model_files = [
        ("z1_focused", best_solutions_dir / "cts_z1_focused_best.json"),
        ("balanced", best_solutions_dir / "cts_balanced_best.json"),
        ("z3_focused", best_solutions_dir / "cts_z3_focused_best.json"),
    ]
    
    for model_name, file_path in model_files:
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue
        
        solution_data = load_json(file_path)
        solutions[model_name] = solution_data.get("solution", solution_data)
        print(f"Loaded {model_name} from {file_path}")
    
    if not solutions:
        print("Error: No solutions loaded!")
        sys.exit(1)
    
    # Generate weighted waiting plot
    output_path = plots_dir / "weighted_waiting_times_abc_balance_models"
    generate_weighted_waiting_plot(
        solutions=solutions,
        depots=depots,
        stops_by_id=stops_by_id,
        output_path=output_path,
        city_name="daejeon"
    )
    
    print(f"\nGenerated: {output_path}.png")

if __name__ == "__main__":
    main()


