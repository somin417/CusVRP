#!/usr/bin/env python3
"""
Smoke test for proposed algorithm on real GPKG data.
"""

import argparse
import sys
import json
import csv
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vrp_fairness.data import load_stops_from_gpkg
from src.vrp_fairness.vroom_vrp import solve_multi_depot
from src.vrp_fairness.proposed_algorithm import proposed_algorithm
from src.vrp_fairness.objectives import (
    compute_waiting_times, compute_Z1, compute_Z2, compute_Z3, compute_combined_Z
)
from src.vrp_fairness.osrm_provider import create_osrm_providers
from src.vrp_fairness.inavi import iNaviCache


def main():
    parser = argparse.ArgumentParser(description="Test proposed algorithm on real data")
    parser.add_argument("--gpkg", type=str, default="data/yuseong_housing_3__point.gpkg", help="GeoPackage file path")
    parser.add_argument("--layer", type=str, default="yuseong_housing_2__point", help="Layer name")
    parser.add_argument("--n", type=int, default=30, help="Number of stops to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    
    # Load stops
    if args.gpkg:
        stops = load_stops_from_gpkg(
            gpkg_path=args.gpkg,
            layer=args.layer,
            n=args.n,
            seed=args.seed,
            service_time_s=300
        )
        print(f"Loaded {len(stops)} stops from GPKG")
    else:
        print("Error: --gpkg required")
        sys.exit(1)
    
    # Set n_i from households if available, otherwise 1
    for stop in stops:
        if not hasattr(stop, 'meta'):
            stop.meta = {}
        if not hasattr(stop, 'households'):
            stop.households = stop.meta.get("n_i", 1)
    
    # Define depots (2 depots)
    import random
    random.seed(args.seed)
    bounds = {
        "lat_min": min(s.lat for s in stops),
        "lat_max": max(s.lat for s in stops),
        "lon_min": min(s.lon for s in stops),
        "lon_max": max(s.lon for s in stops)
    }
    
    depots = [
        {"id": "DC1", "lat": bounds["lat_min"] + 0.3 * (bounds["lat_max"] - bounds["lat_min"]),
         "lon": bounds["lon_min"] + 0.3 * (bounds["lon_max"] - bounds["lon_min"])},
        {"id": "DC2", "lat": bounds["lat_min"] + 0.7 * (bounds["lat_max"] - bounds["lat_min"]),
         "lon": bounds["lon_min"] + 0.7 * (bounds["lon_max"] - bounds["lon_min"])}
    ]
    
    # Vehicles (3 per depot)
    vehicles = {}
    for depot in depots:
        vehicles[depot["id"]] = [
            {"id": f"{depot['id']}_v{i+1}", "capacity": 100, "depot_id": depot["id"]}
            for i in range(3)
        ]
    
    # Assign stops to depots (simple nearest)
    from src.vrp_fairness.assignment import assign_stops_to_depots, create_osrm_time_provider
    cache = iNaviCache(approx_mode=False)
    time_provider_latlon = create_osrm_time_provider(cache)
    stops_by_depot = assign_stops_to_depots(stops, depots, time_provider_latlon)
    
    # Convert to dict format
    stops_by_depot_dict = {}
    stops_by_id = {}
    for depot_id, assigned_stops in stops_by_depot.items():
        stops_by_depot_dict[depot_id] = [
            {
                "id": stop.id,
                "lat": stop.lat,
                "lon": stop.lon,
                "demand": stop.demand,
                "service_time_s": stop.service_time,
                "households": stop.households
            }
            for stop in assigned_stops
        ]
        for stop in assigned_stops:
            stops_by_id[stop.id] = {
                "lat": stop.lat,
                "lon": stop.lon,
                "demand": stop.demand,
                "service_time_s": stop.service_time,
                "households": stop.households
            }
    
    # Create OSRM providers
    time_provider, distance_provider = create_osrm_providers(depots, stops_by_id, cache)
    
    # Run baseline
    print("\n=== Running Baseline ===")
    baseline = solve_multi_depot(
        depots=depots,
        vehicles_by_depot=vehicles,
        stops_by_depot=stops_by_depot_dict,
        request_geometry=True
    )
    baseline["depots"] = depots
    baseline["stops_dict"] = stops_by_id
    
    waiting_baseline = compute_waiting_times(baseline, stops_by_id, time_provider)
    Z1_baseline = compute_Z1(waiting_baseline, stops_by_id)
    Z2_baseline = compute_Z2(baseline, distance_provider, time_provider, args.use_distance)
    Z3_baseline = compute_Z3(waiting_baseline, stops_by_id)
    Z_baseline = compute_combined_Z(Z1_baseline, Z2_baseline, Z3_baseline,
                                    Z1_baseline, Z2_baseline, Z3_baseline,
                                    args.alpha, args.beta, args.gamma)
    
    print(f"Baseline: Z={Z_baseline:.3f}, Z1={Z1_baseline:.1f}, Z2={Z2_baseline:.1f}, Z3={Z3_baseline:.1f}")
    
    # Run proposed
    print("\n=== Running Proposed Algorithm ===")
    proposed, debug = proposed_algorithm(
        depots=depots,
        vehicles=vehicles,
        stops_by_depot=stops_by_depot_dict,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        eps=args.eps,
        iters=args.iters,
        seed=args.seed,
        time_provider=time_provider,
        distance_provider=distance_provider,
        normalize="baseline",
        use_distance_objective=args.use_distance,
        enforce_capacity=args.enforce_capacity
    )
    proposed["depots"] = depots
    proposed["stops_dict"] = stops_by_id
    
    waiting_proposed = compute_waiting_times(proposed, stops_by_id, time_provider)
    Z1_proposed = compute_Z1(waiting_proposed, stops_by_id)
    Z2_proposed = compute_Z2(proposed, distance_provider, time_provider, args.use_distance)
    Z3_proposed = compute_Z3(waiting_proposed, stops_by_id)
    Z_proposed = compute_combined_Z(Z1_proposed, Z2_proposed, Z3_proposed,
                                     Z1_baseline, Z2_baseline, Z3_baseline,
                                     args.alpha, args.beta, args.gamma)
    
    print(f"Proposed: Z={Z_proposed:.3f}, Z1={Z1_proposed:.1f}, Z2={Z2_proposed:.1f}, Z3={Z3_proposed:.1f}")
    
    # Improvements
    print("\n=== Improvements ===")
    print(f"Z:  {Z_baseline:.3f} -> {Z_proposed:.3f} ({((Z_baseline-Z_proposed)/Z_baseline*100):.1f}% improvement)")
    print(f"Z1: {Z1_baseline:.1f} -> {Z1_proposed:.1f} ({((Z1_baseline-Z1_proposed)/Z1_baseline*100):.1f}% improvement)")
    print(f"Z2: {Z2_baseline:.1f} -> {Z2_proposed:.1f} ({((Z2_baseline-Z2_proposed)/Z2_baseline*100):.1f}% improvement)")
    print(f"Z3: {Z3_baseline:.1f} -> {Z3_proposed:.1f} ({((Z3_baseline-Z3_proposed)/Z3_baseline*100):.1f}% improvement)")
    
    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "solutions").mkdir(exist_ok=True)
    (output_dir / "traces").mkdir(exist_ok=True)
    
    run_id = f"seed{args.seed}_n{args.n}"
    
    # Save solution
    solution_file = output_dir / "solutions" / f"{run_id}_proposed.json"
    with open(solution_file, 'w') as f:
        json.dump(proposed, f, indent=2)
    print(f"\nSolution saved: {solution_file}")
    
    # Save trace
    trace_file = output_dir / "traces" / f"{run_id}_proposed.csv"
    with open(trace_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "Z", "Z1", "Z2", "Z3", "accepted", "k_removed"])
        writer.writeheader()
        writer.writerows(debug["trace"])
    print(f"Trace saved: {trace_file}")
    
    print("\n" + "=" * 80)
    print("Example command:")
    print(f"python scripts/test_proposed_on_realdata.py --gpkg {args.gpkg} --layer {args.layer} "
          f"--n {args.n} --alpha {args.alpha} --beta {args.beta} --gamma {args.gamma} "
          f"--eps {args.eps} --iters {args.iters} --seed {args.seed}")
    print("=" * 80)


if __name__ == "__main__":
    main()

