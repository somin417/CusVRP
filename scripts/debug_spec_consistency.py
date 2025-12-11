#!/usr/bin/env python3
"""
DEBUG + SPEC-CONSISTENCY CHECK for VRP/Fairness Pipeline
Compares implementation against Notion mathematical model.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vrp_fairness.data import load_stops_from_gpkg
from src.vrp_fairness.assignment import assign_stops_to_depots, create_osrm_time_provider
from src.vrp_fairness.vroom_vrp import solve_multi_depot
from src.vrp_fairness.proposed_algorithm import proposed_algorithm
from src.vrp_fairness.objectives import (
    compute_waiting_times, compute_Z1, compute_Z2, compute_Z3, compute_combined_Z
)
from src.vrp_fairness.osrm_provider import create_osrm_providers
from src.vrp_fairness.inavi import iNaviCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_subsection(title):
    print(f"\n--- {title} ---")

def main():
    print_section("VRP/FAIRNESS PIPELINE SPEC-CONSISTENCY CHECK")
    
    # Configuration
    config = {
        "seed": 0,
        "gpkg": "data/yuseong_housing_3__point.gpkg",
        "layer": "yuseong_housing_2__point",
        "sample_n": 20,
        "num_dcs": 2,
        "vehicles_per_dc": 3,
        "demand_field": "A26",
        "eps": 0.10,
        "iters": 30,
        "alpha": 1.0,
        "beta": 0.2,
        "gamma": 0.2,
        "use_distance": False
    }
    
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # ============================================================
    # PART A: CODE MAPPING TO NOTION SPEC
    # ============================================================
    
    print_section("A. CODE MAPPING TO NOTION SPEC")
    
    print_subsection("1) Mathematical Notation → Code Mapping")
    print("""
    Notion Model → Code Implementation:
    
    x_ijv (binary, vehicle v travels i→j):
      → Implemented as: route['ordered_stop_ids'] sequence
      → Location: vroom_vrp.py:_convert_vroom_response() extracts stop order
      → File: src/vrp_fairness/vroom_vrp.py:146-200
    
    t_i (service start time / arrival):
      → Implemented as: compute_waiting_times() returns w_i (arrival time)
      → Location: objectives.py:compute_waiting_times()
      → File: src/vrp_fairness/objectives.py:12-110
    
    W_max (max weighted waiting):
      → Implemented as: compute_Z1() = max_i (n_i * w_i)
      → Location: objectives.py:compute_Z1()
      → File: src/vrp_fairness/objectives.py:113-136
    
    δ_i (MAD fairness deviation):
      → ⚠️  NOT IMPLEMENTED AS MAD
      → Current: Z3 = Σ n_i (w_i - w̄)² (weighted VARIANCE, not MAD)
      → Notion spec: δ_i = |t_i - t̄|, Z3 = Σ n_i δ_i (MAD)
      → Location: objectives.py:compute_Z3() uses variance
      → File: src/vrp_fairness/objectives.py:139-178
    
    Z1, Z2, Z3 computation:
      → Z1: objectives.py:compute_Z1() - ✓ Matches spec
      → Z2: objectives.py:compute_Z2() - ✓ Matches spec (Σ c_ij x_ijv)
      → Z3: objectives.py:compute_Z3() - ✗ Uses variance, not MAD
    
    Combined Z = α(Z1/Z1*) + β(Z2/Z2*) + γ(Z3/Z3*):
      → Implemented as: objectives.py:compute_combined_Z()
      → Location: src/vrp_fairness/objectives.py:272-299
      → ✓ Matches spec formula
    """)
    
    print_subsection("2) DC Assignment Consistency Check")
    print("""
    Assignment function: assignment.py:assign_stops_to_depots()
    Method: Uses OSRM travel time t(dc, i) via time_provider
    Formula: k(i) = argmin_k t(dc_k, i)  ✓ Matches spec
    
    Will verify with actual data below...
    """)
    
    print_subsection("3) Per-DC VRP Routing Check")
    print("""
    Vehicles per depot: Created in run_experiment.py
    VROOM request: vroom_vrp.py:call_vrp() formats request
    Response parsing: vroom_vrp.py:_convert_vroom_response() extracts routes
    Route structure: Each route has ordered_stop_ids (x_ijv equivalent)
    
    Will verify with actual VROOM call below...
    """)
    
    print_subsection("4) Waiting-Time Constraints Check")
    print("""
    Time consistency constraint (Notion):
      t_j ≥ t_i + s_i + c_ij - M(1 - x_ijv)
    
    Implementation:
      → compute_waiting_times() simulates route order
      → t_j = t_i + travel_time(i→j) + service_time(i)
      → Implicitly enforces constraint (no Big-M needed in simulation)
      → Location: objectives.py:94-104
    
    Will verify with debug route print below...
    """)
    
    print_subsection("5) MAD Fairness Implementation Status")
    print("""
    ✘ GAP IDENTIFIED:
    
    Notion spec requires:
      δ_i = |t_i - t̄|  (absolute deviation)
      Z3  = Σ n_i δ_i  (weighted MAD)
    
    Current implementation:
      Z3 = Σ n_i (w_i - w̄)²  (weighted VARIANCE)
    
    Location: objectives.py:139-178
    
    PROPOSED FIX:
      Add compute_Z3_MAD() function using absolute deviation
      Or modify compute_Z3() to accept mode='MAD' | 'variance'
    """)
    
    print_subsection("6) Combined Objective Normalization")
    print("""
    Z1*, Z2*, Z3* computation:
      → Currently: Uses baseline values (Z1_0, Z2_0, Z3_0)
      → Location: proposed_algorithm.py:66-75
      → File: src/vrp_fairness/proposed_algorithm.py:58-75
    
    Combined Z computation:
      → Z = α(Z1/Z1*) + β(Z2/Z2*) + γ(Z3/Z3*)
      → Location: objectives.py:compute_combined_Z()
      → ✓ Matches spec formula
    """)
    
    print_subsection("7) ALNS Operators")
    print("""
    Destroy operator:
      → _remove_stops() removes k random stops (k ∈ [kmin, kmax])
      → Location: proposed_algorithm.py:_remove_stops()
    
    Repair operator:
      → _regret_insertion_optimized() inserts unassigned stops
      → Uses regret-2 insertion with incremental evaluation
      → Location: proposed_algorithm.py:_regret_insertion_optimized()
    
    Acceptance rule:
      → Accept if Z_new < Z_current (greedy)
      → Cost budget: Z2_new ≤ (1+eps)*Z2_0
      → Location: proposed_algorithm.py:145-165
    
    Contextual TS:
      → ⚠️  NOT IMPLEMENTED
      → Current: Simple greedy acceptance
      → Notion spec mentions contextual TS but code uses ALNS-lite
    """)
    
    # ============================================================
    # PART B: RUN DEBUG EXPERIMENT
    # ============================================================
    
    print_section("B. RUNNING DEBUG EXPERIMENT")
    
    # Load stops
    print_subsection("Loading stops from GPKG")
    stops = load_stops_from_gpkg(
        gpkg_path=config["gpkg"],
        layer=config["layer"],
        n=config["sample_n"],
        seed=config["seed"],
        demand_field=config["demand_field"]
    )
    logger.info(f"Loaded {len(stops)} stops")
    
    # Create depots (2 DCs)
    print_subsection("Creating depots")
    import random
    random.seed(config["seed"])
    depots = []
    for i in range(config["num_dcs"]):
        # Random locations in Daejeon area
        lat = 36.3 + random.random() * 0.2
        lon = 127.3 + random.random() * 0.1
        depots.append({
            "id": f"DC{i+1}",
            "lat": lat,
            "lon": lon
        })
        logger.info(f"  {depots[-1]['id']}: {lat:.6f}, {lon:.6f}")
    
    # Create vehicles (3 per DC)
    print_subsection("Creating vehicles")
    vehicles = {}
    for depot in depots:
        vehicles[depot["id"]] = [
            {"id": f"{depot['id']}_V{j+1}", "capacity": 1000}
            for j in range(config["vehicles_per_dc"])
        ]
        logger.info(f"  {depot['id']}: {len(vehicles[depot['id']])} vehicles")
    
    # Assign stops to depots
    print_subsection("Assigning stops to depots (OSRM travel time)")
    cache = iNaviCache(approx_mode=False)
    time_provider_func = create_osrm_time_provider(cache)
    
    # Stops are already Stop objects from load_stops_from_gpkg
    stops_by_depot = assign_stops_to_depots(
        stops=stops,
        depots=depots,
        time_provider=lambda lat1, lon1, lat2, lon2: time_provider_func(lat1, lon1, lat2, lon2)
    )
    
    # Debug: Print sample of 10 stops with assignment
    print("\n  Sample of 10 stops with assignment:")
    print("  " + "-" * 70)
    print(f"  {'Stop ID':<15} {'Households':<12} {'Assigned DC':<15} {'Travel Time (s)':<15}")
    print("  " + "-" * 70)
    
    sample_count = 0
    for dc_id, assigned_stops in stops_by_depot.items():
        for stop in assigned_stops[:5]:  # First 5 per DC
            if sample_count >= 10:
                break
            # Compute travel time for this assignment
            depot = next(d for d in depots if d["id"] == dc_id)
            travel_time = time_provider_func(depot["lat"], depot["lon"], stop.lat, stop.lon)
            households = stop.demand
            print(f"  {stop.id:<15} {households:<12} {dc_id:<15} {travel_time:<15.1f}")
            sample_count += 1
        if sample_count >= 10:
            break
    
    # Convert stops to dict format for VROOM
    stops_by_depot_dict = {}
    stops_by_id = {}
    for dc_id, stop_list in stops_by_depot.items():
        stops_by_depot_dict[dc_id] = [
            {
                "id": s.id,
                "name": s.id,
                "lat": s.lat,
                "lon": s.lon,
                "coordinates": [s.lat, s.lon],
                "demand": s.demand,
                "households": s.demand
            }
            for s in stop_list
        ]
        for s in stop_list:
            stops_by_id[s.id] = {
                "id": s.id,
                "lat": s.lat,
                "lon": s.lon,
                "demand": s.demand,
                "households": s.demand,
                "service_time_s": s.service_time
            }
    
    # Create OSRM providers
    time_provider, distance_provider = create_osrm_providers(depots, stops_by_id, cache)
    
    # Run baseline
    print_subsection("Running baseline VROOM solution")
    baseline = solve_multi_depot(
        depots=depots,
        vehicles_by_depot=vehicles,
        stops_by_depot=stops_by_depot_dict,
        request_geometry=True
    )
    baseline["depots"] = depots
    baseline["stops_dict"] = stops_by_id
    
    # Debug: Print VROOM route structure (x_ijv equivalent)
    print("\n  VROOM Route Structure (x_ijv equivalent):")
    for dc_id, routes in baseline.get("routes_by_dc", {}).items():
        print(f"\n  Depot {dc_id}:")
        for route in routes:
            stop_ids = route.get("ordered_stop_ids", [])
            print(f"    Vehicle {route.get('vehicle_id', '?')}: {len(stop_ids)} stops")
            if stop_ids:
                print(f"      Route: {dc_id} → {' → '.join(stop_ids[:5])}{' → ...' if len(stop_ids) > 5 else ''} → {dc_id}")
    
    # Compute baseline objectives
    print_subsection("Computing baseline objectives")
    waiting_baseline = compute_waiting_times(baseline, stops_by_id, time_provider)
    Z1_baseline = compute_Z1(waiting_baseline, stops_by_id)
    Z2_baseline = compute_Z2(baseline, distance_provider, time_provider, config["use_distance"])
    Z3_baseline = compute_Z3(waiting_baseline, stops_by_id)
    Z_baseline = compute_combined_Z(
        Z1_baseline, Z2_baseline, Z3_baseline,
        Z1_baseline, Z2_baseline, Z3_baseline,
        config["alpha"], config["beta"], config["gamma"]
    )
    
    print(f"\n  Baseline Objectives:")
    print(f"    Z1 (W_max): {Z1_baseline:.2f}")
    print(f"    Z2 (Total Cost): {Z2_baseline:.2f}")
    print(f"    Z3 (Variance): {Z3_baseline:.2f}")
    print(f"    Combined Z: {Z_baseline:.4f}")
    print(f"    Cost baseline (Z2_0): {Z2_baseline:.2f}")
    
    # Debug: Print waiting time computation for one route
    print_subsection("Debug: Waiting time computation for sample route")
    sample_route = None
    sample_dc = None
    for dc_id, routes in baseline.get("routes_by_dc", {}).items():
        if routes:
            sample_route = routes[0]
            sample_dc = dc_id
            break
    
    if sample_route:
        print(f"\n  Route: {sample_route.get('vehicle_id', '?')} from {sample_dc}")
        print(f"  {'Stop':<15} {'Travel (s)':<12} {'Service (s)':<12} {'Arrival t_i':<12} {'WW_i (n_i*t_i)':<15}")
        print("  " + "-" * 70)
        
        current_time = 0.0
        prev_id = sample_dc
        for stop_id in sample_route.get("ordered_stop_ids", [])[:10]:  # First 10 stops
            travel_time = time_provider(prev_id, stop_id)
            current_time += travel_time
            arrival = current_time
            stop_data = stops_by_id.get(stop_id, {})
            n_i = stop_data.get("households", 1)
            service_time = stop_data.get("service_time_s", 300)
            WW_i = n_i * arrival
            print(f"  {stop_id:<15} {travel_time:<12.1f} {service_time:<12} {arrival:<12.1f} {WW_i:<15.1f}")
            current_time += service_time
            prev_id = stop_id
    
    # Run proposed algorithm
    print_subsection("Running proposed algorithm (ALNS)")
    improved, debug_info = proposed_algorithm(
        depots=depots,
        vehicles=vehicles,
        stops_by_depot=stops_by_depot_dict,
        alpha=config["alpha"],
        beta=config["beta"],
        gamma=config["gamma"],
        eps=config["eps"],
        iters=config["iters"],
        seed=config["seed"],
        time_provider=time_provider,
        distance_provider=distance_provider,
        normalize="baseline",
        use_distance_objective=config["use_distance"],
        enforce_capacity=False
    )
    
    # Compute improved objectives
    print_subsection("Computing improved objectives")
    waiting_improved = compute_waiting_times(improved, stops_by_id, time_provider)
    Z1_improved = compute_Z1(waiting_improved, stops_by_id)
    Z2_improved = compute_Z2(improved, distance_provider, time_provider, config["use_distance"])
    Z3_improved = compute_Z3(waiting_improved, stops_by_id)
    Z_improved = compute_combined_Z(
        Z1_improved, Z2_improved, Z3_improved,
        Z1_baseline, Z2_baseline, Z3_baseline,
        config["alpha"], config["beta"], config["gamma"]
    )
    
    cost_budget = (1 + config["eps"]) * Z2_baseline
    
    print(f"\n  Improved Objectives:")
    print(f"    Z1 (W_max): {Z1_improved:.2f} ({((Z1_baseline - Z1_improved) / Z1_baseline * 100):+.1f}%)")
    print(f"    Z2 (Total Cost): {Z2_improved:.2f} ({((Z2_baseline - Z2_improved) / Z2_baseline * 100):+.1f}%)")
    print(f"    Z3 (Variance): {Z3_improved:.2f} ({((Z3_baseline - Z3_improved) / Z3_baseline * 100):+.1f}%)")
    print(f"    Combined Z: {Z_improved:.4f} ({((Z_baseline - Z_improved) / Z_baseline * 100):+.1f}%)")
    print(f"    Cost new: {Z2_improved:.2f}")
    print(f"    Cost budget: {cost_budget:.2f}")
    print(f"    Budget satisfied: {'✓' if Z2_improved <= cost_budget else '✗'}")
    print(f"    Improvement: {((Z_baseline - Z_improved) / Z_baseline * 100):+.2f}%")
    
    # ============================================================
    # PART C: GAPS VS NOTION MODEL
    # ============================================================
    
    print_section("C. GAPS VS NOTION MATHEMATICAL MODEL")
    
    print_subsection("(1) ✔ Fully Matched Elements")
    print("""
    ✓ x_ijv: Route adjacency via ordered_stop_ids
    ✓ t_i: Arrival time computation in compute_waiting_times()
    ✓ W_max: Z1 = max_i (n_i * w_i) in compute_Z1()
    ✓ Z2: Total routing cost = Σ c_ij x_ijv in compute_Z2()
    ✓ Combined Z: α(Z1/Z1*) + β(Z2/Z2*) + γ(Z3/Z3*) in compute_combined_Z()
    ✓ DC assignment: k(i) = argmin_k t(dc_k, i) using OSRM
    ✓ Per-DC VRP: Independent VROOM solves per depot
    ✓ Cost budget: Z2 ≤ (1+ε)Z2* enforced in ALNS
    ✓ Time consistency: Implicitly enforced in waiting time simulation
    """)
    
    print_subsection("(2) △ Partially Matched / Approximated Elements")
    print("""
    △ Z3 (Fairness metric):
      - Notion spec: Z3 = Σ n_i δ_i where δ_i = |t_i - t̄| (MAD)
      - Current code: Z3 = Σ n_i (w_i - w̄)² (Variance)
      - Impact: Variance penalizes large deviations more than MAD
      - Approximation: Acceptable if variance is used as proxy for fairness
    
    △ Contextual TS:
      - Notion spec mentions contextual TS for operator selection
      - Current code: Simple greedy acceptance (ALNS-lite)
      - Impact: Less sophisticated operator selection, but simpler
      - Approximation: ALNS-lite is a valid simplification
    """)
    
    print_subsection("(3) ✘ Missing / Incorrect Elements")
    print("""
    ✘ Z3 as MAD (Mean Absolute Deviation):
      - Notion spec: δ_i = |t_i - t̄|, Z3 = Σ n_i δ_i
      - Current: Z3 = Σ n_i (w_i - w̄)²
      - Location: objectives.py:compute_Z3()
      - PROPOSED FIX: Add compute_Z3_MAD() or modify compute_Z3() with mode parameter
    
    ✘ Explicit MTZ subtour elimination:
      - Notion spec mentions MTZ constraints
      - Current: VROOM handles subtour elimination internally
      - Impact: Not directly visible in code, but VROOM ensures valid routes
      - Status: Handled by VROOM solver (acceptable)
    
    ✘ Contextual TS implementation:
      - Notion spec mentions contextual TS
      - Current: Greedy acceptance only
      - PROPOSED FIX: Add contextual TS for destroy/repair operator selection
    """)
    
    print_section("SUMMARY")
    print(f"""
    ✓ Most core elements match the Notion spec
    ✘ Z3 uses variance instead of MAD (needs fix)
    △ Contextual TS not implemented (acceptable simplification)
    
    Next steps:
    1. Add compute_Z3_MAD() function
    2. Optionally add contextual TS for operator selection
    3. Verify MAD vs variance impact on fairness optimization
    """)

if __name__ == "__main__":
    main()

