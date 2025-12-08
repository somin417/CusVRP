"""
Proposed ALNS-lite algorithm for combined objective optimization.
"""

import random
import logging
from typing import Dict, List, Any, Callable, Optional, Tuple
from .vroom_vrp import solve_multi_depot
from .objectives import (
    compute_waiting_times, compute_Z1, compute_Z2, compute_Z3, compute_combined_Z
)

logger = logging.getLogger(__name__)


def proposed_algorithm(
    depots: List[Dict[str, Any]],
    vehicles: Dict[str, List[Dict[str, Any]]],
    stops_by_depot: Dict[str, List[Dict[str, Any]]],
    *,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    eps: float = 0.1,
    iters: int = 300,
    seed: int = 0,
    time_provider: Callable[[str, str], float],
    distance_provider: Optional[Callable[[str, str], float]] = None,
    normalize: str = "baseline",
    use_distance_objective: bool = False,
    enforce_capacity: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    ALNS-lite algorithm optimizing combined objective Z.
    
    Returns:
        (best_solution, debug_dict)
    """
    random.seed(seed)
    
    # A) Baseline
    logger.info("Computing baseline solution...")
    baseline = solve_multi_depot(
        depots=depots,
        vehicles_by_depot=vehicles,
        stops_by_depot=stops_by_depot,
        request_geometry=True
    )
    
    # Build stops_by_id from baseline
    stops_by_id = baseline.get("stops_dict", {})
    
    # Ensure baseline has depots for objectives computation
    if "depots" not in baseline:
        baseline["depots"] = depots
    
    # Compute baseline objectives
    waiting_baseline = compute_waiting_times(baseline, stops_by_id, time_provider)
    Z1_0 = compute_Z1(waiting_baseline, stops_by_id)
    Z2_0 = compute_Z2(baseline, distance_provider, time_provider, use_distance_objective)
    Z3_0 = compute_Z3(waiting_baseline, stops_by_id)
    
    logger.info(f"Baseline: Z1={Z1_0:.1f}, Z2={Z2_0:.1f}, Z3={Z3_0:.1f}")
    
    # Normalizers
    if normalize == "baseline":
        Z1_star, Z2_star, Z3_star = Z1_0, Z2_0, Z3_0
    elif normalize == "best_known":
        # Quick runs for Z1* and Z3*
        logger.info("Computing best_known normalizers...")
        # TODO: Implement fairness_run and variance_run
        Z1_star, Z3_star = Z1_0, Z3_0  # Placeholder
        Z2_star = Z2_0
    else:
        Z1_star, Z2_star, Z3_star = Z1_0, Z2_0, Z3_0
    
    # B) Search loop
    # Deep copy baseline and remove geometry (will be recalculated if needed)
    import copy
    current_solution = copy.deepcopy(baseline)
    for dc_id, routes in current_solution.get("routes_by_dc", {}).items():
        for route in routes:
            if "geometry" in route:
                del route["geometry"]
            if "legs" in route:
                del route["legs"]
    
    best_solution = copy.deepcopy(current_solution)
    best_Z = compute_combined_Z(Z1_0, Z2_0, Z3_0, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
    
    trace = []
    kmin, kmax = 2, max(5, len(stops_by_id) // 10)
    
    # Progress printing intervals
    progress_interval = max(1, iters // 20)  # Print ~20 times total
    if iters <= 100:
        progress_interval = max(1, iters // 10)  # For small iterations, print ~10 times
    
    logger.info(f"Starting ALNS search: {iters} iterations")
    logger.info(f"Baseline: Z={compute_combined_Z(Z1_0, Z2_0, Z3_0, Z1_star, Z2_star, Z3_star, alpha, beta, gamma):.3f} "
                f"(Z1={Z1_0:.1f}, Z2={Z2_0:.1f}, Z3={Z3_0:.1f})")
    
    # Cache for incremental evaluation
    current_waiting_cache = None
    current_Z2_cache = None
    
    for iteration in range(iters):
        # Destroy: remove k stops biased to large WW_i
        if current_waiting_cache is None:
            waiting = compute_waiting_times(current_solution, stops_by_id, time_provider)
        else:
            waiting = current_waiting_cache
        WW = {}
        for stop_id, w_i in waiting.items():
            stop_data = stops_by_id.get(stop_id, {})
            n_i = stop_data.get("households", stop_data.get("meta", {}).get("n_i", 1)) or 1
            WW[stop_id] = n_i * w_i
        
        # Select k stops to remove (biased to high WW)
        k = random.randint(kmin, kmax)
        all_stops = list(WW.keys())
        if len(all_stops) < k:
            k = len(all_stops)
        
        # Weighted selection
        weights = [WW.get(sid, 0) + 1e-6 for sid in all_stops]
        removed = random.choices(all_stops, weights=weights, k=k)
        
        # Create partial solution (remove stops)
        partial_solution = _remove_stops(current_solution, removed)
        
        # Repair: optimized regret-2 insertion
        new_solution, new_waiting_cache, new_Z2_cache = _regret_insertion_optimized(
            partial_solution, removed, depots, vehicles, stops_by_id,
            time_provider, distance_provider, use_distance_objective,
            Z1_star, Z2_star, Z3_star, alpha, beta, gamma, Z2_0, eps, enforce_capacity,
            current_waiting_cache, current_Z2_cache
        )
        
        # Evaluate
        waiting_new = new_waiting_cache if new_waiting_cache else compute_waiting_times(new_solution, stops_by_id, time_provider)
        Z1_new = compute_Z1(waiting_new, stops_by_id)
        Z2_new = new_Z2_cache if new_Z2_cache is not None else compute_Z2(new_solution, distance_provider, time_provider, use_distance_objective)
        Z3_new = compute_Z3(waiting_new, stops_by_id)
        Z_new = compute_combined_Z(Z1_new, Z2_new, Z3_new, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
        
        # Acceptance
        accepted = False
        if Z_new < best_Z:
            best_Z = Z_new
            best_solution = new_solution.copy()
            accepted = True
        
        # Compare with current solution
        if current_waiting_cache is None:
            waiting_current = compute_waiting_times(current_solution, stops_by_id, time_provider)
        else:
            waiting_current = current_waiting_cache
        Z1_current = compute_Z1(waiting_current, stops_by_id)
        Z2_current = current_Z2_cache if current_Z2_cache is not None else compute_Z2(current_solution, distance_provider, time_provider, use_distance_objective)
        Z3_current = compute_Z3(waiting_current, stops_by_id)
        Z_current = compute_combined_Z(Z1_current, Z2_current, Z3_current, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
        
        if Z_new <= Z_current:
            current_solution = new_solution
            current_waiting_cache = new_waiting_cache
            current_Z2_cache = new_Z2_cache
            accepted = True
        
        trace.append({
            "iter": iteration,
            "Z": Z_new,
            "Z1": Z1_new,
            "Z2": Z2_new,
            "Z3": Z3_new,
            "accepted": accepted,
            "k_removed": k
        })
        
        # Progress printing
        should_print = (iteration + 1) % progress_interval == 0 or (iteration + 1) == iters
        if should_print:
            progress_pct = 100 * (iteration + 1) / iters
            improvement = 100 * (1 - best_Z / compute_combined_Z(Z1_0, Z2_0, Z3_0, Z1_star, Z2_star, Z3_star, alpha, beta, gamma))
            logger.info(
                f"[{progress_pct:5.1f}%] Iter {iteration+1}/{iters}: "
                f"Z={Z_new:.3f} (best={best_Z:.3f}, "
                f"Z1={Z1_new:.1f}, Z2={Z2_new:.1f}, Z3={Z3_new:.1f}, "
                f"improvement={improvement:+.1f}%, accepted={accepted})"
            )
    
    # Final summary
    final_waiting = compute_waiting_times(best_solution, stops_by_id, time_provider)
    final_Z1 = compute_Z1(final_waiting, stops_by_id)
    final_Z2 = compute_Z2(best_solution, distance_provider, time_provider, use_distance_objective)
    final_Z3 = compute_Z3(final_waiting, stops_by_id)
    final_Z = compute_combined_Z(final_Z1, final_Z2, final_Z3, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
    baseline_Z = compute_combined_Z(Z1_0, Z2_0, Z3_0, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
    improvement = 100 * (1 - final_Z / baseline_Z)
    
    logger.info("=" * 70)
    logger.info("ALNS Search Complete!")
    logger.info(f"Baseline: Z={baseline_Z:.3f} (Z1={Z1_0:.1f}, Z2={Z2_0:.1f}, Z3={Z3_0:.1f})")
    logger.info(f"Best:     Z={final_Z:.3f} (Z1={final_Z1:.1f}, Z2={final_Z2:.1f}, Z3={final_Z3:.1f})")
    logger.info(f"Improvement: {improvement:+.2f}%")
    logger.info("=" * 70)
    
    debug = {
        "trace": trace,
        "baseline_Z": baseline_Z,
        "best_Z": final_Z,
        "normalizers": {"Z1_star": Z1_star, "Z2_star": Z2_star, "Z3_star": Z3_star}
    }
    
    return best_solution, debug


def _remove_stops(solution: Dict[str, Any], stop_ids: List[str]) -> Dict[str, Any]:
    """Remove stops from solution."""
    new_solution = solution.copy()
    new_routes_by_dc = {}
    
    for dc_id, routes in solution.get("routes_by_dc", {}).items():
        new_routes = []
        for route in routes:
            new_stop_ids = [sid for sid in route.get("ordered_stop_ids", []) if sid not in stop_ids]
            if new_stop_ids:
                new_route = route.copy()
                new_route["ordered_stop_ids"] = new_stop_ids
                # Remove geometry - it's from baseline and incorrect for new stop order
                if "geometry" in new_route:
                    del new_route["geometry"]
                if "legs" in new_route:
                    del new_route["legs"]
                new_routes.append(new_route)
        if new_routes:
            new_routes_by_dc[dc_id] = new_routes
    
    new_solution["routes_by_dc"] = new_routes_by_dc
    return new_solution


def _compute_route_waiting_incremental(
    route: Dict[str, Any],
    depot_id: str,
    stops_by_id: Dict[str, Dict[str, Any]],
    time_provider: Callable[[str, str], float],
    service_time_field: str = "service_time_s"
) -> Dict[str, float]:
    """Compute waiting times for a single route (incremental)."""
    waiting_times = {}
    stop_ids = route.get("ordered_stop_ids", [])
    if not stop_ids:
        return waiting_times
    
    current_time = 0.0
    prev_id = depot_id
    
    for stop_id in stop_ids:
        travel_time = time_provider(prev_id, stop_id)
        current_time += travel_time
        waiting_times[stop_id] = current_time
        
        service_time = stops_by_id.get(stop_id, {}).get(service_time_field, 0) or 0
        current_time += service_time
        prev_id = stop_id
    
    return waiting_times


def _compute_route_Z2_incremental(
    route: Dict[str, Any],
    depot_id: str,
    stops_by_id: Dict[str, Dict[str, Any]],
    distance_provider: Optional[Callable[[str, str], float]],
    time_provider: Callable[[str, str], float],
    use_distance: bool
) -> float:
    """Compute Z2 (routing cost) for a single route (incremental)."""
    stop_ids = route.get("ordered_stop_ids", [])
    if not stop_ids:
        return 0.0
    
    total_cost = 0.0
    prev_id = depot_id
    
    for stop_id in stop_ids:
        if use_distance and distance_provider:
            cost = distance_provider(prev_id, stop_id)
        else:
            cost = time_provider(prev_id, stop_id)
        total_cost += cost
        prev_id = stop_id
    
    # Return to depot
    if use_distance and distance_provider:
        total_cost += distance_provider(prev_id, depot_id)
    else:
        total_cost += time_provider(prev_id, depot_id)
    
    return total_cost


def _regret_insertion_optimized(
    partial_solution: Dict[str, Any],
    unassigned: List[str],
    depots: List[Dict[str, Any]],
    vehicles: Dict[str, List[Dict[str, Any]]],
    stops_by_id: Dict[str, Dict[str, Any]],
    time_provider: Callable[[str, str], float],
    distance_provider: Optional[Callable[[str, str], float]],
    use_distance: bool,
    Z1_star: float, Z2_star: float, Z3_star: float,
    alpha: float, beta: float, gamma: float,
    Z2_0: float, eps: float, enforce_capacity: bool,
    current_waiting_cache: Optional[Dict[str, float]],
    current_Z2_cache: Optional[float]
) -> Tuple[Dict[str, Any], Dict[str, float], float]:
    """
    Optimized regret-2 insertion with incremental evaluation.
    
    Returns:
        (solution, waiting_cache, Z2_cache)
    """
    solution = partial_solution.copy()
    
    # Build depot lookup
    depot_map = {d.get("id"): d for d in depots}
    
    # Pre-compute route-level waiting times and Z2 for unchanged routes
    route_waiting_cache = {}
    route_Z2_cache = {}
    total_Z2 = 0.0
    
    for dc_id, routes in solution.get("routes_by_dc", {}).items():
        depot_id = dc_id
        for route_idx, route in enumerate(routes):
            route_key = (dc_id, route_idx)
            route_waiting = _compute_route_waiting_incremental(route, depot_id, stops_by_id, time_provider)
            route_waiting_cache[route_key] = route_waiting
            route_Z2 = _compute_route_Z2_incremental(route, depot_id, stops_by_id, distance_provider, time_provider, use_distance)
            route_Z2_cache[route_key] = route_Z2
            total_Z2 += route_Z2
    
    # Build full waiting cache from route caches
    full_waiting_cache = {}
    for route_waiting in route_waiting_cache.values():
        full_waiting_cache.update(route_waiting)
    
    for stop_id in unassigned:
        best_insertion = None
        best_cost = float('inf')
        
        stop_data = stops_by_id.get(stop_id, {})
        stop_lat = stop_data.get("lat")
        stop_lon = stop_data.get("lon")
        
        # Try all positions in all routes
        for dc_id, routes in solution.get("routes_by_dc", {}).items():
            depot_id = dc_id
            depot = depot_map.get(dc_id, {})
            
            for route_idx, route in enumerate(routes):
                stop_ids = route.get("ordered_stop_ids", [])
                route_key = (dc_id, route_idx)
                
                # Get cached values for this route
                base_route_waiting = route_waiting_cache[route_key].copy()
                base_route_Z2 = route_Z2_cache[route_key]
                
                # Try promising positions only (limit to reduce computation)
                # Strategy: try only best positions (first, last, and 1-2 random)
                positions_to_try = []
                
                # Always try first and last positions (most promising)
                if len(stop_ids) > 0:
                    positions_to_try.append(0)
                    positions_to_try.append(len(stop_ids))
                else:
                    positions_to_try.append(0)
                
                # Try only 1-2 random positions for diversity (reduced from 3)
                if len(stop_ids) > 2:
                    num_random = min(2, max(1, len(stop_ids) // 4))  # Reduced: max 2 random
                    if num_random > 0:
                        random_positions = random.sample(range(1, len(stop_ids)), min(num_random, len(stop_ids) - 1))
                        positions_to_try.extend(random_positions)
                
                # Remove duplicates and sort
                positions_to_try = sorted(set(positions_to_try))
                
                # Limit total positions to try (safety check)
                if len(positions_to_try) > 5:
                    positions_to_try = positions_to_try[:5]
                
                for pos in positions_to_try:
                    # Quick Z2 check first (fast filter)
                    # Estimate Z2 change for this insertion
                    if pos == 0:
                        # Insert at start: depot -> stop -> first_stop
                        prev_id = depot_id
                        next_id = stop_ids[0] if stop_ids else depot_id
                    elif pos == len(stop_ids):
                        # Insert at end: last_stop -> stop -> depot
                        prev_id = stop_ids[-1] if stop_ids else depot_id
                        next_id = depot_id
                    else:
                        # Insert in middle: prev_stop -> stop -> next_stop
                        prev_id = stop_ids[pos - 1]
                        next_id = stop_ids[pos]
                    
                    # Compute Z2 delta for this route
                    if use_distance and distance_provider:
                        cost_prev_to_stop = distance_provider(prev_id, stop_id)
                        cost_stop_to_next = distance_provider(stop_id, next_id)
                        cost_prev_to_next = distance_provider(prev_id, next_id)
                    else:
                        cost_prev_to_stop = time_provider(prev_id, stop_id)
                        cost_stop_to_next = time_provider(stop_id, next_id)
                        cost_prev_to_next = time_provider(prev_id, next_id)
                    
                    Z2_delta = cost_prev_to_stop + cost_stop_to_next - cost_prev_to_next
                    test_route_Z2 = base_route_Z2 + Z2_delta
                    test_total_Z2 = total_Z2 - base_route_Z2 + test_route_Z2
                    
                    # Fast filter: check budget constraint early
                    if test_total_Z2 > (1 + eps) * Z2_0:
                        continue
                    
                    # Create test route
                    test_stop_ids = stop_ids[:pos] + [stop_id] + stop_ids[pos:]
                    test_route = route.copy()
                    test_route["ordered_stop_ids"] = test_stop_ids
                    
                    # Incremental waiting time computation for this route only
                    test_route_waiting = _compute_route_waiting_incremental(test_route, depot_id, stops_by_id, time_provider)
                    
                    # Merge with other routes' waiting times (incremental update)
                    # Only update changed stops instead of full copy
                    test_waiting = dict(full_waiting_cache)  # Shallow copy is faster
                    # Remove old route waiting times
                    for old_stop_id in base_route_waiting:
                        test_waiting.pop(old_stop_id, None)
                    # Add new route waiting times
                    test_waiting.update(test_route_waiting)
                    
                    # Compute objectives (only for this route change)
                    # Note: Z1 and Z3 still need full waiting dict, but we've optimized the merge
                    Z1 = compute_Z1(test_waiting, stops_by_id)
                    Z2 = test_total_Z2
                    Z3 = compute_Z3(test_waiting, stops_by_id)
                    
                    cost = compute_combined_Z(Z1, Z2, Z3, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_insertion = (dc_id, route_idx, pos, test_route_waiting, test_route_Z2)
        
        # Apply best insertion
        if best_insertion:
            dc_id, route_idx, pos, new_route_waiting, new_route_Z2 = best_insertion
            route = solution["routes_by_dc"][dc_id][route_idx]
            route["ordered_stop_ids"].insert(pos, stop_id)
            
            # Update caches
            route_key = (dc_id, route_idx)
            route_waiting_cache[route_key] = new_route_waiting
            old_route_Z2 = route_Z2_cache[route_key]
            route_Z2_cache[route_key] = new_route_Z2
            total_Z2 = total_Z2 - old_route_Z2 + new_route_Z2
            
            # Update full waiting cache
            for old_stop_id in list(full_waiting_cache.keys()):
                if old_stop_id in base_route_waiting:
                    del full_waiting_cache[old_stop_id]
            full_waiting_cache.update(new_route_waiting)
            
            # Remove geometry when route changes
            if "geometry" in route:
                del route["geometry"]
            if "legs" in route:
                del route["legs"]
    
    # Clean up all geometry from solution
    for dc_id, routes in solution.get("routes_by_dc", {}).items():
        for route in routes:
            if "geometry" in route:
                del route["geometry"]
            if "legs" in route:
                del route["legs"]
    
    return solution, full_waiting_cache, total_Z2


# Keep old function for backward compatibility (but mark as deprecated)
def _regret_insertion(
    partial_solution: Dict[str, Any],
    unassigned: List[str],
    depots: List[Dict[str, Any]],
    vehicles: Dict[str, List[Dict[str, Any]]],
    stops_by_id: Dict[str, Dict[str, Any]],
    time_provider: Callable[[str, str], float],
    distance_provider: Optional[Callable[[str, str], float]],
    use_distance: bool,
    Z1_star: float, Z2_star: float, Z3_star: float,
    alpha: float, beta: float, gamma: float,
    Z2_0: float, eps: float, enforce_capacity: bool
) -> Dict[str, Any]:
    """Legacy regret insertion (deprecated - use _regret_insertion_optimized)."""
    solution, _, _ = _regret_insertion_optimized(
        partial_solution, unassigned, depots, vehicles, stops_by_id,
        time_provider, distance_provider, use_distance,
        Z1_star, Z2_star, Z3_star, alpha, beta, gamma, Z2_0, eps, enforce_capacity,
        None, None
    )
    return solution
