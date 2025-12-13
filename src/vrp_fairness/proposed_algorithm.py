"""
Proposed ALNS-lite algorithm for combined objective optimization.
"""

import random
import logging
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple
from .vroom_vrp import solve_multi_depot
from .objectives import (
    compute_waiting_times, compute_Z1, compute_Z2, compute_Z3, compute_combined_Z
)

logger = logging.getLogger(__name__)


def get_stop_load(stop_data: Dict[str, Any]) -> int:
    """
    Return n_i = households / demand / meta.n_i, default 1.
    This is the household weight used in Z1/Z3, worst_k, and capacity.
    
    Args:
        stop_data: Stop data dictionary
    
    Returns:
        Integer load (household count) for the stop
    """
    n_i = stop_data.get(
        "households",
        stop_data.get("demand", stop_data.get("meta", {}).get("n_i", 1)),
    )
    if n_i is None or n_i == 0:
        return 1
    return int(n_i)


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
    enforce_capacity: bool = False,
    operator_mode: str = "fixed"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    ALNS-lite algorithm optimizing combined objective Z.
    
    Args:
        operator_mode: "fixed" (default) or "cts" (Contextual Thompson Sampling)
    
    Returns:
        (best_solution, debug_dict)
    """
    random.seed(seed)
    rng = np.random.default_rng(seed)
    
    # Define operator pairs for CTS
    operator_pairs = [
        ("worst_k", "regret2"),
        ("worst_k", "best_insert"),
        ("cluster_k", "regret2"),
        ("cluster_k", "best_insert"),
        ("random_k", "regret2"),
        ("random_k", "best_insert"),
    ]
    
    # Initialize CTS bandit if needed
    bandit = None
    if operator_mode == "cts":
        from .bandit_cts import ContextualTSBandit, CTSConfig
        context_dim = 8  # From build_context
        config = CTSConfig(dim=context_dim, lambda_reg=1e-3, noise_var=1.0)
        bandit = ContextualTSBandit(n_arms=len(operator_pairs), config=config)
        logger.info(f"Initialized CTS bandit with {len(operator_pairs)} operator pairs")
    
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
    # Deep copy baseline and remove geometry and cached cost fields (will be recalculated if needed)
    import copy
    current_solution = copy.deepcopy(baseline)
    for dc_id, routes in current_solution.get("routes_by_dc", {}).items():
        for route in routes:
            if "geometry" in route:
                del route["geometry"]
            if "legs" in route:
                del route["legs"]
            # Remove cached cost fields so compute_Z2 always recalculates from provider
            if "total_duration" in route:
                del route["total_duration"]
            if "total_distance" in route:
                del route["total_distance"]
    
    best_solution = copy.deepcopy(current_solution)
    best_Z = compute_combined_Z(Z1_0, Z2_0, Z3_0, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
    best_iteration = -1  # Track which iteration found the best solution
    best_Z1_at_update = Z1_0  # Track Z1 when best was updated (for verification)
    best_Z2_at_update = Z2_0
    best_Z3_at_update = Z3_0
    
    # Get list of all stops that should be assigned (from baseline solution)
    # This excludes depots which are not in routes
    depot_ids = {d.get("id") for d in depots}
    expected_stop_ids = set()
    for routes in baseline.get("routes_by_dc", {}).values():
        for route in routes:
            for stop_id in route.get("ordered_stop_ids", []):
                if stop_id not in depot_ids:
                    expected_stop_ids.add(stop_id)
    
    trace = []
    # Restrict destroy size to keep runtime reasonable
    kmin, kmax = 2, min(5, max(5, len(stops_by_id) // 10))
    
    # Progress printing intervals
    progress_interval = max(1, iters // 50)  # Print ~10 times total
    
    logger.info(f"Starting ALNS search: {iters} iterations")
    logger.info(f"Baseline: Z={compute_combined_Z(Z1_0, Z2_0, Z3_0, Z1_star, Z2_star, Z3_star, alpha, beta, gamma):.3f} "
                f"(Z1={Z1_0:.1f}, Z2={Z2_0:.1f}, Z3={Z3_0:.1f})")
    
    # Cache for incremental evaluation
    current_waiting_cache = None
    current_Z2_cache = None
    
    # Baseline metrics for context building
    baseline_metrics = {"Z1_0": Z1_0, "Z2_0": Z2_0, "Z3_0": Z3_0}
    cost_budget = (1 + eps) * Z2_0
    
    # Backup best solution periodically (every 10% of iterations or when best is updated)
    backup_interval = max(1, iters // 10)
    best_solution_backup = None  # Will be set by run_experiment if backup directory is provided
    
    for iteration in range(iters):
        # Destroy: remove k stops biased to large WW_i
        # Always recompute waiting to ensure all depots/stops and service times are included
        waiting = compute_waiting_times(current_solution, stops_by_id, time_provider)
        WW = {}
        for stop_id, w_i in waiting.items():
            stop_data = stops_by_id.get(stop_id, {})
            n_i = get_stop_load(stop_data)
            WW[stop_id] = n_i * w_i
        
        # Select operator pair (CTS or fixed)
        if operator_mode == "cts" and bandit is not None:
            # Build context
            context = build_context(
                current_solution, waiting, stops_by_id, depots, time_provider,
                baseline_metrics, Z1_star, Z2_star, Z3_star, Z2_0, eps,
                iteration, iters
            )
            
            # Select arm using CTS
            arm_idx = bandit.select_arm(context, rng)
            destroy_op, repair_op = operator_pairs[arm_idx]
            chosen_arm_name = f"{destroy_op}+{repair_op}"
        else:
            # Fixed mode: use worst_k + regret2 (original behavior)
            destroy_op, repair_op = "worst_k", "regret2"
            chosen_arm_name = "fixed"
        
        # Select k stops to remove
        k = random.randint(kmin, kmax)
        all_stops = list(WW.keys())
        if len(all_stops) < k:
            k = len(all_stops)
        
        # Apply destroy operator
        if destroy_op == "worst_k":
            partial_solution, removed = _remove_stops_worst_k(
                current_solution, waiting, stops_by_id, k
            )
        elif destroy_op == "cluster_k":
            partial_solution, removed = _remove_stops_cluster_k(
                current_solution, waiting, stops_by_id, k
            )
        elif destroy_op == "random_k":
            partial_solution, removed = _remove_stops_random_k(
                current_solution, waiting, k
            )
        else:
            # Fallback to worst_k
            partial_solution, removed = _remove_stops_worst_k(
                current_solution, waiting, stops_by_id, k
            )
        
        # Apply repair operator
        if repair_op == "regret2":
            new_solution, new_waiting_cache, new_Z2_cache = _regret_insertion_optimized(
                partial_solution, removed, depots, vehicles, stops_by_id,
                time_provider, distance_provider, use_distance_objective,
                Z1_star, Z2_star, Z3_star, alpha, beta, gamma, Z2_0, eps, enforce_capacity,
                current_waiting_cache, current_Z2_cache
            )
        elif repair_op == "best_insert":
            new_solution, new_waiting_cache, new_Z2_cache = _repair_best_insertion(
                partial_solution, removed, depots, vehicles, stops_by_id,
                time_provider, distance_provider, use_distance_objective,
                Z1_star, Z2_star, Z3_star, alpha, beta, gamma, Z2_0, eps, enforce_capacity,
                current_waiting_cache, current_Z2_cache
            )
        else:
            # Fallback to regret2
            new_solution, new_waiting_cache, new_Z2_cache = _regret_insertion_optimized(
                partial_solution, removed, depots, vehicles, stops_by_id,
                time_provider, distance_provider, use_distance_objective,
                Z1_star, Z2_star, Z3_star, alpha, beta, gamma, Z2_0, eps, enforce_capacity,
                current_waiting_cache, current_Z2_cache
            )

        # Evaluate
        # Always recompute waiting_new (do not rely on cached incremental waiting)
        waiting_new = compute_waiting_times(new_solution, stops_by_id, time_provider)
        
        # Check for unassigned stops: compare assigned stops with expected stops from baseline
        assigned_stops = set()
        for routes in new_solution.get("routes_by_dc", {}).values():
            for route in routes:
                for stop_id in route.get("ordered_stop_ids", []):
                    if stop_id not in depot_ids:
                        assigned_stops.add(stop_id)
        
        missing_stops = expected_stop_ids - assigned_stops
        missing_after_repair = len(missing_stops)
        
        # If any stops are missing, force-insert them ignoring budget
        if missing_stops:
            new_solution, new_waiting_cache, new_Z2_cache = _force_insert_all(
                new_solution, list(missing_stops), depots, vehicles, stops_by_id,
                time_provider, distance_provider, use_distance_objective,
                current_waiting_cache, current_Z2_cache
            )
            # Recompute waiting after force insert
            waiting_new = new_waiting_cache if new_waiting_cache else compute_waiting_times(new_solution, stops_by_id, time_provider)
            # Re-check after force insert (should be 0 now)
            assigned_stops = set()
            for routes in new_solution.get("routes_by_dc", {}).values():
                for route in routes:
                    for stop_id in route.get("ordered_stop_ids", []):
                        if stop_id not in depot_ids:
                            assigned_stops.add(stop_id)
            missing_after_repair = len(expected_stop_ids - assigned_stops)
        
        Z1_new = compute_Z1(waiting_new, stops_by_id)
        Z2_new = new_Z2_cache if new_Z2_cache is not None else compute_Z2(new_solution, distance_provider, time_provider, use_distance_objective)
        Z3_new = compute_Z3(waiting_new, stops_by_id)
        Z_new = compute_combined_Z(Z1_new, Z2_new, Z3_new, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)

        # Final check: penalize if still missing (shouldn't happen after force_insert, but safety check)
        if missing_after_repair > 0:
            # Large penalty to prevent accepting incomplete solutions
            Z_new = float("inf")
        
        # Acceptance
        accepted = False
        if Z_new < best_Z:
            old_best_Z = best_Z
            best_Z = Z_new
            # Deep copy to prevent best_solution from being modified when current_solution changes
            best_solution = copy.deepcopy(new_solution)
            # Recompute waiting for best_solution to ensure full coverage and service times
            best_waiting = compute_waiting_times(best_solution, stops_by_id, time_provider)
            best_iteration = iteration
            best_Z1_at_update = compute_Z1(best_waiting, stops_by_id)
            best_Z2_at_update = Z2_new
            best_Z3_at_update = compute_Z3(best_waiting, stops_by_id)
            accepted = True
            logger.debug(
                f"Best solution updated at iter {iteration}: Z={old_best_Z:.6f} -> {best_Z:.6f} "
                f"(Z1={best_Z1_at_update:.1f}, Z2={best_Z2_at_update:.1f}, Z3={best_Z3_at_update:.1f})"
            )
            # Update backup whenever best is found with recomputed waiting
            best_solution_backup = {
                "solution": copy.deepcopy(best_solution),
                "waiting": best_waiting,
                "Z": best_Z,
                "Z1": best_Z1_at_update,
                "Z2": best_Z2_at_update,
                "Z3": best_Z3_at_update,
                "iteration": best_iteration
            }
        
        # Compare with current solution
        # Always recompute waiting_current (do not rely on cached incremental waiting)
        waiting_current = compute_waiting_times(current_solution, stops_by_id, time_provider)
        Z1_current = compute_Z1(waiting_current, stops_by_id)
        Z2_current = current_Z2_cache if current_Z2_cache is not None else compute_Z2(current_solution, distance_provider, time_provider, use_distance_objective)
        Z3_current = compute_Z3(waiting_current, stops_by_id)
        Z_current = compute_combined_Z(Z1_current, Z2_current, Z3_current, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
        
        if Z_new <= Z_current:
            # Deep copy to prevent current_solution from being modified when new_solution changes
            current_solution = copy.deepcopy(new_solution)
            current_waiting_cache = new_waiting_cache
            current_Z2_cache = new_Z2_cache
            accepted = True
        
        # Update CTS bandit if enabled
        reward = 0.0
        if operator_mode == "cts" and bandit is not None:
            reward = compute_reward(
                Z1_current, Z1_new, Z2_current, Z2_new,
                Z1_0, cost_budget, lambda_penalty=1.0
            )
            # Build context for update (use current solution context)
            context = build_context(
                current_solution, waiting_current, stops_by_id, depots, time_provider,
                baseline_metrics, Z1_star, Z2_star, Z3_star, Z2_0, eps,
                iteration, iters
            )
            bandit.update(arm_idx, context, reward)
        
        trace.append({
            "iter": iteration,
            "Z": Z_new,
            "Z1": Z1_new,
            "Z2": Z2_new,
            "Z3": Z3_new,
            "accepted": accepted,
            "k_removed": k,
            "chosen_arm": chosen_arm_name if operator_mode == "cts" else None,
            "reward": reward if operator_mode == "cts" else None,
            "missing": missing_after_repair
        })
        
        # Periodic backup of best solution (every backup_interval iterations)
        if (iteration + 1) % backup_interval == 0 and best_solution_backup is not None:
            # Backup is handled by run_experiment.py if backup_dir is provided
            pass
        
        # Progress printing
        should_print = (iteration + 1) % progress_interval == 0 or (iteration + 1) == iters
        if should_print:
            progress_pct = 100 * (iteration + 1) / iters
            improvement = 100 * (1 - best_Z / compute_combined_Z(Z1_0, Z2_0, Z3_0, Z1_star, Z2_star, Z3_star, alpha, beta, gamma))
            elapsed = (iteration + 1) / max(1, progress_pct / 100.0)
            remaining_iters = iters - (iteration + 1)
            eta_iters = remaining_iters / max(1, progress_interval)
            logger.info(
                f"[{progress_pct:5.1f}%] Iter {iteration+1}/{iters}: "
                f"Z={Z_new:.3f} (best={best_Z:.3f}, "
                    f"Z1={Z1_new:.1f}, Z2={Z2_new:.1f}, Z3={Z3_new:.1f}, "
                    f"missing={missing_after_repair}, "
                    f"improvement={improvement:+.1f}%, accepted={accepted}); "
                f"ETA ~{remaining_iters} iters"
            )
    
    # Final summary - verify best_solution matches best_Z (use full waiting recomputation)
    final_waiting = compute_waiting_times(best_solution, stops_by_id, time_provider)
    final_Z1 = compute_Z1(final_waiting, stops_by_id)
    final_Z2 = compute_Z2(best_solution, distance_provider, time_provider, use_distance_objective)
    final_Z3 = compute_Z3(final_waiting, stops_by_id)
    final_Z = compute_combined_Z(final_Z1, final_Z2, final_Z3, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
    baseline_Z = compute_combined_Z(Z1_0, Z2_0, Z3_0, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
    improvement = 100 * (1 - final_Z / baseline_Z)
    
    # Verify best_solution matches best_Z (with tolerance for floating point)
    if abs(final_Z - best_Z) > 0.001:
        logger.error("=" * 70)
        logger.error("CRITICAL BUG: best_solution does not match best_Z!")
        logger.error(f"  best_Z (tracked) = {best_Z:.6f}")
        logger.error(f"  final_Z (computed from best_solution) = {final_Z:.6f}")
        logger.error(f"  Difference = {abs(final_Z - best_Z):.6f}")
        logger.error(f"  Best was found at iteration {best_iteration}")
        logger.error(f"  Z1 at update: {best_Z1_at_update:.1f}, Z1 now: {final_Z1:.1f}")
        logger.error(f"  Z2 at update: {best_Z2_at_update:.1f}, Z2 now: {final_Z2:.1f}")
        logger.error(f"  Z3 at update: {best_Z3_at_update:.1f}, Z3 now: {final_Z3:.1f}")
        logger.error("=" * 70)
        # Use the stored best metrics (at update) to keep consistency
        final_Z1 = best_Z1_at_update
        final_Z2 = best_Z2_at_update
        final_Z3 = best_Z3_at_update
        final_Z = best_Z
        # If we have waiting at update, prefer it
        if best_solution_backup and best_solution_backup.get("waiting"):
            final_waiting = best_solution_backup["waiting"]

    # Sanity check: all stops that appear in routes must have waiting times
    depot_ids = {d.get("id") for d in depots}
    expected_stop_ids = set()
    for routes in baseline.get("routes_by_dc", {}).values():
        for route in routes:
            for sid in route.get("ordered_stop_ids", []):
                if sid not in depot_ids:
                    expected_stop_ids.add(sid)
    missing_in_waiting = expected_stop_ids - set(final_waiting.keys())
    if missing_in_waiting:
        logger.error(f"compute_waiting_times BUG: missing waiting for stops: {sorted(missing_in_waiting)}")
    # Sanity check: missing stops in waiting
    depot_ids = {d.get("id") for d in depots}
    expected_stops = set()
    for routes in baseline.get("routes_by_dc", {}).values():
        for route in routes:
            for sid in route.get("ordered_stop_ids", []):
                if sid not in depot_ids:
                    expected_stops.add(sid)
    missing_in_waiting = expected_stops - set(final_waiting.keys())
    if missing_in_waiting:
        logger.error(f"compute_waiting_times BUG: missing waiting for stops: {sorted(missing_in_waiting)}")
    
    logger.info("=" * 70)
    logger.info("ALNS Search Complete!")
    logger.info(f"Baseline: Z={baseline_Z:.3f} (Z1={Z1_0:.1f}, Z2={Z2_0:.1f}, Z3={Z3_0:.1f})")
    logger.info(f"Best:     Z={final_Z:.3f} (Z1={final_Z1:.1f}, Z2={final_Z2:.1f}, Z3={final_Z3:.1f})")
    logger.info(f"Best found at iteration: {best_iteration}")
    logger.info(f"Improvement: {improvement:+.2f}%")
    logger.info("=" * 70)
    
    debug = {
        "trace": trace,
        "baseline_Z": baseline_Z,
        "best_Z": final_Z,
        "normalizers": {"Z1_star": Z1_star, "Z2_star": Z2_star, "Z3_star": Z3_star},
        "baseline_solution": baseline,  # Include baseline solution for comparison
        "baseline_objectives": {
            "Z1": Z1_0,
            "Z2": Z2_0,
            "Z3": Z3_0
        },
        "best_objectives": {
            "Z1": final_Z1,
            "Z2": final_Z2,
            "Z3": final_Z3
        },
        "best_iteration": best_iteration,  # Track which iteration found the best
        "best_Z_tracked": best_Z,  # The tracked best_Z value
        "best_solution_backup": best_solution_backup,  # Backup of best solution with metadata
        "best_waiting": best_solution_backup.get("waiting") if best_solution_backup else None,
        "operator_mode": operator_mode
    }
    
    # Add CTS summary if enabled
    if operator_mode == "cts" and bandit is not None:
        debug["cts_summary"] = bandit.get_posterior_summary()
    
    return best_solution, debug


def _force_insert_all(
    solution: Dict[str, Any],
    missing_stops: List[str],
    depots: List[Dict[str, Any]],
    vehicles: Dict[str, List[Dict[str, Any]]],
    stops_by_id: Dict[str, Dict[str, Any]],
    time_provider: Callable[[str, str], float],
    distance_provider: Optional[Callable[[str, str], float]],
    use_distance: bool,
    current_waiting_cache: Optional[Dict[str, float]],
    current_Z2_cache: Optional[float]
) -> Tuple[Dict[str, Any], Dict[str, float], float]:
    """
    Fallback: ensure all stops are inserted, ignoring budget constraints.
    Picks cheapest insertion by time/distance for each missing stop.
    """
    # Reuse regret insertion without budget filter by temporarily setting eps high
    return _regret_insertion_optimized(
        solution,
        missing_stops,
        depots,
        vehicles,
        stops_by_id,
        time_provider,
        distance_provider,
        use_distance,
        Z1_star=1.0,
        Z2_star=1.0,
        Z3_star=1.0,
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        Z2_0=float("inf"),  # disable budget
        eps=1e9,            # huge slack
        enforce_capacity=False,
        current_waiting_cache=current_waiting_cache,
        current_Z2_cache=current_Z2_cache
    )


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
                # Remove geometry and cached cost fields - they're from baseline and incorrect for new stop order
                if "geometry" in new_route:
                    del new_route["geometry"]
                if "legs" in new_route:
                    del new_route["legs"]
                # Remove cached cost fields so compute_Z2 recalculates from provider
                if "total_duration" in new_route:
                    del new_route["total_duration"]
                if "total_distance" in new_route:
                    del new_route["total_distance"]
                new_routes.append(new_route)
        if new_routes:
            new_routes_by_dc[dc_id] = new_routes
    
    new_solution["routes_by_dc"] = new_routes_by_dc
    return new_solution


def _remove_stops_worst_k(
    solution: Dict[str, Any],
    waiting: Dict[str, float],
    stops_by_id: Dict[str, Dict[str, Any]],
    k: int
) -> Tuple[Dict[str, Any], List[str]]:
    """Remove k stops with highest weighted waiting times."""
    WW = {}
    for stop_id, w_i in waiting.items():
        stop_data = stops_by_id.get(stop_id, {})
        n_i = get_stop_load(stop_data)
        WW[stop_id] = n_i * w_i
    
    # Sort by WW (descending) and take top k
    sorted_stops = sorted(WW.items(), key=lambda x: x[1], reverse=True)
    removed = [stop_id for stop_id, _ in sorted_stops[:k]]
    
    return _remove_stops(solution, removed), removed


def _remove_stops_cluster_k(
    solution: Dict[str, Any],
    waiting: Dict[str, float],
    stops_by_id: Dict[str, Dict[str, Any]],
    k: int
) -> Tuple[Dict[str, Any], List[str]]:
    """Remove k stops in a cluster (randomly select a stop, then remove nearby stops)."""
    all_stops = list(waiting.keys())
    if len(all_stops) < k:
        k = len(all_stops)
    
    # Randomly select a seed stop
    seed_stop = random.choice(all_stops)
    removed = [seed_stop]
    
    # Get remaining stops sorted by distance from seed (approximate by lat/lon)
    seed_data = stops_by_id.get(seed_stop, {})
    seed_lat = seed_data.get("lat", 0)
    seed_lon = seed_data.get("lon", 0)
    
    remaining = [s for s in all_stops if s != seed_stop]
    # Sort by approximate distance (Manhattan distance in lat/lon)
    remaining.sort(key=lambda s: abs(stops_by_id.get(s, {}).get("lat", 0) - seed_lat) + 
                                 abs(stops_by_id.get(s, {}).get("lon", 0) - seed_lon))
    
    # Take k-1 nearest stops
    removed.extend(remaining[:k-1])
    
    return _remove_stops(solution, removed), removed


def _remove_stops_random_k(
    solution: Dict[str, Any],
    waiting: Dict[str, float],
    k: int
) -> Tuple[Dict[str, Any], List[str]]:
    """Remove k random stops."""
    all_stops = list(waiting.keys())
    if len(all_stops) < k:
        k = len(all_stops)
    
    removed = random.sample(all_stops, k)
    return _remove_stops(solution, removed), removed


def _repair_best_insertion(
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
    Best insertion repair: insert each stop at the position that minimizes combined Z.
    Simplified version without regret-2 (faster but less sophisticated).
    """
    # Use regret-2 as fallback (it's already optimized)
    return _regret_insertion_optimized(
        partial_solution, unassigned, depots, vehicles, stops_by_id,
        time_provider, distance_provider, use_distance,
        Z1_star, Z2_star, Z3_star, alpha, beta, gamma, Z2_0, eps, enforce_capacity,
        current_waiting_cache, current_Z2_cache
    )


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
    
    # Step 2A: Compute per-route loads (for capacity enforcement)
    route_loads: Dict[Tuple[str, int], int] = {}
    Q = 100  # Fixed capacity per route
    for dc_id, routes in solution.get("routes_by_dc", {}).items():
        for route_idx, route in enumerate(routes):
            load = 0
            for sid in route.get("ordered_stop_ids", []):
                stop_data = stops_by_id.get(sid, {})
                load += get_stop_load(stop_data)
            route_loads[(dc_id, route_idx)] = load
    
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
                    # Step 2B: Capacity filter on candidate insertions (only when enforce_capacity=True)
                    route_key = (dc_id, route_idx)
                    current_load = route_loads.get(route_key, 0)
                    stop_data_for_load = stops_by_id.get(stop_id, {})
                    n_i = get_stop_load(stop_data_for_load)
                    
                    if enforce_capacity and current_load + n_i > Q:
                        # This insertion would violate capacity; skip this candidate
                        continue
                    
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
                    
                    # Budget constraint removed - allow all insertions to ensure feasibility
                    # (Budget can still be used for acceptance criteria if needed)
                    
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
        
        # Step 2D: Behavior when no capacity-feasible insertion exists (enforce_capacity=True)
        if not best_insertion and enforce_capacity:
            logger.warning(f"No capacity-feasible insertion found for stop {stop_id} with Q={Q}")
            # Leave stop unassigned - will be handled by _force_insert_all later
        
        # Apply best insertion
        if best_insertion:
            dc_id, route_idx, pos, new_route_waiting, new_route_Z2 = best_insertion
            route = solution["routes_by_dc"][dc_id][route_idx]
            route["ordered_stop_ids"].insert(pos, stop_id)
            
            # Step 2C: Update route_loads when best insertion is applied
            route_key = (dc_id, route_idx)
            stop_data_for_load = stops_by_id.get(stop_id, {})
            n_i = get_stop_load(stop_data_for_load)
            route_loads[route_key] = route_loads.get(route_key, 0) + n_i
            
            # Update caches
            route_waiting_cache[route_key] = new_route_waiting
            old_route_Z2 = route_Z2_cache[route_key]
            route_Z2_cache[route_key] = new_route_Z2
            total_Z2 = total_Z2 - old_route_Z2 + new_route_Z2
            
            # Update full waiting cache
            for old_stop_id in list(full_waiting_cache.keys()):
                if old_stop_id in base_route_waiting:
                    del full_waiting_cache[old_stop_id]
            full_waiting_cache.update(new_route_waiting)
            
            # Remove geometry and cached cost fields when route changes
            if "geometry" in route:
                del route["geometry"]
            if "legs" in route:
                del route["legs"]
            # Remove cached cost fields so compute_Z2 recalculates from provider
            if "total_duration" in route:
                del route["total_duration"]
            if "total_distance" in route:
                del route["total_distance"]
    
    # Clean up all geometry and cached cost fields from solution
    for dc_id, routes in solution.get("routes_by_dc", {}).items():
        for route in routes:
            if "geometry" in route:
                del route["geometry"]
            if "legs" in route:
                del route["legs"]
            # Remove cached cost fields so compute_Z2 recalculates from provider
            if "total_duration" in route:
                del route["total_duration"]
            if "total_distance" in route:
                del route["total_distance"]
    
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


# ============================================================================
# Context and Reward Functions for Contextual Thompson Sampling
# ============================================================================

def build_context(
    solution: Dict[str, Any],
    waiting: Dict[str, float],
    stops_by_id: Dict[str, Dict[str, Any]],
    depots: List[Dict[str, Any]],
    time_provider: Callable[[str, str], float],
    baseline_metrics: Dict[str, float],
    Z1_star: float,
    Z2_star: float,
    Z3_star: float,
    Z2_0: float,
    eps: float,
    current_iter: int,
    max_iter: int
) -> np.ndarray:
    """
    Build context vector from current solution.
    
    Context features:
        x[0]: Z1 / Z1_star (normalized max weighted waiting)
        x[1]: Z2 / Z2_star (normalized routing cost)
        x[2]: Z3 / Z3_star (normalized fairness)
        x[3]: cost_slack = max(0, (budget - cost) / baseline_cost)
        x[4]: dc_imbalance = std(route_duration) / mean(route_duration)
        x[5]: tail_ratio = top-10% WW mean / overall WW mean
        x[6]: boundary_ratio = fraction of boundary stops
        x[7]: iter_frac = current_iter / max_iter
    
    Args:
        solution: Current solution
        waiting: Waiting times dict
        stops_by_id: Stop data dict
        depots: List of depot dicts
        time_provider: Time provider function
        baseline_metrics: Dict with Z1_0, Z2_0, Z3_0
        Z1_star, Z2_star, Z3_star: Normalizers
        Z2_0: Baseline cost
        eps: Cost budget parameter
        current_iter: Current iteration number
        max_iter: Maximum iterations
    
    Returns:
        Context vector (8,)
    """
    # Compute current objectives
    Z1 = compute_Z1(waiting, stops_by_id)
    Z2 = compute_Z2(solution, None, time_provider, False)  # Use time, not distance
    Z3 = compute_Z3(waiting, stops_by_id)
    
    # Normalized objectives
    x0 = Z1 / Z1_star if Z1_star > 0 else 1.0
    x1 = Z2 / Z2_star if Z2_star > 0 else 1.0
    x2 = Z3 / Z3_star if Z3_star > 0 else 1.0
    
    # Cost slack: how much budget is remaining
    cost_budget = (1 + eps) * Z2_0
    cost_slack = max(0.0, (cost_budget - Z2) / Z2_0) if Z2_0 > 0 else 0.0
    x3 = cost_slack
    
    # DC imbalance: coefficient of variation of route durations
    route_durations = []
    for dc_id, routes in solution.get("routes_by_dc", {}).items():
        depot_id = dc_id
        for route in routes:
            stop_ids = route.get("ordered_stop_ids", [])
            if not stop_ids:
                continue
            duration = 0.0
            prev_id = depot_id
            for stop_id in stop_ids:
                duration += time_provider(prev_id, stop_id)
                service_time = stops_by_id.get(stop_id, {}).get("service_time_s", 0) or 0
                duration += service_time
                prev_id = stop_id
            duration += time_provider(prev_id, depot_id)  # Return to depot
            route_durations.append(duration)
    
    if route_durations and len(route_durations) > 1:
        mean_dur = np.mean(route_durations)
        std_dur = np.std(route_durations)
        x4 = std_dur / mean_dur if mean_dur > 0 else 0.0
    else:
        x4 = 0.0
    
    # Tail ratio: top-10% weighted waiting / overall mean
    WW = {}
    for stop_id, w_i in waiting.items():
        stop_data = stops_by_id.get(stop_id, {})
        n_i = get_stop_load(stop_data)
        WW[stop_id] = n_i * w_i
    
    if WW:
        ww_values = sorted(WW.values(), reverse=True)
        n_top10 = max(1, len(ww_values) // 10)
        top10_mean = np.mean(ww_values[:n_top10])
        overall_mean = np.mean(ww_values)
        x5 = top10_mean / overall_mean if overall_mean > 0 else 1.0
    else:
        x5 = 1.0
    
    # Boundary ratio: fraction of stops where second-best depot is within 10% of best
    # Approximate by checking if stop is close to multiple depots
    boundary_count = 0
    total_stops = len(waiting)
    
    if total_stops > 0 and len(depots) > 1:
        for stop_id in waiting.keys():
            stop_data = stops_by_id.get(stop_id, {})
            stop_lat = stop_data.get("lat")
            stop_lon = stop_data.get("lon")
            
            if stop_lat is None or stop_lon is None:
                continue
            
            # Compute travel times to all depots
            depot_times = []
            for depot in depots:
                depot_id = depot.get("id", "depot")
                try:
                    time = time_provider(depot_id, stop_id)
                    depot_times.append((depot_id, time))
                except:
                    continue
            
            if len(depot_times) >= 2:
                depot_times.sort(key=lambda x: x[1])
                best_time = depot_times[0][1]
                second_best_time = depot_times[1][1]
                
                # Check if second-best is within 10% of best
                if best_time > 0 and (second_best_time - best_time) / best_time <= 0.1:
                    boundary_count += 1
        
        x6 = boundary_count / total_stops if total_stops > 0 else 0.0
    else:
        x6 = 0.0
    
    # Iteration fraction
    x7 = current_iter / max_iter if max_iter > 0 else 0.0
    
    return np.array([x0, x1, x2, x3, x4, x5, x6, x7], dtype=np.float32)


def compute_reward(
    prev_Z1: float,
    new_Z1: float,
    prev_Z2: float,
    new_Z2: float,
    baseline_Z1: float,
    cost_budget: float,
    lambda_penalty: float = 1.0
) -> float:
    """
    Compute reward for operator selection.
    
    Reward:
        - If cost_new <= cost_budget:
            r = max(0.0, (prev_Z1 - new_Z1) / baseline_Z1)
        - Else:
            r = -lambda_penalty * max(0.0, new_Z2 / cost_budget - 1.0)
    
    Args:
        prev_Z1: Previous Z1 value
        new_Z1: New Z1 value
        prev_Z2: Previous Z2 value
        new_Z2: New Z2 value
        baseline_Z1: Baseline Z1 for normalization
        cost_budget: Cost budget threshold
        lambda_penalty: Penalty multiplier for budget violations
    
    Returns:
        Reward value (clipped to [-1, 1])
    """
    if new_Z2 <= cost_budget:
        # Reward based on Z1 improvement
        if baseline_Z1 > 0:
            reward = max(0.0, (prev_Z1 - new_Z1) / baseline_Z1)
        else:
            reward = 0.0
    else:
        # Penalty for budget violation
        violation = max(0.0, new_Z2 / cost_budget - 1.0)
        reward = -lambda_penalty * violation
    
    # Clip to [-1, 1] for stability
    return np.clip(reward, -1.0, 1.0)
