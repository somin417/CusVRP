"""
Objective functions for VRP fairness optimization.
Uses OSRM/VROOM outputs only (no iNavi).
"""

import logging
from typing import Dict, List, Any, Callable, Optional

logger = logging.getLogger(__name__)


def compute_waiting_times(
    solution: Dict[str, Any],
    stops_by_id: Dict[str, Dict[str, Any]],
    time_provider: Callable[[str, str], float],
    service_time_field: str = "service_time_s"
) -> Dict[str, float]:
    """
    Compute waiting time w_i for each stop by simulating route order.
    
    Args:
        solution: Solution dict with 'routes_by_dc' or 'routes'
        stops_by_id: Dict mapping stop_id to stop data
        time_provider: Function (from_id, to_id) -> travel_time_seconds
        service_time_field: Field name for service time in stop data
    
    Returns:
        Dict mapping stop_id -> waiting_time_seconds (w_i)
    """
    waiting_times = {}
    
    # Get routes from solution
    routes = []
    if "routes_by_dc" in solution:
        for dc_routes in solution["routes_by_dc"].values():
            routes.extend(dc_routes)
    elif "routes" in solution:
        routes = solution["routes"]
    
    # Get depot info
    depots = solution.get("depots", [])
    depot_ids = {d.get("id", "depot") for d in depots} if isinstance(depots, list) else set()
    
    # Build depot lookup from solution
    depots = solution.get("depots", [])
    depot_id_map = {}
    if isinstance(depots, list):
        for depot in depots:
            depot_id_map[depot.get("id", "depot")] = depot
    elif isinstance(depots, dict):
        depot_id_map = depots
    
    # Also check routes_by_dc keys as depot IDs
    if "routes_by_dc" in solution:
        for dc_id in solution["routes_by_dc"].keys():
            if dc_id not in depot_id_map:
                # Try to find depot with this ID
                if isinstance(depots, list):
                    for depot in depots:
                        if depot.get("id") == dc_id:
                            depot_id_map[dc_id] = depot
                            break
    
    for route in routes:
        stop_ids = route.get("ordered_stop_ids", [])
        if not stop_ids:
            continue
        
        # Find depot for this route - try multiple sources
        depot_id = route.get("dc_id") or route.get("depot_id")
        
        # If not found, try to infer from routes_by_dc structure
        if not depot_id and "routes_by_dc" in solution:
            for dc_id, dc_routes in solution["routes_by_dc"].items():
                if route in dc_routes:
                    depot_id = dc_id
                    break
        
        # Last resort: use first depot or "depot"
        if not depot_id:
            if depot_id_map:
                depot_id = list(depot_id_map.keys())[0]
            else:
                depot_id = "depot"
        
        current_time = 0.0  # Start at depot at time 0
        
        # Process route: depot -> stop1 -> stop2 -> ... -> depot
        prev_id = depot_id
        for stop_id in stop_ids:
            if stop_id in depot_ids:
                continue  # Skip if it's a depot
            
            # Travel time from previous location to this stop
            travel_time = time_provider(prev_id, stop_id)
            current_time += travel_time
            
            # Waiting time is arrival time (before service)
            waiting_times[stop_id] = current_time
            
            # Add service time
            stop_data = stops_by_id.get(stop_id, {})
            service_time = stop_data.get(service_time_field, stop_data.get("service_time", 0))
            current_time += service_time
            
            prev_id = stop_id
        
        # Return to depot (optional, not needed for waiting times)
    
    return waiting_times


def compute_Z1(waiting: Dict[str, float], stops_by_id: Dict[str, Dict[str, Any]]) -> float:
    """
    Compute Z1 = max_i (n_i * w_i) - maximum weighted waiting time.
    
    Args:
        waiting: Dict mapping stop_id -> waiting_time_seconds
        stops_by_id: Dict mapping stop_id -> stop data (with households/n_i)
    
    Returns:
        Maximum weighted waiting time
    """
    max_weighted_wait = 0.0
    
    for stop_id, w_i in waiting.items():
        stop_data = stops_by_id.get(stop_id, {})
        # Get n_i from households or meta, default 1
        n_i = stop_data.get("households", stop_data.get("meta", {}).get("n_i", 1))
        if n_i is None or n_i == 0:
            n_i = 1
        
        weighted_wait = n_i * w_i
        max_weighted_wait = max(max_weighted_wait, weighted_wait)
    
    return max_weighted_wait


def compute_Z3(waiting: Dict[str, float], stops_by_id: Dict[str, Dict[str, Any]]) -> float:
    """
    Compute Z3 = sum_i n_i * (w_i - w_bar)^2 - weighted variance.
    
    Args:
        waiting: Dict mapping stop_id -> waiting_time_seconds
        stops_by_id: Dict mapping stop_id -> stop data
    
    Returns:
        Weighted variance of waiting times
    """
    # Compute weighted mean w_bar
    total_weight = 0.0
    weighted_sum = 0.0
    
    for stop_id, w_i in waiting.items():
        stop_data = stops_by_id.get(stop_id, {})
        n_i = stop_data.get("households", stop_data.get("meta", {}).get("n_i", 1))
        if n_i is None or n_i == 0:
            n_i = 1
        
        total_weight += n_i
        weighted_sum += n_i * w_i
    
    if total_weight == 0:
        return 0.0
    
    w_bar = weighted_sum / total_weight
    
    # Compute weighted variance
    weighted_variance = 0.0
    for stop_id, w_i in waiting.items():
        stop_data = stops_by_id.get(stop_id, {})
        n_i = stop_data.get("households", stop_data.get("meta", {}).get("n_i", 1))
        if n_i is None or n_i == 0:
            n_i = 1
        
        weighted_variance += n_i * (w_i - w_bar) ** 2
    
    return weighted_variance


def compute_Z2(
    solution: Dict[str, Any],
    distance_provider: Optional[Callable[[str, str], float]] = None,
    time_provider: Optional[Callable[[str, str], float]] = None,
    use_distance: bool = False
) -> float:
    """
    Compute Z2 = total routing cost (distance or time).
    
    Args:
        solution: Solution dict with routes
        distance_provider: Function (from_id, to_id) -> distance_meters
        time_provider: Function (from_id, to_id) -> time_seconds
        use_distance: If True, use distance; else use time
    
    Returns:
        Total routing cost
    """
    total_cost = 0.0
    
    routes = []
    if "routes_by_dc" in solution:
        for dc_routes in solution["routes_by_dc"].values():
            routes.extend(dc_routes)
    elif "routes" in solution:
        routes = solution["routes"]
    
    # Get depot info
    depots = solution.get("depots", [])
    depot_ids = {d.get("id", "depot") for d in depots} if isinstance(depots, list) else set()
    
    # Build depot lookup
    depot_id_map = {}
    if isinstance(depots, list):
        for depot in depots:
            depot_id_map[depot.get("id", "depot")] = depot
    if "routes_by_dc" in solution:
        for dc_id in solution["routes_by_dc"].keys():
            if dc_id not in depot_id_map and isinstance(depots, list):
                for depot in depots:
                    if depot.get("id") == dc_id:
                        depot_id_map[dc_id] = depot
                        break
    
    for route in routes:
        stop_ids = route.get("ordered_stop_ids", [])
        if not stop_ids:
            continue
        
        # Find depot for this route
        depot_id = route.get("dc_id") or route.get("depot_id")
        if not depot_id and "routes_by_dc" in solution:
            for dc_id, dc_routes in solution["routes_by_dc"].items():
                if route in dc_routes:
                    depot_id = dc_id
                    break
        if not depot_id:
            depot_id = list(depot_id_map.keys())[0] if depot_id_map else "depot"
        
        # If route has total_duration/total_distance, use it
        if use_distance and "total_distance" in route:
            total_cost += route["total_distance"]
        elif not use_distance and "total_duration" in route:
            total_cost += route["total_duration"]
        else:
            # Compute by summing leg costs
            prev_id = depot_id
            for stop_id in stop_ids:
                if stop_id in depot_ids:
                    continue
                
                if use_distance and distance_provider:
                    cost = distance_provider(prev_id, stop_id)
                elif not use_distance and time_provider:
                    cost = time_provider(prev_id, stop_id)
                else:
                    logger.warning(f"Cannot compute Z2: missing provider for {use_distance=}")
                    cost = 0.0
                
                total_cost += cost
                prev_id = stop_id
            
            # Return to depot
            if use_distance and distance_provider:
                total_cost += distance_provider(prev_id, depot_id)
            elif not use_distance and time_provider:
                total_cost += time_provider(prev_id, depot_id)
    
    return total_cost


def compute_combined_Z(
    Z1: float,
    Z2: float,
    Z3: float,
    Z1_star: float,
    Z2_star: float,
    Z3_star: float,
    alpha: float,
    beta: float,
    gamma: float
) -> float:
    """
    Compute combined objective: Z = alpha*(Z1/Z1*) + beta*(Z2/Z2*) + gamma*(Z3/Z3*).
    
    Args:
        Z1, Z2, Z3: Current objective values
        Z1_star, Z2_star, Z3_star: Normalization values
        alpha, beta, gamma: Weight coefficients
    
    Returns:
        Combined objective value
    """
    # Avoid division by zero
    Z1_norm = Z1 / Z1_star if Z1_star > 0 else Z1
    Z2_norm = Z2 / Z2_star if Z2_star > 0 else Z2
    Z3_norm = Z3 / Z3_star if Z3_star > 0 else Z3
    
    return alpha * Z1_norm + beta * Z2_norm + gamma * Z3_norm

