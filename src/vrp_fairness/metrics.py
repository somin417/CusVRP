"""
Metrics calculation for VRP solutions.
"""

import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RouteMetrics:
    """Metrics for a single route."""
    vehicle_id: int
    stop_ids: List[str]
    arrival_times: Dict[str, float]  # stop_id -> arrival time
    waiting_times: Dict[str, float]  # stop_id -> waiting time
    total_duration: float
    total_distance: float


@dataclass
class SolutionMetrics:
    """Overall solution metrics."""
    # Waiting time statistics
    W_max: float  # Maximum waiting time
    W_mean: float  # Mean waiting time
    W_p95: float  # 95th percentile waiting time
    W_top10_mean: float  # Mean of top 10% waiting times
    
    # Cost metrics
    total_cost: float  # Total travel time/distance
    total_distance: float
    total_duration: float
    
    # Balance metrics
    driver_balance: float  # Max - min route duration
    route_durations: List[float]  # Duration per route
    
    # Counts
    num_vehicles_used: int
    num_stops_served: int


def calculate_route_metrics(
    route: Dict[str, Any],
    stops_dict: Dict[str, Dict[str, Any]],
    time_matrix: Dict[tuple, int],
    depot_id: str = "Depot"
) -> RouteMetrics:
    """
    Calculate metrics for a single route.
    
    Args:
        route: Route dict with 'ordered_stop_ids', 'total_duration', 'total_distance'
        stops_dict: Dictionary mapping stop_id to stop data (with lat, lon, service_time)
        time_matrix: Dictionary mapping (stop1_id, stop2_id) to travel time
        depot_id: ID of the depot
    
    Returns:
        RouteMetrics object
    """
    stop_ids = route["ordered_stop_ids"]
    arrival_times = {}
    waiting_times = {}
    
    current_time = 0.0  # Departure time from depot is 0
    
    for i in range(len(stop_ids) - 1):
        from_stop = stop_ids[i]
        to_stop = stop_ids[i + 1]
        
        # Travel time
        travel_time = time_matrix.get((from_stop, to_stop), 0)
        current_time += travel_time
        
        # If not depot, record arrival and add service time
        if to_stop != depot_id:
            arrival_times[to_stop] = current_time
            waiting_times[to_stop] = current_time
            
            # Add service time
            service_time = stops_dict.get(to_stop, {}).get("service_time", 300)
            current_time += service_time
    
    return RouteMetrics(
        vehicle_id=route.get("vehicle_id", 0),
        stop_ids=stop_ids,
        arrival_times=arrival_times,
        waiting_times=waiting_times,
        total_duration=route.get("total_duration", 0),
        total_distance=route.get("total_distance", 0.0)
    )


def calculate_solution_metrics(
    routes: List[Dict[str, Any]],
    stops_dict: Dict[str, Dict[str, Any]],
    time_matrix: Dict[tuple, int],
    depot_id: str = "Depot"
) -> SolutionMetrics:
    """
    Calculate overall solution metrics.
    
    Args:
        routes: List of route dictionaries
        stops_dict: Dictionary mapping stop_id to stop data
        time_matrix: Dictionary mapping (stop1_id, stop2_id) to travel time
        depot_id: ID of the depot
    
    Returns:
        SolutionMetrics object
    """
    all_waiting_times = []
    route_durations = []
    total_distance = 0.0
    total_duration = 0.0
    all_stop_ids = set()
    
    for route in routes:
        route_metrics = calculate_route_metrics(route, stops_dict, time_matrix, depot_id)
        route_durations.append(route_metrics.total_duration)
        total_distance += route_metrics.total_distance
        total_duration += route_metrics.total_duration
        
        for stop_id, wait_time in route_metrics.waiting_times.items():
            all_waiting_times.append(wait_time)
            all_stop_ids.add(stop_id)
    
    if not all_waiting_times:
        # No stops served
        return SolutionMetrics(
            W_max=0.0, W_mean=0.0, W_p95=0.0, W_top10_mean=0.0,
            total_cost=0.0, total_distance=0.0, total_duration=0.0,
            driver_balance=0.0, route_durations=[],
            num_vehicles_used=0, num_stops_served=0
        )
    
    # Waiting time statistics
    W_max = max(all_waiting_times)
    W_mean = statistics.mean(all_waiting_times)
    
    sorted_waiting = sorted(all_waiting_times, reverse=True)
    n_top10 = max(1, len(sorted_waiting) // 10)
    W_top10_mean = statistics.mean(sorted_waiting[:n_top10])
    
    # 95th percentile
    if len(sorted_waiting) > 0:
        p95_idx = int(0.95 * len(sorted_waiting))
        W_p95 = sorted_waiting[p95_idx] if p95_idx < len(sorted_waiting) else sorted_waiting[-1]
    else:
        W_p95 = 0.0
    
    # Driver balance (max - min route duration)
    driver_balance = max(route_durations) - min(route_durations) if route_durations else 0.0
    
    return SolutionMetrics(
        W_max=W_max,
        W_mean=W_mean,
        W_p95=W_p95,
        W_top10_mean=W_top10_mean,
        total_cost=total_duration,  # Using duration as cost
        total_distance=total_distance,
        total_duration=total_duration,
        driver_balance=driver_balance,
        route_durations=route_durations,
        num_vehicles_used=len(routes),
        num_stops_served=len(all_stop_ids)
    )


def metrics_to_dict(metrics: SolutionMetrics) -> Dict[str, Any]:
    """Convert SolutionMetrics to dictionary for JSON serialization."""
    return {
        "W_max": metrics.W_max,
        "W_mean": metrics.W_mean,
        "W_p95": metrics.W_p95,
        "W_top10_mean": metrics.W_top10_mean,
        "total_cost": metrics.total_cost,
        "total_distance": metrics.total_distance,
        "total_duration": metrics.total_duration,
        "driver_balance": metrics.driver_balance,
        "route_durations": metrics.route_durations,
        "num_vehicles_used": metrics.num_vehicles_used,
        "num_stops_served": metrics.num_stops_served
    }

