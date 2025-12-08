"""
Stop-to-depot assignment using OSRM travel time.
"""

import logging
from typing import List, Dict, Callable, Optional
from .data import Stop
from .inavi import get_leg_route, iNaviCache

logger = logging.getLogger(__name__)


def create_osrm_time_provider(cache: Optional[iNaviCache] = None) -> Callable[[float, float, float, float], float]:
    """
    Create a time provider function that uses OSRM (via get_leg_route).
    
    Args:
        cache: Optional iNaviCache instance for caching
    
    Returns:
        Function (lat1, lon1, lat2, lon2) -> travel_time_seconds
    """
    if cache is None:
        cache = iNaviCache()
    
    def time_provider(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Get OSRM travel time between two points."""
        leg = get_leg_route(
            origin=(lat1, lon1),
            dest=(lat2, lon2),
            cache=cache
        )
        return leg.travel_time_s
    
    return time_provider


def assign_stops_to_depots(
    stops: List[Stop],
    depots: List[Dict[str, any]],
    time_provider: Callable[[float, float, float, float], float],
    cache: Optional[Dict] = None
) -> Dict[str, List[Stop]]:
    """
    Assign each stop to the nearest depot by OSRM travel time.
    
    Args:
        stops: List of Stop objects
        depots: List of depot dicts with 'id', 'lat', 'lon'
        time_provider: Function (lat1, lon1, lat2, lon2) -> travel_time_seconds
        cache: Optional cache dict for time lookups
    
    Returns:
        Dictionary mapping depot_id -> list of assigned stops
    """
    if cache is None:
        cache = {}
    
    assignments: Dict[str, List[Stop]] = {depot["id"]: [] for depot in depots}
    
    logger.info(f"Assigning {len(stops)} stops to {len(depots)} depots using OSRM travel time")
    
    for stop in stops:
        best_depot_id = None
        best_time = float('inf')
        
        for depot in depots:
            cache_key = (depot["id"], stop.id)
            if cache_key in cache:
                travel_time = cache[cache_key]
            else:
                travel_time = time_provider(
                    depot["lat"], depot["lon"],
                    stop.lat, stop.lon
                )
                cache[cache_key] = travel_time
            
            if travel_time < best_time:
                best_time = travel_time
                best_depot_id = depot["id"]
        
        if best_depot_id:
            assignments[best_depot_id].append(stop)
            logger.debug(f"Assigned stop {stop.id} to depot {best_depot_id} (time: {best_time:.1f}s)")
        else:
            logger.warning(f"Failed to assign stop {stop.id} to any depot")
    
    # Log assignments
    for depot_id, assigned_stops in assignments.items():
        logger.info(f"  Depot {depot_id}: {len(assigned_stops)} stops")
    
    return assignments

